import torch
import torch.optim as optim
import json
import os
import random
import numpy as np
import progressbar
from types import SimpleNamespace
from model import VQ3DDecoder, VQ3DEncoder, CodeBook, VQVAE3D
from dataloader.dataset import vqvae_collate_fn
from utils import load_dataset, normalize_frames, save_vq_image, save_vq_losses
from torch.utils.data import DataLoader
from model import VQTransformer

def main():
    with open("config_vqvae_2.json", "r") as file:
        opt = json.load(file)
        opt = SimpleNamespace(**opt)

    with open("config_vq_transformer_2.json", "r") as file:
        opt2 = json.load(file)
        opt2 = SimpleNamespace(**opt2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved_transformer = torch.load('%s/model.pth' % opt2.model_dir, map_location=device)
    opt_vqvae = saved_transformer['opt_vqvae']
    print(opt_vqvae)
    vqvae_model = saved_transformer['vqvae']
    for param in vqvae_model.parameters():
        param.requires_grad = False
    predictor = saved_transformer["transformer"]
    down_to = opt_vqvae.image_size / (opt_vqvae.downsample ** 2)
    vq_transformer = VQTransformer(vqvae_model, predictor, t=int(opt_vqvae.time_window), hw_prime=int(down_to))

    if opt.model_dir != '':
        # load model and continue training from checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        saved_model = torch.load('%s/model.pth' % opt.model_dir, map_location=device)
        epochs = opt.epochs
        optimizer = opt.optimizer
        model_dir = opt.model_dir
        opt = saved_model['opt']
        print(opt)
        opt.epochs = epochs
        opt.optimizer = optimizer
        opt.model_dir = model_dir
        opt.log_dir = '%s/continued' % opt.log_dir

    else:
        down_to = opt.image_size / (opt.downsample**2)
        name = f'vqvae-{opt.image_size}x{opt.image_size}-down_to={down_to}x{down_to}-' \
               f'time_wind={opt.time_window}-emb_size={opt.vq_emb_size}-codebook_size={opt.codebook_num_emb}-' \
               f'n_res={opt.n_res}-bias={opt.conv_bias}'
        dataset = "road_only" if opt.channels == 1 else "with_objects"
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, dataset + "_" + opt.train_on, name)


    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

    dtype = torch.FloatTensor
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = opt.seed
    if torch.cuda.is_available():
        print("Random Seed: ", seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dtype = torch.cuda.FloatTensor
    else:
        print("CUDA WAS NOT AVAILABLE!")
        print("Didn't seed...")

    print(opt)

    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % opt.optimizer)

    if opt.model_dir != '':
        encoder = saved_model['encoder']
        codebook = saved_model['codebook']
        decoder = saved_model['decoder']
    else:
        encoder = VQ3DEncoder(opt.channels, opt.vq_emb_size, opt.time_window, opt.downsample, opt.n_res, opt.conv_bias)
        codebook = CodeBook(opt.codebook_num_emb, opt.vq_emb_size)
        decoder = VQ3DDecoder(opt.vq_emb_size, opt.channels, opt.time_window, opt.downsample, opt.n_res, opt.conv_bias)
        # init weights maybe?

    model = VQVAE3D(encoder, decoder, codebook).to(device)
    optimizer = opt.optimizer(model.parameters(), lr=opt.lr)

    train_data, test_data = load_dataset(opt, is_vqvae=True)
    collate_fn = vqvae_collate_fn
    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True,
                             collate_fn=collate_fn)

    def get_training_batch():
        while True:
            for sequence in train_loader:
                # data between -1 and 1
                batch = (normalize_frames(dtype, sequence) - 0.5) * 2
                # returns frames
                yield batch

    training_batch_generator = get_training_batch()

    def get_testing_batch():
        while True:
            for sequence in test_loader:
                # data between -1 and 1
                batch = (normalize_frames(dtype, sequence) - 0.5) * 2
                # returns frames
                yield batch

    testing_batch_generator = get_testing_batch()

    def train(x):
        losses = model.loss(x)
        loss = losses["loss"]
        rec_loss = losses["recon_loss"]
        vq_loss = losses["vq_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return rec_loss.item(), vq_loss.item()

    def train_epoch():
        epoch_size = len(train_loader)
        recon_losses = list()
        vq_losses = list()
        progress = progressbar.ProgressBar(max_value=epoch_size).start()
        for i in range(epoch_size):
            progress.update(i + 1)
            x = next(training_batch_generator)
            rec_loss, vq_loss = train(x)
            recon_losses.append(rec_loss)
            vq_losses.append(vq_loss)
        return recon_losses, vq_losses

    def evaluate():
        epoch_size = len(test_loader)
        recon_losses = list()
        vq_losses = list()
        progress = progressbar.ProgressBar(max_value=epoch_size).start()
        with torch.no_grad():
            for i in range(epoch_size):
                progress.update(i + 1)
                x = next(testing_batch_generator)
                loss = model.loss(x)
                rec_loss, vq_loss = loss["recon_loss"].item(), loss["vq_loss"].item()
                recon_losses.append(rec_loss)
                vq_losses.append(vq_loss)
        return np.mean(np.array(recon_losses)), np.mean(np.array(vq_losses))

    def reconstruct(epoch_idx, num_sample):
        with torch.no_grad():
            for i in range(num_sample):
                # bs * c * t * h * w
                x = next(testing_batch_generator)[0].unsqueeze(0)
                x_tilde, _ = model(x)

                model.eval()
                vq_transformer.eval()
                """
                
                
                for name, param in model.encoder.named_parameters():
                    print(f"Name: {name}")
                    print(f"Type: {type(param)}")
                    print(f"Shape: {param.shape}")
                    print(f"Values: \n{param.data}")
                    print()

                print("XXXXXXXXX", end="\n\n\n\n\n")

                for name, param in vq_transformer.vqvae.encoder.named_parameters():
                    print(f"Name: {name}")
                    print(f"Type: {type(param)}")
                    print(f"Shape: {param.shape}")
                    print(f"Values: \n{param.data}")
                    print()
                """
                compare_models(model.encoder, vq_transformer.vqvae.encoder)

                enc_normal = model.encode_code(x).flatten()
                enc_transformer = vq_transformer.encode_to_indices(x)
                print(enc_normal)
                print(enc_transformer)
                assert False


                # become c * t * h * w
                x = x.detach().cpu().squeeze(0)
                x_tilde = x_tilde.detach().cpu().squeeze(0)

                # scale again (from -1 to 1 to 0 to 255)
                x = (((x + 1) * 128) // 1).clamp(0, 255)
                x_tilde = (((x_tilde + 1) * 128) // 1).clamp(0, 255)

                out_file = os.path.join(opt.log_dir, f"gen/epoch_{epoch_idx}-gen_{i + 1}.png")
                save_vq_image(out_file, x, x_tilde)

    recons_minibatch_loss = list()
    vq_minibatch_loss = list()
    recons_test_loss = list()
    vq_test_loss = list()
    for epoch in range(1, opt.epochs + 1):
        print(f"Epoch {epoch} / {opt.epochs}")
        model.train()
        # train epoch returns results of all mini-batches
        rec_batch_ls, vq_batch_ls = train_epoch()
        recons_minibatch_loss.append(rec_batch_ls)
        vq_minibatch_loss.append(vq_batch_ls)

        model.eval()
        # evaluate returns the mean results
        t_recons_loss, t_vq_loss = evaluate()
        recons_test_loss.append(t_recons_loss)
        vq_test_loss.append(t_vq_loss)

        loss_file = os.path.join(opt.log_dir, "loss/out.json")
        save_vq_losses(loss_file, recons_minibatch_loss, vq_minibatch_loss, recons_test_loss, vq_test_loss)

    reconstruct(101, 10)

def compare_models(model1, model2):
    model1_items = {**{name: param for name, param in model1.named_parameters()},
                    **{name: buff for name, buff in model1.named_buffers()}}
    model2_items = {**{name: param for name, param in model2.named_parameters()},
                    **{name: buff for name, buff in model2.named_buffers()}}

    models_identical = True

    # Check model1 items against model2
    for name, item1 in model1_items.items():
        if name in model2_items:
            item2 = model2_items[name]
            if not torch.equal(item1, item2):
                print(f"Mismatch found at: {name}")
                print(f"Model1 item ({name}): \n{item1}")
                print(f"Model2 item ({name}): \n{item2}")
                models_identical = False
        else:
            print(f"Parameter/buffer {name} found in Model 1 not in Model 2")
            models_identical = False

    # Check for any extra items in model2 not present in model1
    for name in model2_items:
        if name not in model1_items:
            print(f"Parameter/buffer {name} found in Model 2 not in Model 1")
            models_identical = False

    if models_identical:
        print("Models are identical.")
    else:
        print("Models are not identical.")

if __name__ == '__main__':
    main()
