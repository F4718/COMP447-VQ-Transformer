from model import ActionConditionedTransformer, VQVAE3D
import json
import os
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.optim as optim
from model import VQTransformer
from utils import load_dataset, normalize_data, reshape_data, save_transformer_losses, save_as_gif, save_as_image
from data_preprocess.preprocess_data import create_xy_bins
from torch.utils.data import DataLoader
from dataloader.dataset import TransformerCollateFunction
import progressbar
import matplotlib.pyplot as plt

with open("config_vq_transformer_2.json", "r") as file:
    opt = json.load(file)
    opt = SimpleNamespace(**opt)

# load model and continue training from checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
saved_model = torch.load('%s/model.pth' % opt.model_dir, map_location=device)
epochs = opt.epochs
optimizer = opt.optimizer
model_dir = opt.model_dir
opt = saved_model['opt']

# load the saved vqvae with the model
opt_vqvae = saved_model['opt_vqvae']
vqvae_model = saved_model['vqvae']
for param in vqvae_model.parameters():
    param.requires_grad = False

opt.epochs = epochs
opt.optimizer = optimizer
opt.model_dir = model_dir
predictor = saved_model["transformer"]

down_to = opt_vqvae.image_size / (opt_vqvae.downsample ** 2)
vq_transformer = VQTransformer(vqvae_model, predictor, t=int(opt_vqvae.time_window), hw_prime=int(down_to))

min_delta, max_delta = -2, 2
bin_lims_x, bin_lims_y = create_xy_bins(min_delta, max_delta, (opt.num_x_bins, opt.num_y_bins))
collate_fn = TransformerCollateFunction(bin_lims_x, bin_lims_y)

train_data, test_data = load_dataset(opt, is_vqvae=False)
train_loader = DataLoader(train_data,
                          batch_size=opt.batch_size,
                          shuffle=False,
                          drop_last=True,
                          pin_memory=True,
                          collate_fn=collate_fn)
test_loader = DataLoader(test_data,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True,
                         collate_fn=collate_fn)

dtype1 = torch.FloatTensor
dtype2 = torch.IntTensor


def get_training_batch():
    while True:
        for sequence in train_loader:
            x, actions = normalize_data(dtype1, dtype2, sequence)
            # x: seq_len * bs * c * h * w
            # actions: seq_len * bs * 8
            batch = reshape_data(opt, x, actions)
            # returns x, actions
            # x: num_group, bs, c, t, h, w
            # actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
            yield batch


training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            x, actions = normalize_data(dtype1, dtype2, sequence)
            # x: seq_len * bs * c * h * w
            # actions: seq_len * bs * 8
            batch = reshape_data(opt, x, actions)
            # returns x, actions
            # x: num_group, bs, c, t, h, w
            # actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
            yield batch


testing_batch_generator = get_testing_batch()

x, actions = next(training_batch_generator)


def sample(x, actions):
    with torch.no_grad():

        # take the first batch elements
        x = x[:, 0].unsqueeze(1)  # num_group, 1, c, t, h, w
        actions = actions[:, 0].unsqueeze(1)

        # form gt_sequence
        num_g, _, c, t, h, w = x.shape
        gt_sequence = x.cpu().permute(0, 1, 3, 4, 5, 2).squeeze(1).view(-1, h, w, c)  # (num_groups * t) * h * w * c
        gt_sequence = gt_sequence.clamp(-1, 1)
        gt_sequence = (((gt_sequence + 1) / 2 * 255) // 1)

        # get the initial seq element
        initial_x = x[0].to(device)
        encoded_x = vq_transformer.encode_to_indices(initial_x)  # bs * (seq_len = t * h' * w')

        # flatten the quantized indices
        pred_sequences = list()
        pred_sequences.append(encoded_x)
        # predict next sequences based on the predictions, not gt of course
        for group_num in range(len(actions)):
            past_actions = actions[group_num].to(device)
            encoded_future_x, _ = vq_transformer.forward_on_indices(encoded_x, past_actions)
            pred_sequences.append(encoded_future_x)
            encoded_x = encoded_future_x

        print(pred_sequences)
        reconstructions = list()
        for seq in pred_sequences:
            rec = vq_transformer.decode_from_indices(seq)  # bs * t * h * w * c (bs is 1)
            rec = torch.from_numpy(rec)
            reconstructions.append(rec.squeeze(0))

        reconstructions = torch.concat(reconstructions, dim=0)  # (num_groups * t) * h * w * c
        reconstructions = reconstructions.clamp(-1, 1)
        reconstructions = (((reconstructions + 1) / 2 * 255) // 1)

x = x[:, 1]  # num_group, 1, c, t, h, w
print(x.shape)
# print(x.shape)
vq_transformer.eval()
# print(vqvae_model.encode_code(x).view(x.shape[0], -1))
# sample(x, actions)
e1 = vq_transformer.encode_to_indices(x)
d1 = vq_transformer.decode_from_indices(e1)
d1 = torch.from_numpy(d1)
bs, t, h, w, c = d1.shape
d1 = d1.view(-1, h, w, c)
d1 = d1.clamp(-1, 1)
d1 = (((d1 + 1) / 2 * 255) // 1)
d1 = d1.numpy()
# Number of frames
t = d1.shape[0]
fig, axes = plt.subplots(1, t, figsize=(t * 5, 5))  # Adjust the figure size as needed
for i in range(t):
    ax = axes[i]
    ax.imshow(d1[i])
    ax.axis('off')
    ax.set_title(f'Frame {i + 1}')
plt.tight_layout()
plt.show()


vq_transformer.train()
# print(vqvae_model.encode_code(x).view(x.shape[0], -1))
# sample(x, actions)
e3 = vq_transformer.encode_to_indices(x)
d3 = vq_transformer.decode_from_indices(e3)
d3 = torch.from_numpy(d3)
bs, t, h, w, c = d3.shape
d3 = d3.view(-1, h, w, c)
d3 = d3.clamp(-1, 1)
d3 = (((d3 + 1) / 2 * 255) // 1)
d3 = d3.numpy()
t = d3.shape[0]
fig, axes = plt.subplots(1, t, figsize=(t * 5, 5))  # Adjust the figure size as needed
for i in range(t):
    ax = axes[i]
    ax.imshow(d3[i])
    ax.axis('off')
    ax.set_title(f'Frame {i + 1}')
plt.tight_layout()
plt.show()








