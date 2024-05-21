import torch
from PIL import Image
import matplotlib.pyplot as plt


def reconstruct(epoch_idx, num_samples):
    for i in range(num_samples):
        t = 2
        x = torch.zeros(t, 3, 4, 4) - 0.5
        x_tilde = torch.ones(t, 3, 4, 4) / 2

        # become t * c * h * w
        x = x.detach().cpu()
        x_tilde = x_tilde.detach().cpu()

        # scale again (from -1 to 1 to 0 to 255)
        x = (((x + 1) * 128) // 1).clamp(1, 255).to(torch.uint8)
        x_tilde = (((x_tilde + 1) * 128) // 1).clamp(1, 255).to(torch.uint8)
        # Concatenate all x and x_tilde images side by side
        x_concat = torch.cat([x[i] for i in range(t)], dim=1)  # Horizontally
        x_tilde_concat = torch.cat([x_tilde[i] for i in range(t)], dim=1)  # Horizontally

        # Concatenate x and x_tilde vertically
        final_image = torch.cat((x_concat, x_tilde_concat), dim=2)  # Vertically

        # Convert to PIL image to save
        final_image = final_image.permute(1, 2, 0)  # Change from C*H*W to H*W*C for PIL
        new_image = torch.zeros(final_image.shape[0], final_image.shape[1], 3).to(torch.uint8)
        for i in range(final_image.shape[2]):
            print(i)
            print(final_image[:, :, -1 - i])
            new_image[:, :, 2 - i] = final_image[:, :, -1 - i]
        print(new_image.shape)
        plt.imshow(final_image.numpy())
        plt.axis('off')  # Turn off axis numbers and ticks

        plt.show()
        plt.imshow(new_image.numpy())
        plt.axis('off')  # Turn off axis numbers and ticks

        plt.show()

        new_image = Image.fromarray(new_image.numpy(), 'RGB')

        # Save the image
        new_image.save('concatenated_image.png')
        plt.imshow(new_image)
        plt.axis('off')  # Turn off axis numbers and ticks

        plt.show()


reconstruct(1, 1)
