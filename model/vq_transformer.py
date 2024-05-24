import torch
import torch.nn as nn
import math
import numpy as np


class VQTransformer(nn.Module):
    def __init__(self, vqvae, transformer):
        super(VQTransformer, self).__init__()
        self.vqvae = vqvae
        self.predictor = transformer

    def forward(self, x, action=None):
        """
        x:          bs * c * t * h * w
        action:     bs * t * n or bs * n or None?
        """

    def forward_on_indices(self, encoded_x, actions):
        """
        x:          bs * (seq_len = t * h' * w')
        action:     bs * t * n or bs * n or None?
        """
        pass

    def cross_entropy_loss(self, cond_x, future_x, actions):
        """
        x:          bs * c * t * h * w (image itself)
        action:     bs * t * n or bs * n or None?
        """
        pass
