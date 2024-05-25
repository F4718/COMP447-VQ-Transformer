import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class VQTransformer(nn.Module):
    def __init__(self, vqvae, transformer, t, hw_prime):
        super(VQTransformer, self).__init__()
        self.vqvae = vqvae
        self.predictor = transformer
        self.t = t
        self.hw_prime = hw_prime

    def forward(self, x, action=None):
        """
        x:          bs * c * t * h * w
        action:     bs * t * n or bs * n or None?
        """

    def encode_to_indices(self, x):
        # x: bs * c * t * h * w
        bs = x.shape[0]
        x = self.vqvae.encode_code(x).view(bs, -1)
        return x  # bs * (seq_len = t * h' * w')

    def forward_on_indices(self, encoded_x, actions):
        """
        encoded_x:   bs * (seq_len = t * h' * w')
        actions:     bs * t * n or bs * n or None?
        """
        pred_logits = self.predictor(encoded_x, actions)    # bs * seq_len * vocab_size
        pred_indices = torch.argmax(pred_logits, dim=-1)    # bs * seq_len
        return pred_indices, pred_logits

    def decode_from_indices(self, h):
        # h: bs * seq_len
        bs = h.shape[0]
        h = h.view(bs, self.t, self.hw_prime, self.hw_prime)
        x = self.vqvae.decode_code(h)   # bs * t * h * w * c
        return x

    def cross_entropy_loss(self, cond_x, future_x, actions):
        """
        cond_x, future_x:    bs * c * t * h * w (image itself)
        actions:             bs * t * n or bs * n or None?
        """
        encoded_past = self.encode_to_indices(cond_x)
        encoded_future = self.encode_to_indices(future_x)

        pred_indices, pred_logits = self.forward_on_indices(encoded_past, actions)
        loss = F.cross_entropy(pred_logits, encoded_future)
        return loss
