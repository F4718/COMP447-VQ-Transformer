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
        action:     bs *
        """
