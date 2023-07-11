#!/usr/bin/env python3
# Author: Joel Ye

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskMode(Enum):
    neuron = "neuron"
    timestep = "timestep"


class Masker(nn.Module):
    def __init__(
        self, mask_mode: MaskMode = MaskMode.timestep, mask_ratio: float = 0.2
    ):
        super(Masker, self).__init__()

        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio

    def forward(self, X):
        assert X.ndim == 3
        if self.mask_mode == MaskMode.neuron:
            # We mask one neuron at a time
            # Neurons are along dimension 1
            mask = torch.rand(1, X.shape[1], 1) < self.mask_ratio
            return torch.expand_copy(mask, X.shape)
        elif self.mask_mode == MaskMode.timestep:
            # We mask one timestep at a time
            # Timesteps are along dimension 2
            mask = torch.rand(1, 1, X.shape[2]) < self.mask_ratio
            return torch.expand_copy(mask, X.shape)
