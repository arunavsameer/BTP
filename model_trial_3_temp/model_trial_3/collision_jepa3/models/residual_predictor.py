"""Per-horizon residual latent predictor: Zhat_h = Z_t + f_h(Z_t)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualPredictor(nn.Module):
    def __init__(self, z_channels: int = 32):
        super().__init__()
        hid = 96
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, z_channels, kernel_size=1),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return z_t + self.net(z_t)
