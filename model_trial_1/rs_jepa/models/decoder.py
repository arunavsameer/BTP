"""Heatmap decoder: Z (C x grid x grid) -> H (grid x grid) in [0, 1]."""

from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapDecoder(nn.Module):
    def __init__(self, z_channels: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        h = torch.sigmoid(h)
        return h.squeeze(1)


class HeatmapDelta(nn.Module):
    """Predict ΔH in [-1, 1] from a latent residual (copy + Δ → future heat)."""

    def __init__(self, z_channels: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z)).squeeze(1)
