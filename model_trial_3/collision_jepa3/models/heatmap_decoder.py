"""Current heatmap decoder and per-horizon tanh delta heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapDecoder(nn.Module):
    def __init__(self, z_channels: int = 32, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z)).squeeze(1)


class HeatmapDelta(nn.Module):
    def __init__(self, z_channels: int = 32, hidden: int = 48):
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


def copy_residual_future(h_now: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Hhat_h = clamp(sg(Hhat_now) + Delta_h, 0, 1). Shared by all horizons."""
    return (h_now.detach() + delta).clamp(0.0, 1.0)
