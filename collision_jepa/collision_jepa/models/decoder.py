"""5x5 heatmap decoder: Z (grid x grid x z_channels) -> H (grid x grid) in [0, 1].

Deliberately tiny. It is trained with the teacher (Stage A) and then *frozen* and
reused by the student, so the student is forced to make its own Z meaningful rather
than relying on a decoder that can compensate for a bad Z.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapDecoder(nn.Module):
    def __init__(self, z_channels: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, z_channels, grid, grid] -> [B, grid, grid] in [0, 1]."""
        h = self.net(z)
        h = torch.sigmoid(h)
        return h.squeeze(1)
