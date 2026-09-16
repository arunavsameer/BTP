"""5x5 heatmap decoder and occupancy head.

Trained with the teacher (Stage A), then frozen and reused by the student so Z
has to carry collision meaning rather than a flexible decoder covering for it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapDecoder(nn.Module):
    """Z (grid x grid x z_channels) -> H (grid x grid) in [0, 1].

    3x3 convs let a cell use its neighbors (corridor vs object) instead of
    forcing every Z vector to be collision-complete in isolation.
    """

    def __init__(self, z_channels: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, z_channels, grid, grid] -> [B, grid, grid] in [0, 1]."""
        h = self.net(z)
        h = torch.sigmoid(h)
        return h.squeeze(1)


class OccupancyHead(nn.Module):
    """Last ``occ_channels`` of Z -> per-cell occupancy logits [B, grid, grid]."""

    def __init__(self, occ_channels: int = 4):
        super().__init__()
        self.occ_channels = int(occ_channels)
        self.proj = nn.Conv2d(self.occ_channels, 1, kernel_size=1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z[:, -self.occ_channels :]).squeeze(1)

    def prob(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(z))
