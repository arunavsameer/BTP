"""Stage 0 baseline: tiny CNN -> future 5x5 heatmap directly (no V-JEPA, no Z).

This proves whether cheap RGB frames + feature differences can beat the copy
baseline before we invest in the V-JEPA teacher. Same front-end as the student so
the numbers are comparable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tiny_cnn import TinyCNN


class Stage0Model(nn.Module):
    def __init__(self, width: int = 32, feature_grid: int = 5):
        super().__init__()
        self.cnn = TinyCNN(width=width, feature_grid=feature_grid)
        c = self.cnn.feat_channels
        # Input is [F3, D1, D2] concatenated on channels.
        self.head = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, kernel_size=1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, n_frames(=3), 3, H, W] -> future heatmap [B, grid, grid]."""
        b, n, c, h, w = frames.shape
        feats = self.cnn(frames.reshape(b * n, c, h, w))
        feats = feats.reshape(b, n, feats.shape[1], feats.shape[2], feats.shape[3])
        f1, f2, f3 = feats[:, 0], feats[:, 1], feats[:, 2]
        d1 = f3 - f2
        d2 = f3 - f1
        x = torch.cat([f3, d1, d2], dim=1)
        out = self.head(x)
        return torch.sigmoid(out).squeeze(1)
