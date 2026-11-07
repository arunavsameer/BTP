"""Tiny 1×1 head: 16×16 fine features → 5×5 counterfactual variance in [0, 0.25]."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceHead(nn.Module):
    """One conv on the frozen fine map. Max-pool matches G's heatmap pooling."""

    def __init__(self, in_channels: int = 128, out_grid: int = 5, max_var: float = 0.25):
        super().__init__()
        self.out_grid = int(out_grid)
        self.max_var = float(max_var)
        self.conv = nn.Conv2d(int(in_channels), 1, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, -4.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        v_fine = self.max_var * torch.sigmoid(self.conv(feat))
        if v_fine.shape[-2] == self.out_grid and v_fine.shape[-1] == self.out_grid:
            return v_fine.squeeze(1)
        return F.adaptive_max_pool2d(v_fine, self.out_grid).squeeze(1)
