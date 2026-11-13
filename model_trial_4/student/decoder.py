"""Pair-CF heatmap decoder (frozen). Copied so Trial 4 does not import Trial 3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _spatial_heatmap_tower(z_channels: int, hidden: int, dilated: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(z_channels, hidden, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    ]
    if dilated:
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.Conv2d(hidden, 1, kernel_size=1))
    return nn.Sequential(*layers)


class HeatmapDecoder(nn.Module):
    def __init__(
        self,
        z_channels: int = 128,
        hidden: int = 48,
        dilated: bool = True,
        out_grid: int = 5,
        pool: str = "max",
    ):
        super().__init__()
        self.dilated = bool(dilated)
        self.out_grid = int(out_grid)
        self.pool = str(pool)
        self.net = _spatial_heatmap_tower(z_channels, hidden, self.dilated)

    def forward_fine_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Layer-4 1×1 conv logits at 16×16, before AdaptiveMaxPool2d(5)."""
        return self.net(z)

    def forward_with_logits(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_fine = self.forward_fine_logits(z)
        if logits_fine.shape[-2] != self.out_grid or logits_fine.shape[-1] != self.out_grid:
            pooled = F.adaptive_max_pool2d(logits_fine, self.out_grid)
        else:
            pooled = logits_fine
        h = torch.sigmoid(pooled).squeeze(1)
        return h, pooled.squeeze(1)

    def forward_checkpoints(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Student training heads: 16×16 logits + 5×5 probability + 5×5 logits."""
        logits_fine = self.forward_fine_logits(z)
        if logits_fine.shape[-2] != self.out_grid or logits_fine.shape[-1] != self.out_grid:
            pooled = F.adaptive_max_pool2d(logits_fine, self.out_grid)
        else:
            pooled = logits_fine
        p = torch.sigmoid(pooled).squeeze(1)
        return {
            "Z_16": logits_fine,
            "P_5": p,
            "Hhat_1s": p,
            "Hhat_1s_logits": pooled.squeeze(1),
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.forward_with_logits(z)
        return h
