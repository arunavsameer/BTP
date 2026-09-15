"""Tiny depthwise-separable CNN backbone.

Maps one RGB frame [B, 3, H, W] to a feature map pooled to feature_grid x feature_grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, groups=groups, bias=False)
        # GroupNorm: more stable than BatchNorm on small batches / mixed scenes.
        ng = 8 if cout >= 8 else 1
        while cout % ng != 0:
            ng //= 2
        self.bn = nn.GroupNorm(ng, cout)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWSeparable(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, k=3, stride=stride, groups=cin)
        self.pw = ConvBNAct(cin, cout, k=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class TinyCNN(nn.Module):
    def __init__(self, width: int = 32, feature_grid: int = 5):
        super().__init__()
        self.feature_grid = feature_grid
        self.feat_channels = width * 4

        self.stem = ConvBNAct(3, width, k=3, stride=2)
        self.block1 = DWSeparable(width, width * 2, stride=2)
        self.block2 = DWSeparable(width * 2, width * 2, stride=2)
        self.block3 = DWSeparable(width * 2, width * 4, stride=2)
        self.block4 = DWSeparable(width * 4, width * 4, stride=1)

    def _pool_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        g = self.feature_grid
        x = F.interpolate(x, size=(2 * g, 2 * g), mode="bilinear", align_corners=False)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self._pool_to_grid(x)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
