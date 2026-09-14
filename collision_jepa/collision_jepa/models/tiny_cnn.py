"""A tiny depthwise-separable CNN backbone for the student.

Maps one RGB frame [B, 3, H, W] to a compact feature map pooled to a fixed
``feature_grid x feature_grid`` (default 5x5) so it aligns with the collision grid.
Designed to stay well under ~1M params so the full student is wearable-friendly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWSeparable(nn.Module):
    """Depthwise 3x3 (optionally strided) + pointwise 1x1."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, k=3, stride=stride, groups=cin)
        self.pw = ConvBNAct(cin, cout, k=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class TinyCNN(nn.Module):
    """Shared per-frame encoder.

    Output: [B, feat_channels, feature_grid, feature_grid].
    """

    def __init__(self, width: int = 32, feature_grid: int = 5):
        super().__init__()
        self.feature_grid = feature_grid
        self.feat_channels = width * 4

        self.stem = ConvBNAct(3, width, k=3, stride=2)          # H/2
        self.block1 = DWSeparable(width, width * 2, stride=2)   # H/4
        self.block2 = DWSeparable(width * 2, width * 2, stride=2)  # H/8
        self.block3 = DWSeparable(width * 2, width * 4, stride=2)  # H/16
        self.block4 = DWSeparable(width * 4, width * 4, stride=1)

    def _pool_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Pool an arbitrary spatial map down to feature_grid x feature_grid.

        We avoid ``adaptive_avg_pool2d`` because the legacy ONNX exporter rejects it
        when the input size is not an integer multiple of the output (e.g. 8 -> 5).
        Resizing to 2*grid then a 2x2 average is equivalent-in-spirit, has no
        parameters, and exports cleanly (Resize + AveragePool).
        """
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
        x = self._pool_to_grid(x)
        return x


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
