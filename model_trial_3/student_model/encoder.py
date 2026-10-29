"""Tiny (2+1)D CNN that emits G's 128×16×16 fine map from 16 RGB frames."""

from __future__ import annotations

import torch
import torch.nn as nn

from collision_jepa3.models.spatial_adapter import TemporalSoftmaxPool


class InvertedResidual2p1d(nn.Module):
    """MobileNetV2-style residual with separate spatial and temporal depthwise convs."""

    def __init__(self, channels: int, expand: int = 4, time_dilation: int = 1, gn_groups: int = 8):
        super().__init__()
        mid = int(channels) * int(expand)
        d = int(time_dilation)
        self.pw1 = nn.Conv3d(channels, mid, kernel_size=1)
        self.dw_s = nn.Conv3d(mid, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=mid)
        self.dw_t = nn.Conv3d(
            mid,
            mid,
            kernel_size=(3, 1, 1),
            padding=(d, 0, 0),
            dilation=(d, 1, 1),
            groups=mid,
        )
        self.pw2 = nn.Conv3d(mid, channels, kernel_size=1)
        self.gn = nn.GroupNorm(int(gn_groups), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.pw1(x))
        y = self.dw_s(y)
        y = self.dw_t(y)
        y = self.pw2(y)
        y = self.gn(y)
        return self.act(x + y)


class StudentEncoder(nn.Module):
    """RGB [B,3,T,H,W] → fine map [B,128,16,16] on V-JEPA's tubelet grid."""

    def __init__(
        self,
        width: int = 64,
        out_channels: int = 128,
        temporal_groups: int = 8,
        n_low: int = 2,
        n_high: int = 6,
        high_expand: int = 3,
        high_dilations: tuple[int, ...] = (1, 2, 1, 2, 1, 2),
    ):
        super().__init__()
        if len(high_dilations) != int(n_high):
            raise ValueError(f"high_dilations {high_dilations} != n_high {n_high}")
        self.out_channels = int(out_channels)
        self.stem = nn.Sequential(
            nn.Conv3d(3, width, kernel_size=(2, 16, 16), stride=(2, 16, 16)),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.low = nn.Sequential(
            *[InvertedResidual2p1d(width, expand=4, time_dilation=1) for _ in range(int(n_low))]
        )
        self.proj = nn.Sequential(
            nn.Conv3d(width, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.high = nn.Sequential(
            *[
                InvertedResidual2p1d(out_channels, expand=int(high_expand), time_dilation=d)
                for d in high_dilations
            ]
        )
        self.extra = InvertedResidual2p1d(out_channels, expand=2, time_dilation=1)
        self.time_pool = TemporalSoftmaxPool(out_channels, int(temporal_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, T, H, W] ImageNet-normalised. Returns [B, 128, 16, 16]."""
        y = self.stem(x)
        y = self.low(y)
        y = self.proj(y)
        y = self.high(y)
        y = self.extra(y)
        return self.time_pool(y)
