"""Dense tubelet tokens → Z_t [B, 32, 5, 5] with a lightweight temporal adapter."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, groups_norm: int = 8):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel, 1, 1),
            padding=(pad, 0, 0),
            dilation=(dilation, 1, 1),
            groups=channels,
        )
        self.pw = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(groups_norm, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        return x + y


class SpatialTemporalAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        temporal_groups: int,
        spatial: int,
        z_channels: int = 32,
        feature_grid: int = 5,
        projection_channels: int = 128,
        temporal_blocks: int = 2,
        temporal_kernel: int = 3,
        temporal_dilation: int = 1,
        group_norm_groups: int = 8,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.temporal_groups = int(temporal_groups)
        self.spatial = int(spatial)
        self.z_channels = int(z_channels)
        self.feature_grid = int(feature_grid)
        c = int(projection_channels)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(self.hidden_size, c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[
                TemporalResidualBlock(c, kernel=temporal_kernel, dilation=temporal_dilation, groups_norm=group_norm_groups)
                for _ in range(int(temporal_blocks))
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c, self.z_channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(self.z_channels)

    def expected_tokens(self) -> int:
        return self.temporal_groups * self.spatial * self.spatial

    def tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, C] → [B, T', C, S, S] without dropping an arbitrary prefix/suffix."""
        b, n, c = tokens.shape
        expected = self.expected_tokens()
        if n != expected:
            raise RuntimeError(
                f"token count {n} != T'*S*S = {self.temporal_groups}*{self.spatial}*{self.spatial}={expected}. "
                "Refusing to slice tokens to force a reshape."
            )
        if c != self.hidden_size:
            raise RuntimeError(f"token hidden {c} != adapter hidden_size {self.hidden_size}")
        grid = tokens.reshape(b, self.temporal_groups, self.spatial, self.spatial, c)
        return grid.permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        grid = self.tokens_to_grid(tokens)  # B, T', C, S, S
        b, t_prime, c, s, _ = grid.shape
        x = grid.reshape(b * t_prime, c, s, s)
        x = self.spatial_proj(x)
        x = x.reshape(b, t_prime, x.shape[1], s, s).permute(0, 2, 1, 3, 4).contiguous()
        x = self.temporal(x)  # B, 128, T', S, S
        mean_f = x.mean(dim=2)
        last_f = x[:, :, -1]
        mixed = self.mix(torch.cat([mean_f, last_f], dim=1))
        pooled = F.adaptive_avg_pool2d(mixed, self.feature_grid)
        z = self.bottleneck(pooled)
        z = z.permute(0, 2, 3, 1)
        z = self.norm(z)
        z = z.permute(0, 3, 1, 2).contiguous()
        if z.shape != (b, self.z_channels, self.feature_grid, self.feature_grid):
            raise RuntimeError(f"Z_t shape {tuple(z.shape)} != {[b, self.z_channels, self.feature_grid, self.feature_grid]}")
        return z
