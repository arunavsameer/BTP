"""ConvNeXt-V2 3D blocks: depthwise conv + LayerNorm + 4x expand + GELU + GRN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse import apply_submanifold, resize_mask


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LayerNorm3d(nn.Module):
    """Channels-first LayerNorm over C (ConvNeXt style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.bias = nn.Parameter(torch.zeros(int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        w = self.weight.view(1, -1, 1, 1, 1)
        b = self.bias.view(1, -1, 1, 1, 1)
        return w * x + b


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.bias = nn.Parameter(torch.zeros(int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class GRN3d(nn.Module):
    """Global Response Normalization over T,H,W. Active sites only if ``mask`` is set."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, int(dim), 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, int(dim), 1, 1, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        else:
            m = resize_mask(mask, x.shape[2:])
            gx = torch.sqrt((x * m).pow(2).sum(dim=(2, 3, 4), keepdim=True) + 1e-12)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class GRN2d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, int(dim), 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, int(dim), 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block3d(nn.Module):
    """3D ConvNeXt-V2 block with optional submanifold masking."""

    def __init__(self, dim: int, drop_path: float = 0.0, kernel: tuple[int, int, int] = (3, 7, 7)):
        super().__init__()
        pad = tuple(k // 2 for k in kernel)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel, padding=pad, groups=dim)
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN3d(4 * dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = apply_submanifold(x, mask)
        x = self.dwconv(x)
        x = apply_submanifold(x, mask)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, mask)
        x = self.pwconv2(x)
        x = apply_submanifold(x, mask)
        return residual + self.drop_path(x)


class ConvNeXtV2Block2d(nn.Module):
    """Plain 2D ConvNeXt-V2 block used as the FCMAE decoder."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN2d(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(self.dwconv(x))))))
        return residual + self.drop_path(x)


class Downsample3d(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, stride: tuple[int, int, int]):
        super().__init__()
        self.norm = LayerNorm3d(dim_in)
        k = tuple(int(s) for s in stride)
        self.conv = nn.Conv3d(dim_in, dim_out, kernel_size=k, stride=k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.norm(x)
        x = apply_submanifold(x, mask)
        x = self.conv(x)
        if mask is None:
            return x, None
        mask = resize_mask(mask, x.shape[2:])
        x = apply_submanifold(x, mask)
        return x, mask


class TemporalSoftmaxPool(nn.Module):
    """Same recency-biased softmax pool the Pair-CF adapter uses."""

    def __init__(self, channels: int, temporal_groups: int):
        super().__init__()
        hidden = max(int(channels) // 4, 8)
        self.score = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )
        recency = torch.linspace(-1.0, 1.0, int(temporal_groups)).view(1, 1, int(temporal_groups), 1, 1)
        self.recency = nn.Parameter(recency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[2]
        recency = self.recency
        if recency.shape[2] != t:
            recency = F.interpolate(recency, size=(t, 1, 1), mode="nearest")
        attn = torch.softmax(self.score(x) + recency, dim=2)
        return (x * attn).sum(dim=2)
