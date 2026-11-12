"""Identity-init JEPA predictor on the 16×16 fine map."""

from __future__ import annotations

import torch
import torch.nn as nn


def _sincos_2d(grid: int, channels: int) -> torch.Tensor:
    assert channels % 4 == 0
    s = int(grid)
    c = int(channels)
    y, x = torch.meshgrid(torch.arange(s).float(), torch.arange(s).float(), indexing="ij")
    y = y / max(s - 1, 1)
    x = x / max(s - 1, 1)
    omega = torch.exp(torch.arange(0, c // 4).float() * (-torch.log(torch.tensor(10000.0)) / (c // 4)))
    out = torch.cat(
        [
            torch.sin(y.unsqueeze(-1) * omega),
            torch.cos(y.unsqueeze(-1) * omega),
            torch.sin(x.unsqueeze(-1) * omega),
            torch.cos(x.unsqueeze(-1) * omega),
        ],
        dim=-1,
    )
    return out.permute(2, 0, 1).unsqueeze(0)


class DWSepRes(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c = int(channels)
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        self.pw = nn.Conv2d(c, c, kernel_size=1)
        self.gn = nn.GroupNorm(8, c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.gn(self.pw(self.dw(x))))


class JEPAPredictor(nn.Module):
    def __init__(self, channels: int = 128, grid: int = 16):
        super().__init__()
        self.block = DWSepRes(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.register_buffer("pos", _sincos_2d(grid, channels), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.pos.to(device=x.device, dtype=x.dtype)
        if mask is not None:
            h = h * mask
        return x + self.proj(self.block(h))
