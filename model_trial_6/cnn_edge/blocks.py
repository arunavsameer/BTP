"""CollisionStudentV2 blocks: causal (2+1)D stem, recency pool, bottleneck MHSA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn_groups(channels: int, max_groups: int = 8) -> int:
    c = int(channels)
    for g in (max_groups, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def sincos_2d(grid: int, channels: int) -> torch.Tensor:
    """Fixed 2D sine-cosine PE as tokens [1, S*S, C]."""
    assert int(channels) % 4 == 0
    s = int(grid)
    c = int(channels)
    y, x = torch.meshgrid(torch.arange(s).float(), torch.arange(s).float(), indexing="ij")
    y = y / max(s - 1, 1)
    x = x / max(s - 1, 1)
    omega = torch.exp(torch.arange(0, c // 4).float() * (-torch.log(torch.tensor(10000.0)) / (c // 4)))
    pos = torch.cat(
        [
            torch.sin(y.unsqueeze(-1) * omega),
            torch.cos(y.unsqueeze(-1) * omega),
            torch.sin(x.unsqueeze(-1) * omega),
            torch.cos(x.unsqueeze(-1) * omega),
        ],
        dim=-1,
    )
    return pos.reshape(s * s, c).unsqueeze(0)


class ConvGNAct3d(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        kernel: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.gn = nn.GroupNorm(gn_groups(cout), cout)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class CausalDW3d(nn.Module):
    """Depthwise temporal k=3, left-padded so the kernel never sees future frames."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        d = int(dilation)
        self.left = (3 - 1) * d
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=0,
            dilation=(d, 1, 1),
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 0, 0, 0, self.left, 0)))


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hid = max(int(channels) // int(reduction), 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hid),
            nn.GELU(),
            nn.Linear(hid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class SqueezeExcite3d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hid = max(int(channels) // int(reduction), 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hid),
            nn.GELU(),
            nn.Linear(hid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * w


class InvRes2p1d(nn.Module):
    """Expand → spatial DW (optional stride 2) → causal temporal DW k=3 → project."""

    def __init__(
        self,
        cin: int,
        cout: int,
        expand: int = 2,
        spatial_stride: int = 1,
        residual: bool = False,
        mid: int | None = None,
    ):
        super().__init__()
        hid = int(mid) if mid is not None else int(cin) * int(expand)
        s = int(spatial_stride)
        self.pw1 = nn.Conv3d(cin, hid, kernel_size=1, bias=False)
        self.dw_s = nn.Conv3d(
            hid, hid, kernel_size=(1, 3, 3), stride=(1, s, s), padding=(0, 1, 1), groups=hid, bias=False
        )
        self.dw_t = CausalDW3d(hid, dilation=1)
        self.pw2 = nn.Conv3d(hid, cout, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(gn_groups(cout), cout)
        self.act = nn.GELU()
        self.residual = bool(residual) and int(cin) == int(cout) and s == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.pw1(x))
        y = self.dw_s(y)
        y = self.dw_t(y)
        y = self.act(self.gn(self.pw2(y)))
        if self.residual:
            y = y + x
        return y


class CausalInvResSE(nn.Module):
    """Inverted residual at 128-d with causal dilated time DW, then SE, then skip."""

    def __init__(self, channels: int = 128, mid: int = 192, dilation: int = 1):
        super().__init__()
        c, h = int(channels), int(mid)
        self.pw1 = nn.Conv3d(c, h, kernel_size=1, bias=False)
        self.dw_s = nn.Conv3d(h, h, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=h, bias=False)
        self.dw_t = CausalDW3d(h, dilation=int(dilation))
        self.pw2 = nn.Conv3d(h, c, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(gn_groups(c), c)
        self.act = nn.GELU()
        self.se = SqueezeExcite3d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.pw1(x))
        y = self.dw_s(y)
        y = self.dw_t(y)
        y = self.act(self.gn(self.pw2(y)))
        return x + self.se(y)


class RecencyAttnPool(nn.Module):
    """Softmax over T=8 from 1×1 scores plus a learned recency bias (later frames higher)."""

    def __init__(self, channels: int, time: int = 8):
        super().__init__()
        self.score = nn.Conv2d(int(channels), 1, kernel_size=1)
        # Walkthrough init: roughly [-1.4, …, 0] so the present is preferred at start.
        self.recency = nn.Parameter(torch.linspace(-1.4, 0.0, int(time)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        xt = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        s = self.score(xt).view(b, t, h, w) + self.recency.view(1, t, 1, 1)
        attn = torch.softmax(s, dim=1)
        return (x * attn.unsqueeze(1)).sum(dim=2)


class BottleneckMHSA(nn.Module):
    """LN → QKV 128→192→64 (4 heads × 16) → out 64→128 → FFN 128→155→128."""

    def __init__(self, dim: int = 128, inner: int = 64, heads: int = 4, ffn_hidden: int = 155):
        super().__init__()
        if inner % heads != 0:
            raise ValueError(f"inner {inner} not divisible by heads {heads}")
        self.heads = int(heads)
        self.head_dim = int(inner) // int(heads)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, inner * 3)
        self.proj = nn.Linear(inner, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, dim)
        self.act = nn.GELU()

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        b, n, _ = tok.shape
        h, d = self.heads, self.head_dim
        x = self.norm1(tok)
        qkv = self.qkv(x).view(b, n, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) * (d ** -0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, h * d)
        tok = tok + self.proj(out)
        y = self.fc2(self.act(self.fc1(self.norm2(tok))))
        return tok + y


class LayerScaleDWBlock(nn.Module):
    """DW 3×3 → GN → GELU → PW → SE, residual scaled by γ (init 0.001)."""

    def __init__(self, channels: int = 128, gamma_init: float = 0.001):
        super().__init__()
        c = int(channels)
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.gn = nn.GroupNorm(gn_groups(c), c)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.se = SqueezeExcite2d(c)
        self.gamma = nn.Parameter(torch.full((c, 1, 1), float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.se(self.pw(self.act(self.gn(self.dw(x)))))
        return x + self.gamma * y
