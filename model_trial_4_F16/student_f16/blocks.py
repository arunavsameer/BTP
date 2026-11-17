"""Building blocks for CollisionStudentV2. Conv + GN uses bias=False."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn_groups(channels: int) -> int:
    for g in (16, 8, 4, 2, 1):
        if int(channels) % g == 0:
            return g
    return 1


class SpatialGN3d(nn.Module):
    """GroupNorm over H×W only so time steps do not share statistics (causal)."""

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(int(num_groups), int(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
        y = self.gn(y)
        return y.view(n, t, c, h, w).permute(0, 2, 1, 3, 4)


def causal_time_pad(x: torch.Tensor, kernel: int = 3, dilation: int = 1) -> torch.Tensor:
    pad_left = int(dilation) * (int(kernel) - 1)
    return F.pad(x, (0, 0, 0, 0, pad_left, 0))


class ConvGNAct3d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel, stride=1, padding=0, dilation=1, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(
            cin,
            cout,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.gn = SpatialGN3d(gn_groups(cout), cout)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class InvRes2p1d(nn.Module):
    """(2+1)D inverted residual. Temporal DW is causal (no future frames)."""

    def __init__(
        self,
        cin: int,
        cout: int,
        mid: int | None = None,
        expand: int = 2,
        spatial_stride: int = 1,
        time_dilation: int = 1,
        residual: bool = False,
    ):
        super().__init__()
        mid = int(cin) * int(expand) if mid is None else int(mid)
        s = int(spatial_stride)
        d = int(time_dilation)
        self.residual = bool(residual) and cin == cout and s == 1
        self.pw1 = ConvGNAct3d(cin, mid, 1, act=True)
        self.dw_s = ConvGNAct3d(mid, mid, (1, 3, 3), stride=(1, s, s), padding=(0, 1, 1), groups=mid, act=True)
        self.dw_t = nn.Conv3d(mid, mid, kernel_size=(3, 1, 1), padding=0, dilation=(d, 1, 1), groups=mid, bias=False)
        self.gn_t = SpatialGN3d(gn_groups(mid), mid)
        self.act_t = nn.GELU()
        self.pw2 = nn.Conv3d(mid, cout, kernel_size=1, bias=False)
        self.dilation = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw1(x)
        y = self.dw_s(y)
        y = causal_time_pad(y, kernel=3, dilation=self.dilation)
        y = self.act_t(self.gn_t(self.dw_t(y)))
        y = self.pw2(y)
        if self.residual:
            y = y + x
        return y


class SE3d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(3, 4), keepdim=True)
        s = self.fc2(self.act(self.fc1(s))).sigmoid()
        return x * s


class SE2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = self.fc2(self.act(self.fc1(s))).sigmoid()
        return x * s


class CausalInvResSE(nn.Module):
    def __init__(self, channels: int = 128, mid: int = 192, dilation: int = 1, se_r: int = 8):
        super().__init__()
        self.block = InvRes2p1d(channels, channels, mid=mid, spatial_stride=1, time_dilation=dilation, residual=False)
        self.se = SE3d(channels, reduction=se_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se(self.block(x))


class RecencyAttnPool(nn.Module):
    def __init__(self, channels: int, time: int = 8):
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1)
        recency = torch.linspace(-1.4, 0.0, int(time)).view(1, 1, int(time), 1, 1)
        self.recency_bias = nn.Parameter(recency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[2]
        bias = self.recency_bias
        if bias.shape[2] != t:
            raise RuntimeError(f"recency T={bias.shape[2]} != {t}")
        attn = torch.softmax(self.score(x) + bias, dim=2)
        return (x * attn).sum(dim=2)


def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    assert dim % 4 == 0
    y, x = torch.meshgrid(torch.arange(grid).float(), torch.arange(grid).float(), indexing="ij")
    y = y / max(grid - 1, 1)
    x = x / max(grid - 1, 1)
    omega = torch.exp(torch.arange(0, dim // 4).float() * (-torch.log(torch.tensor(10000.0)) / (dim // 4)))
    out = torch.cat(
        [
            torch.sin(y.unsqueeze(-1) * omega),
            torch.cos(y.unsqueeze(-1) * omega),
            torch.sin(x.unsqueeze(-1) * omega),
            torch.cos(x.unsqueeze(-1) * omega),
        ],
        dim=-1,
    )
    return out.view(grid * grid, dim)


class BottleneckMHSA(nn.Module):
    def __init__(self, dim: int = 128, attn_dim: int = 64, heads: int = 4, ffn_hidden: int = 128):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(attn_dim) // int(heads)
        if self.head_dim * int(heads) != int(attn_dim):
            raise ValueError("attn_dim must divide heads")
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=False)
        self.proj = nn.Linear(attn_dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        qkv = self.qkv(self.ln1(x)).view(b, n, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q * (self.head_dim ** -0.5)) @ k.transpose(-2, -1), dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(b, n, h * self.head_dim)
        x = x + self.proj(y)
        x = x + self.ffn(self.ln2(x))
        return x


class LayerScaleDWBlock(nn.Module):
    def __init__(self, channels: int = 128, se_r: int = 8, gamma_init: float = 1e-3):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.gn = nn.GroupNorm(gn_groups(channels), channels)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.se = SE2d(channels, reduction=se_r)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.se(self.pw(self.act(self.gn(self.dw(x)))))
        return x + self.gamma * y
