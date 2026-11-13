"""CollisionStudentV3: V2 backbone plus three cheap priors.

Does not replace CollisionStudentV2. Stage-2 fusion is
`[F_stem | Δ1 | Δ3 | (x,y,r)]` = 291 channels → 128, then a learnable
`[1,128,8,1,1]` timestamp is added. Extra trainable params: ~12.7k (Δ3)
+ ~384 (CoordConv in the 1×1 fuse) + 1024 (time embed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from student.v2.blocks import (
    BottleneckMHSA,
    CausalInvResSE,
    ConvGNAct3d,
    InvRes2p1d,
    LayerScaleDWBlock,
    RecencyAttnPool,
    sincos_2d,
)
from student.v2.model import CollisionStudentV2, IMAGENET_MEAN, IMAGENET_STD, n_params

STEM_C = 96
MOTION_C = 128
TIME = 8
GRID = 16


def coord_xyr(grid: int = GRID) -> torch.Tensor:
    """Constant FOE-centered maps: x, y in [-1,1], r = sqrt(x^2+y^2). Shape [1,3,1,H,W]."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, int(grid)),
        torch.linspace(-1.0, 1.0, int(grid)),
        indexing="ij",
    )
    rr = torch.sqrt(xx * xx + yy * yy)
    return torch.stack([xx, yy, rr], dim=0).view(1, 3, 1, int(grid), int(grid))


def causal_delta(f: torch.Tensor, lag: int) -> torch.Tensor:
    """F_t - F_{t-lag}, zeros for the first `lag` steps (no future)."""
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    zeros = torch.zeros_like(f[:, :, :lag])
    return torch.cat([zeros, f[:, :, lag:] - f[:, :, :-lag]], dim=2)


class StudentV3Backbone(nn.Module):
    def __init__(self, ffn_hidden: int = 155):
        super().__init__()
        self.stem_1a = ConvGNAct3d(3, 16, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), act=True)
        self.stem_1b = InvRes2p1d(16, 32, expand=2, spatial_stride=2, residual=False)
        self.stem_1c = InvRes2p1d(32, 64, expand=2, spatial_stride=2, residual=False)
        self.stem_1d = InvRes2p1d(64, STEM_C, expand=2, spatial_stride=2, residual=False)
        # 96+96+96+3 = 291
        self.vel_fuse = ConvGNAct3d(STEM_C * 3 + 3, MOTION_C, 1, act=True)
        self.time_embed = nn.Parameter(torch.zeros(1, MOTION_C, TIME, 1, 1))
        self.causal_1 = CausalInvResSE(MOTION_C, mid=192, dilation=1)
        self.causal_2 = CausalInvResSE(MOTION_C, mid=192, dilation=2)
        self.present = nn.Sequential(
            nn.Conv2d(MOTION_C, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.history_pool = RecencyAttnPool(MOTION_C, time=TIME)
        self.history = nn.Sequential(
            nn.Conv2d(MOTION_C, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.mhsa_1 = BottleneckMHSA(MOTION_C, 64, 4, ffn_hidden=ffn_hidden)
        self.mhsa_2 = BottleneckMHSA(MOTION_C, 64, 4, ffn_hidden=ffn_hidden)
        self.jepa = nn.Sequential(LayerScaleDWBlock(MOTION_C), LayerScaleDWBlock(MOTION_C))
        self.register_buffer("pos", sincos_2d(GRID, MOTION_C), persistent=False)
        self.register_buffer("coord", coord_xyr(GRID), persistent=False)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem_1a(x)
        y = self.stem_1b(y)
        y = self.stem_1c(y)
        return self.stem_1d(y)

    def motion(self, f_stem: torch.Tensor) -> torch.Tensor:
        delta1 = causal_delta(f_stem, 1)
        delta3 = causal_delta(f_stem, 3)
        coord = self.coord.to(device=f_stem.device, dtype=f_stem.dtype).expand(
            f_stem.shape[0], -1, f_stem.shape[2], -1, -1
        )
        fused = self.vel_fuse(torch.cat([f_stem, delta1, delta3, coord], dim=1))
        te = self.time_embed
        if te.shape[2] != fused.shape[2]:
            raise RuntimeError(f"time_embed T={te.shape[2]} != {fused.shape[2]}")
        fused = fused + te.to(dtype=fused.dtype)
        return self.causal_2(self.causal_1(fused))

    def readout(self, f_motion: torch.Tensor) -> torch.Tensor:
        a = self.present(f_motion[:, :, -1])
        b = self.history(self.history_pool(f_motion))
        return torch.cat([a, b], dim=1)

    def predict(self, f_ctx: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_ctx.shape
        tok = f_ctx.flatten(2).transpose(1, 2) + self.pos.to(device=f_ctx.device, dtype=f_ctx.dtype)
        tok = self.mhsa_2(self.mhsa_1(tok))
        grid = tok.transpose(1, 2).view(b, c, h, w)
        return self.jepa(grid)

    def forward(self, x_cthw: torch.Tensor) -> dict[str, torch.Tensor]:
        f_stem = self.stem(x_cthw)
        f_motion = self.motion(f_stem)
        f_ctx = self.readout(f_motion)
        f_pred = self.predict(f_ctx)
        return {"F_stem": f_stem, "F_motion": f_motion, "F_ctx": f_ctx, "F_pred": f_pred, "F": f_pred}


class CollisionStudentV3(CollisionStudentV2):
    def __init__(self, decoder: nn.Module, ffn_hidden: int = 155):
        nn.Module.__init__(self)
        self.backbone = StudentV3Backbone(ffn_hidden=ffn_hidden)
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)
