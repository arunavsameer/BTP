"""CollisionStudentV6: V2 stack with ResidualGatedFusionV6 at Stage 2.

Drops V3 Δ3. Keeps V2 stem, causal SE, dual readout, MHSA predictor.
Stage-2 entry is Δ1 + ego (H,W pool) + CoordConv, with unattenuated F_stem residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from student.v2.model import CollisionStudentV2, IMAGENET_MEAN, IMAGENET_STD, StudentV2Backbone, n_params

__all__ = ["CollisionStudentV6", "ResidualGatedFusionV6", "StudentV6Backbone", "n_params"]

DECAY_START_EPOCH = 20
DECODER_LR_MULT = 0.15
UNFREEZE_WARMUP_EPOCHS = 2
GRID = 16
TIME = 8
STEM_C = 96
MOTION_C = 128


def decoder_lr(epoch: int, backbone_lr: float) -> float:
    if int(epoch) <= DECAY_START_EPOCH:
        return 0.0
    ramp = min(1.0, (int(epoch) - DECAY_START_EPOCH) / float(UNFREEZE_WARMUP_EPOCHS))
    return float(backbone_lr) * DECODER_LR_MULT * ramp


def coord_xyr(grid: int = GRID) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, int(grid)),
        torch.linspace(-1.0, 1.0, int(grid)),
        indexing="ij",
    )
    rr = torch.sqrt(xx * xx + yy * yy)
    return torch.stack([xx, yy, rr], dim=0).view(1, 3, 1, int(grid), int(grid))


class ResidualGatedFusionV6(nn.Module):
    """Stage-2 entry: Δ1 only, ego pooled over H,W, F_stem + G_t ⊙ M_t."""

    def __init__(self, in_ch: int = 96, out_ch: int = 128, init_gate_bias: float = -0.5):
        super().__init__()
        self.motion_proj = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_ch),
            nn.SiLU(inplace=True),
        )
        self.ego_proj = nn.Sequential(
            nn.Conv3d(in_ch, 24, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.gate_conv = nn.Conv3d(in_ch * 2 + 24 + 3, in_ch, kernel_size=1, bias=True)
        nn.init.constant_(self.gate_conv.bias, init_gate_bias)
        self.stage2_in = nn.Sequential(
            nn.Conv3d(in_ch + 3, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, f_stem: torch.Tensor, d1: torch.Tensor, coords: torch.Tensor):
        m_t = self.motion_proj(d1)
        ego_pool = d1.mean(dim=(-2, -1), keepdim=True)
        ego_feat = self.ego_proj(ego_pool).expand(-1, -1, -1, m_t.size(-2), m_t.size(-1))
        gate_in = torch.cat([f_stem, d1, ego_feat, coords], dim=1)
        g_t = torch.sigmoid(self.gate_conv(gate_in))
        f_fused = f_stem + g_t * m_t
        out = self.stage2_in(torch.cat([f_fused, coords], dim=1))
        return out, g_t


class StudentV6Backbone(StudentV2Backbone):
    def __init__(self, ffn_hidden: int = 155, init_gate_bias: float = -0.5):
        super().__init__(ffn_hidden=ffn_hidden)
        del self.vel_fuse
        self.gated_fusion = ResidualGatedFusionV6(STEM_C, MOTION_C, init_gate_bias=init_gate_bias)
        self.time_embed = nn.Parameter(torch.zeros(1, MOTION_C, TIME, 1, 1))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.register_buffer("coord", coord_xyr(GRID), persistent=False)

    def motion(self, f_stem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(f_stem[:, :, :1])
        d1 = torch.cat([zeros, f_stem[:, :, 1:] - f_stem[:, :, :-1]], dim=2)
        coords = self.coord.to(device=f_stem.device, dtype=f_stem.dtype).expand(
            f_stem.shape[0], -1, f_stem.shape[2], -1, -1
        )
        fused, gate = self.gated_fusion(f_stem, d1, coords)
        te = self.time_embed
        if te.shape[2] != fused.shape[2]:
            raise RuntimeError(f"time_embed T={te.shape[2]} != {fused.shape[2]}")
        fused = fused + te.to(dtype=fused.dtype)
        return self.causal_2(self.causal_1(fused)), gate

    def forward(self, x_cthw: torch.Tensor) -> dict[str, torch.Tensor]:
        f_stem = self.stem(x_cthw)
        f_motion, gate = self.motion(f_stem)
        f_ctx = self.readout(f_motion)
        f_pred = self.predict(f_ctx)
        return {
            "F_stem": f_stem,
            "F_motion": f_motion,
            "F_ctx": f_ctx,
            "F_pred": f_pred,
            "F": f_pred,
            "G_t": gate,
        }


class CollisionStudentV6(CollisionStudentV2):
    def __init__(self, decoder: nn.Module, ffn_hidden: int = 155, init_gate_bias: float = -0.5):
        nn.Module.__init__(self)
        self.backbone = StudentV6Backbone(ffn_hidden=ffn_hidden, init_gate_bias=init_gate_bias)
        self.decoder = decoder
        self.decoder_unfrozen = False
        self.set_decoder_trainable(False)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def backbone_parameters(self):
        yield from self.backbone.parameters()

    def set_decoder_trainable(self, on: bool) -> None:
        self.decoder_unfrozen = bool(on)
        for p in self.decoder.parameters():
            p.requires_grad_(self.decoder_unfrozen)
        if self.decoder_unfrozen:
            self.decoder.train()
        else:
            self.decoder.eval()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if not self.decoder_unfrozen:
            self.decoder.eval()
        return self
