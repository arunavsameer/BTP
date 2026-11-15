"""CollisionStudentV4: V3 backbone with GatedMotionFusion instead of raw concat.

Does not replace CollisionStudentV3. CoordConv, Δ1/Δ3, time embed, and
Causal (2+1)D + SE blocks stay. Only Stage-2 entry `[F, Δ1, Δ3, coords]`
1×1 fuse is swapped for a squeezed ego-motion gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from student.v2.model import CollisionStudentV2, IMAGENET_MEAN, IMAGENET_STD, n_params
from student.v3.model import MOTION_C, STEM_C, StudentV3Backbone, causal_delta

__all__ = ["CollisionStudentV4", "GatedMotionFusion", "StudentV4Backbone", "n_params"]


class GatedMotionFusion(nn.Module):
    """
    Replaces raw concat+conv at Stage 2 entry.
    Inputs:
        f:      [B, 96, 8, 16, 16]  (appearance from pyramid stem)
        d1, d3: [B, 96, 8, 16, 16]  (causal 1-step and 3-step motion deltas)
        coords: [B,  3, 8, 16, 16]  (CoordConv x, y, r maps)
    Output:
        out:    [B, 128, 8, 16, 16] (ready for Causal (2+1)D + SE blocks)
    """

    def __init__(self, in_ch: int = 96, out_ch: int = 128, init_gate_bias: float = -2.0):
        super().__init__()
        motion_ch = in_ch * 2

        self.motion_proj = nn.Sequential(
            nn.Conv3d(motion_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_ch),
            nn.SiLU(inplace=True),
        )
        self.ego_proj = nn.Sequential(
            nn.Conv3d(motion_ch, 24, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
        )
        gate_in_ch = in_ch + motion_ch + 24 + 3
        self.gate_conv = nn.Conv3d(gate_in_ch, in_ch, kernel_size=1, bias=True)
        nn.init.constant_(self.gate_conv.bias, init_gate_bias)
        self.stage2_in = nn.Sequential(
            nn.Conv3d(in_ch + 3, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(
        self, f: torch.Tensor, d1: torch.Tensor, d3: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deltas = torch.cat([d1, d3], dim=1)
        m_t = self.motion_proj(deltas)
        ego_pool = deltas.mean(dim=(-2, -1), keepdim=True)
        ego_feat = self.ego_proj(ego_pool).expand(-1, -1, -1, m_t.shape[-2], m_t.shape[-1])
        gate_in = torch.cat([f, deltas, ego_feat, coords], dim=1)
        g_t = torch.sigmoid(self.gate_conv(gate_in))
        f_fused = f + g_t * m_t
        out = self.stage2_in(torch.cat([f_fused, coords], dim=1))
        return out, g_t


class StudentV4Backbone(StudentV3Backbone):
    def __init__(self, ffn_hidden: int = 155, init_gate_bias: float = -2.0):
        super().__init__(ffn_hidden=ffn_hidden)
        del self.vel_fuse
        self.gated_fuse = GatedMotionFusion(STEM_C, MOTION_C, init_gate_bias=init_gate_bias)

    def motion(self, f_stem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta1 = causal_delta(f_stem, 1)
        delta3 = causal_delta(f_stem, 3)
        coord = self.coord.to(device=f_stem.device, dtype=f_stem.dtype).expand(
            f_stem.shape[0], -1, f_stem.shape[2], -1, -1
        )
        fused, gate = self.gated_fuse(f_stem, delta1, delta3, coord)
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


class CollisionStudentV4(CollisionStudentV2):
    def __init__(self, decoder: nn.Module, ffn_hidden: int = 155, init_gate_bias: float = -2.0):
        nn.Module.__init__(self)
        self.backbone = StudentV4Backbone(ffn_hidden=ffn_hidden, init_gate_bias=init_gate_bias)
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)
