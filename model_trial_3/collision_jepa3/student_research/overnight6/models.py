"""Student backbones for overnight Pair-CF experiments. Decoder is attached by the trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from collision_jepa3.models.vjepa_backbone import IMAGENET_MEAN, IMAGENET_STD
from student_model.encoder import StudentEncoder, InvertedResidual2p1d
from student_model.student import JEPAPredictor


class ChannelProj(nn.Module):
    def __init__(self, c: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, 0, 0] = torch.eye(c)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AuxTierHead(nn.Module):
    def __init__(self, c: int = 128, hidden: int = 64, n_class: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_class),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class RecentBranch(nn.Module):
    """Latest 4 frames → [B,128,16,16]."""

    def __init__(self, width: int = 48, out_channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, width, kernel_size=(2, 16, 16), stride=(2, 16, 16)),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            InvertedResidual2p1d(width, expand=3, time_dilation=1),
            InvertedResidual2p1d(width, expand=3, time_dilation=1),
        )
        self.to_fine = nn.Sequential(
            nn.Conv3d(width, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 16, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.to_fine(self.body(self.stem(x)))
        return self.pool(y).squeeze(2)


class StudentOvernight(nn.Module):
    def __init__(self, decoder: nn.Module, *, variant: str, use_aux: bool = False, dual: bool = False):
        super().__init__()
        self.variant = str(variant)
        self.dual = bool(dual)
        self.encoder = StudentEncoder(out_channels=128)
        self.predictor = JEPAPredictor(128, grid=16)
        self.proj = ChannelProj(128)
        self.aux = AuxTierHead(128) if use_aux else None
        self.recent = RecentBranch() if dual else None
        self.gate = nn.Sequential(nn.Conv2d(256, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 1), nn.Sigmoid()) if dual else None
        self.decoder = decoder
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean.to(dtype=clip.dtype, device=clip.device)) / self._std.to(
            dtype=clip.dtype, device=clip.device
        )

    def encode_full(self, clip: torch.Tensor) -> torch.Tensor:
        x = self.normalize(clip).permute(0, 2, 1, 3, 4).contiguous()
        return self.encoder(x)

    def encode_recent(self, clip: torch.Tensor) -> torch.Tensor:
        x = self.normalize(clip[:, -4:]).permute(0, 2, 1, 3, 4).contiguous()
        return self.recent(x)

    def fuse(self, full: torch.Tensor, recent: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([full, recent], dim=1))
        return g * recent + (1.0 - g) * full

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = self.encode_full(clip)
        if self.dual:
            rec = self.encode_recent(clip)
            fused = self.fuse(ctx, rec)
            f_dec = fused
        else:
            rec = None
            fused = None
            f_dec = self.proj(ctx)
        pred = self.predictor(ctx)
        h, logits = self.decoder.forward_with_logits(f_dec)
        out = {
            "F_ctx": ctx,
            "F": f_dec,
            "F_pred": pred,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
        }
        if rec is not None:
            out["F_recent"] = rec
            out["F_fused"] = fused
        if self.aux is not None:
            out["aux_logits"] = self.aux(ctx.detach())
        return out

    def student_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith("decoder."):
                continue
            yield p

    def param_audit(self) -> dict:
        def n(m):
            return 0 if m is None else int(sum(p.numel() for p in m.parameters()))

        total = int(sum(p.numel() for p in self.parameters()))
        return {
            "variant": self.variant,
            "total": total,
            "trainable_now": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "encoder": n(self.encoder),
            "predictor": n(self.predictor),
            "proj_1x1": n(self.proj),
            "aux_head": n(self.aux),
            "recent": n(self.recent),
            "gate": n(self.gate),
            "decoder": n(self.decoder),
        }


def build_model(mid: int, decoder: nn.Module) -> StudentOvernight:
    if mid == 5:
        return StudentOvernight(decoder, variant="m5_multiscale", use_aux=False, dual=True)
    if mid == 6:
        return StudentOvernight(decoder, variant="m6_taskaware", use_aux=True, dual=False)
    names = {
        1: "m1_stop_floor",
        2: "m2_ordinal",
        3: "m3_asymmetric",
        4: "m4_parent_recovery",
    }
    return StudentOvernight(decoder, variant=names[mid], use_aux=False, dual=False)
