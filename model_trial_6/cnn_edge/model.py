"""CollisionStudentV2 + Canny: pyramid stem, causal motion, dual readout, MHSA predictor.

Input is 16 cyan-on-RGB frames (still 3 channels). Frozen Pair-CF decoder maps
F_pred [B,128,16,16] → Ĥ 5×5. Decoder weights are never trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    BottleneckMHSA,
    CausalInvResSE,
    ConvGNAct3d,
    InvRes2p1d,
    LayerScaleDWBlock,
    RecencyAttnPool,
    sincos_2d,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def n_params(mod: nn.Module, *, trainable: bool | None = None) -> int:
    if trainable is None:
        return int(sum(p.numel() for p in mod.parameters()))
    return int(sum(p.numel() for p in mod.parameters() if p.requires_grad is trainable))


class StudentV2Backbone(nn.Module):
    def __init__(self, ffn_hidden: int = 155):
        super().__init__()
        self.stem_1a = ConvGNAct3d(3, 16, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), act=True)
        self.stem_1b = InvRes2p1d(16, 32, expand=2, spatial_stride=2, residual=False)
        self.stem_1c = InvRes2p1d(32, 64, expand=2, spatial_stride=2, residual=False)
        self.stem_1d = InvRes2p1d(64, 96, expand=2, spatial_stride=2, residual=False, mid=128)
        self.vel_fuse = ConvGNAct3d(192, 128, 1, act=True)
        self.causal_1 = CausalInvResSE(128, mid=192, dilation=1)
        self.causal_2 = CausalInvResSE(128, mid=192, dilation=2)
        self.present = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.history_pool = RecencyAttnPool(128, time=8)
        self.history = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.mhsa_1 = BottleneckMHSA(128, 64, 4, ffn_hidden=ffn_hidden)
        self.mhsa_2 = BottleneckMHSA(128, 64, 4, ffn_hidden=ffn_hidden)
        self.refine = nn.Sequential(LayerScaleDWBlock(128), LayerScaleDWBlock(128))
        self.register_buffer("pos", sincos_2d(16, 128), persistent=False)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem_1a(x)
        y = self.stem_1b(y)
        y = self.stem_1c(y)
        return self.stem_1d(y)

    def motion(self, f_stem: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(f_stem[:, :, :1])
        delta = torch.cat([zeros, f_stem[:, :, 1:] - f_stem[:, :, :-1]], dim=2)
        fused = self.vel_fuse(torch.cat([f_stem, delta], dim=1))
        return self.causal_2(self.causal_1(fused))

    def readout(self, f_motion: torch.Tensor) -> torch.Tensor:
        present = self.present(f_motion[:, :, -1])
        history = self.history(self.history_pool(f_motion))
        return torch.cat([present, history], dim=1)

    def predict(self, f_ctx: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_ctx.shape
        tok = f_ctx.flatten(2).transpose(1, 2) + self.pos.to(device=f_ctx.device, dtype=f_ctx.dtype)
        tok = self.mhsa_2(self.mhsa_1(tok))
        grid = tok.transpose(1, 2).reshape(b, c, h, w)
        return self.refine(grid)

    def forward(self, x_cthw: torch.Tensor) -> dict[str, torch.Tensor]:
        f_stem = self.stem(x_cthw)
        f_motion = self.motion(f_stem)
        f_ctx = self.readout(f_motion)
        f_pred = self.predict(f_ctx)
        return {"F_stem": f_stem, "F_motion": f_motion, "F_ctx": f_ctx, "F_pred": f_pred, "F": f_pred}


class CollisionStudentV2(nn.Module):
    """Canny-RGB clip [B,16,3,256,256] → F_pred → frozen Pair-CF Ĥ 5×5."""

    def __init__(self, decoder: nn.Module, ffn_hidden: int = 155):
        super().__init__()
        self.backbone = StudentV2Backbone(ffn_hidden=ffn_hidden)
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean.to(dtype=clip.dtype, device=clip.device)) / self._std.to(
            dtype=clip.dtype, device=clip.device
        )

    def decode(self, f_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ĥ 5×5, 5×5 logits, and 16×16 decoder logits (Z16). Decoder stays frozen."""
        logits_fine = self.decoder.net(f_pred)
        pooled = F.adaptive_max_pool2d(logits_fine, 5) if logits_fine.shape[-1] != 5 else logits_fine
        h = torch.sigmoid(pooled).squeeze(1)
        return h, pooled.squeeze(1), logits_fine.squeeze(1)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.normalize(clip).permute(0, 2, 1, 3, 4).contiguous()
        out = self.backbone(x)
        h, logits, z16 = self.decode(out["F_pred"])
        out["Hhat_1s"] = h
        out["Hhat_1s_logits"] = logits
        out["Z16"] = z16
        return out

    def param_audit(self) -> dict:
        return {
            "total": n_params(self),
            "trainable": n_params(self, trainable=True),
            "frozen": n_params(self, trainable=False),
            "backbone": n_params(self.backbone),
            "decoder": n_params(self.decoder),
        }


# Train-script alias used by the older compact-CNN layout.
CollisionCNNEdge = CollisionStudentV2
