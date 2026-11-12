"""RGB student: encoder + predictor trainable, Pair-CF decoder frozen."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import StudentEncoder
from .predictor import JEPAPredictor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StudentLatPairCF(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.encoder = StudentEncoder(out_channels=128)
        self.predictor = JEPAPredictor(128, grid=16)
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

    def encode_context(self, clip: torch.Tensor) -> torch.Tensor:
        x = self.normalize(clip).permute(0, 2, 1, 3, 4).contiguous()
        return self.encoder(x)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(clip)
        pred = self.predictor(ctx)
        h, logits = self.decoder.forward_with_logits(ctx)
        return {
            "F_ctx": ctx,
            "F": ctx,
            "F_pred": pred,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
        }

    def param_audit(self) -> dict:
        def n(mod):
            return int(sum(p.numel() for p in mod.parameters()))

        total = int(sum(p.numel() for p in self.parameters()))
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "encoder": n(self.encoder),
            "predictor": n(self.predictor),
            "decoder": n(self.decoder),
        }
