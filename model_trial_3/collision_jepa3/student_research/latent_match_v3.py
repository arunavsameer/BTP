"""LatentMatch V3: shared per-frame CNN + Conv3D temporal fuse + frozen parent decoder + ordinal head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from collision_jepa3.student_research.fdec_student import decoder_state_hash
from collision_jepa3.student_research.student_final.model import OrdinalSeverityHead


class DWBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.gn = nn.GroupNorm(max(1, cout // 8), cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pw(self.dw(x))))


class SharedFrameEncoder(nn.Module):
    """Independent per-frame encoder → 128×16×16 (20×20 then adaptive pool)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )
        self.b1 = DWBlock(32, 64, stride=2)
        self.b2 = DWBlock(64, 96, stride=2)
        self.b3 = DWBlock(96, 128, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b3(self.b2(self.b1(self.stem(x))))
        if y.shape[-1] != 16 or y.shape[-2] != 16:
            y = F.adaptive_avg_pool2d(y, 16)
        return y


class TemporalFusion(nn.Module):
    """Learned softmax mix over T frame features (16 scalars)."""

    def __init__(self, channels: int = 128, n_frames: int = 16) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.logits = nn.Parameter(torch.zeros(n_frames))
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: [B, C, T, 16, 16]
        w = torch.softmax(self.logits, dim=0).view(1, 1, -1, 1, 1)
        fused = (stacked * w).sum(dim=2)
        return self.proj(fused)


class StudentLatentMatchV3(nn.Module):
    EXPECTED_DECODER_HASH = "77f78c8714667617"

    def __init__(self, parent_decoder: nn.Module, n_frames: int = 16):
        super().__init__()
        self.n_frames = n_frames
        self.encoder = SharedFrameEncoder()
        self.temporal = TemporalFusion(128, n_frames)
        self.decoder = parent_decoder
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.severity = OrdinalSeverityHead(fine_channels=128, grid=5)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c, h, w = clip.shape
        frames = clip.reshape(b * t, c, h, w)
        feats = self.encoder(frames).reshape(b, t, 128, 16, 16).permute(0, 2, 1, 3, 4)
        f_s = self.temporal(feats)
        hmap, logits = self.decoder.forward_with_logits(f_s)
        sev = self.severity(f_s, hmap)
        return {"F_student": f_s, "Hhat_1s": hmap, "Hhat_1s_logits": logits, **sev}

    def param_audit(self) -> dict[str, Any]:
        def n(m):
            return int(sum(p.numel() for p in m.parameters()))

        enc = n(self.encoder)
        tmp = n(self.temporal)
        dec = n(self.decoder)
        sev = n(self.severity)
        total = enc + tmp + dec + sev
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return {
            "encoder": enc,
            "temporal_fusion": tmp,
            "parent_decoder_frozen": dec,
            "severity_head": sev,
            "total_including_frozen": total,
            "trainable": trainable,
            "under_750k": total < 750_000,
            "decoder_hash": decoder_state_hash(self.decoder),
        }
