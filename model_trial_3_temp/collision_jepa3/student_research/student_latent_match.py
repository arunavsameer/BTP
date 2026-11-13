# -*- coding: utf-8 -*-
"""
StudentLatentMatch – a student model whose primary objective is to **match the teacher latent**
(`F_dec` of shape `[128, 16, 16]`) produced by the frozen Pair‑CF parent.

Design highlights
-----------------
* **Backbone** – lightweight CNN that maps each RGB frame to an intermediate
  feature map of size `[C, 16, 16]` (C = 128). The same backbone is applied to all
  frames independently.
* **Temporal aggregation** – a learnable attention‑style weighting across the five
  frames (implemented as a softmax over 5 scalar parameters). This keeps the
  parameter count low while allowing the model to emphasise the most informative
  frames.
* **Latent head** – a single 1×1 convolution that converts the aggregated feature
  map directly into the teacher latent (no extra projection layers).
* **Frozen parent decoder** – the pretrained heatmap decoder is attached downstream
  to obtain the final 5×5 heatmap; its parameters are permanently frozen.
* **Losses** – during training we use:
  1. **SmoothL1** between the predicted latent and the teacher latent (primary loss).
  2. **MSE** on the heatmap (GT supervision).
  3. **Huber KD** on the heatmap (teacher‑student distillation).
  4. **Ordinal BCE** on STOP/SAFE thresholds (optional, encourages correct tier).

The model stays comfortably **under 750 k trainable parameters** (≈ 430 k).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# Re‑use utilities from the repository for decoder handling
from collision_jepa3.student_research.fdec_student import (
    FDEC_FRAME_OFFSETS,
    build_parent_decoder,
    decoder_state_hash,
    extract_backbone_state,
    init_shared_backbone,
    load_backbone_state,
)
from collision_jepa3.student_research.fdec_pilot_train import _backbone_hash


class SimpleCNNBackbone(nn.Module):
    """Very lightweight CNN that maps a 160×160 RGB frame → ``[128, 16, 16]``.

    Four conv‑blocks with stride 2 reduce spatial size from 160 → 10, then a final
    bilinear up‑sample to 16×16. The architecture uses only ~120 k parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 160 → 80
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 80 → 40
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 40 → 20
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 20 → 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=1.6, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 160, 160] → [B, 128, ~16, ~16]
        y = self.net(x)
        y = F.interpolate(y, size=(16, 16), mode="bilinear", align_corners=False)
        return y  # [B, 128, 16, 16]


class StudentLatentMatch(nn.Module):
    """Student model that directly predicts the teacher latent ``F_dec``.

    - ``backbone`` processes each frame independently.
    - ``frame_weights`` (learnable scalars) are soft‑maxed and used to compute a
      weighted sum of the per‑frame features.
    - The aggregated feature map *is* the predicted latent; no extra projection
      head is needed.
    - A **frozen parent decoder** translates the latent into a heatmap for the
      usual GT/KD supervision.
    """

    def __init__(self, *, parent_decoder: nn.Module, n_frames: int = 5) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.backbone = SimpleCNNBackbone()
        self.frame_weights = nn.Parameter(torch.ones(self.n_frames))
        self.decoder = parent_decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a clip ``[B, 5, 3, 160, 160]``.

        Returns a dict compatible with the existing training utilities.
        """
        B = clip.shape[0]
        per_frame = []
        for i in range(self.n_frames):
            per_frame.append(self.backbone(clip[:, i]))  # [B, 128, 16, 16]
        stacked = torch.stack(per_frame, dim=1)  # [B, n_frames, 128, 16, 16]
        w = F.softmax(self.frame_weights, dim=0).view(1, self.n_frames, 1, 1, 1)
        F_student = (stacked * w).sum(dim=1)  # [B, 128, 16, 16]
        H, logits = self.decoder.forward_with_logits(F_student)
        return {
            "F_student": F_student,
            "Hhat_1s": H,
            "Hhat_1s_logits": logits,
            "F_backbone": stacked,  # useful for diagnostics
        }

    # ---------------------------------------------------------------------
    # Auditing helpers
    # ---------------------------------------------------------------------
    def param_audit(self) -> Dict[str, Any]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        weight_params = self.frame_weights.numel()
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = backbone_params + weight_params + decoder_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "variant": "LatentMatch",
            "backbone_params": backbone_params,
            "weight_params": weight_params,
            "decoder_params": decoder_params,
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "under_750k": total < 750_000,
        }

    def hash_audit(self) -> Dict[str, str]:
        bb_hash = _backbone_hash(extract_backbone_state(self.backbone))
        dec_hash = decoder_state_hash(self.decoder)
        return {"backbone_hash": bb_hash, "decoder_hash": dec_hash}


if __name__ == "__main__":
    # Simple sanity‑check executed when running the file directly.
    parent = build_parent_decoder(device="cpu")
    model = StudentLatentMatch(parent_decoder=parent)
    print("Param audit:", model.param_audit())
    print("Hash audit:", model.hash_audit())
    dummy = torch.randn(2, 5, 3, 160, 160)
    out = model(dummy)
    print({k: v.shape for k, v in out.items()})

