# -*- coding: utf-8 -*-
"""
New Student Model (StudentNew)
================================
This model is **independent** from the existing S0/S1/S2 implementations.
It processes a 5‑frame RGB clip (shape ``[B, 5, 3, 160, 160]``) with a **lightweight shared 2‑D CNN** that
produces a latent tensor ``F_student`` of shape ``[B, 128, 16, 16]``.
The latent is then fed into the **frozen Pair‑CF parent decoder** to obtain the 5×5 heatmap.

Key design differences from S0/S1/S2
-----------------------------------
* **Temporal weighting** – a learned scalar weight per frame (5 parameters) is applied after the
  per‑frame CNN, instead of concatenating features along the channel dimension.
* **No explicit spatial projection module** – the CNN directly outputs the required spatial grid.
* **Parameter budget** – ~420 k trainable parameters, well under the 750 k limit.
* **Efficient inference** – a single shared CNN forward pass per frame, followed by a cheap weighted sum.

The architecture mirrors the training pipeline used by the other pilots (GT + KD + F‑dec losses)
but can be trained from scratch while still leveraging the **pre‑trained frozen parent decoder**.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict

# Re‑use utilities from the existing code base
from collision_jepa3.student_research.fdec_student import (
    FDEC_FRAME_OFFSETS,
    build_parent_decoder,
    decoder_state_hash,
    extract_backbone_state,
    init_shared_backbone,
    load_backbone_state,
)
from collision_jepa3.student_research.fdec_pilot_train import _backbone_hash


class SharedCNN(nn.Module):
    """Lightweight shared 2‑D CNN that maps a single RGB frame → ``[128, 16, 16]``.

    The network consists of four convolutional blocks with stride 2 to down‑sample
    from 160×160 to roughly 16×16.  All convolutions use ``kernel_size=3`` and
    ``padding=1`` to preserve centre alignment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 160 → 80
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 80 → 40
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 40 → 20
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 20 → 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Final up‑sample to 16×16 (nearest neighbour for simplicity)
            nn.Upsample(scale_factor=1.6, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape ``[B, 128, 16, 16]``.

        Args:
            x: ``[B, 3, 160, 160]`` RGB frame.
        """
        y = self.blocks(x)
        # The up‑sample may produce 16×16 (or 16×16±1 due to rounding).  Crop/pad to exact size.
        y = F.interpolate(y, size=(16, 16), mode="bilinear", align_corners=False)
        return y


class StudentNew(nn.Module):
    """Student model used for the *new* experiment.

    - ``backbone`` – shared CNN applied independently to each of the 5 frames.
    - ``frame_weights`` – learnable scalar weights (5) that are soft‑maxed to combine the
      per‑frame latents into a single ``F_student``.
    - ``decoder`` – frozen parent decoder (heatmap decoder) copied from the Pair‑CF
      treatment checkpoint.
    """

    def __init__(self, *, parent_decoder: nn.Module, n_frames: int = 5) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.backbone = SharedCNN()
        # One scalar per frame – we keep it as a Parameter and apply softmax in forward
        self.frame_weights = nn.Parameter(torch.ones(self.n_frames))
        self.decoder = parent_decoder
        # Freeze decoder permanently
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a clip ``[B, 5, 3, 160, 160]``.

        Returns a dict with keys used by the training loop:
        ``F_student`` (latent), ``Hhat_1s`` (heatmap), ``Hhat_1s_logits`` and ``F_backbone``
        (raw per‑frame latent stack for debugging).
        """
        B = clip.shape[0]
        per_frame = []  # list of [B, 128, 16, 16]
        for i in range(self.n_frames):
            frame = clip[:, i]  # [B, 3, 160, 160]
            per_frame.append(self.backbone(frame))
        # Stack → [B, n_frames, 128, 16, 16]
        stacked = torch.stack(per_frame, dim=1)
        # Softmax over frames
        w = F.softmax(self.frame_weights, dim=0)  # [n_frames]
        w = w.view(1, self.n_frames, 1, 1, 1)
        F_student = (stacked * w).sum(dim=1)  # [B, 128, 16, 16]
        # Decoder forward (returns heatmap and logits)
        H, logits = self.decoder.forward_with_logits(F_student)
        return {
            "F_student": F_student,
            "Hhat_1s": H,
            "Hhat_1s_logits": logits,
            "F_backbone": stacked,  # useful for diagnostics / visualisation
        }

    # ---------------------------------------------------------------------
    # Utility helpers for auditing
    # ---------------------------------------------------------------------
    def param_audit(self) -> Dict[str, Any]:
        """Return a detailed parameter count and budget check.

        ``under_750k`` is ``True`` when the total number of parameters is < 750 000.
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        proj_params = self.frame_weights.numel()
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = backbone_params + proj_params + decoder_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "variant": "NewStudent",
            "backbone_params": backbone_params,
            "proj_params": proj_params,
            "decoder_params": decoder_params,
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "under_750k": total < 750_000,
        }

    def hash_audit(self) -> Dict[str, str]:
        """Compute SHA‑256 hashes for backbone, decoder and the full model state.
        Useful for reproducibility checks.
        """
        bb_hash = _backbone_hash(extract_backbone_state(self.backbone))
        dec_hash = decoder_state_hash(self.decoder)
        return {"backbone_hash": bb_hash, "decoder_hash": dec_hash}


if __name__ == "__main__":
    # Simple sanity check – instantiate and print audit information
    cfg_path = Path(__file__).parents[3] / "configs" / "default.yaml"
    # Load a dummy config just to resolve device – we rely on utils in the repo.
    # In practice the training script will supply the proper config.
    dummy = torch.rand(1, 5, 3, 160, 160)
    parent = build_parent_decoder(device="cpu")
    model = StudentNew(parent_decoder=parent)
    print("Parameter audit:", model.param_audit())
    print("Hash audit:", model.hash_audit())

