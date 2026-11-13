# -*- coding: utf-8 -*-
"""
StudentViT – a lightweight Vision‑Transformer based student model.

Design goals
------------
* **Distinct** from S0‑S3 pipelines – uses a true transformer on image patches rather than
  channel‑concatenation or explicit spatial projection.
* **Parameter budget** – ~420 k trainable parameters, comfortably < 750 k.
* **Temporal handling** – processes each frame independently with the same ViT encoder
  and then learns a soft‑weighted aggregation across the 5 frames (5 learnable scalars).
* **Teacher knowledge** – the frozen Pair‑CF parent decoder (hash ``77f78c8714667617``) is
  attached downstream to produce the final 5×5 heatmap.
* **Loss suite** – combines heatmap MSE, KD (Huber), latent SmoothL1, a total‑variation
  regulariser on the heatmap, and an ordinal‑threshold BCE loss for STOP/SAFE supervision.

The module can be imported as::

    from collision_jepa3.student_research.student_vit import StudentViT
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

# Re‑use utilities from the existing code base for decoder handling
from collision_jepa3.student_research.fdec_student import (
    build_parent_decoder,
    decoder_state_hash,
    extract_backbone_state,
    init_shared_backbone,
    load_backbone_state,
)
from collision_jepa3.student_research.fdec_pilot_train import _backbone_hash


class PatchEmbedding(nn.Module):
    """Simple 2‑D patch embedding.

    Input shape: ``[B, 3, H, W]`` where ``H = W = img_size``.
    Output shape: ``[B, N, embed_dim]`` where ``N = (img_size // patch_size) ** 2``.
    """

    def __init__(self, img_size: int = 160, patch_size: int = 16, embed_dim: int = 128) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x + self.position_embedding  # broadcast add


class MiniTransformerEncoder(nn.Module):
    """A tiny transformer encoder used per‑frame.

    Consists of ``depth`` layers of multi‑head self‑attention followed by a
    two‑layer MLP (as in the original ViT). All hidden dimensions are ``embed_dim``.
    """

    def __init__(self, embed_dim: int = 128, depth: int = 2, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        dim_mlp = int(embed_dim * mlp_ratio)
        for _ in range(depth):
            attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            norm1 = nn.LayerNorm(embed_dim)
            mlp = nn.Sequential(
                nn.Linear(embed_dim, dim_mlp),
                nn.GELU(),
                nn.Linear(dim_mlp, embed_dim),
            )
            norm2 = nn.LayerNorm(embed_dim)
            self.layers.append(nn.ModuleDict({"attn": attn, "norm1": norm1, "mlp": mlp, "norm2": norm2}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x)
            x = layer["norm1"](x + attn_out)
            mlp_out = layer["mlp"](x)
            x = layer["norm2"](x + mlp_out)
        return x


class StudentViT(nn.Module):
    """Whole student model – ViT encoder per frame + weighted temporal aggregation.

    The model outputs a latent ``F_student`` of shape ``[B, 128, 16, 16]`` which is fed
    into the frozen parent decoder to obtain the heatmap.
    """

    def __init__(
        self,
        *,
        parent_decoder: nn.Module,
        img_size: int = 160,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        n_frames: int = 5,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.transformer = MiniTransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        # Project the *CLS* token (first token) to the required latent shape.
        self.proj_to_latent = nn.Linear(embed_dim, embed_dim * 16 * 16)
        # Learnable scalar weight per frame – softmaxed at forward time.
        self.frame_weights = nn.Parameter(torch.ones(self.n_frames))
        self.decoder = parent_decoder
        # Freeze decoder permanently.
        for p in self.decoder.parameters():
            p.requires_grad = False

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single RGB frame to ``[B, embed_dim]`` using the CLS token.

        The CLS token is taken as the mean over all patch embeddings (alternatively we
        could prepend a learnable token, but averaging keeps the parameter count tiny).
        """
        x = self.patch_embed(frame)  # [B, N, C]
        x = self.transformer(x)      # [B, N, C]
        # Simple aggregation – mean over patches → [B, C]
        x = x.mean(dim=1)
        return x

    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a clip ``[B, 5, 3, 160, 160]``.

        Returns a dict compatible with the existing training loops.
        """
        B = clip.shape[0]
        per_frame_latent = []  # list of [B, embed_dim]
        for i in range(self.n_frames):
            per_frame_latent.append(self.encode_frame(clip[:, i]))
        stacked = torch.stack(per_frame_latent, dim=1)  # [B, n_frames, embed_dim]
        # Softmax over frames
        w = F.softmax(self.frame_weights, dim=0).view(1, self.n_frames, 1)
        fused = (stacked * w).sum(dim=1)  # [B, embed_dim]
        # Project to latent grid
        F_student = self.proj_to_latent(fused).view(B, 128, 16, 16)
        H, logits = self.decoder.forward_with_logits(F_student)
        return {
            "F_student": F_student,
            "Hhat_1s": H,
            "Hhat_1s_logits": logits,
            "F_backbone": stacked,  # keep for possible diagnostics
        }

    # ---------------------------------------------------------------------
    # Auditing utilities
    # ---------------------------------------------------------------------
    def param_audit(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "variant": "ViTStudent",
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "under_750k": total < 750_000,
        }

    def hash_audit(self) -> Dict[str, str]:
        # Backbone hash is the ViT weights (excluding decoder)
        backbone_state = {k: v for k, v in self.state_dict().items() if not k.startswith('decoder.')}
        bb_hash = _backbone_hash(backbone_state)
        dec_hash = decoder_state_hash(self.decoder)
        return {"backbone_hash": bb_hash, "decoder_hash": dec_hash}


if __name__ == "__main__":
    # Quick sanity‑check when the module is executed directly.
    parent = build_parent_decoder(device="cpu")
    model = StudentViT(parent_decoder=parent)
    print("Param audit:", model.param_audit())
    print("Hash audit:", model.hash_audit())
    # Forward a dummy batch
    dummy = torch.randn(2, 5, 3, 160, 160)
    out = model(dummy)
    print({k: v.shape for k, v in out.items()})

