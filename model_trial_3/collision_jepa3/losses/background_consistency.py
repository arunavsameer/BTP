"""Per-cell cosine distance between matched background latents."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def background_consistency_loss(z_a: torch.Tensor, z_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """z: [B, 32, 5, 5] → mean_ij (1 - cos)."""
    a = z_a.float().permute(0, 2, 3, 1)
    b = z_b.float().permute(0, 2, 3, 1)
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    cos = (a_n * b_n).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()
