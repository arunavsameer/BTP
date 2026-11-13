"""Paired collision/miss variance regression for U1."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_var_loss(
    v_hat: torch.Tensor,
    v_true: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: float = 9.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    if float(mask.sum()) <= 0:
        return v_hat.new_zeros(())
    pn = 4.0 * v_hat
    tn = 4.0 * v_true
    w = torch.ones_like(tn)
    w = w + float(pos_weight) * (tn > (4.0 * float(eps))).to(dtype=tn.dtype)
    w = w * mask.to(dtype=tn.dtype).view(-1, 1, 1)
    loss = F.smooth_l1_loss(pn, tn, reduction="none")
    return (loss * w).sum() / w.sum().clamp_min(1.0)
