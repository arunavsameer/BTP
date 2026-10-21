"""Heatmap reconstruction (soft BCE or weighted MSE) and false-positive penalty."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def row_priority_weights(row_weights: list[float], device, dtype) -> torch.Tensor:
    w = torch.tensor(row_weights, device=device, dtype=dtype)
    w = w / w.mean().clamp(min=1e-6)
    return w


def cell_threat_weights(
    target: torch.Tensor,
    floor: float = 0.05,
    scale: float = 4.0,
    gamma: float = 2.0,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """w_ij = floor + scale * H_ij^gamma, optionally mixed with extra, then mean-normalized per sample."""
    tgt = target.float()
    w = floor + scale * tgt.clamp(min=0.0).pow(gamma)
    if extra is not None:
        w = w + scale * extra.float().abs().pow(gamma)
    valid_mean = w.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return w / valid_mean


def appear_cell_weight(
    h_now: torch.Tensor,
    h_future: torch.Tensor,
    *,
    now_threshold: float = 0.1,
    future_threshold: float = 0.5,
    weight: float = 8.0,
) -> torch.Tensor:
    """Extra multiplier on cells empty now that are occupied at the future map.

    1.0 everywhere except cells with ``h_now < now_threshold`` and
    ``h_future >= future_threshold``, which get ``weight``. Stacks on top of
    occupied ``pos_weight`` in ``soft_bce``.
    """
    w = torch.ones_like(h_future, dtype=torch.float32)
    if float(weight) == 1.0:
        return w
    appear = (h_now.float() < float(now_threshold)) & (h_future.float() >= float(future_threshold))
    return torch.where(appear, w * float(weight), w)


def soft_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    pos_threshold: float = 0.1,
    pos_weight: float = 2.0,
    cell_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-cell BCE with soft GT in [0, 1]. No row weights.

    Averaged over cells. Occupied cells (GT > pos_threshold) get ``pos_weight``
    vs 1.0 on empty cells. ``cell_weight`` is an optional per-cell multiplier
    (empty→occupied extra). ``sample_mask`` is the ambiguity include-mask
    (True = keep the sample).
    """
    pred = pred.float().clamp(eps, 1.0 - eps)
    target = target.float().clamp(0.0, 1.0)
    bce = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
    w = torch.ones_like(bce)
    if float(pos_weight) != 1.0:
        w = torch.where(target > float(pos_threshold), w * float(pos_weight), w)
    if cell_weight is not None:
        w = w * cell_weight.to(dtype=w.dtype, device=w.device)
    if sample_mask is not None:
        sm = sample_mask.to(dtype=w.dtype, device=w.device).view(-1, 1, 1)
        w = w * sm
        if float(sample_mask.sum()) <= 0:
            return pred.new_zeros(())
    denom = w.sum().clamp(min=eps)
    return (bce * w).sum() / denom


def maxpool3x3(x: torch.Tensor) -> torch.Tensor:
    """Per-cell max over the 3×3 neighbourhood, including the center. Same spatial size."""
    return F.max_pool2d(x.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)


def blob_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    occupied_threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """BCE on 3×3 max(pred) vs 3×3 max(GT), only where the GT neighbourhood is occupied.

    Lighting the true cell satisfies this. Lighting only a neighbour still does.
    Empty 3×3s are excluded so the term cannot push random blobs onto empty maps.
    """
    p = maxpool3x3(pred).clamp(eps, 1.0 - eps)
    t = maxpool3x3(target).clamp(0.0, 1.0)
    keep = t >= float(occupied_threshold)
    if sample_mask is not None:
        keep = keep & sample_mask.to(dtype=torch.bool, device=pred.device).view(-1, 1, 1)
    if not bool(keep.any()):
        return pred.new_zeros(())
    bce = -(t * p.log() + (1.0 - t) * (1.0 - p).log())
    return bce[keep].mean()


def target_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    cell_mask: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    floor: float = 0.05,
    scale: float = 4.0,
    gamma: float = 2.0,
    extra: torch.Tensor | None = None,
    recall_beta: float = 1.0,
    recall_hi: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted MSE. Weights come from the target, never from the prediction."""
    pred = pred.float()
    target = target.float()
    se = (pred - target) ** 2
    w = cell_threat_weights(target, floor=floor, scale=scale, gamma=gamma, extra=extra)
    if row_weights is not None:
        rw = row_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1)
        w = w * rw
    beta = float(recall_beta)
    if beta != 1.0:
        under_high = ((target >= float(recall_hi)) & (pred.detach() < target)).to(dtype=w.dtype)
        w = w * (1.0 + (beta - 1.0) * under_high)
    if cell_mask is not None:
        w = w * cell_mask.to(dtype=w.dtype, device=w.device)
    if sample_mask is not None:
        sm = sample_mask.to(dtype=w.dtype, device=w.device).view(-1, 1, 1)
        w = w * sm
        if float(sample_mask.sum()) <= 0:
            return pred.new_zeros(())
    denom = w.sum().clamp(min=eps)
    return (se * w).sum() / denom


def false_positive_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    safe_threshold: float = 0.05,
    margin: float = 0.10,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    safe = target < safe_threshold
    if sample_mask is not None:
        safe = safe & sample_mask.to(dtype=torch.bool, device=pred.device).view(-1, 1, 1)
    if not bool(safe.any()):
        return pred.new_zeros(())
    excess = torch.relu(pred - margin)
    return (excess.pow(2)[safe]).mean()
