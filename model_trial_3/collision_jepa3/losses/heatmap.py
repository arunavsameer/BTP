"""Target-weighted spatial heatmap MSE and false-positive penalty."""

from __future__ import annotations

import torch


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
