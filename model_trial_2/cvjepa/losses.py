"""Loss functions: heatmap, false-positive, JEPA, warning alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_heatmap_weights(
    grid: int,
    center: float = 2.2,
    ring: float = 1.5,
    edge: float = 1.0,
    corner: float = 0.65,
    normalize: bool = True,
) -> torch.Tensor:
    w = torch.zeros(grid, grid)
    c = (grid - 1) * 0.5
    last = grid - 1
    for i in range(grid):
        for j in range(grid):
            cheb = max(abs(i - c), abs(j - c))
            is_corner = i in (0, last) and j in (0, last)
            is_border = i in (0, last) or j in (0, last)
            if cheb < 0.51:
                w[i, j] = center
            elif cheb < 1.51:
                w[i, j] = ring
            elif is_corner:
                w[i, j] = corner
            elif is_border:
                w[i, j] = edge
            else:
                w[i, j] = ring
    if normalize:
        w = w / w.mean().clamp(min=1e-6)
    return w


def weighted_focal_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    focal_weight: float = 1.0,
    gamma: float = 2.0,
    change: torch.Tensor | None = None,
    change_weight: float = 0.0,
    fn_weight: float = 0.0,
    cell_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    se = (pred - target) ** 2
    tgt = torch.clamp(target, 0.0, 1.0)
    weight = 1.0 + focal_weight * tgt.pow(gamma)
    if change is not None and change_weight > 0:
        weight = weight + change_weight * torch.clamp(change, 0.0, 1.0)
    if fn_weight > 0:
        under = torch.relu(target - pred)
        weight = weight + fn_weight * under * (tgt > 0.3).float()
    if cell_weights is not None:
        weight = weight * cell_weights.to(dtype=weight.dtype, device=weight.device)
    return (se * weight).mean()


def false_positive_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    safe_threshold: float = 0.20,
    center_cols: torch.Tensor | None = None,
    center_mult: float = 1.0,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    safe = (target < safe_threshold).float()
    weight = safe
    if center_cols is not None and float(center_mult) != 1.0:
        col_w = pred.new_ones(pred.shape[-1])
        col_w[center_cols.long()] = float(center_mult)
        weight = weight * col_w.view(1, 1, -1)
    return (pred.clamp(min=0.0).pow(2) * weight).mean()


def jepa_distance(z_hat: torch.Tensor, z: torch.Tensor, cos_weight: float = 1.0) -> torch.Tensor:
    """Match student Zhat+ to teacher Z. Scale by teacher std so heatmap loss is not drowned."""
    scale = z.detach().flatten(1).std(dim=1, keepdim=True).clamp_min(1e-3)
    scale = scale.view(-1, 1, 1, 1)
    mse = F.mse_loss(z_hat.float() / scale, z.float() / scale)
    cos = F.cosine_similarity(z_hat.float(), z.float(), dim=1)
    return mse + cos_weight * (1.0 - cos).mean()


def _group_max(weighted: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    if cols.numel() == 0:
        return weighted.new_zeros(weighted.shape[0])
    return weighted.index_select(2, cols).amax(dim=(1, 2))


def warning_scores(
    heatmap: torch.Tensor,
    row_weights: torch.Tensor,
    left_cols: torch.Tensor,
    center_cols: torch.Tensor,
    right_cols: torch.Tensor,
) -> torch.Tensor:
    weighted = heatmap * row_weights.view(1, -1, 1)
    left = _group_max(weighted, left_cols)
    center = _group_max(weighted, center_cols)
    right = _group_max(weighted, right_cols)
    return torch.stack([left, center, right], dim=1)


def warning_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    row_weights: torch.Tensor,
    left_cols: torch.Tensor,
    center_cols: torch.Tensor,
    right_cols: torch.Tensor,
    caution_threshold: float = 0.45,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    ps = warning_scores(pred, row_weights, left_cols, center_cols, right_cols)
    ts = warning_scores(target, row_weights, left_cols, center_cols, right_cols)
    mse = F.mse_loss(ps, ts)
    peak_fn = torch.relu(ts.amax(dim=1) - ps.amax(dim=1)).pow(2)
    threat = (ts.amax(dim=1) >= caution_threshold).float()
    if float(threat.sum()) > 0:
        peak = (peak_fn * threat).sum() / threat.sum().clamp(min=1.0)
    else:
        peak = peak_fn.mean()
    logits = ps / 0.08
    labels = ts.argmax(dim=1)
    if float(threat.sum()) > 0:
        ce = F.cross_entropy(logits, labels, reduction="none")
        ce = (ce * threat).sum() / threat.sum().clamp(min=1.0)
    else:
        ce = logits.new_zeros(())
    over = torch.relu(ps.amax(dim=1) - ts.amax(dim=1)).pow(2)
    quiet = (ts.amax(dim=1) < caution_threshold).float()
    if float(quiet.sum()) > 0:
        over_pen = (over * quiet).sum() / quiet.sum().clamp(min=1.0)
    else:
        over_pen = pred.new_zeros(())
    return mse + peak + 0.25 * ce + over_pen
