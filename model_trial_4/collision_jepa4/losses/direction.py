"""Directional LEFT/CENTER/RIGHT alignment via log-sum-exp scores."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _group_score(weighted: torch.Tensor, cols: list[int], temperature: float) -> torch.Tensor:
    if not cols:
        return weighted.new_zeros(weighted.shape[0])
    cells = weighted.index_select(2, torch.tensor(cols, device=weighted.device, dtype=torch.long))
    flat = cells.reshape(cells.shape[0], -1)
    return temperature * torch.logsumexp(flat / max(temperature, 1e-6), dim=1)


def direction_scores(
    heatmap: torch.Tensor,
    row_weights: torch.Tensor,
    left_cols: list[int],
    center_cols: list[int],
    right_cols: list[int],
    temperature: float = 0.08,
) -> torch.Tensor:
    hm = heatmap.float()
    rw = row_weights.to(device=hm.device, dtype=hm.dtype).view(1, -1, 1)
    rw = rw / rw.mean().clamp(min=1e-6)
    weighted = hm * rw
    left = _group_score(weighted, left_cols, temperature)
    center = _group_score(weighted, center_cols, temperature)
    right = _group_score(weighted, right_cols, temperature)
    return torch.stack([left, center, right], dim=1)


def direction_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    row_weights: torch.Tensor,
    left_cols: list[int],
    center_cols: list[int],
    right_cols: list[int],
    sample_mask: torch.Tensor | None = None,
    temperature: float = 0.08,
    threat_threshold: float = 0.25,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    ps = direction_scores(pred, row_weights, left_cols, center_cols, right_cols, temperature)
    ts = direction_scores(target, row_weights, left_cols, center_cols, right_cols, temperature)
    threat = ts.amax(dim=1) >= threat_threshold
    if sample_mask is not None:
        threat = threat & sample_mask.to(dtype=torch.bool, device=pred.device)
    if not bool(threat.any()):
        return pred.new_zeros(())
    return F.smooth_l1_loss(ps[threat], ts[threat])
