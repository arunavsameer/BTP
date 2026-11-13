"""Ordinal + spatial + teacher KD losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  return F.binary_cross_entropy(pred, target, reduction="none")


def weighted_ordinal_loss(
  q_warn: torch.Tensor,
  q_stop: torch.Tensor,
  p_gt: torch.Tensor,
  group_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
  y_warn = ((p_gt - 0.20) / 0.10).clamp(0.0, 1.0)
  y_stop = ((p_gt - 0.55) / 0.10).clamp(0.0, 1.0)
  gw = group_weight / group_weight.mean().clamp_min(1e-6)

  def branch(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pos = (gw * y).sum()
    neg = (gw * (1.0 - y)).sum()
    total = pos + neg + 1e-6
    w_pos = torch.minimum(torch.tensor(3.0, device=q.device), torch.sqrt(total / (2.0 * pos + 1e-6)))
    w_neg = torch.minimum(torch.tensor(3.0, device=q.device), torch.sqrt(total / (2.0 * neg + 1e-6)))
    bce = _soft_bce(q, y)
    w = torch.where(y > 0.5, w_pos, w_neg)
    return (bce * w * gw).mean()

  l_warn = branch(q_warn, y_warn)
  l_stop = branch(q_stop, y_stop)
  total = 0.5 * (l_warn + l_stop)
  return total, {"L_ord_warn": float(l_warn.detach()), "L_ord_stop": float(l_stop.detach()), "L_ord": float(total.detach())}


def spatial_gt_loss(pred: torch.Tensor, target: torch.Tensor, group_weight: torch.Tensor) -> torch.Tensor:
  huber = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05)
  w_bar = (0.05 + 4.0 * target**2).mean().clamp_min(1e-6)
  w_ij = (0.05 + 4.0 * target**2) / w_bar
  valid = torch.isfinite(target).float()
  gw = group_weight.view(-1, 1, 1) / group_weight.mean().clamp_min(1e-6)
  return (huber * w_ij * valid * gw).mean()


def teacher_kd_loss(pred: torch.Tensor, teacher: torch.Tensor, group_weight: torch.Tensor) -> torch.Tensor:
  huber = F.smooth_l1_loss(pred, teacher, reduction="none", beta=0.1)
  gw = group_weight.view(-1, 1, 1, 1) / group_weight.mean().clamp_min(1e-6)
  return (huber * gw).mean()


def combined_loss(
  out: dict,
  batch: dict,
  *,
  lambda_gt: float,
  lambda_kd: float,
) -> tuple[torch.Tensor, dict[str, float]]:
  gw = batch["group_weight"]
  l_ord, terms = weighted_ordinal_loss(out["q_warn"], out["q_stop"], batch["p_gt"], gw)
  l_gt = spatial_gt_loss(out["Hhat_1s"], batch["gt_1s"], gw)
  l_kd = torch.tensor(0.0, device=l_ord.device)
  if lambda_kd > 0 and "H_teacher" in batch:
    mask = batch.get("do_kd")
    if mask is not None and mask.any():
      idx = mask
      if idx.all():
        l_kd = teacher_kd_loss(out["Hhat_1s"], batch["H_teacher"], gw)
      else:
        l_kd = teacher_kd_loss(out["Hhat_1s"][idx], batch["H_teacher"], gw[idx])
  total = l_ord + lambda_gt * l_gt + lambda_kd * l_kd
  terms.update(
    {
      "L_gt": float(l_gt.detach()),
      "L_kd": float(l_kd.detach()),
      "L_total": float(total.detach()),
      "lambda_gt": float(lambda_gt),
      "lambda_kd": float(lambda_kd),
    }
  )
  return total, terms
