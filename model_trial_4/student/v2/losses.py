"""3-checkpoint distillation: 5×5 focal + 16×16 BCE + 0.5 feature alignment."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def get_lambda_feat(
    epoch: int,
    start_epoch: int = 20,
    duration: int = 25,
    lam_start: float = 0.50,
    lam_end: float = 0.05,
) -> float:
    """Cosine ramp of the feature-loss weight. Holds `lam_start` through
    `start_epoch`, reaches `lam_end` at start_epoch+duration, then holds."""
    if int(epoch) <= int(start_epoch) or int(duration) <= 0:
        return float(lam_start)
    end_epoch = int(start_epoch) + int(duration)
    if int(epoch) >= end_epoch:
        return float(lam_end)
    progress = (int(epoch) - int(start_epoch)) / float(duration)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(lam_end + (lam_start - lam_end) * cosine_decay)


def feat_loss(f_pred: torch.Tensor, f_teacher: torch.Tensor) -> torch.Tensor:
    p = f_pred.float().flatten(1)
    t = f_teacher.float().detach().flatten(1)
    cos = 1.0 - F.cosine_similarity(p, t, dim=1).mean()
    sl1 = F.smooth_l1_loss(f_pred.float(), f_teacher.float().detach(), beta=0.1)
    return cos + 0.5 * sl1


def logits16_loss(z_student: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
    target = torch.sigmoid(z_teacher.float().detach())
    return F.binary_cross_entropy_with_logits(z_student.float(), target)


def cell_hazard_weight(p_teacher: torch.Tensor) -> torch.Tensor:
    y = p_teacher.float()
    w = torch.ones_like(y)
    w = torch.where(y >= 0.25, torch.full_like(w, 2.0), w)
    w = torch.where(y >= 0.60, torch.full_like(w, 4.0), w)
    return w


def focal_5x5(p_student: torch.Tensor, p_teacher: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    y = p_teacher.float().detach().clamp(0.0, 1.0)
    p = p_student.float().clamp(1e-6, 1.0 - 1e-6)
    bce = -(y * p.log() + (1.0 - y) * (1.0 - p).log())
    p_t = p * y + (1.0 - p) * (1.0 - y)
    w = cell_hazard_weight(y) * (1.0 - p_t).clamp(min=0.0).pow(float(gamma))
    return (w * bce).mean()


def student_v2_loss(out, batch, *, w_5: float = 1.0, w_16: float = 1.0, w_feat: float = 0.5):
    l_feat = feat_loss(out["F_pred"], batch["F_teacher"])
    l_16 = logits16_loss(out["Z_student_16"], batch["Z_teacher_16"])
    l_5 = focal_5x5(out["P_student_5"], batch["P_teacher_5"])
    total = float(w_5) * l_5 + float(w_16) * l_16 + float(w_feat) * l_feat
    den = total.detach().clamp_min(1e-8)
    c5, c16, cf = float(w_5) * l_5, float(w_16) * l_16, float(w_feat) * l_feat
    terms = {
        "L_5x5": float(l_5.detach()),
        "L_16x16": float(l_16.detach()),
        "L_feat": float(l_feat.detach()),
        "L_5x5_contrib": float(c5.detach()),
        "L_16x16_contrib": float(c16.detach()),
        "L_feat_contrib": float(cf.detach()),
        "share_5x5": float((c5.detach() / den).clamp(0, 1)),
        "share_16x16": float((c16.detach() / den).clamp(0, 1)),
        "share_feat": float((cf.detach() / den).clamp(0, 1)),
        "L_total": float(total.detach()),
        "lam_feat": float(w_feat),
    }
    return total, terms
