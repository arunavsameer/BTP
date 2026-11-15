"""V5 distillation: QFL 5×5 + hazard-weighted 16×16 + λ_feat floor 0.10."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def get_lambda_feat_v5(
    epoch: int,
    start_epoch: int = 20,
    duration: int = 25,
    lam_start: float = 0.50,
    lam_end: float = 0.10,
) -> float:
    """0.50 through epoch 20, cosine to 0.10 over 25 epochs, then hold."""
    if int(epoch) <= int(start_epoch) or int(duration) <= 0:
        return float(lam_start)
    end_epoch = int(start_epoch) + int(duration)
    if int(epoch) >= end_epoch:
        return float(lam_end)
    progress = (int(epoch) - int(start_epoch)) / float(duration)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(lam_end + (lam_start - lam_end) * cosine_decay)


def student_v5_loss(out, batch, *, w_feat: float, gamma: float = 2.0):
    logits_5 = out["Hhat_1s_logits"]
    p_tea_5 = batch["P_teacher_5"].float().detach()
    if logits_5.ndim == 3:
        logits_5 = logits_5.unsqueeze(1)
    if p_tea_5.ndim == 3:
        p_tea_5 = p_tea_5.unsqueeze(1)
    p_tea_5 = p_tea_5.clamp(0.0, 1.0)

    p5 = torch.sigmoid(logits_5)
    bce5 = F.binary_cross_entropy_with_logits(logits_5.float(), p_tea_5, reduction="none")
    h5 = torch.ones_like(p_tea_5)
    h5 = torch.where(p_tea_5 >= 0.25, torch.full_like(h5, 2.5), h5)
    h5 = torch.where(p_tea_5 >= 0.60, torch.full_like(h5, 8.0), h5)
    h5 = torch.where((p_tea_5 >= 0.60) & (p5 < p_tea_5), h5 * 1.5, h5)
    w5 = h5 * torch.abs(p_tea_5 - p5).pow(float(gamma))
    l_5x5_cell = (w5 * bce5).mean()
    p5_pool = F.avg_pool2d(p5, kernel_size=3, stride=1, padding=1)
    y5_pool = F.avg_pool2d(p_tea_5, kernel_size=3, stride=1, padding=1)
    l_5x5_nbr = F.smooth_l1_loss(p5_pool, y5_pool, beta=0.05)
    l_5 = l_5x5_cell + 0.30 * l_5x5_nbr

    logits_16 = out["Z_student_16"].float()
    z_tea = batch["Z_teacher_16"].float().detach()
    y16 = torch.sigmoid(z_tea)
    bce16 = F.binary_cross_entropy_with_logits(logits_16, y16, reduction="none")
    h16 = torch.ones_like(y16)
    h16 = torch.where(y16 >= 0.25, torch.full_like(h16, 2.5), h16)
    h16 = torch.where(y16 >= 0.60, torch.full_like(h16, 6.0), h16)
    l_16 = (h16 * bce16).mean()

    f_pred = out["F_pred"].float()
    f_tea = batch["F_teacher"].float().detach()
    cos_sim = F.cosine_similarity(f_pred, f_tea, dim=1).mean()
    l_feat = (1.0 - cos_sim) + 0.5 * F.smooth_l1_loss(f_pred, f_tea, beta=0.1)

    total = l_5 + l_16 + float(w_feat) * l_feat
    den = total.detach().clamp_min(1e-8)
    c5, c16, cf = l_5, l_16, float(w_feat) * l_feat
    terms = {
        "L_5x5": float(l_5.detach()),
        "L_5x5_cell": float(l_5x5_cell.detach()),
        "L_5x5_nbr": float(l_5x5_nbr.detach()),
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
