"""StudentV2 distillation: feature match + 5×5 focal + light STOP/SAFE hinges.

Walkthrough: L = Lfocal(H, P_teacher_5) + LBCE(Z16, Z_teacher_16) + 0.5 (Lcos + 0.5 SmoothL1).
Parent cache has F_dec and H_teacher, not Z16 (Z_t.npy is the 32×5×5 adapter bottleneck).
So Z16 BCE is omitted. Whitened L1 is optional when trial-5 fdec_channel_stats.npz is loaded.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from .paths import TRIAL3, TRIAL5

sys.path.insert(0, str(TRIAL3))
sys.path.insert(0, str(TRIAL5))
from collision_jepa3.losses.heatmap import safe_peak_hinge, stop_peak_hinge  # noqa: E402


def feat_align(f_pred: torch.Tensor, f_teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = f_pred.float().flatten(1)
    t = f_teacher.float().detach().flatten(1)
    l_cos = 1.0 - F.cosine_similarity(p, t, dim=1).mean()
    l_sl1 = F.smooth_l1_loss(f_pred.float(), f_teacher.float().detach(), beta=0.1)
    return l_cos + 0.5 * l_sl1, l_cos, l_sl1


def whitened_l1(
    f_pred: torch.Tensor, f_teacher: torch.Tensor, mu: torch.Tensor | None, sigma: torch.Tensor | None
) -> torch.Tensor:
    if mu is None or sigma is None:
        return f_pred.new_zeros(())
    fs = (f_pred.float() - mu) / (sigma + 1e-6)
    ft = (f_teacher.float().detach() - mu) / (sigma + 1e-6)
    return (fs - ft).abs().mean()


def cell_hazard_weight(p_teacher: torch.Tensor) -> torch.Tensor:
    y = p_teacher.float()
    w = torch.ones_like(y)
    w = torch.where(y >= 0.25, torch.full_like(w, 2.0), w)
    w = torch.where(y >= 0.60, torch.full_like(w, 4.0), w)
    return w


def focal_heatmap(p_student: torch.Tensor, p_teacher: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Primary 5×5 term: focal BCE vs teacher heatmap (not GT)."""
    y = p_teacher.float().detach().clamp(0.0, 1.0)
    p = p_student.float().clamp(1e-6, 1.0 - 1e-6)
    bce = -(y * p.log() + (1.0 - y) * (1.0 - p).log())
    p_t = p * y + (1.0 - p) * (1.0 - y)
    w = cell_hazard_weight(y) * (1.0 - p_t).clamp(min=0.0).pow(float(gamma))
    return (w * bce).mean()


def student_v2_loss(
    out: dict,
    batch: dict,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    *,
    w_focal: float = 1.0,
    w_feat: float = 0.5,
    w_white: float = 0.25,
    w_stop: float = 0.25,
    w_safe: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    fs, ft = out["F_pred"] if "F_pred" in out else out["F"], batch["F_teacher"]
    l_feat, l_cos, l_sl1 = feat_align(fs, ft)
    l_white = whitened_l1(fs, ft, mu, sigma)
    l_focal = focal_heatmap(out["Hhat_1s"].float(), batch["H_teacher"].float())

    gt = batch["gt_1s"].float()
    h = out["Hhat_1s"].float()
    amb = batch.get("amb_1s")
    keep = None if amb is None else (~amb.to(dtype=torch.bool))
    l_stop = stop_peak_hinge(h, gt, sample_mask=keep, stop_threshold=0.60, margin=0.05)
    l_safe = safe_peak_hinge(h, gt, sample_mask=keep, caution_threshold=0.25, margin=0.0)

    total = (
        float(w_focal) * l_focal
        + float(w_feat) * l_feat
        + float(w_white) * l_white
        + float(w_stop) * l_stop
        + float(w_safe) * l_safe
    )
    terms = {
        "L_focal": float(l_focal.detach()),
        "L_feat": float(l_feat.detach()),
        "L_cos": float(l_cos.detach()),
        "L_sl1": float(l_sl1.detach()),
        "L_white": float(l_white.detach()),
        "L_stop": float(l_stop.detach()),
        "L_safe": float(l_safe.detach()),
        "L_total": float(total.detach()),
    }
    return total, terms


# Back-compat name used by the first trial-6 sketch.
cnn_edge_loss = student_v2_loss
