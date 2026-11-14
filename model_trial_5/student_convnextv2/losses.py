"""Latent matching to parent F_dec plus Pair-CF teacher heatmap terms.

Latent terms copy the M1 student recipe that actually matched ``F_dec``.
Heatmap terms reuse the parent treatment losses (BCE-on-logits, blob BCE,
STOP/SAFE hinges, false-positive, direction) so the frozen decoder stays
calibrated while the backbone is pulled onto the cached latents.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from collision_jepa3.losses.direction import direction_alignment_loss
from collision_jepa3.losses.heatmap import (
    appear_cell_weight,
    blob_bce_logits,
    false_positive_penalty,
    safe_peak_hinge,
    small_blob_cell_weight,
    soft_bce_logits,
    stop_peak_hinge,
)
from student_model.loss import blob_weights, relation_loss, spatial_cosine, weighted_l1

STOP, THREAT, SAFE = 2, 1, 0


def peak_class_t(gt: torch.Tensor) -> torch.Tensor:
    p = gt.flatten(1).max(dim=1).values
    y = torch.zeros(gt.size(0), dtype=torch.long, device=gt.device)
    y = torch.where(p >= 0.25, torch.full_like(y, THREAT), y)
    y = torch.where(p >= 0.60, torch.full_like(y, STOP), y)
    return y


def selective_kd(h: torch.Tensor, ht: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    yg, yp = peak_class_t(gt), peak_class_t(ht)
    agree = (yg == yp).to(h.dtype)
    den = agree.sum().clamp_min(1.0)
    return ((h - ht).abs() * agree.view(-1, 1, 1)).sum() / (den * 25.0)


def convnext_student_loss(
    out: dict,
    batch: dict,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    w_feat: float = 1.0,
    w_cos: float = 0.30,
    w_rel: float = 0.20,
    w_1s: float = 0.50,
    w_blob: float = 0.15,
    w_stop: float = 0.50,
    w_safe: float = 0.25,
    w_fp: float = 0.05,
    w_dir: float = 0.03,
    w_kd: float = 0.30,
    task_scale: float = 0.50,
) -> tuple[torch.Tensor, dict[str, float]]:
    fs = out["F"]
    ft = batch["F_teacher"]
    gt = batch["gt_1s"].float()
    ht = batch["H_teacher"].float()
    h = out["Hhat_1s"].float()
    logits = out["Hhat_1s_logits"].float()
    h_now = batch.get("h_now")
    amb = batch.get("amb_1s")
    keep = None if amb is None else (~amb.to(dtype=torch.bool))

    w_spatial = blob_weights(
        ft,
        h_now,
        gt,
        energy_alpha=2.0,
        appear_boost=4.0,
        stop_boost=2.0,
    )
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    l_feat = weighted_l1(fs_n, ft_n, w_spatial)
    l_cos = spatial_cosine(fs, ft, w_spatial)
    l_rel = relation_loss(fs, ft)

    appear = None
    if h_now is not None:
        appear = appear_cell_weight(h_now.float(), gt, now_threshold=0.1, future_threshold=0.5, weight=8.0)
    small = small_blob_cell_weight(gt, stop_threshold=0.60, occupied_threshold=0.5, max_cells=4, weight=2.0)
    cell_w = small if appear is None else (appear * small)

    l_1s = soft_bce_logits(
        logits,
        gt,
        sample_mask=keep,
        pos_threshold=0.1,
        pos_weight=2.0,
        cell_weight=cell_w,
    )
    l_blob = blob_bce_logits(logits, gt, sample_mask=keep, occupied_threshold=0.5)
    l_stop = stop_peak_hinge(h, gt, sample_mask=keep, stop_threshold=0.60, margin=0.05)
    l_safe = safe_peak_hinge(h, gt, sample_mask=keep, caution_threshold=0.25, margin=0.0)
    l_fp = false_positive_penalty(h, gt, sample_mask=keep, safe_threshold=0.05, margin=0.10)
    row_w = torch.ones(5, device=h.device, dtype=torch.float32)
    l_dir = direction_alignment_loss(
        h,
        gt,
        row_weights=row_w,
        left_cols=[0, 1],
        center_cols=[2],
        right_cols=[3, 4],
        sample_mask=keep,
        temperature=0.08,
        threat_threshold=0.25,
    )
    l_kd = selective_kd(h, ht, gt)

    l_latent = float(w_feat) * l_feat + float(w_cos) * l_cos + float(w_rel) * l_rel
    l_task = (
        float(w_1s) * l_1s
        + float(w_blob) * l_blob
        + float(w_stop) * l_stop
        + float(w_safe) * l_safe
        + float(w_fp) * l_fp
        + float(w_dir) * l_dir
        + float(w_kd) * l_kd
    )
    total = l_latent + float(task_scale) * l_task
    terms = {
        "L_feat": float(l_feat.detach()),
        "L_cos": float(l_cos.detach()),
        "L_rel": float(l_rel.detach()),
        "L_1s": float(l_1s.detach()),
        "L_blob": float(l_blob.detach()),
        "L_stop": float(l_stop.detach()),
        "L_safe": float(l_safe.detach()),
        "L_fp": float(l_fp.detach()),
        "L_dir": float(l_dir.detach()),
        "L_kd": float(l_kd.detach()),
        "L_latent": float(l_latent.detach()),
        "L_task": float(l_task.detach()),
        "L_total": float(total.detach()),
    }
    return total, terms
