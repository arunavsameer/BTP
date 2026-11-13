"""Collision-JEPA S3 — Train Student From Scratch With Direct STOP/SAFE Supervision.

Core module: differentiable peak, two-threshold ordinal targets, balanced weights,
initialization audit, gradient calibration, and statistical comparison.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, HorizonAccumulator, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.student_research.cnn import ArchConfig, StudentRGBNet
from collision_jepa3.student_research.fdec_student import (
    FDEC_FRAME_OFFSETS,
    FDecNormStats,
    FDecSpatialProjection,
    base_concat_config,
    build_parent_decoder,
    decoder_state_hash,
    extract_backbone_state,
    init_shared_backbone,
    load_backbone_state,
)
from collision_jepa3.student_research.fdec_pilot_train import _backbone_hash, _huber_kd
from collision_jepa3.student_research.parent import load_pair_cf_parent, resolve_parent_ckpt


class StudentS3(nn.Module):
    """S3 Student: Fresh Base CNN backbone + fresh spatial projection + frozen parent decoder."""

    def __init__(
        self,
        arch: ArchConfig | None = None,
        n_frames: int = 5,
        *,
        parent_decoder: nn.Module,
        proj_out_grid: int = 16,
    ):
        super().__init__()
        arch = arch or base_concat_config()
        self.backbone = StudentRGBNet(arch, n_frames=n_frames, input_size=160)
        self.proj = FDecSpatialProjection(channels=arch.fine_channels, out_grid=proj_out_grid)
        self.decoder = parent_decoder
        self.fine_channels = arch.fine_channels

        # Permanently freeze decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

    def spatial_features(self, clip: torch.Tensor) -> torch.Tensor:
        return self.backbone._spatial_feats(clip)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        fine = self.spatial_features(clip)
        f_s = self.proj(fine)
        h, logits = self.decoder.forward_with_logits(f_s)
        return {
            "F_student": f_s,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
            "F_backbone": fine,
        }

    def param_audit(self) -> dict[str, Any]:
        enc = sum(p.numel() for p in self.backbone.stem.parameters())
        enc += sum(p.numel() for p in self.backbone.encoder.parameters())
        enc += sum(p.numel() for p in self.backbone.head.parameters())
        proj = sum(p.numel() for p in self.proj.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        total = enc + proj + dec
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "variant": "S3",
            "backbone_params": enc,
            "projection_params": proj,
            "decoder_params": dec,
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "under_700k": bool(total < 700000),
        }


def compute_module_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def smooth_peak(h: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    """Differentiable peak approximation using temperature-scaled spatial softmax.
    h: [B, 5, 5] -> p_s: [B] in [0, 1]
    """
    b = h.shape[0]
    flat = h.view(b, -1)
    w = torch.softmax(flat / float(tau), dim=-1)
    return (flat * w).sum(dim=-1)


def compute_ordinal_targets(p_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft ordinal targets from GT peak:
    y25 = clamp((p_gt - 0.20) / 0.10, 0, 1)
    y60 = clamp((p_gt - 0.55) / 0.10, 0, 1)
    """
    y25 = torch.clamp((p_gt - 0.20) / 0.10, 0.0, 1.0)
    y60 = torch.clamp((p_gt - 0.55) / 0.10, 0.0, 1.0)
    return y25, y60


def compute_ordinal_logits(p_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted threshold logits:
    z25 = (p_s - 0.25) / 0.03
    z60 = (p_s - 0.60) / 0.03
    """
    z25 = (p_s - 0.25) / 0.03
    z60 = (p_s - 0.60) / 0.03
    return z25, z60


def compute_balanced_weights(
    train_rows: list[dict],
    train_peaks_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute scenario- and class-balanced weights for threshold 0.25 and 0.60 across training data.

    Criteria:
    1. Treat scenario_group_id as the independent weighting unit (1 / n_g).
    2. Prevent large scenario groups from dominating.
    3. Balance positive and negative target contributions across groups.
    4. Distribute contributions within each scenario group.
    5. Preserve soft targets around the boundaries.
    6. Normalize final sample weights to mean 1.
    """
    n_samples = len(train_rows)
    y25 = np.clip((train_peaks_gt - 0.20) / 0.10, 0.0, 1.0)
    y60 = np.clip((train_peaks_gt - 0.55) / 0.10, 0.0, 1.0)

    group_to_indices = defaultdict(list)
    for idx, r in enumerate(train_rows):
        group_to_indices[r["scenario_group_id"]].append(idx)

    def _calc_weights(y: np.ndarray) -> tuple[np.ndarray, float, float]:
        w_raw = np.zeros(n_samples, dtype=np.float64)
        W_pos = 0.0
        W_neg = 0.0
        for g, idxs in group_to_indices.items():
            ng = len(idxs)
            for i in idxs:
                W_pos += y[i] / ng
                W_neg += (1.0 - y[i]) / ng

        for g, idxs in group_to_indices.items():
            ng = len(idxs)
            for i in idxs:
                w_raw[i] = (1.0 / ng) * ((y[i] / (2.0 * max(W_pos, 1e-6))) + ((1.0 - y[i]) / (2.0 * max(W_neg, 1e-6))))

        w_norm = w_raw / max(float(w_raw.mean()), 1e-6)
        return w_norm, W_pos, W_neg

    w25, Wpos25, Wneg25 = _calc_weights(y25)
    w60, Wpos60, Wneg60 = _calc_weights(y60)

    groups_pos_25 = set(train_rows[i]["scenario_group_id"] for i in range(n_samples) if y25[i] > 0)
    groups_neg_25 = set(train_rows[i]["scenario_group_id"] for i in range(n_samples) if y25[i] < 1)
    groups_pos_60 = set(train_rows[i]["scenario_group_id"] for i in range(n_samples) if y60[i] > 0)
    groups_neg_60 = set(train_rows[i]["scenario_group_id"] for i in range(n_samples) if y60[i] < 1)

    audit = {
        "n_samples": n_samples,
        "n_independent_scenario_groups": len(group_to_indices),
        "threshold_025": {
            "effective_positive_contribution": float(np.sum(w25 * y25) / np.sum(w25)),
            "effective_negative_contribution": float(np.sum(w25 * (1 - y25)) / np.sum(w25)),
            "groups_contributing_positive": len(groups_pos_25),
            "groups_contributing_negative": len(groups_neg_25),
            "weight_min": float(w25.min()),
            "weight_max": float(w25.max()),
            "weight_std": float(w25.std()),
        },
        "threshold_060": {
            "effective_positive_contribution": float(np.sum(w60 * y60) / np.sum(w60)),
            "effective_negative_contribution": float(np.sum(w60 * (1 - y60)) / np.sum(w60)),
            "groups_contributing_positive": len(groups_pos_60),
            "groups_contributing_negative": len(groups_neg_60),
            "weight_min": float(w60.min()),
            "weight_max": float(w60.max()),
            "weight_std": float(w60.std()),
            "stop_groups_limitation": (
                f"STOP examples present in {len(groups_pos_60)} of {len(group_to_indices)} groups"
            ),
        },
    }
    return w25.astype(np.float32), w60.astype(np.float32), audit


def build_data_audit(
    train_rows: list[dict],
    train_gt: np.ndarray,
    warning: dict,
    verified_pairs_train: int,
) -> dict[str, Any]:
    peaks = train_gt.max(axis=(1, 2))
    tiers = []
    stop_thr = float(warning.get("stop_threshold", 0.60))
    caution_thr = float(warning.get("caution_threshold", 0.25))

    for p in peaks:
        if p >= stop_thr:
            tiers.append("STOP")
        elif p >= caution_thr:
            tiers.append("THREAT")
        else:
            tiers.append("SAFE")

    counts = {c: int(tiers.count(c)) for c in ("SAFE", "THREAT", "STOP")}

    groups_by_tier = {c: set() for c in ("SAFE", "THREAT", "STOP")}
    for t, r in zip(tiers, train_rows):
        groups_by_tier[t].add(r["scenario_group_id"])

    pre_post = {c: {"pre_fork": 0, "post_fork": 0} for c in ("SAFE", "THREAT", "STOP")}
    for t, r in zip(tiers, train_rows):
        k = "pre_fork" if r.get("pre_fork") else "post_fork"
        pre_post[t][k] += 1

    peak_dists = {}
    for c in ("SAFE", "THREAT", "STOP"):
        sub = peaks[np.array(tiers) == c]
        if len(sub) > 0:
            qs = np.percentile(sub, [10, 25, 50, 75, 90, 95, 99])
            peak_dists[c] = {
                "count": int(len(sub)),
                "min": float(sub.min()),
                "p10": float(qs[0]),
                "p25": float(qs[1]),
                "p50": float(qs[2]),
                "p75": float(qs[3]),
                "p90": float(qs[4]),
                "p95": float(qs[5]),
                "p99": float(qs[6]),
                "max": float(sub.max()),
                "mean": float(sub.mean()),
                "std": float(sub.std()),
            }

    near_25 = int(np.sum((peaks >= 0.20) & (peaks <= 0.30)))
    near_60 = int(np.sum((peaks >= 0.55) & (peaks <= 0.65)))

    all_groups = set(r["scenario_group_id"] for r in train_rows)

    return {
        "n_train_samples": len(train_rows),
        "class_counts": counts,
        "independent_scenario_groups_by_class": {c: len(groups_by_tier[c]) for c in counts},
        "total_independent_scenario_groups": len(all_groups),
        "pre_post_fork_counts_by_class": pre_post,
        "gt_peak_distributions_by_class": peak_dists,
        "boundary_samples": {
            "near_0_25_range_0_20_to_0_30": near_25,
            "near_0_60_range_0_55_to_0_65": near_60,
        },
        "verified_background_pairs_train": verified_pairs_train,
        "notes": [
            "GT classes assigned strictly by classify_heatmap thresholds (caution=0.25, stop=0.60)",
            f"STOP class appears in {len(groups_by_tier['STOP'])} of {len(all_groups)} train scenario groups",
            "Scenario group balancing applied in ordinal loss to normalize group mass",
        ],
    }


def _grad_norm_and_flat(params: list[torch.nn.Parameter]) -> tuple[float, torch.Tensor]:
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return 0.0, torch.zeros(1)
    flat = torch.cat(grads)
    return float(flat.norm(2).item()), flat


def calibrate_gradients_s3(
    model: StudentS3,
    batches: list[dict],
    norm: FDecNormStats,
    w25_table: torch.Tensor,
    w60_table: torch.Tensor,
    device: str,
    alpha_kd: float = 0.5,
) -> dict[str, Any]:
    """Perform gradient calibration across 4 training batches.
    Target:
      eta_F * G_F = 0.25 * G_dense
      lambda_ordinal * G_ordinal = 0.5 * G_dense (or 0.25 if conflict)
    """
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    g_dense_list, g_fdec_list, g_ord_list = [], [], []
    g_25_list, g_60_list = [], []
    cos_dense_ord_list, cos_dense_fdec_list = [], []

    for b_idx, raw_batch in enumerate(batches):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
        rids = batch["row_id"].cpu()
        w25_b = w25_table[rids].to(device)
        w60_b = w60_table[rids].to(device)

        # 1. Forward
        out = model(batch["clip"])
        h_s = out["Hhat_1s"]
        gt_1s = batch["gt_1s"]
        h_teacher = batch["H_teacher"]
        f_s = out["F_student"]
        f_t = batch["F_teacher"]

        # Dense loss
        l_gt = F.mse_loss(h_s, gt_1s)
        l_kd = _huber_kd(h_s, h_teacher)
        l_dense = l_gt + alpha_kd * l_kd

        # F_dec loss
        f_sn = (f_s - norm.mu) / (norm.sigma + 1e-6)
        f_tn = (f_t - norm.mu) / (norm.sigma + 1e-6)
        l_fdec = F.smooth_l1_loss(f_sn, f_tn)

        # Ordinal loss
        p_s = smooth_peak(h_s, tau=0.02)
        p_gt = gt_1s.view(gt_1s.shape[0], -1).amax(dim=-1)
        y25, y60 = compute_ordinal_targets(p_gt)
        z25, z60 = compute_ordinal_logits(p_s)
        l25 = F.binary_cross_entropy_with_logits(z25, y25, weight=w25_b)
        l60 = F.binary_cross_entropy_with_logits(z60, y60, weight=w60_b)
        l_ord = 0.5 * (l25 + l60)

        # Compute gradients
        model.zero_grad(set_to_none=True)
        l_dense.backward(retain_graph=True)
        gn_dense, flat_dense = _grad_norm_and_flat(trainable_params)

        model.zero_grad(set_to_none=True)
        l_fdec.backward(retain_graph=True)
        gn_fdec, flat_fdec = _grad_norm_and_flat(trainable_params)

        model.zero_grad(set_to_none=True)
        l_ord.backward(retain_graph=True)
        gn_ord, flat_ord = _grad_norm_and_flat(trainable_params)

        model.zero_grad(set_to_none=True)
        l25.backward(retain_graph=True)
        gn_25, _ = _grad_norm_and_flat(trainable_params)

        model.zero_grad(set_to_none=True)
        l60.backward(retain_graph=True)
        gn_60, _ = _grad_norm_and_flat(trainable_params)

        cos_ord = float(F.cosine_similarity(flat_dense.unsqueeze(0), flat_ord.unsqueeze(0)).item())
        cos_fdec = float(F.cosine_similarity(flat_dense.unsqueeze(0), flat_fdec.unsqueeze(0)).item())

        g_dense_list.append(gn_dense)
        g_fdec_list.append(gn_fdec)
        g_ord_list.append(gn_ord)
        g_25_list.append(gn_25)
        g_60_list.append(gn_60)
        cos_dense_ord_list.append(cos_ord)
        cos_dense_fdec_list.append(cos_fdec)

    model.zero_grad(set_to_none=True)

    mean_dense = float(np.mean(g_dense_list))
    mean_fdec = float(np.mean(g_fdec_list))
    mean_ord = float(np.mean(g_ord_list))
    mean_25 = float(np.mean(g_25_list))
    mean_60 = float(np.mean(g_60_list))
    mean_cos_ord = float(np.mean(cos_dense_ord_list))
    mean_cos_fdec = float(np.mean(cos_dense_fdec_list))

    # Calibration targets
    # Part 4: eta_F * G_F = 0.25 * G_dense
    eta_f = 0.25 * (mean_dense / max(mean_fdec, 1e-8))

    # Part 7: lambda * G_ord = 0.5 * G_dense (or 0.25 if conflict)
    has_conflict = bool(mean_cos_ord < 0.0)
    target_ord_ratio = 0.25 if has_conflict else 0.50
    lambda_ordinal = target_ord_ratio * (mean_dense / max(mean_ord, 1e-8))

    return {
        "n_calibration_batches": len(batches),
        "mean_grad_norm_dense": mean_dense,
        "mean_grad_norm_fdec": mean_fdec,
        "mean_grad_norm_ordinal": mean_ord,
        "mean_grad_norm_threshold_025": mean_25,
        "mean_grad_norm_threshold_060": mean_60,
        "mean_cosine_dense_ordinal": mean_cos_ord,
        "mean_cosine_dense_fdec": mean_cos_fdec,
        "gradient_conflict_detected": has_conflict,
        "target_ordinal_gradient_ratio": target_ord_ratio,
        "target_fdec_gradient_ratio": 0.25,
        "eta_f_selected": float(eta_f),
        "lambda_ordinal_selected": float(lambda_ordinal),
        "notes": [
            "Calibrated exclusively on 4 training batches prior to training step 0",
            f"F_dec scaled to 0.25 dense norm: eta_F = {eta_f:.6f}",
            f"Ordinal scaled to {target_ord_ratio:.2f} dense norm (conflict={has_conflict}): lambda = {lambda_ordinal:.6f}",
        ],
    }


def classify_peak_tier(peak_val: float, warning: dict) -> str:
    stop_thr = float(warning.get("stop_threshold", 0.60))
    caution_thr = float(warning.get("caution_threshold", 0.25))
    if peak_val >= stop_thr:
        return "STOP"
    if peak_val >= caution_thr:
        return "THREAT"
    return "SAFE"


def evaluate_predictions_comprehensive(
    preds: np.ndarray,
    gts: np.ndarray,
    parent_hms: np.ndarray,
    rows: list[dict],
    warning: dict,
    verified_dev_pairs: set[tuple],
) -> dict[str, Any]:
    """Compute all Part 10 primary, fork, pair, and ranking metrics."""
    n = len(preds)
    stop_thr = float(warning.get("stop_threshold", 0.60))
    caution_thr = float(warning.get("caution_threshold", 0.25))

    peaks_pred = preds.max(axis=(1, 2))
    peaks_gt = gts.max(axis=(1, 2))

    pred_tiers = [classify_peak_tier(p, warning) for p in peaks_pred]
    gt_tiers = [classify_peak_tier(p, warning) for p in peaks_gt]

    # Confusion matrix (rows: GT SAFE, THREAT, STOP; cols: Pred SAFE, THREAT, STOP)
    tiers_order = ["SAFE", "THREAT", "STOP"]
    cm = {gt_t: {pred_t: 0 for pred_t in tiers_order} for gt_t in tiers_order}
    for gt_t, pred_t in zip(gt_tiers, pred_tiers):
        cm[gt_t][pred_t] += 1

    # Counts
    n_gt_stop = sum(1 for t in gt_tiers if t == "STOP")
    n_gt_threat = sum(1 for t in gt_tiers if t == "THREAT")
    n_gt_safe = sum(1 for t in gt_tiers if t == "SAFE")

    n_pred_stop = sum(1 for t in pred_tiers if t == "STOP")
    n_pred_threat = sum(1 for t in pred_tiers if t == "THREAT")
    n_pred_safe = sum(1 for t in pred_tiers if t == "SAFE")

    # Primary recalls
    tp_stop = cm["STOP"]["STOP"]
    stop_recall = tp_stop / max(n_gt_stop, 1)

    tp_threat = cm["THREAT"]["THREAT"]
    threat_recall = tp_threat / max(n_gt_threat, 1)

    tp_safe = cm["SAFE"]["SAFE"]
    safe_recall = tp_safe / max(n_gt_safe, 1)

    # Precision & False alarm
    stop_precision = tp_stop / max(n_pred_stop, 1) if n_pred_stop > 0 else 0.0
    safe_false_alarm_rate = (n_gt_safe - tp_safe) / max(n_gt_safe, 1)

    # Misses to SAFE
    frac_gt_stop_as_safe = cm["STOP"]["SAFE"] / max(n_gt_stop, 1)
    frac_gt_threat_as_safe = cm["THREAT"]["SAFE"] / max(n_gt_threat, 1)

    # Balanced recall
    if stop_recall > 0 and safe_recall > 0:
        balanced_recall = float((2 * stop_recall * safe_recall) / (stop_recall + safe_recall))
    else:
        balanced_recall = 0.0

    # Continuous ranking metrics
    valids = np.ones_like(gts, dtype=bool)
    op = operating_metrics(preds, gts, valids, hi=0.5)
    recall_at_5fa = float(op.get("recall_at_5fa", 0.0))
    ap_high = float(op.get("ap_high", 0.0))

    # AUC
    stop_mask = (np.array(gt_tiers) == "STOP") | (np.array(gt_tiers) == "SAFE")
    if np.sum(np.array(gt_tiers) == "STOP") > 0 and np.sum(np.array(gt_tiers) == "SAFE") > 0:
        y_stop = (np.array(gt_tiers)[stop_mask] == "STOP").astype(int)
        auc_stop_safe = float(roc_auc_score(y_stop, peaks_pred[stop_mask]))
    else:
        auc_stop_safe = float("nan")

    threat_mask = (np.array(gt_tiers) == "THREAT") | (np.array(gt_tiers) == "SAFE")
    if np.sum(np.array(gt_tiers) == "THREAT") > 0 and np.sum(np.array(gt_tiers) == "SAFE") > 0:
        y_threat = (np.array(gt_tiers)[threat_mask] == "THREAT").astype(int)
        auc_threat_safe = float(roc_auc_score(y_threat, peaks_pred[threat_mask]))
    else:
        auc_threat_safe = float("nan")

    gt_mse = float(np.mean((preds - gts) ** 2))
    parent_mae = float(np.mean(np.abs(preds - parent_hms)))

    # Peak distribution by GT class
    peak_distributions = {}
    for c in tiers_order:
        sub = peaks_pred[np.array(gt_tiers) == c]
        if len(sub) > 0:
            qs = np.percentile(sub, [10, 25, 50, 75, 90, 95, 99])
            peak_distributions[c] = {
                "count": int(len(sub)),
                "min": float(sub.min()),
                "p10": float(qs[0]),
                "p25": float(qs[1]),
                "p50": float(qs[2]),
                "p75": float(qs[3]),
                "p90": float(qs[4]),
                "p95": float(qs[5]),
                "p99": float(qs[6]),
                "max": float(sub.max()),
                "mean": float(sub.mean()),
                "std": float(sub.std()),
                "frac_ge_0_25": float(np.mean(sub >= caution_thr)),
                "frac_ge_0_60": float(np.mean(sub >= stop_thr)),
            }

    # Fork evaluation
    pre_fork = np.array([bool(r.get("pre_fork")) for r in rows])
    post_fork = ~pre_fork

    def _fork_metrics(mask):
        sub_gt = [gt_tiers[i] for i in range(n) if mask[i]]
        sub_pred = [pred_tiers[i] for i in range(n) if mask[i]]
        n_s = sum(1 for t in sub_gt if t == "STOP")
        tp_s = sum(1 for g, p in zip(sub_gt, sub_pred) if g == "STOP" and p == "STOP")
        n_sf = sum(1 for t in sub_gt if t == "SAFE")
        tp_sf = sum(1 for g, p in zip(sub_gt, sub_pred) if g == "SAFE" and p == "SAFE")
        r_stop = tp_s / max(n_s, 1) if n_s > 0 else float("nan")
        r_safe = tp_sf / max(n_sf, 1) if n_sf > 0 else float("nan")
        far = (n_sf - tp_sf) / max(n_sf, 1) if n_sf > 0 else float("nan")
        return {"stop_recall": r_stop, "safe_recall": r_safe, "false_alarm_rate": far, "n_stop": n_s, "n_safe": n_sf}

    fork_stats = {
        "pre_fork": _fork_metrics(pre_fork),
        "post_fork": _fork_metrics(post_fork),
    }

    # Background pair evaluation
    by_pair_key: dict[tuple, list[dict]] = defaultdict(list)
    for i, r in enumerate(rows):
        pid = r.get("background_pair_id") or ""
        if pid:
            k = (pid, int(r["frame"]))
            by_pair_key[k].append({"peak": peaks_pred[i], "tier": pred_tiers[i], "gt_tier": gt_tiers[i]})

    gaps, flips = [], 0
    both_correct_stop, both_correct_safe = 0, 0
    n_pairs_evaluated = 0

    for k, vals in by_pair_key.items():
        if len(vals) != 2:
            continue
        if k not in verified_dev_pairs:
            continue
        pa, pb = vals[0]["peak"], vals[1]["peak"]
        ta, tb = vals[0]["tier"], vals[1]["tier"]
        ga, gb = vals[0]["gt_tier"], vals[1]["gt_tier"]

        gaps.append(abs(pa - pb))
        if (pa >= stop_thr) != (pb >= stop_thr):
            flips += 1
        if ga == "STOP" and gb == "STOP" and ta == "STOP" and tb == "STOP":
            both_correct_stop += 1
        if ga == "SAFE" and gb == "SAFE" and ta == "SAFE" and tb == "SAFE":
            both_correct_safe += 1
        n_pairs_evaluated += 1

    gaps_arr = np.array(gaps, dtype=float) if gaps else np.array([])
    pair_metrics = {
        "n_verified_pairs_evaluated": n_pairs_evaluated,
        "median_peak_gap": float(np.median(gaps_arr)) if len(gaps_arr) else float("nan"),
        "p90_peak_gap": float(np.quantile(gaps_arr, 0.9)) if len(gaps_arr) else float("nan"),
        "warning_tier_flips": int(flips),
        "warning_tier_flip_rate": float(flips / max(n_pairs_evaluated, 1)),
        "both_background_correct_stop_count": int(both_correct_stop),
        "both_background_correct_safe_count": int(both_correct_safe),
        "both_background_correct_stop_rate": float(both_correct_stop / max(n_pairs_evaluated, 1)),
        "both_background_correct_safe_rate": float(both_correct_safe / max(n_pairs_evaluated, 1)),
    }

    return {
        "stop_recall": stop_recall,
        "safe_recall": safe_recall,
        "threat_recall": threat_recall,
        "stop_precision": stop_precision,
        "safe_false_alarm_rate": safe_false_alarm_rate,
        "fraction_gt_stop_as_safe": frac_gt_stop_as_safe,
        "fraction_gt_threat_as_safe": frac_gt_threat_as_safe,
        "balanced_stop_safe_recall": balanced_recall,
        "recall_at_5fa": recall_at_5fa,
        "ap_high": ap_high,
        "auc_stop_vs_safe": auc_stop_safe,
        "auc_threat_vs_safe": auc_threat_safe,
        "gt_heatmap_mse": gt_mse,
        "parent_heatmap_mae": parent_mae,
        "confusion_matrix": cm,
        "peak_distributions_by_class": peak_distributions,
        "fork_evaluation": fork_stats,
        "pair_evaluation": pair_metrics,
        "n_samples": n,
    }


def paired_scenario_bootstrap(
    preds_s3: np.ndarray,
    preds_s2: np.ndarray,
    preds_parent: np.ndarray,
    gts: np.ndarray,
    rows: list[dict],
    warning: dict,
    n_resamples: int = 2000,
    seed: int = 1337,
) -> dict[str, Any]:
    """Paired scenario-group bootstrap comparing S3 vs S2 and S3 vs Parent."""
    group_map = defaultdict(list)
    for i, r in enumerate(rows):
        group_map[r["scenario_group_id"]].append(i)

    groups = sorted(group_map.keys())
    n_groups = len(groups)
    rng = np.random.default_rng(seed)

    diffs_vs_s2 = {
        "stop_recall": [],
        "safe_recall": [],
        "balanced_recall": [],
        "post_fork_stop_recall": [],
        "safe_false_alarm_rate": [],
    }
    diffs_vs_parent = {
        "stop_recall": [],
        "safe_recall": [],
        "balanced_recall": [],
        "post_fork_stop_recall": [],
        "safe_false_alarm_rate": [],
    }

    stop_thr = float(warning.get("stop_threshold", 0.60))
    caution_thr = float(warning.get("caution_threshold", 0.25))

    p3_peaks = preds_s3.max(axis=(1, 2))
    p2_peaks = preds_s2.max(axis=(1, 2))
    pp_peaks = preds_parent.max(axis=(1, 2))
    gt_peaks = gts.max(axis=(1, 2))

    post_mask = np.array([not bool(r.get("pre_fork")) for r in rows])

    for _ in range(n_resamples):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        idx_list = []
        for g in sampled_groups:
            idx_list.extend(group_map[g])
        idx = np.array(idx_list, dtype=int)

        def _get_metrics(p_sub, gt_sub, post_sub):
            n_stop = np.sum(gt_sub >= stop_thr)
            n_safe = np.sum(gt_sub < caution_thr)
            tp_stop = np.sum((gt_sub >= stop_thr) & (p_sub >= stop_thr))
            tp_safe = np.sum((gt_sub < caution_thr) & (p_sub < caution_thr))
            r_stop = float(tp_stop / max(n_stop, 1))
            r_safe = float(tp_safe / max(n_safe, 1))
            far_safe = float((n_safe - tp_safe) / max(n_safe, 1))
            bal = float((2 * r_stop * r_safe) / (r_stop + r_safe)) if (r_stop > 0 and r_safe > 0) else 0.0

            n_post_stop = np.sum((gt_sub >= stop_thr) & post_sub)
            tp_post_stop = np.sum((gt_sub >= stop_thr) & post_sub & (p_sub >= stop_thr))
            r_post_stop = float(tp_post_stop / max(n_post_stop, 1)) if n_post_stop > 0 else float("nan")

            return {
                "stop_recall": r_stop,
                "safe_recall": r_safe,
                "balanced_recall": bal,
                "post_fork_stop_recall": r_post_stop,
                "safe_false_alarm_rate": far_safe,
            }

        m3 = _get_metrics(p3_peaks[idx], gt_peaks[idx], post_mask[idx])
        m2 = _get_metrics(p2_peaks[idx], gt_peaks[idx], post_mask[idx])
        mp = _get_metrics(pp_peaks[idx], gt_peaks[idx], post_mask[idx])

        for k in diffs_vs_s2:
            diffs_vs_s2[k].append(m3[k] - m2[k])
            diffs_vs_parent[k].append(m3[k] - mp[k])

    def _summarize(diff_dict):
        res = {}
        for k, v in diff_dict.items():
            arr = np.array([x for x in v if not np.isnan(x)], dtype=float)
            if len(arr) > 0:
                res[k] = {
                    "mean_diff": float(arr.mean()),
                    "ci_95_lower": float(np.percentile(arr, 2.5)),
                    "ci_95_upper": float(np.percentile(arr, 97.5)),
                    "p_improvement": float(np.mean(arr > 0)),
                }
            else:
                res[k] = {"mean_diff": float("nan"), "ci_95_lower": float("nan"), "ci_95_upper": float("nan")}
        return res

    summary_s2 = _summarize(diffs_vs_s2)
    summary_parent = _summarize(diffs_vs_parent)

    ci_bal = summary_s2["balanced_recall"]
    overlaps_zero = bool(ci_bal["ci_95_lower"] <= 0.0 <= ci_bal["ci_95_upper"])

    return {
        "n_resamples": n_resamples,
        "n_scenario_groups": n_groups,
        "diff_s3_minus_s2": summary_s2,
        "diff_s3_minus_parent": summary_parent,
        "balanced_recall_s3_vs_s2_ci_overlaps_zero": overlaps_zero,
    }
