"""Shared utilities for pair-conditioned counterfactual diagnostics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA

from collision_jepa3.analysis.channel_icc import anchor_feature_vec
from collision_jepa3.data.dataset import TeacherDataset
from collision_jepa3.data.pairing import background_pair_valid, heatmaps_disagree
from collision_jepa3.evaluation.metrics import STOP, SAFE, CAUTION, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics

GT_TOL = 1e-3


@dataclass
class PairRow:
    pair_key: str
    clip_a: str
    clip_b: str
    anchor: int
    scenario_group_id: str
    background_pair_id: str
    outcome_type: str
    background_a: str
    background_b: str
    pre_fork: bool
    gt_peak: float
    gt_occ_cells: int
    gt_occ_rows: list[int]
    gt_occ_cols: list[int]
    lighting_a: str | None
    lighting_b: str | None
    scenario_type: str


def pooled_F(tokens: np.ndarray) -> np.ndarray:
    return anchor_feature_vec(tokens).astype(np.float32)


def broadcast_channel_delta(tokens: np.ndarray, delta_c: np.ndarray) -> np.ndarray:
    """tokens [T,C], delta_c [C] — same correction on every token position."""
    t = np.asarray(tokens, dtype=np.float32)
    d = np.asarray(delta_c, dtype=np.float32).reshape(1, -1)
    return t - d


def add_channel_delta(tokens: np.ndarray, delta_c: np.ndarray) -> np.ndarray:
    t = np.asarray(tokens, dtype=np.float32)
    d = np.asarray(delta_c, dtype=np.float32).reshape(1, -1)
    return t + d


def proj_subspace(vec: np.ndarray, U: np.ndarray) -> np.ndarray:
    """U [k,C]. Returns P @ vec = U.T @ (U @ vec)."""
    v = np.asarray(vec, dtype=np.float64)
    coeff = U @ v
    return (coeff @ U).astype(np.float32)


def pair_shrink_d(D: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    return (float(lam) * proj_subspace(D, U)).astype(np.float32)


def global_shrink_delta(F: np.ndarray, mu: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    return (float(lam) * proj_subspace(F - mu, U)).astype(np.float32)


def random_orthonormal_U(c_dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((c_dim, k))
    Q, _ = np.linalg.qr(M, mode="reduced")
    return Q.T.astype(np.float64)  # [k,C]


def fit_pair_pca(D_train: np.ndarray, k: int) -> dict[str, np.ndarray]:
    mu_d = D_train.mean(axis=0)
    X = D_train - mu_d
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(X)
    U = pca.components_.astype(np.float64)
    return {
        "mu_d": mu_d.astype(np.float32),
        "U": U,
        "eigenvalues": pca.explained_variance_.astype(np.float64),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
    }


def participation_ratio(evals: np.ndarray) -> float:
    lam = np.clip(np.asarray(evals, dtype=np.float64), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return 0.0
    return float((s * s) / np.square(lam).sum())


def bootstrap_subspace_overlap(D: np.ndarray, k: int, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    _, U_ref, _ = fit_pca_full(D, k)
    overlaps = []
    for _ in range(n_boot):
        idx = rng.choice(D.shape[0], size=D.shape[0], replace=True)
        _, U_b, _ = fit_pca_full(D[idx], k)
        M = U_ref @ U_b.T
        s = np.linalg.svd(M, compute_uv=False)
        overlaps.append(float(np.mean(np.clip(s, 0.0, 1.0) ** 2)))
    return {"mean": float(np.mean(overlaps)), "std": float(np.std(overlaps))}


def fit_pca_full(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = fit_pair_pca(D, k)
    return fit["mu_d"], fit["U"], fit["explained_variance_ratio"]


def _occ_meta(hm: np.ndarray, thr: float = 0.05) -> tuple[float, int, list[int], list[int]]:
    x = np.asarray(hm, dtype=np.float64)
    peak = float(x.max())
    occ = x >= thr
    rows, cols = np.where(occ)
    return peak, int(occ.sum()), sorted(set(rows.tolist())), sorted(set(cols.tolist()))


def enumerate_pairs(ds: TeacherDataset, split_name: str) -> dict[str, Any]:
    recs = ds.records_by_id
    groups: dict[tuple, list[str]] = defaultdict(list)
    for clip_id, t in ds.index:
        rec = recs[clip_id]
        if not rec.background_pair_id:
            continue
        groups[(rec.background_pair_id, rec.outcome_type, int(t))].append(clip_id)

    valid: list[PairRow] = []
    invalid_reasons: dict[str, int] = defaultdict(int)
    mismatched_gt = 0
    for (bpid, outcome, t), cids in groups.items():
        if len(cids) != 2:
            invalid_reasons[f"n_clips_{len(cids)}"] += 1
            continue
        a, b = sorted(cids)
        ra, rb = recs[a], recs[b]
        if ra.background_id == rb.background_id:
            invalid_reasons["same_background"] += 1
            continue
        if ra.scenario_group_id != rb.scenario_group_id:
            invalid_reasons["scenario_mismatch"] += 1
            continue
        fork = int(ra.fork_frame or rb.fork_frame or 225)
        hm_a = ds._heatmaps[a]
        hm_b = ds._heatmaps[b]
        if not background_pair_valid(ra, rb, hm_a, hm_b, t, ds.horizons, ds.bg_tol):
            mismatched_gt += 1
            continue
        h1a = ds.h1s_at_anchor(a, t)
        h1b = ds.h1s_at_anchor(b, t)
        if heatmaps_disagree(h1a, h1b, GT_TOL):
            mismatched_gt += 1
            continue
        peak, occ, rr, cc = _occ_meta(h1a)
        pk = f"{bpid}|{outcome}|{t}"
        valid.append(
            PairRow(
                pair_key=pk,
                clip_a=a,
                clip_b=b,
                anchor=t,
                scenario_group_id=ra.scenario_group_id,
                background_pair_id=bpid,
                outcome_type=outcome,
                background_a=ra.background_id,
                background_b=rb.background_id,
                pre_fork=t < fork,
                gt_peak=peak,
                gt_occ_cells=occ,
                gt_occ_rows=rr,
                gt_occ_cols=cc,
                lighting_a=ra.lighting,
                lighting_b=rb.lighting,
                scenario_type=ra.scenario_type,
            )
        )
    return {
        "split": split_name,
        "n_valid_pairs": len(valid),
        "mismatched_gt": mismatched_gt,
        "invalid_reasons": dict(invalid_reasons),
        "pairs": valid,
    }


@dataclass
class WindowRow:
    clip_id: str
    anchor: int
    pair_key: str
    pair_side: str  # "a" | "b" | "oracle"
    scenario_group_id: str
    pre_fork: bool
    gt_peak: float
    mate_clip_id: str


def build_val_windows(pairs: list[PairRow]) -> list[WindowRow]:
    rows: list[WindowRow] = []
    for p in pairs:
        for side, cid in (("a", p.clip_a), ("b", p.clip_b)):
            mate = p.clip_b if side == "a" else p.clip_a
            rows.append(
                WindowRow(
                    clip_id=cid,
                    anchor=p.anchor,
                    pair_key=p.pair_key,
                    pair_side=side,
                    scenario_group_id=p.scenario_group_id,
                    pre_fork=p.pre_fork,
                    gt_peak=p.gt_peak,
                    mate_clip_id=mate,
                )
            )
    return rows


def warn_stats(pred: np.ndarray, gt: np.ndarray, warn: dict) -> dict[str, float]:
    caution = float(warn["caution_threshold"])
    stop = float(warn["stop_threshold"])
    pr_peak = pred.reshape(len(pred), -1).max(axis=1)
    gt_peak = gt.reshape(len(gt), -1).max(axis=1)

    def band(p):
        out = np.zeros(p.shape[0], dtype=np.int64)
        out[p >= caution] = 1
        out[p >= stop] = 2
        return out

    gtf = band(gt_peak)
    prf = band(pr_peak)
    stop_m = gtf == STOP
    safe_m = gtf == SAFE
    threat_m = gtf == CAUTION
    return {
        "stop_recall": float((prf[stop_m] == STOP).mean()) if stop_m.any() else float("nan"),
        "safe_recall": float((prf[safe_m] == SAFE).mean()) if safe_m.any() else float("nan"),
        "threat_recall": float((prf[threat_m] >= CAUTION).mean()) if threat_m.any() else float("nan"),
        "warning_flip_rate": float((gtf != prf).mean()),
        "stop_to_safe": float(((gtf == STOP) & (prf == SAFE)).mean()),
        "safe_to_stop": float(((gtf == SAFE) & (prf == STOP)).mean()),
    }


def cell_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    op = operating_metrics(pred, gt, valid)
    mse = float(op["mse"])
    return {
        "recall_at_5fa": float(op["recall_at_5fa"]),
        "ap_high": float(op["ap_high"]),
        "mse": mse,
    }


def horizon_style_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    pre_fork: np.ndarray,
    warn: dict,
) -> dict[str, Any]:
    cell = cell_metrics(pred, gt, valid)
    frame = warn_stats(pred, gt, warn)
    # pre/post fork stop recall
    caution = float(warn["caution_threshold"])
    stop = float(warn["stop_threshold"])
    pr_peak = pred.reshape(len(pred), -1).max(axis=1)
    gt_peak = gt.reshape(len(gt), -1).max(axis=1)
    gt_stop = gt_peak >= stop
    pr_stop = pr_peak >= stop
    pre_sr = float(pr_stop[gt_stop & pre_fork].mean()) if (gt_stop & pre_fork).any() else float("nan")
    post_sr = float(pr_stop[gt_stop & (~pre_fork)].mean()) if (gt_stop & (~pre_fork)).any() else float("nan")
    return {**cell, **frame, "pre_fork_stop_recall": pre_sr, "post_fork_stop_recall": post_sr}


def pair_robustness(pairs: list[PairRow], pred_by_key: dict[tuple[str, int], np.ndarray]) -> dict[str, float]:
    gaps, peak_d, flips = [], [], []
    warn = {"caution_threshold": 0.25, "stop_threshold": 0.60, "row_weights": [1.0] * 5, "left_cols": [0, 1], "center_cols": [2], "right_cols": [3, 4]}
    for p in pairs:
        ha = pred_by_key[(p.clip_a, p.anchor)]
        hb = pred_by_key[(p.clip_b, p.anchor)]
        gaps.append(float(np.abs(ha - hb).mean()))
        pa, pb = float(ha.max()), float(hb.max())
        peak_d.append(abs(pa - pb))
        _, sa, _ = classify_heatmap(ha, warn)
        _, sb, _ = classify_heatmap(hb, warn)
        flips.append(int(sa != sb))
    gaps = np.asarray(gaps)
    peak_d = np.asarray(peak_d)
    return {
        "mean_pair_gap": float(gaps.mean()),
        "median_pair_gap": float(np.median(gaps)),
        "p90_pair_gap": float(np.percentile(gaps, 90)),
        "warning_flip_rate": float(np.mean(flips)),
        "mean_peak_diff": float(peak_d.mean()),
    }


def scenario_bootstrap_ci(
    groups: np.ndarray,
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 500,
    seed: int = 1337,
) -> dict[str, float]:
    """groups and values same length; stat_fn applied to resampled group means."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    g2v: dict[str, list[float]] = defaultdict(list)
    for g, v in zip(groups, values, strict=True):
        g2v[str(g)].append(float(v))
    group_ids = list(g2v.keys())
    means = {g: float(np.mean(v)) for g, v in g2v.items()}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(group_ids, size=len(group_ids), replace=True)
        boots.append(float(np.mean([means[g] for g in pick])))
    arr = np.asarray(boots)
    return {
        "mean": float(np.mean(arr)),
        "p2.5": float(np.percentile(arr, 2.5)),
        "p97.5": float(np.percentile(arr, 97.5)),
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
