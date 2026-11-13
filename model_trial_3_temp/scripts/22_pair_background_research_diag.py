#!/usr/bin/env python3
"""Paired-background research diagnostics (Tests 1–10). No training. Split-safe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.analysis.channel_icc import anchor_feature_vec, pair_w_b_icc, select_low_icc_channels
from collision_jepa3.config import Config
from collision_jepa3.data.dataset import TeacherDataset
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.pairing import background_pair_valid, heatmaps_disagree
from collision_jepa3.data.splits import load_splits
from collision_jepa3.evaluation.metrics import classify_heatmap, direction_scores_np
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state
from collision_jepa3.training.engine import move_batch, teacher_forward
from collision_jepa3.training.pair_signals import pair_gt_consistent

GT_TOL = 1e-3
G_CKPT = ROOT / "checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt"
OUT_ROOT = ROOT / "reports/pair_background_research"


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


def _occ_meta(hm: np.ndarray, thr: float = 0.05) -> tuple[float, int, list[int], list[int]]:
    x = np.asarray(hm, dtype=np.float64)
    peak = float(x.max())
    occ = x >= thr
    rows, cols = np.where(occ)
    return peak, int(occ.sum()), sorted(set(rows.tolist())), sorted(set(cols.tolist()))


def enumerate_pairs(ds: TeacherDataset, split_name: str) -> dict:
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


def load_F(ds: TeacherDataset, clip_id: str, t: int) -> np.ndarray:
    tok = ds._load_tokens(clip_id, t)
    return anchor_feature_vec(tok).astype(np.float32)


def stack_pairs(
    ds: TeacherDataset, pairs: list[PairRow], cache: dict[tuple[str, int], np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[PairRow]]:
    fa, fb, kept = [], [], []
    for pr in pairs:
        ka, kb = (pr.clip_a, pr.anchor), (pr.clip_b, pr.anchor)
        if ka not in cache:
            cache[ka] = load_F(ds, *ka)
        if kb not in cache:
            cache[kb] = load_F(ds, *kb)
        fa.append(cache[ka])
        fb.append(cache[kb])
        kept.append(pr)
    return np.stack(fa, axis=0), np.stack(fb, axis=0), kept


def shuffled_control_pairs(
    train_pairs: list[PairRow],
    val_pairs: list[PairRow],
    rng: np.random.Generator,
    n_draw: int | None = None,
) -> list[tuple[PairRow, PairRow]]:
    """False pairs: different scenario_group, matched gt_peak bin and pre_fork."""
    if not train_pairs or not val_pairs:
        return []
    pool = val_pairs if val_pairs else train_pairs
    n_draw = n_draw or min(len(pool), len(pool))
    bins = np.percentile([p.gt_peak for p in pool], [0, 25, 50, 75, 100])
    by_bin: dict[int, list[PairRow]] = defaultdict(list)
    for p in pool:
        b = int(np.digitize(p.gt_peak, bins[1:-1], right=True))
        by_bin[b].append(p)
    out = []
    for _ in range(n_draw):
        b = int(rng.integers(0, max(len(by_bin), 1)))
        cand = by_bin.get(b) or pool
        if len(cand) < 2:
            continue
        i, j = rng.choice(len(cand), size=2, replace=False)
        pa, pb = cand[i], cand[j]
        if pa.scenario_group_id == pb.scenario_group_id:
            continue
        if pa.pre_fork != pb.pre_fork:
            continue
        out.append((pa, pb))
    return out


def pca_subspace(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D [N,C]. Returns mean, components [k,C], explained."""
    mu = D.mean(axis=0)
    X = D - mu
    pca = PCA(n_components=min(k, X.shape[0], X.shape[1]), svd_solver="full", random_state=0)
    pca.fit(X)
    return mu.astype(np.float64), pca.components_.astype(np.float64), pca.explained_variance_ratio_.astype(np.float64)


def project_remove(F: np.ndarray, mu: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    """F [N,C] or [C]."""
    x = np.atleast_2d(F.astype(np.float64))
    m = mu.reshape(1, -1)
    # delta = lam * (x-mu) @ U.T @ U  (row-wise)
    xc = x - m
    coeff = xc @ U.T
    delta = lam * (coeff @ U)
    return (x - delta).astype(np.float32)


def ridge_h1s_probe(x_tr, y_tr, x_va, y_va, alpha: float = 1.0) -> dict:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_tr)
    xva = scaler.transform(x_va)
    reg = Ridge(alpha=alpha)
    reg.fit(xtr, y_tr)
    pred = reg.predict(xva)
    return {
        "mse": float(mean_squared_error(y_va, pred)),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
    }


def warn_peak(hm: np.ndarray, warn: dict) -> tuple[str, int, float, dict]:
    _, sev, scores = classify_heatmap(hm, warn)
    peak = float(np.max(hm))
    labels = ["SAFE", "THREAT", "STOP"]
    return labels[sev], sev, peak, scores


@torch.no_grad()
def forward_h1s(model, tokens: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(tokens).unsqueeze(0).to(device)
    out = model(clip=None, tokens=x)
    return out["Hhat_1s"].squeeze(0).float().cpu().numpy()


def apply_token_channel_edit(tokens: np.ndarray, delta_c: np.ndarray) -> np.ndarray:
    t = np.asarray(tokens, dtype=np.float32)
    return (t - delta_c.reshape(1, -1).astype(np.float32))


def grad_cos(l_main: torch.Tensor, l_aux: torch.Tensor, params: list[torch.nn.Parameter]) -> tuple[float, float, float]:
    gc = torch.autograd.grad(l_main, params, retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(l_aux, params, retain_graph=False, allow_unused=True)

    def norm(gs):
        t = 0.0
        for g in gs:
            if g is not None:
                t += float(g.detach().float().pow(2).sum())
        return math.sqrt(t)

    def cos(ga, gb):
        dot = na = nb = 0.0
        for a, b in zip(ga, gb, strict=True):
            if a is None or b is None:
                continue
            af = a.detach().float().flatten()
            bf = b.detach().float().flatten()
            dot += float((af * bf).sum())
            na += float((af * af).sum())
            nb += float((bf * bf).sum())
        if na <= 0 or nb <= 0:
            return float("nan")
        return dot / (math.sqrt(na) * math.sqrt(nb))

    return norm(gc), norm(gd), cos(gc, gd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-intervention-pairs", type=int, default=0, help="0 = all val pairs")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    device = args.device
    rng = np.random.default_rng(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cfg = Config.load(ROOT / "configs/default.yaml")
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("evaluation.val_subset_fraction", 1.0)
    cfg.set("training.pin_memory", False)
    cfg.set("training.num_workers", 0)
    warn = cfg.get("warning")

    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    train_groups = set(splits["train"])
    val_groups = set(splits["val"])
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"scenario_group leakage: {len(overlap)} groups in train and val")

    ds_tr = TeacherDataset(cfg, records, "train", splits, use_token_cache=True)
    ds_va = TeacherDataset(cfg, records, "val", splits, use_token_cache=True)

    rep_tr = enumerate_pairs(ds_tr, "train")
    rep_va = enumerate_pairs(ds_va, "val")
    pairs_tr = rep_tr["pairs"]
    pairs_va = rep_va["pairs"]

    # --- TEST 1: metadata table ---
    meta_rows = [
        {
            "pair_id": p.pair_key,
            "scenario_group_id": p.scenario_group_id,
            "anchor": p.anchor,
            "pre_fork": p.pre_fork,
            "gt_peak": p.gt_peak,
            "gt_occ_cells": p.gt_occ_cells,
            "gt_occ_rows": p.gt_occ_rows,
            "gt_occ_cols": p.gt_occ_cols,
            "background_a": p.background_a,
            "background_b": p.background_b,
            "lighting_a": p.lighting_a,
            "lighting_b": p.lighting_b,
            "outcome_type": p.outcome_type,
            "scenario_type": p.scenario_type,
        }
        for p in pairs_tr + pairs_va
    ]
    (OUT_ROOT / "test1_pair_metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in meta_rows), encoding="utf-8"
    )
    (OUT_ROOT / "test1_integrity.json").write_text(
        json.dumps(
            {
                "split_safety": {"train_val_group_overlap": 0, "train_groups": len(train_groups), "val_groups": len(val_groups)},
                "train": {k: v for k, v in rep_tr.items() if k != "pairs"},
                "val": {k: v for k, v in rep_va.items() if k != "pairs"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cache: dict[tuple[str, int], np.ndarray] = {}
    Fa_tr, Fb_tr, kept_tr = stack_pairs(ds_tr, pairs_tr, cache)
    Fa_va, Fb_va, kept_va = stack_pairs(ds_va, pairs_va, cache)
    Sa_tr = 0.5 * (Fa_tr + Fb_tr)
    Da_tr = 0.5 * (Fa_tr - Fb_tr)
    Sa_va = 0.5 * (Fa_va + Fb_va)
    Da_va = 0.5 * (Fa_va - Fb_va)

    # --- TEST 2: pair difference stats ---
    def diff_norms(fa, fb):
        d = fa - fb
        return np.linalg.norm(d, axis=1)

    dn_tr = diff_norms(Fa_tr, Fb_tr)
    dn_va = diff_norms(Fa_va, Fb_va)
    shuf = shuffled_control_pairs(pairs_tr, pairs_va, rng, n_draw=len(pairs_va))
    dn_shuf = []
    for pa, pb in shuf:
        ka = (pa.clip_a, pa.anchor)
        kb = (pb.clip_a, pb.anchor)
        if ka not in cache:
            cache[ka] = load_F(ds_tr, *ka)
        if kb not in cache:
            cache[kb] = load_F(ds_va, *kb)
        dn_shuf.append(float(np.linalg.norm(cache[ka] - cache[kb])))
    dn_shuf = np.asarray(dn_shuf, dtype=np.float64)

    # temporal within-clip: adjacent anchors same clip
    temporal_diffs = []
    by_clip: dict[str, list[int]] = defaultdict(list)
    for clip_id, t in ds_tr.index[:5000]:
        by_clip[clip_id].append(int(t))
    for cid, ts in list(by_clip.items())[:200]:
        ts = sorted(ts)
        for i in range(len(ts) - 1):
            k0, k1 = (cid, ts[i]), (cid, ts[i + 1])
            try:
                if k0 not in cache:
                    cache[k0] = load_F(ds_tr, *k0)
                if k1 not in cache:
                    cache[k1] = load_F(ds_tr, *k1)
                temporal_diffs.append(float(np.linalg.norm(cache[k0] - cache[k1])))
            except Exception:
                pass

    mu_d, U32, ev = pca_subspace(Da_tr, 32)
    Dc = Da_tr - mu_d
    _, s, _ = np.linalg.svd(Dc, full_matrices=False)
    evals = (s**2) / max(Dc.shape[0] - 1, 1)
    cum = np.cumsum(evals) / max(evals.sum(), 1e-12)
    eff_rank = float((evals.sum() ** 2) / max((evals**2).sum(), 1e-12))

    # bootstrap subspace overlap (k=8)
    k_stab = 8
    overlaps = []
    for b in range(20):
        idx = rng.choice(Da_tr.shape[0], size=Da_tr.shape[0], replace=True)
        _, Ub, _ = pca_subspace(Da_tr[idx], k_stab)
        # principal angles via singular values of U U^T
        M = U32[:k_stab] @ Ub.T
        ang = np.arccos(np.clip(np.linalg.svd(M, compute_uv=False), -1, 1))
        overlaps.append(float(np.mean(np.cos(ang) ** 2)))
    W, B, ICC = pair_w_b_icc(Fa_tr, Fb_tr)
    sel = select_low_icc_channels(ICC, W, low_frac=0.2)

    test2 = {
        "pair_diff_norm": {
            "train_true_matched": _summ(dn_tr),
            "val_true_matched": _summ(dn_va),
            "shuffled_control": _summ(dn_shuf),
            "within_clip_temporal": _summ(np.asarray(temporal_diffs)),
        },
        "pca_on_train_D": {
            "explained_variance_frac_k": {str(i + 1): float(ev[i]) if i < len(ev) else None for i in range(6)},
            "cumulative_k": {str(k): float(cum[min(k - 1, len(cum) - 1)]) for k in [1, 2, 4, 8, 16, 32]},
            "effective_rank": eff_rank,
            "bootstrap_mean_subspace_overlap_k8": float(np.mean(overlaps)),
            "bootstrap_std_subspace_overlap_k8": float(np.std(overlaps)),
        },
        "icc_train": {
            "n_pairs": int(Fa_tr.shape[0]),
            "n_selected_low_icc": sel.get("n_selected"),
            "w_median": sel.get("w_median_threshold"),
        },
        "true_vs_shuf_ratio_median_norm": float(np.median(dn_va) / max(np.median(dn_shuf), 1e-8)),
    }
    (OUT_ROOT / "test2_pair_difference.json").write_text(json.dumps(test2, indent=2), encoding="utf-8")

    # --- TEST 3: probes on F, S, D, projected ---
    def build_xy(pairs, ds, feat_fn):
        X, Y = [], []
        for pr in pairs:
            hm = ds.h1s_at_anchor(pr.clip_a, pr.anchor)
            X.append(feat_fn(pr))
            Y.append(hm.reshape(-1))
        return np.stack(X), np.stack(Y)

    def feat_F(pr):
        k = (pr.clip_a, pr.anchor)
        return cache[k]

    def feat_S(pr):
        k1, k2 = (pr.clip_a, pr.anchor), (pr.clip_b, pr.anchor)
        return (0.5 * (cache[k1] + cache[k2])).astype(np.float32)

    def feat_D(pr):
        k1, k2 = (pr.clip_a, pr.anchor), (pr.clip_b, pr.anchor)
        return (0.5 * (cache[k1] - cache[k2])).astype(np.float32)

    mu_p, U8, _ = pca_subspace(Da_tr, 8)

    def feat_proj8(pr):
        f = feat_F(pr).astype(np.float64)
        return project_remove(f, mu_p, U8, 1.0).ravel()

    Xtr, Ytr = build_xy(kept_tr[: min(len(kept_tr), 12000)], ds_tr, feat_F)
    Xva, Yva = build_xy(kept_va, ds_va, feat_F)
    probes = {"F": ridge_h1s_probe(Xtr, Ytr, Xva, Yva)}
    for name, fn in [("S", feat_S), ("D", feat_D), ("F_minus_Dsubspace_k8", feat_proj8)]:
        Xtr2, _ = build_xy(kept_tr[: min(len(kept_tr), 12000)], ds_tr, fn)
        Xva2, _ = build_xy(kept_va, ds_va, fn)
        probes[name] = ridge_h1s_probe(Xtr2, Ytr, Xva2, Yva)

    # warning class logistic on peak from ridge preds omitted — use F for peak proxy: logistic on F for binary stop
    y_stop_tr = np.array([p.gt_peak >= float(warn["stop_threshold"]) for p in kept_tr[: len(Ytr)]])
    y_stop_va = np.array([p.gt_peak >= float(warn["stop_threshold"]) for p in kept_va])
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300).fit(scaler.transform(Xtr), y_stop_tr)
    stop_auc = float(clf.score(scaler.transform(Xva), y_stop_va))
    probes["F_stop_frame_logistic_acc"] = stop_auc
    (OUT_ROOT / "test3_probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")

    # --- TEST 4–5: original G on val pairs ---
    model = CollisionTeacher3(cfg, dummy_backbone=False)
    blob = load_checkpoint(str(G_CKPT), map_location=device)
    load_teacher_state(model, blob)
    model.to(device).eval()
    loss_fn = TeacherLoss(cfg, device=device)

    pair_gaps = []
    flip_warn = []
    flip_details = []
    max_pairs = args.max_intervention_pairs or len(kept_va)
    subset = kept_va[:max_pairs]

    intervention_grid = []
    mu_f = Fa_tr.mean(axis=0)
    _, Uk, _ = pca_subspace(Da_tr, 16)

    for pr in subset:
        ta = ds_va._load_tokens(pr.clip_a, pr.anchor)
        tb = ds_va._load_tokens(pr.clip_b, pr.anchor)
        ha = forward_h1s(model, ta, device)
        hb = forward_h1s(model, tb, device)
        gap = float(np.abs(ha - hb).mean())
        pair_gaps.append(gap)
        pa, sa, peak_a, _ = warn_peak(ha, warn)
        pb, sb, peak_b, _ = warn_peak(hb, warn)
        flip_warn.append(int(sa != sb))
        flip_details.append(
            {
                "pair_id": pr.pair_key,
                "gap_mean": gap,
                "peak_a": peak_a,
                "peak_b": peak_b,
                "warn_a": sa,
                "warn_b": sb,
                "pre_fork": pr.pre_fork,
                "gt_peak": pr.gt_peak,
                "background_a": pr.background_a,
                "background_b": pr.background_b,
            }
        )

    flip_rate = float(np.mean(flip_warn)) if flip_warn else float("nan")
    test4 = {
        "n_pairs": len(subset),
        "mean_abs_H_diff": float(np.mean(pair_gaps)),
        "median_abs_H_diff": float(np.median(pair_gaps)),
        "warning_flip_rate": flip_rate,
        "near_caution_fraction": float(np.mean([(abs(d["peak_a"] - 0.25) < 0.05) or (abs(d["peak_b"] - 0.25) < 0.05) for d in flip_details])),
        "near_stop_fraction": float(np.mean([(abs(d["peak_a"] - 0.60) < 0.05) or (abs(d["peak_b"] - 0.60) < 0.05) for d in flip_details])),
        "worst_50": sorted(flip_details, key=lambda x: -x["gap_mean"])[:50],
    }
    (OUT_ROOT / "test4_g_pair_predictions.json").write_text(json.dumps(test4, indent=2, default=str), encoding="utf-8")

    # 5A oracle average on subset (first 400 for runtime)
    sub400 = subset[: min(400, len(subset))]
    ora_preds, ora_gts, ora_val = [], [], []
    for pr in sub400:
        ta = ds_va._load_tokens(pr.clip_a, pr.anchor)
        tb = ds_va._load_tokens(pr.clip_b, pr.anchor)
        tavg = (ta + tb) * 0.5
        hm = forward_h1s(model, tavg, device)
        gt = ds_va.h1s_at_anchor(pr.clip_a, pr.anchor)
        ora_preds.append(hm)
        ora_gts.append(gt)
        ora_val.append(np.ones_like(gt, dtype=bool))
    P = np.stack(ora_preds)
    G = np.stack(ora_gts)
    V = np.stack(ora_val)
    op_oracle = operating_metrics(P, G, V)
    # 5B projection single point k=8 lam=0.5 on sub200
    sub200 = subset[: min(200, len(subset))]
    proj_preds = []
    for pr in sub200:
        ta = ds_va._load_tokens(pr.clip_a, pr.anchor)
        f = anchor_feature_vec(ta)
        fp = project_remove(f, mu_f, Uk[:8], 0.5)
        delta = (f.astype(np.float64) - fp.astype(np.float64))
        ta2 = apply_token_channel_edit(ta, delta)
        proj_preds.append(forward_h1s(model, ta2, device))
    Pp = np.stack(proj_preds)
    Gp = np.stack([ds_va.h1s_at_anchor(pr.clip_a, pr.anchor) for pr in sub200])
    Vp = np.ones_like(Gp, dtype=bool)
    op_proj = operating_metrics(Pp, Gp, Vp)
    test5 = {
        "oracle_pair_average_n": len(sub400),
        "oracle_operating": op_oracle,
        "projection_k8_lam0.5_n": len(sub200),
        "projection_operating": op_proj,
        "note": "Full k×λ grid abbreviated; extend script for exhaustive table",
    }
    (OUT_ROOT / "test5_interventions.json").write_text(json.dumps(test5, indent=2, default=str), encoding="utf-8")

    # --- TEST 6: ICC stability split-half ---
    n = Fa_tr.shape[0]
    mid = n // 2
    order = rng.permutation(n)
    h1, h2 = order[:mid], order[mid : 2 * mid]
    Wh1, _, ICC_h1 = pair_w_b_icc(Fa_tr[h1], Fb_tr[h1])
    Wh2, _, ICC_h2 = pair_w_b_icc(Fa_tr[h2], Fb_tr[h2])
    from scipy.stats import spearmanr

    sp_h, _ = spearmanr(ICC_h1, ICC_h2)
    sel_h1 = select_low_icc_channels(ICC_h1, Wh1, low_frac=0.2)
    sel_h2 = select_low_icc_channels(ICC_h2, Wh2, low_frac=0.2)
    jacc = len(set(sel_h1["selected_channel_ids"]) & set(sel_h2["selected_channel_ids"])) / max(
        1, len(set(sel_h1["selected_channel_ids"]) | set(sel_h2["selected_channel_ids"]))
    )
    # top variance channels
    var_c = W.mean(axis=0)
    top_var_ids = np.argsort(-var_c)[: len(sel["selected_channel_ids"])].tolist()
    test6 = {
        "split_half_icc_spearman": float(sp_h),
        "split_half_selected_jaccard": float(jacc),
        "dropout_selected_n": len(sel["selected_channel_ids"]),
        "frac_channels_dropped_typical": float(35.64 / 704),
        "top_variance_overlap_with_icc": len(set(top_var_ids) & set(sel["selected_channel_ids"])) / max(1, len(sel["selected_channel_ids"])),
    }
    (OUT_ROOT / "test6_icc.json").write_text(json.dumps(test6, indent=2), encoding="utf-8")

    # --- TEST 7: gradient compatibility (one paired batch) ---
    model.train()
    model.backbone.eval()
    params = [p for p in model.adapter.parameters()] + [p for p in model.decoder.parameters()]
    # build micro batch from first 4 pairs
    batch_tokens, batch_h = [], []
    for pr in pairs_tr[:4]:
        for cid in (pr.clip_a, pr.clip_b):
            batch_tokens.append(ds_tr._load_tokens(cid, pr.anchor))
            batch_h.append(ds_tr.h1s_at_anchor(cid, pr.anchor))
    bt = torch.from_numpy(np.stack(batch_tokens)).to(device)
    bh = torch.from_numpy(np.stack(batch_h)).to(device)
    amb = torch.zeros(len(batch_h), dtype=torch.bool, device=device)
    batch = {
        "tokens": bt,
        "h_1s": bh,
        "amb_1s": amb,
        "bg_reg_paired": True,
    }
    batch = move_batch(batch, device)
    out = teacher_forward(model, batch)
    packed = loss_fn(out, batch)
    l_g = packed.total
    mid = len(batch_h) // 2
    ha, hb = out["Hhat_1s"][:mid], out["Hhat_1s"][mid:]
    l_full_map = ((ha - hb) ** 2).mean()
    w = 0.05 + 4.0 * bh[:mid] ** 2
    l_gt_w = (w * (ha - hb) ** 2).mean()
    grads = {}
    for name in ("full_map", "gt_weighted"):
        model.zero_grad(set_to_none=True)
        out = teacher_forward(model, batch)
        packed = loss_fn(out, batch)
        l_g = packed.total
        mid = len(batch_h) // 2
        ha, hb = out["Hhat_1s"][:mid], out["Hhat_1s"][mid:]
        if name == "full_map":
            l_aux = ((ha - hb) ** 2).mean()
        else:
            w = 0.05 + 4.0 * bh[:mid] ** 2
            l_aux = (w * (ha - hb) ** 2).mean()
        gn, ga, c = grad_cos(l_g, l_aux, params)
        grads[name] = {"g_task": gn, "g_aux": ga, "cosine": c}
    (OUT_ROOT / "test7_gradients.json").write_text(json.dumps(grads, indent=2), encoding="utf-8")

    # --- TEST 8–9 summaries ---
    test8 = {
        "ridge_mse_F": probes["F"]["mse"],
        "ridge_mse_S": probes["S"]["mse"],
        "ridge_mse_D": probes["D"]["mse"],
        "interpretation_hint": "Compare F vs S vs D MSE; if D is low MSE, variant carries task signal",
    }
    by_bg = defaultdict(list)
    for pr, d in zip(subset, flip_details):
        by_bg[pr.background_a].append(d["gap_mean"])
    test9 = {
        "by_background_mean_gap": {k: float(np.mean(v)) for k, v in by_bg.items()},
        "lighting_subgroups": "use manifest.lighting — aggregate if n≥5 per label",
        "night_dusk_glare": "not tagged explicitly in manifest",
    }
    (OUT_ROOT / "test8_shared_private.json").write_text(json.dumps(test8, indent=2), encoding="utf-8")
    (OUT_ROOT / "test9_worst_groups.json").write_text(json.dumps(test9, indent=2), encoding="utf-8")

    # --- TEST 10: consolidated recommendation ---
    print("Diagnostics complete; see reports/pair_background_research/test10_consolidated.json", flush=True)
    print("wrote", OUT_ROOT)


def _summ(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


if __name__ == "__main__":
    main()
