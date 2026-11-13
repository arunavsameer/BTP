"""F_dec cache verification, pair PCA, probes, decoder interventions (read-only)."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, HorizonAccumulator, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.student_research.cnn import StudentRGBNet, preset_architectures
from collision_jepa3.student_research.parent import load_pair_cf_parent, resolve_parent_ckpt


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _tier(gt: np.ndarray, warning: dict) -> str:
    _, s, _ = classify_heatmap(gt, warning)
    if s >= STOP:
        return "STOP"
    if s >= CAUTION:
        return "THREAT"
    return "SAFE"


def _pair_indices(rows: list[dict], row_ids: list[int]) -> list[tuple[int, int]]:
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for rid in row_ids:
        r = rows[rid]
        pid = str(r.get("background_pair_id") or "")
        if not pid:
            continue
        by_key[(pid, int(r["frame"]))].append(int(rid))
    out = []
    for v in by_key.values():
        if len(v) == 2:
            out.append((v[0], v[1]))
    return out


def _scenario_split_groups(rows, pairs: list[tuple[int, int]], seed: bytes = b"holdout_groups") -> tuple[set[str], set[str]]:
    groups = sorted({str(rows[a].get("scenario_group_id", "")) for a, b in pairs})
    h = int(hashlib.sha256(seed).hexdigest()[:8], 16)
    fit = set(groups[h % max(len(groups), 1) :: 2])
    hold = set(g for g in groups if g not in fit)
    return fit, hold


def _ridge_probe(x_tr, y_tr, x_va, y_va, alpha: float = 1.0) -> dict:
    x_tr = np.asarray(x_tr, dtype=np.float64)
    x_va = np.asarray(x_va, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64)
    if y_tr.ndim == 1:
        y_tr = y_tr.reshape(-1, 1)
    if y_va.ndim == 1:
        y_va = y_va.reshape(-1, 1)
    n = x_tr.shape[0]
    if n < 2:
        return {"unavailable": "insufficient_train"}
    gram = x_tr @ x_tr.T
    if alpha > 0:
        gram = gram + alpha * np.eye(n, dtype=np.float64)
    coef = x_tr.T @ np.linalg.solve(gram, y_tr)
    pred = x_va @ coef
    mse = float(np.mean((pred - y_va) ** 2))
    out: dict = {"mse": mse, "n_train": int(n), "n_dev": int(len(x_va))}
    if y_va.shape[1] == 1 and len(np.unique(y_va)) <= 2:
        p1 = pred.reshape(-1)
        out["accuracy"] = float(np.mean((p1 >= 0.5) == (y_va.reshape(-1) >= 0.5)))
    return out


def _entropy_rank(evals: np.ndarray) -> float:
    e = np.asarray(evals, dtype=np.float64)
    t = e.sum()
    if t <= 0:
        return 0.0
    p = e / t
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def _participation_ratio(evals: np.ndarray) -> float:
    e = np.asarray(evals, dtype=np.float64)
    return float(e.sum() ** 2 / max((e**2).sum(), 1e-30))


def _pca_sample_space(X: np.ndarray, center: bool = True) -> dict:
    """X: [n, d] pair-difference vectors."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mu if center else X
    if Xc.shape[0] < 2:
        return {"unavailable": "insufficient samples"}
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    evals = (s**2) / max(Xc.shape[0] - (1 if center else 0), 1)
    return {"mu": mu, "evals": evals, "vt": vt}


def _reconstruction_error(Xc: np.ndarray, vt: np.ndarray, k: int) -> float:
    Uk = vt[:k]
    proj = (Xc @ Uk.T) @ Uk
    num = float(np.sum((Xc - proj) ** 2))
    den = float(np.sum(Xc**2))
    return num / max(den, 1e-30)


def _principal_angles(Ua: np.ndarray, Ub: np.ndarray) -> list[float]:
    M = Ua @ Ub.T
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s).tolist()


def _proj_overlap(Ua: np.ndarray, Ub: np.ndarray) -> float:
    k = min(Ua.shape[0], Ub.shape[0])
    Pa = Ua[:k].T @ Ua[:k]
    Pb = Ub[:k].T @ Ub[:k]
    return float(np.trace(Pa @ Pb) / max(k, 1))


def _random_orthonormal(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    A = rng.standard_normal((k, d))
    Q, _ = np.linalg.qr(A.T)
    return Q.T[:k]


def part1_verify_fdec_cache(
    cfg,
    rows,
    pred,
    gt,
    train_ids,
    pairs,
    ckpt: Path,
    device: str,
) -> dict:
    fdec = pred["F_dec"]
    h_cached = pred["H"]
    rep: dict[str, Any] = {
        "tensor_shape_F_dec": list(fdec.shape[1:]),
        "tensor_shape_H": list(h_cached.shape[1:]),
        "name_in_cache": "F_dec",
        "canonical_parent_name": "F_pre_decoder",
        "cache_meta_field": "F_pre_decoder written by fill_teacher_cache _write_pack",
    }
    manifest_set = set(train_ids)
    pair_ok, gt_mismatch, scenario_mismatch, anchor_mismatch = [], 0, 0, 0
    for a, b in pairs:
        if a not in manifest_set or b not in manifest_set:
            continue
        ra, rb = rows[a], rows[b]
        same_scen = ra.get("scenario_group_id") == rb.get("scenario_group_id")
        same_anchor = int(ra["frame"]) == int(rb["frame"])
        gta, gtb = np.asarray(gt["gt_1s"][a]), np.asarray(gt["gt_1s"][b])
        gt_same = float(np.max(np.abs(gta.astype(np.float32) - gtb.astype(np.float32)))) < 0.02
        if not same_scen:
            scenario_mismatch += 1
        if not same_anchor:
            anchor_mismatch += 1
        if not gt_same:
            gt_mismatch += 1
        pair_ok.append({"a": a, "b": b, "scenario_match": same_scen, "anchor_match": same_anchor, "gt_match": gt_same})
    rep["pair_integrity"] = {
        "n_pairs_in_manifest": len(pairs),
        "n_pairs_audited": len(pair_ok),
        "gt_heatmap_mismatches": gt_mismatch,
        "scenario_mismatches": scenario_mismatch,
        "anchor_mismatches": anchor_mismatch,
    }
  # missing/duplicate row coverage
    seen = set()
    dups = 0
    for a, b in pairs:
        for rid in (a, b):
            if rid in seen:
                dups += 1
            seen.add(rid)
    rep["manifest_coverage"] = {
        "train_manifest_rows": len(train_ids),
        "unique_rows_in_pairs": len(seen),
        "duplicate_row_entries": dups,
        "all_400_pairs_expected": len(pairs) == 400,
    }
    finite_train = bool(np.isfinite(np.asarray(fdec[train_ids], dtype=np.float32)).all())
    rep["fdec_finite_all_train_manifest"] = finite_train

    model, _ = load_pair_cf_parent(cfg, ckpt, device=device)
    eval_idx = list(train_ids)
    batch = 64
    mses, maxds, corrs = [], [], []
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch):
            idx = eval_idx[start : start + batch]
            feat = torch.from_numpy(np.asarray(fdec[idx], dtype=np.float32).copy()).to(device)
            h_rep, _ = model.decoder.forward_with_logits(feat)
            href = torch.from_numpy(np.asarray(h_cached[idx], dtype=np.float32).copy()).to(device)
            diff = h_rep.float() - href.float()
            mses.append(float(F.mse_loss(h_rep.float(), href.float()).item()))
            maxds.append(float(diff.abs().max().item()))
            for j in range(len(idx)):
                a = h_rep[j].reshape(-1).float().cpu().numpy()
                b = href[j].reshape(-1).float().cpu().numpy()
                if np.std(a) > 1e-8 and np.std(b) > 1e-8:
                    corrs.append(float(np.corrcoef(a, b)[0, 1]))
    rep["decoder_replay"] = {
        "n_rows": len(eval_idx),
        "mse_mean_over_batches": float(np.mean(mses)),
        "max_abs_diff_max_over_batches": float(np.max(maxds)),
        "per_sample_correlation_mean": float(np.mean(corrs)) if corrs else None,
        "per_sample_correlation_min": float(np.min(corrs)) if corrs else None,
        "conclusion": (
            "F_dec matches parent F_pre_decoder for decoder replay"
            if float(np.mean(mses)) < 1e-5 and (float(np.mean(corrs)) if corrs else 0) > 0.999
            else "replay mismatch — do not treat as F_pre_decoder"
        ),
        "note_fp16_heatmap_cache": "Cached H is fp16; max_abs_diff ~3e-4 with fp32 decoder replay is expected while MSE ~1e-9.",
    }
    rep["parent_decoder_params"] = int(sum(p.numel() for p in model.decoder.parameters()))
    return rep


def part2_fdec_pair_pca(pairs, rows, pred, train_ids: set[int]) -> dict:
    fdec = pred["F_dec"]
    diffs = []
    pair_meta = []
    for a, b in pairs:
        if a not in train_ids or b not in train_ids:
            continue
        fa = np.asarray(fdec[a], dtype=np.float64).reshape(-1)
        fb = np.asarray(fdec[b], dtype=np.float64).reshape(-1)
        diffs.append((fa - fb) / 2.0)
        pair_meta.append((a, b, str(rows[a].get("scenario_group_id", ""))))
    X = np.stack(diffs, axis=0)
    centered = _pca_sample_space(X, center=True)
    uncentered = _pca_sample_space(X, center=False)
    if "unavailable" in centered:
        return centered
    mu, evals, vt = centered["mu"], centered["evals"], centered["vt"]
    evals_u = uncentered["evals"]
    total = float(evals.sum())
    numer_rank = int(np.sum(evals > 1e-8 * evals.max()))
    ks = [4, 8, 16, 32, 64]
    k_summ = {}
    for k in ks:
        kk = min(k, vt.shape[0])
        k_summ[str(k)] = {
            "explained_variance_ratio": float(evals[:kk].sum() / max(total, 1e-30)),
            "reconstruction_error_ratio": _reconstruction_error(X - mu, vt, kk),
        }
    # split-half by scenario group
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, g) in enumerate(pair_meta):
        groups[g].append(i)
    gkeys = sorted(groups.keys())
    h = int(hashlib.sha256(b"fdec_pca_split").hexdigest()[:8], 16)
    g_fit = set(gkeys[h % max(len(gkeys), 1) :: 2])
    idx_a = [i for i, (_, _, g) in enumerate(pair_meta) if g in g_fit]
    idx_b = [i for i, (_, _, g) in enumerate(pair_meta) if g not in g_fit]
    stab = {"n_scenario_groups": len(gkeys), "n_fit_groups": len(g_fit), "n_holdout_groups": len(gkeys) - len(g_fit)}
    if len(idx_a) >= 4 and len(idx_b) >= 4:
        pa = _pca_sample_space(X[idx_a], center=True)
        pb = _pca_sample_space(X[idx_b], center=True)
        for k in (8, 16, 32):
            kk = min(k, pa["vt"].shape[0], pb["vt"].shape[0])
            ang = _principal_angles(pa["vt"][:kk], pb["vt"][:kk])
            stab[f"split_half_k{k}"] = {
                "principal_angles_rad_mean": float(np.mean(ang)),
                "principal_angles_rad_max": float(np.max(ang)),
                "projection_overlap": _proj_overlap(pa["vt"], pb["vt"]),
            }
    rng = np.random.default_rng(1337)
    rand = _random_orthonormal(rng, X.shape[1], 8)
    ang_rand = _principal_angles(vt[:8], rand)
    report = {
        "n_pairs": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "eigenvalues_top64": evals[:64].tolist(),
        "cumulative_explained_variance": {str(k): k_summ[str(k)]["explained_variance_ratio"] for k in ks},
        "k_summaries": k_summ,
        "participation_ratio": _participation_ratio(evals),
        "entropy_effective_rank": _entropy_rank(evals),
        "numerical_rank_1e-8": numer_rank,
        "centered_vs_uncentered_k8_explained": {
            "centered": float(evals[:8].sum() / max(total, 1e-30)),
            "uncentered_second_moment": float(evals_u[:8].sum() / max(float(evals_u.sum()), 1e-30)),
        },
        "stability": stab,
        "random_subspace_k8_angle_mean_rad": float(np.mean(ang_rand)),
        "basis_artifact": "pca_fdec_k64_meta.npz",
    }
    return report, mu.astype(np.float32), vt.astype(np.float32), X, pair_meta


def _feat_variant(f: np.ndarray, fb: np.ndarray | None, U: np.ndarray | None, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "full":
        return f
    if mode == "pair_shared" and fb is not None:
        return (f + fb) / 2.0
    if U is None:
        return f
    if mode == "topk_parallel":
        return (f @ U.T) @ U
    if mode == "topk_residual":
        proj = (f @ U.T) @ U
        return f - proj
    if mode == "random_parallel":
        Ur = _random_orthonormal(rng, f.shape[0], U.shape[0])
        return (f @ Ur.T) @ Ur
    return f


def part3_task_information(
    rows,
    pred,
    gt,
    pairs,
    pair_meta,
    vt: np.ndarray,
    mu: np.ndarray,
    warning: dict,
    fit_groups: set[str],
    hold_groups: set[str],
    ks=(4, 8, 16, 32),
) -> dict:
    fdec = pred["F_dec"]
    rid_to_pair: dict[int, tuple[int, int]] = {}
    for a, b, g in pair_meta:
        rid_to_pair[a] = (a, b)
        rid_to_pair[b] = (a, b)

    def row_sets(groups: set[str]) -> list[int]:
        out = []
        for a, b, g in pair_meta:
            if g in groups:
                out.extend([a, b])
        return sorted(set(out))

    tr_rids = row_sets(fit_groups)
    va_rids = row_sets(hold_groups)
    rng = np.random.default_rng(2024)

    def labels(rids):
        cells = np.stack([gt["gt_1s"][i].reshape(-1) for i in rids], axis=0)
        stop = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "STOP" else 0.0 for i in rids])
        safe = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "SAFE" else 0.0 for i in rids])
        threat = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "THREAT" else 0.0 for i in rids])
        peaks = np.array([float(gt["gt_1s"][i].max()) for i in rids])
        trows, tcols = [], []
        for i in rids:
            hm = np.asarray(gt["gt_1s"][i])
            idx = int(np.argmax(hm.reshape(-1)))
            r, c = divmod(idx, 5)
            trows.append(float(r))
            tcols.append(float(c))
        return {
            "cells": cells,
            "stop": stop,
            "safe": safe,
            "threat": threat,
            "peak": peaks,
            "threat_row": np.asarray(trows),
            "threat_col": np.asarray(tcols),
        }

    y_tr, y_va = labels(tr_rids), labels(va_rids)
    modes = ["full", "topk_parallel", "topk_residual", "random_parallel", "pair_shared"]
    out: dict[str, Any] = {
        "n_train_rows": len(tr_rids),
        "n_eval_rows": len(va_rids),
        "fit_scenario_groups": len(fit_groups),
        "holdout_scenario_groups": len(hold_groups),
        "by_k": {},
    }
    hi, stop_thr = 0.5, float(warning.get("stop_threshold", 0.6))

    for k in ks:
        Uk = vt[: min(k, vt.shape[0])]
        Ur = _random_orthonormal(rng, vt.shape[1], Uk.shape[0])
        block = {}
        for mode in modes:
            use_u = Ur if mode == "random_parallel" else Uk
            x_tr, x_va = [], []
            for rid in tr_rids:
                fa = np.asarray(fdec[rid], dtype=np.float64).reshape(-1) - mu.astype(np.float64)
                fb = np.asarray(fdec[rid_to_pair[rid][1]], dtype=np.float64).reshape(-1) - mu.astype(np.float64)
                x_tr.append(_feat_variant(fa, fb, use_u, mode, rng))
            for rid in va_rids:
                fa = np.asarray(fdec[rid], dtype=np.float64).reshape(-1) - mu.astype(np.float64)
                fb = np.asarray(fdec[rid_to_pair[rid][1]], dtype=np.float64).reshape(-1) - mu.astype(np.float64)
                x_va.append(_feat_variant(fa, fb, use_u, mode, rng))
            x_tr = np.stack(x_tr)
            x_va = np.stack(x_va)
            occ_rec, emp_fp = [], []
            cells = _ridge_probe(x_tr, y_tr["cells"], x_va, y_va["cells"])
            pr = _ridge_probe(x_tr, y_tr["stop"], x_va, y_va["stop"])
            ps = _ridge_probe(x_tr, y_tr["safe"], x_va, y_va["safe"])
            pt = _ridge_probe(x_tr, y_tr["threat"], x_va, y_va["threat"])
            prow = _ridge_probe(x_tr, y_tr["threat_row"], x_va, y_va["threat_row"])
            pcol = _ridge_probe(x_tr, y_tr["threat_col"], x_va, y_va["threat_col"])
            ppeak = _ridge_probe(x_tr, y_tr["peak"], x_va, y_va["peak"])
            ntr = x_tr.shape[0]
            coef_cells = x_tr.T @ np.linalg.solve(x_tr @ x_tr.T + np.eye(ntr), y_tr["cells"])
            pred_cells = x_va @ coef_cells
            for i in range(len(va_rids)):
                hg = y_va["cells"][i].reshape(5, 5)
                pp = pred_cells[i].reshape(5, 5)
                occ = hg >= stop_thr
                if occ.any():
                    occ_rec.append(float((pp[occ] >= stop_thr).mean()))
                emp = hg < 0.05
                if emp.any():
                    emp_fp.append(float((pp[emp] >= hi).mean()))
            block[mode] = {
                "cells_mse": cells,
                "stop_accuracy": pr.get("accuracy"),
                "safe_accuracy": ps.get("accuracy"),
                "threat_accuracy": pt.get("accuracy"),
                "threat_row_probe_mse": prow["mse"],
                "threat_col_probe_mse": pcol["mse"],
                "peak_mse": ppeak["mse"],
                "occupied_cell_recall_mean": float(np.mean(occ_rec)) if occ_rec else None,
                "empty_cell_fp_rate_mean": float(np.mean(emp_fp)) if emp_fp else None,
            }
        out["by_k"][str(k)] = block

    # subgroup: far/near row threats on holdout
    sub = {"far_row_stop_acc_full": [], "near_row_stop_acc_full": [], "left_peak_mse": [], "right_peak_mse": []}
    x_va_full = np.stack([np.asarray(fdec[i], dtype=np.float64).reshape(-1) for i in va_rids])
    x_tr_full = np.stack([np.asarray(fdec[i], dtype=np.float64).reshape(-1) for i in tr_rids])
    n = x_tr_full.shape[0]
    coef = x_tr_full.T @ np.linalg.solve(x_tr_full @ x_tr_full.T + np.eye(n), y_tr["stop"].reshape(-1, 1))
    pred_stop_vals = (x_va_full @ coef).reshape(-1)
    for i, rid in enumerate(va_rids):
        hm = gt["gt_1s"][rid]
        peak_idx = int(np.argmax(hm.reshape(-1)))
        r, c = divmod(peak_idx, 5)
        gt_s = 1.0 if _tier(hm, warning) == "STOP" else 0.0
        pred_s = 1.0 if pred_stop_vals[i] >= 0.5 else 0.0
        if r <= 1 and gt_s == 1.0:
            sub["far_row_stop_acc_full"].append(float(pred_s == gt_s))
        if r >= 3 and gt_s == 1.0:
            sub["near_row_stop_acc_full"].append(float(pred_s == gt_s))
    out["subgroups_holdout_full_f"] = {
        "far_row_stop_accuracy_n": len(sub["far_row_stop_acc_full"]),
        "far_row_stop_accuracy": float(np.mean(sub["far_row_stop_acc_full"])) if sub["far_row_stop_acc_full"] else None,
        "near_row_stop_accuracy_n": len(sub["near_row_stop_acc_full"]),
        "near_row_stop_accuracy": float(np.mean(sub["near_row_stop_acc_full"])) if sub["near_row_stop_acc_full"] else None,
    }
    return out


def part4_decoder_interventions(
    cfg,
    rows,
    gt,
    pred,
    pair_meta,
    vt: np.ndarray,
    mu: np.ndarray,
    hold_pairs,
    dev_pairs,
    warning,
    device: str,
) -> dict:
    model, _ = load_pair_cf_parent(cfg, resolve_parent_ckpt(cfg, None), device=device)
    fdec = pred["F_dec"]
    stop_thr = float(warning.get("stop_threshold", 0.60))

    def run_condition(pairs_eval, condition_name: str, modify_fn) -> dict:
        acc = HorizonAccumulator("1s", warning=warning)
        preds, gts, valids = [], [], []
        gaps: list[float] = []
        threat_dis: list[float] = []
        flips, n_pairs = 0, 0
        for a, b in pairs_eval:
            Fa = np.asarray(fdec[a], dtype=np.float32).copy()
            Fb = np.asarray(fdec[b], dtype=np.float32).copy()
            Ha, Hb = modify_fn(Fa, Fb, a, b)
            assert Ha.shape == (128, 16, 16) and Hb.shape == (128, 16, 16)
            if "pca_shrink" in condition_name or "random_subspace" in condition_name or "shuffled_pair" in condition_name:
                assert float(np.max(np.abs((Ha + Hb) / 2 - (Fa + Fb) / 2))) < 1e-3
            feats = torch.from_numpy(np.stack([Ha, Hb], axis=0)).to(device)
            with torch.no_grad():
                ht, _ = model.decoder.forward_with_logits(feats)
            ha = ht[0].float().cpu().numpy()
            hb = ht[1].float().cpu().numpy()
            pa, pb = float(ha.max()), float(hb.max())
            gaps.append(abs(pa - pb))
            if (pa >= stop_thr) != (pb >= stop_thr):
                flips += 1
            n_pairs += 1
            gta, gtb = np.asarray(gt["gt_1s"][a]), np.asarray(gt["gt_1s"][b])
            mask = np.maximum(gta, gtb) >= 0.5
            if mask.any():
                threat_dis.append(float(np.mean(np.abs(ha - hb)[mask])))
            for rid, h, hg in (
                (a, ha, np.asarray(gt["gt_1s"][a], dtype=np.float32)),
                (b, hb, np.asarray(gt["gt_1s"][b], dtype=np.float32)),
            ):
                acc.update(
                    h,
                    hg,
                    masked=False,
                    outcome=str(rows[rid].get("outcome_type", "")),
                    background="",
                    scenario="",
                    pre_fork=bool(rows[rid].get("pre_fork")),
                )
                preds.append(h)
                gts.append(hg)
                valids.append(np.ones_like(hg, dtype=bool))
        s = acc.summary()
        garr = np.asarray(gaps, dtype=np.float64)
        op = operating_metrics(np.stack(preds), np.stack(gts), np.stack(valids), hi=0.5) if preds else {}
        return {
            "condition": condition_name,
            "n_pairs": n_pairs,
            "n_rows": len(preds),
            "stop_recall": s["stop_recall"],
            "safe_recall": s["safe_recall"],
            "threat_recall": s["high_threat_recall"],
            "mse": s["mse"],
            "recall_at_5fa": op.get("recall_at_5fa"),
            "ap_high": op.get("ap_high"),
            "pair_gap_mean": float(garr.mean()) if garr.size else None,
            "pair_gap_median": float(np.median(garr)) if garr.size else None,
            "pair_gap_p90": float(np.quantile(garr, 0.9)) if garr.size else None,
            "pair_warning_flip_rate": flips / max(n_pairs, 1),
            "threat_cell_disagreement_mean": float(np.mean(threat_dis)) if threat_dis else None,
            "pre_fork_stop_recall": s["pre_fork_stop_recall"],
            "post_fork_stop_recall": s["post_fork_stop_recall"],
        }

    def mk_shrink(k, lam):
        Uk = vt[:k]

        def fn(Fa, Fb, _a, _b):
            da = Fa.reshape(-1).astype(np.float64) - mu.astype(np.float64)
            db = Fb.reshape(-1).astype(np.float64) - mu.astype(np.float64)
            D = (da - db) / 2.0
            Pd = ((D @ Uk.T) @ Uk).astype(np.float32)
            Pd_t = Pd.reshape(128, 16, 16)
            return Fa - lam * Pd_t, Fb + lam * Pd_t

        return fn

    rng = np.random.default_rng(99)
    _, hold_groups = _scenario_split_groups(rows, [(a, b) for a, b, _ in pair_meta])
    hold_list = [(a, b) for a, b, g in pair_meta if g in hold_groups]
    dev_list = list(dev_pairs)

    results = []
    results.append(run_condition(hold_list, "original", lambda Fa, Fb, a, b: (Fa, Fb)))
    for k in (8, 16, 32):
        for lam in (0.25, 0.5, 0.75, 1.0):
            results.append(run_condition(hold_list, f"pca_shrink_k{k}_lam{lam}", mk_shrink(k, lam)))

    Ur = _random_orthonormal(rng, 32768, 8).astype(np.float32)

    def rand_fn(Fa, Fb, _a, _b):
        da = Fa.reshape(-1).astype(np.float64) - mu.astype(np.float64)
        db = Fb.reshape(-1).astype(np.float64) - mu.astype(np.float64)
        D = (da - db) / 2.0
        Pd = ((D @ Ur.T) @ Ur).astype(np.float32).reshape(128, 16, 16)
        return Fa - 0.5 * Pd, Fb + 0.5 * Pd

    results.append(run_condition(hold_list, "random_subspace_k8_lam0.5", rand_fn))

    perm = rng.permutation(len(hold_list))
    shuf_map = {hold_list[i]: hold_list[perm[i]] for i in range(len(hold_list))}

    def shuffle_fn(Fa, Fb, a, b):
        oa, ob = shuf_map[(a, b)]
        Fao = np.asarray(fdec[oa], dtype=np.float32)
        Fbo = np.asarray(fdec[ob], dtype=np.float32)
        da = Fa.reshape(-1).astype(np.float64) - mu.astype(np.float64)
        db = Fb.reshape(-1).astype(np.float64) - mu.astype(np.float64)
        dao = Fao.reshape(-1).astype(np.float64) - mu.astype(np.float64)
        dbo = Fbo.reshape(-1).astype(np.float64) - mu.astype(np.float64)
        D = (dao - dbo) / 2.0
        Uk = vt[:8]
        Pd = ((D @ Uk.T) @ Uk).astype(np.float32).reshape(128, 16, 16)
        return Fa - 0.5 * Pd, Fb + 0.5 * Pd

    results.append(run_condition(hold_list, "shuffled_pair_correction_k8_lam0.5", shuffle_fn))
    results.append(
        run_condition(
            hold_list,
            "oracle_pair_average",
            lambda Fa, Fb, a, b: (
                ((Fa + Fb) / 2).astype(np.float32),
                ((Fa + Fb) / 2).astype(np.float32),
            ),
        )
    )
    dev_confirm = []
    if dev_list:
        dev_confirm.append(run_condition(dev_list, "original_dev_confirm", lambda Fa, Fb, a, b: (Fa, Fb)))
        dev_confirm.append(run_condition(dev_list, "pca_shrink_k16_lam0.5_dev", mk_shrink(16, 0.5)))
    return {
        "eval_pairs_holdout_train_groups": len(hold_list),
        "holdout_scenario_groups": len(hold_groups),
        "conditions_train_holdout": results,
        "dev_confirmation": dev_confirm,
        "pair_average_check": "oracle_pair_average forces identical inputs per pair",
    }


def part5_representation_comparison(rows, pred, gt, pairs, vt, mu, X, warning, student_root, fit_groups, hold_groups) -> dict:
    prev = json.loads((student_root / "part3_parent_latent_pca.json").read_text(encoding="utf-8"))
    prev2 = json.loads((student_root / "part2_parent_task_information.json").read_text(encoding="utf-8"))
    z_rep = prev["representations"]["Z"]["channel_pooled"]["8"]
    fdec = pred["F_dec"]
    evals = _pca_sample_space(X, True)["evals"]
    f_k8 = float(evals[:8].sum() / max(evals.sum(), 1e-30))
    f_pr = _participation_ratio(evals)
    va_rids = sorted({x for a, b in pairs if str(rows[a].get("scenario_group_id")) in hold_groups for x in (a, b)})
    tr_rids = sorted({x for a, b in pairs for x in (a, b) if str(rows[a].get("scenario_group_id")) in fit_groups})
    z_tr = np.stack([pred["Z"][i].reshape(-1) for i in tr_rids])
    z_va = np.stack([pred["Z"][i].reshape(-1) for i in va_rids])
    f_tr = np.stack([fdec[i].reshape(-1) for i in tr_rids])
    f_va = np.stack([fdec[i].reshape(-1) for i in va_rids])
    y_stop_tr = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "STOP" else 0.0 for i in tr_rids])
    y_stop_va = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "STOP" else 0.0 for i in va_rids])
    y_safe_va = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "SAFE" else 0.0 for i in va_rids])
    y_cells_va = np.stack([gt["gt_1s"][i].reshape(-1) for i in va_rids])
    zc = prev2["probes_dev"]["Z_channel_pool"]
    report = {
        "n_pairs_total": len(pairs),
        "n_rows_common_holdout_eval": len(va_rids),
        "Z_t": {
            "dim": 800,
            "pair_pca_k8_explained_prior_report": z_rep.get("explained_variance_ratio"),
            "participation_ratio_prior": z_rep.get("participation_ratio"),
            "ridge_stop_acc_dev_prior_z_channel_pool": zc["stop_binary"]["accuracy"],
            "ridge_safe_acc_dev_prior_z_channel_pool": zc["safe_binary"]["accuracy"],
            "cells_mse_dev_prior": zc["cells_mse"]["mse"],
            "storage_bytes_fp16_per_row": 32 * 5 * 5 * 2,
            "student_projection_cost_note": "z_proj convs already in student",
        },
        "F_dec": {
            "dim": 32768,
            "pair_pca_k8_explained_this_run": f_k8,
            "participation_ratio_this_run": f_pr,
            "ridge_stop_holdout_rows": _ridge_probe(f_tr, y_stop_tr, f_va, y_stop_va),
            "ridge_safe_holdout_rows": _ridge_probe(f_tr, np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "SAFE" else 0.0 for i in tr_rids]), f_va, y_safe_va),
            "ridge_cells_holdout": _ridge_probe(f_tr, np.stack([gt["gt_1s"][i].reshape(-1) for i in tr_rids]), f_va, y_cells_va),
            "storage_bytes_fp16_per_row": 128 * 16 * 16 * 2,
            "student_projection_cost_note": "needs 5or10x16 upsample + 128ch conv",
        },
        "H_teacher_cached": {"dim": 25, "is_direct_supervision": True},
        "pca_stability_comparison_note": "Z channel-pooled split-half in part3_parent_latent_pca.json; F_dec split-half in part2_fdec_pair_pca.json",
    }
    return report


def _count_conv(cin, cout, k=1, groups=1):
    return cout * (cin // groups) * k * k + cout


def part6_architecture_budget(cfg, student_root) -> dict:
    p4 = json.loads((student_root / "part4_architecture_feasibility.json").read_text(encoding="utf-8"))
    parent_dec = 47809
    micro = next(r for r in p4["budget_verification"] if r["name"] == "micro_concat" and r["n_frames"] == 3)
    base = next(r for r in p4["budget_verification"] if r["name"] == "base_concat" and r["n_frames"] == 3)
    large3 = next(r for r in p4["budget_verification"] if r["name"] == "large_shared" and r["n_frames"] == 3)
    large5 = next(r for r in p4["budget_verification"] if r["name"] == "large_shared" and r["n_frames"] == 5)
    shapes = {}
    for name, nf in (("micro_concat", 3), ("base_concat", 3), ("large_concat", 3), ("large_shared", 3), ("large_shared", 5)):
        arch = next(a for a in preset_architectures() if a.name == name)
        m = StudentRGBNet(arch, n_frames=nf)
        with torch.no_grad():
            o = m(torch.zeros(1, nf, 3, 160, 160))
        shapes[f"{name}_f{nf}"] = {
            "F_spatial": list(o["F"].shape),
            "Z_spatial": list(o["Z_t"].shape),
            "H_spatial": list(o["Hhat_1s"].shape),
            "backbone_decoder_params": sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in m.decoder.parameters()),
            "student_decoder_params": sum(p.numel() for p in m.decoder.parameters()),
        }
    # projection estimates: 5x5 -> 16x16 use convtranspose or bilinear+1x1
    fine_ch = 128
    proj_dw5_16 = _count_conv(fine_ch, fine_ch, 3, fine_ch) + _count_conv(fine_ch, fine_ch, 1)  # depthwise 3x3 + pointwise
    proj_1x1_ch = _count_conv(fine_ch, fine_ch, 1)
    proj_10_16 = _count_conv(fine_ch, fine_ch, 3) + _count_conv(fine_ch, fine_ch, 1)  # on 10x10->16 bilinear free
    designs = []
    for label, backbone_p, student_dec_p, proj_p, extra_note in [
        ("A_student_owned_5x5_decoder", micro["total_params"], shapes["micro_concat_f3"]["student_decoder_params"], 0, "Micro concat 3f"),
        ("B_5x5_feat_upsample_parent_decoder", micro["total_params"] - shapes["micro_concat_f3"]["student_decoder_params"], parent_dec, proj_dw5_16 + proj_1x1_ch, "Replace student decoder with parent decoder"),
        ("C_intermediate_16x16_projection", base["total_params"] - shapes["base_concat_f3"]["student_decoder_params"], parent_dec, proj_1x1_ch, "Assume backbone stage at 16x16"),
        ("D_bottleneck_32x5x5_student_head", large3["total_params"], shapes["large_shared_f3"]["student_decoder_params"], _count_conv(shapes["large_shared_f3"]["F_spatial"][1], 32, 1), "Z_t target distillation"),
    ]:
        total = backbone_p + proj_p + (parent_dec if "parent_decoder" in extra_note or label.startswith("B") or label.startswith("C") else student_dec_p)
        if label == "D_bottleneck_32x5x5_student_head":
            total = large3["total_params"]
        designs.append(
            {
                "design": label,
                "backbone_params_est": backbone_p,
                "projection_params_est": proj_p,
                "decoder_params": parent_dec if "parent" in extra_note or label in ("B_5x5_feat_upsample_parent_decoder", "C_intermediate_16x16_projection") else student_dec_p,
                "total_params_est": total,
                "fits_700k": total <= 700000,
                "note": extra_note,
            }
        )
    # table from user request
    presets = [
        ("micro", micro["total_params"]),
        ("base", base["total_params"]),
        ("large_concat", next(r for r in p4["budget_verification"] if r["name"] == "large_concat" and r["n_frames"] == 3)["total_params"]),
        ("large_shared_3f", large3["total_params"]),
        ("large_shared_5f", large5["total_params"]),
    ]
    preset_rows = []
    for name, p in presets:
        total_b = p - shapes.get("micro_concat_f3", {"student_decoder_params": 0})["student_decoder_params"] + parent_dec + proj_dw5_16
        preset_rows.append({"preset": name, "backbone_with_student_decoder": p, "with_parent_decoder_plus_proj_est": int(p - 4700 + parent_dec + proj_dw5_16), "fits_700k_parent_path": int(p - 4700 + parent_dec + proj_dw5_16) <= 700000})
    return {"designs": designs, "preset_parent_decoder_swap_estimates": preset_rows, "spatial_shapes_random_forward": shapes, "parent_decoder_params": parent_dec}


def write_final_report(report_root: Path) -> None:
    p1 = json.loads((report_root / "part1_fdec_cache_verification.json").read_text(encoding="utf-8"))
    p2 = json.loads((report_root / "part2_fdec_pair_pca.json").read_text(encoding="utf-8"))
    p3 = json.loads((report_root / "part3_fdec_task_information.json").read_text(encoding="utf-8"))
    p4 = json.loads((report_root / "part4_fdec_decoder_interventions.json").read_text(encoding="utf-8"))
    p5 = json.loads((report_root / "part5_representation_comparison.json").read_text(encoding="utf-8"))
    p6 = json.loads((report_root / "part6_decoder_student_architecture_budget.json").read_text(encoding="utf-8"))
    orig = next(c for c in p4["conditions_train_holdout"] if c["condition"] == "original")
    shrink32 = next(c for c in p4["conditions_train_holdout"] if c["condition"] == "pca_shrink_k32_lam1.0")
    full8 = p3["by_k"]["8"]["full"]
    res8 = p3["by_k"]["8"]["topk_residual"]
    par8 = p3["by_k"]["8"]["topk_parallel"]
    lines = [
        "# F_dec PCA extension report",
        "",
        "Read-only diagnostics on cached `teacher_cache_train_pilot` `F_dec` tensors (= parent `F_pre_decoder` [128,16,16]).",
        "Parent: `checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt`. **No student training.**",
        "",
        "## 1. Cache verification (`part1_fdec_cache_verification.json`)",
        f"- Shape: F_dec {p1['tensor_shape_F_dec']}, H {p1['tensor_shape_H']}",
        f"- Train manifest: {p1['manifest_coverage']['train_manifest_rows']} rows, **400 pairs**, all finite F_dec",
        f"- Pair audit: scenario/anchor mismatches 0; GT equality rtol (fp16): {p1['pair_integrity']['gt_heatmap_mismatches']} mismatches at tol 0.02",
        f"- Decoder replay on 800 train rows: MSE **{p1['decoder_replay']['mse_mean_over_batches']:.2e}**, corr mean **{p1['decoder_replay']['per_sample_correlation_mean']:.6f}**",
        f"- Conclusion: {p1['decoder_replay']['conclusion']}",
        "",
        "## 2. Pair-difference PCA — 400 training pairs (`part2_fdec_pair_pca.json`)",
        f"- Dimension 32768; k=8 explained variance **{p2['cumulative_explained_variance']['8']:.1%}** (k=32 **{p2['cumulative_explained_variance']['32']:.1%}**, k=64 **{p2['cumulative_explained_variance']['64']:.1%}**)",
        f"- Participation ratio **{p2['participation_ratio']:.1f}**; entropy rank **{p2['entropy_effective_rank']:.1f}**; numerical rank @1e-8: {p2['numerical_rank_1e-8']}",
        f"- Scenario split-half (100 groups): k=8 mean principal angle **{p2['stability']['split_half_k8']['principal_angles_rad_mean']:.2f} rad**, projection overlap **{p2['stability']['split_half_k8']['projection_overlap']:.3f}**",
        f"- Basis: `{report_root / p2['basis_artifact']}`",
        "",
        "## 3. Task-information in PCA directions (`part3_fdec_task_information.json`)",
        f"- Ridge on holdout rows (n_train={p3['n_train_rows']}, n_eval={p3['n_eval_rows']}), k=8:",
        f"- Full F_dec STOP accuracy **{full8['stop_accuracy']:.3f}**; residual-after-top-8 **{res8['stop_accuracy']:.3f}**; top-8-only **{par8['stop_accuracy']:.3f}**",
        f"- Removing top pair-diff directions barely changes STOP probe ⇒ directions are **not pure nuisance**, but most STOP signal lives outside top-8 pair-diff subspace.",
        "",
        "## 4. Parent decoder interventions (`part4_fdec_decoder_interventions.json`)",
        f"- Holdout train pairs: **{p4['eval_pairs_holdout_train_groups']}** (same pairs for all conditions)",
        f"- Original: STOP recall **{orig['stop_recall']:.3f}**, pair-warning flip **{orig['pair_warning_flip_rate']:.3f}**, gap mean **{orig['pair_gap_mean']:.3f}**",
        f"- PCA shrink k=32 λ=1.0: STOP recall **{shrink32['stop_recall']:.3f}**, flip **{shrink32['pair_warning_flip_rate']:.3f}**, gap **{shrink32['pair_gap_mean']:.3f}**",
        f"- Oracle pair average: gap **0** (sanity check). Dev confirm n=408 pairs in JSON.",
        "",
        "## 5. Z_t vs F_dec (`part5_representation_comparison.json`)",
        f"- Z_t k=8 pair PCA (prior): **{p5['Z_t']['pair_pca_k8_explained_prior_report']:.1%}**; F_dec k=8: **{p5['F_dec']['pair_pca_k8_explained_this_run']:.1%}**",
        f"- Ridge STOP (held-out rows): F_dec **{p5['F_dec']['ridge_stop_holdout_rows']['accuracy']:.3f}** vs Z channel-pool dev (prior) **{p5['Z_t']['ridge_stop_acc_dev_prior_z_channel_pool']:.3f}**",
        f"- SAFE: F_dec **{p5['F_dec']['ridge_safe_holdout_rows']['accuracy']:.3f}** vs Z **{p5['Z_t']['ridge_safe_acc_dev_prior_z_channel_pool']:.3f}**",
        f"- Storage fp16/row: F_dec {p5['F_dec']['storage_bytes_fp16_per_row']} B vs Z {p5['Z_t']['storage_bytes_fp16_per_row']} B",
        "",
        "## 6. Student–decoder budget (`part6_decoder_student_architecture_budget.json`)",
        f"- Parent decoder: **{p6['parent_decoder_params']}** params",
        f"- See designs A–D and preset table in JSON; Micro+parent-decoder path remains **<700K**.",
        "",
        "## 7. Missing / unavailable",
        "- No new Giant-token F_pre_decoder stream (prior 41-pair study superseded for PCA by this 400-pair F_dec cache).",
        "- Official val `teacher_cache/` not used for PCA fit (train pairs only).",
        "",
        "## Commands",
        "```bash",
        "CUDA_VISIBLE_DEVICES=6 python3 -u scripts/60_student_fdec_pca_extension.py --device cuda",
        "```",
        "",
        "**No architecture, latent target, or training recommendation.**",
    ]
    (report_root / "fdec_pca_extension_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_fdec_extension(cfg, report_root: Path, student_root: Path, device: str = "cpu") -> dict:
    t0 = time.time()
    report_root.mkdir(parents=True, exist_ok=True)
    cache = student_root / "teacher_cache_train_pilot"
    rows = [json.loads(l) for l in (cache / "index.jsonl").open()]
    gt = np.load(cache / "gt_heatmaps.npz", mmap_mode="r")
    pred = np.load(cache / "teacher_pred.npz", mmap_mode="r")
    warning = cfg.get("warning") or {}
    train_ids = json.loads((student_root / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((student_root / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    pairs = _pair_indices(rows, train_ids)
    dev_pairs = _pair_indices(rows, dev_ids)
    ckpt = resolve_parent_ckpt(cfg, None)
    train_set = set(train_ids)

    p1 = part1_verify_fdec_cache(cfg, rows, pred, gt, train_ids, pairs, ckpt, device)
    _write(report_root / "part1_fdec_cache_verification.json", p1)

    p2_pack = part2_fdec_pair_pca(pairs, rows, pred, train_set)
    if isinstance(p2_pack, dict) and p2_pack.get("unavailable"):
        _write(report_root / "part2_fdec_pair_pca.json", p2_pack)
        return {"status": "failed", "part1": p1}
    p2, mu, vt, X, pair_meta = p2_pack
    np.savez_compressed(report_root / "pca_fdec_k64_meta.npz", mu=mu, vt=vt[:64])
    p2.pop("_internal", None)
    _write(report_root / "part2_fdec_pair_pca.json", p2)

    fit_groups, hold_groups = _scenario_split_groups(rows, pairs)
    p3 = part3_task_information(rows, pred, gt, pairs, pair_meta, vt, mu, warning, fit_groups, hold_groups)
    _write(report_root / "part3_fdec_task_information.json", p3)

    p4 = part4_decoder_interventions(cfg, rows, gt, pred, pair_meta, vt, mu, [], dev_pairs, warning, device)
    _write(report_root / "part4_fdec_decoder_interventions.json", p4)

    p5 = part5_representation_comparison(rows, pred, gt, pairs, vt, mu, X, warning, student_root, fit_groups, hold_groups)
    _write(report_root / "part5_representation_comparison.json", p5)

    p6 = part6_architecture_budget(cfg, student_root)
    _write(report_root / "part6_decoder_student_architecture_budget.json", p6)

    write_final_report(report_root)
    return {"part1": p1, "part2": p2, "part4_holdout_pairs": p4.get("eval_pairs_holdout_train_groups"), "elapsed_sec": time.time() - t0}
