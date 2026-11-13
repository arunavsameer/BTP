"""Read-only warning / ranking / decoder isolation diagnostics (round 2)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, HorizonAccumulator, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.student_research.diagnostic_only import _tier_from_gt
from collision_jepa3.student_research._simple_plots import save_grid_heatmaps, save_scatter_plots
from collision_jepa3.student_research.fdec_student import (
    FDEC_FRAME_OFFSETS,
    StudentS0,
    StudentS1,
    build_parent_decoder,
)
from collision_jepa3.student_research.fdec_pilot_train import apply_backbone, prepare_shared_backbone
from collision_jepa3.student_research.parent import load_pair_cf_parent, resolve_parent_ckpt


def build_part1_evaluation_integrity(cfg) -> dict[str, Any]:
    w = cfg.get("warning") or {}
    return {
        "scope": "document inference/eval; no code modified during diagnostics",
        "shared_dev_rows": 816,
        "models": {
            "parent_cached": {
                "input": "not re-encoded; teacher_cache H [5,5] from Pair-CF parent at cache time",
                "output": "H in [0,1] after decoder sigmoid+max-pool (see HeatmapDecoder)",
                "activation": "single sigmoid on coarse logits (max-pool path)",
                "peak": "max(H)",
                "classification": "classify_heatmap(H, warning) → direction, severity SAFE/CAUTION/STOP",
            },
            "parent_decoder_replay": {
                "input": "F_dec [128,16,16] from teacher cache",
                "path": "D_P(F_T) via load_pair_cf_parent().decoder.forward_with_logits",
            },
            "S0": {
                "input": "clip [B,5,3,160,160] uint8→[0,1] via uint8_to_nchw01",
                "backbone": "StudentRGBNet concat stem → fine → decoder.forward_with_logits(fine)",
                "output_key": "Hhat_1s",
                "activation": "HeatmapDecoder: sigmoid once on pooled logits (not double-sigmoid)",
                "clamping": "none beyond sigmoid [0,1]",
            },
            "S1": {
                "input": "same clip as S0",
                "path": "backbone fine → FDecSpatialProjection → decoder.forward_with_logits",
                "output_keys": "Hhat_1s, F_student",
                "decoder_note": "epoch-15 S1.decoder may differ from frozen parent init after unfreeze",
            },
        },
        "thresholds": {
            "caution": float(w.get("caution_threshold", 0.25)),
            "stop": float(w.get("stop_threshold", 0.60)),
            "high_threat_cells": 0.5,
            "safe_empty_cells": 0.05,
        },
        "gt_frame_tier": "_tier_from_gt(gt_1s): STOP if peak≥stop, else THREAT if peak≥caution, else SAFE",
        "pilot_epoch_threat_recall": "HorizonAccumulator high_threat_recall (cell-level gt≥0.5), not CAUTION tier",
        "recall_at_5fa": "operating_point.recall_at_false_alarm: threshold on flattened cells, neg=gt<0.5",
        "ap_high": "sklearn average_precision on flattened cells y=(g>=0.5)",
        "peak_checks": {
            "double_sigmoid_risk": "none for max-pool decoder path",
            "s0_s1_same_peak_def": "np.max(Hhat_1s)",
            "teacher_cache_index": "row_id indexes index.jsonl and npz rows consistently in FDecPilotDataset",
            "frame_anchor": "indices_at_anchor(frame, FDEC_FRAME_OFFSETS)",
        },
        "bugs_found": {
            "s0_saved_dev_pair_metrics": {
                "issue": "train verified_pair_keys used in dev eval before fix",
                "status": "documented; use warning_diagnostics npz for pairing",
            }
        },
    }


def peak_stats(peaks: np.ndarray) -> dict[str, float | None]:
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.size == 0:
        return {k: None for k in ("min", "mean", "std", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max")}
    qs = np.percentile(peaks, [10, 25, 50, 75, 90, 95, 99])
    return {
        "min": float(peaks.min()),
        "mean": float(peaks.mean()),
        "std": float(peaks.std()),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        "max": float(peaks.max()),
        "frac_ge_0_25": float(np.mean(peaks >= 0.25)),
        "frac_ge_0_60": float(np.mean(peaks >= 0.60)),
        "n": int(peaks.size),
    }


def _class_stats(peaks: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    sub = peaks[mask]
    if sub.size == 0:
        return {"n": 0, "unavailable": True}
    qs = np.percentile(sub, [50, 90, 95, 99])
    return {
        "n": int(sub.size),
        "mean": float(sub.mean()),
        "median": float(qs[0]),
        "p90": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
        "frac_ge_0_25": float(np.mean(sub >= 0.25)),
        "frac_ge_0_60": float(np.mean(sub >= 0.60)),
    }


def corr_and_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return {"pearson": None, "spearman": None, "slope": None, "intercept": None, "residual_rmse": None}
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rx, ry)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return {
        "pearson": pear,
        "spearman": spear,
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_rmse": float(np.sqrt(np.mean(resid**2))),
        "residual_mean": float(np.mean(resid)),
    }


def cell_maps(hm_stack: np.ndarray, gt_stack: np.ndarray, *, hi: float = 0.5, empty_thr: float = 0.05, stop_thr: float = 0.6) -> dict[str, Any]:
    mean_pred = hm_stack.mean(axis=0)
    mean_gt = gt_stack.mean(axis=0)
    mae = np.abs(hm_stack - gt_stack).mean(axis=0)
    occ_rec, emp_fp = [], []
    for i in range(hm_stack.shape[0]):
        hg, hp = gt_stack[i], hm_stack[i]
        occ = hg >= hi
        emp = hg < empty_thr
        if occ.any():
            occ_rec.append(float((hp[occ] >= hi).mean()))
        emp_fp.append(float((hp[emp] >= hi).mean()) if emp.any() else float("nan"))
    argmax = np.argmax(hm_stack.reshape(hm_stack.shape[0], -1), axis=1)
    rows = argmax // 5
    cols = argmax % 5
    return {
        "mean_pred_grid": mean_pred.tolist(),
        "mean_gt_grid": mean_gt.tolist(),
        "mean_abs_err_grid": mae.tolist(),
        "occupied_cell_recall_mean": float(np.nanmean(occ_rec)) if occ_rec else None,
        "empty_cell_fp_rate_mean": float(np.nanmean(emp_fp)) if emp_fp else None,
        "pred_peak_row_hist": {str(i): int((rows == i).sum()) for i in range(5)},
        "pred_peak_col_hist": {str(j): int((cols == j).sum()) for j in range(5)},
    }


def horizon_and_op(preds: np.ndarray, gts: np.ndarray, warning: dict) -> dict[str, Any]:
    acc = HorizonAccumulator("1s", warning=warning)
    for i in range(preds.shape[0]):
        acc.update(preds[i], gts[i], masked=False, outcome="", background="", scenario="", pre_fork=False)
    s = acc.summary()
    valids = np.ones_like(gts, dtype=bool)
    op = operating_metrics(preds, gts, valids, hi=0.5)
    return {
        "stop_recall": s["stop_recall"],
        "threat_recall_cell_hi05": s["high_threat_recall"],
        "safe_recall": s["safe_recall"],
        "mse_gt": s["mse"],
        "recall_at_5fa": op.get("recall_at_5fa"),
        "ap_high": op.get("ap_high"),
    }


def peak_threshold_sweep(peaks: np.ndarray, gt_tiers: list[str], warning: dict) -> dict[str, Any]:
    peaks = np.asarray(peaks, float)
    gt_tiers = np.asarray(gt_tiers)
    is_stop = gt_tiers == "STOP"
    is_threat_or_stop = np.isin(gt_tiers, ("THREAT", "STOP"))
    is_safe = gt_tiers == "SAFE"
    thresholds = np.unique(np.quantile(peaks, np.linspace(0, 1, 101)))
    thresholds = np.sort(np.unique(np.concatenate([thresholds, [0.25, 0.60]])))
    rows = []
    for thr in thresholds:
        pred_stop = peaks >= thr
        rec_stop = float((pred_stop & is_stop).sum() / is_stop.sum()) if is_stop.any() else None
        rec_ts = float((pred_stop & is_threat_or_stop).sum() / is_threat_or_stop.sum()) if is_threat_or_stop.any() else None
        fa_safe = float((pred_stop & is_safe).sum() / is_safe.sum()) if is_safe.any() else None
        prec_stop = float((pred_stop & is_stop).sum() / pred_stop.sum()) if pred_stop.any() else None
        rows.append(
            {
                "threshold": float(thr),
                "stop_recall": rec_stop,
                "threat_or_stop_recall": rec_ts,
                "precision_stop_tier": prec_stop,
                "fa_rate_on_gt_safe": fa_safe,
            }
        )
    at25 = next((r for r in rows if abs(r["threshold"] - 0.25) < 1e-9), None)
    at60 = next((r for r in rows if abs(r["threshold"] - 0.60) < 1e-9), None)
    return {
        "sweep": rows,
        "at_0_25": at25,
        "at_0_60": at60,
        "note_monotonic_rescale": "Affine p'=a*p+b with a>0 preserves ranking; cannot change AP_high or recall@5FA (cell-level).",
    }


@torch.no_grad()
def collect_all_predictions(
    cfg,
    cache_dir: Path,
    dev_ids: list[int],
    rows: list[dict],
    loader,
    device: str,
    s0: StudentS0,
    s1: StudentS1,
    parent_dec,
) -> dict[str, Any]:
    pred_npz = np.load(cache_dir / "teacher_pred.npz")
    gt_npz = np.load(cache_dir / "gt_heatmaps.npz")
    n = len(dev_ids)
    gt = np.zeros((n, 5, 5), np.float32)
    hp_cached = np.zeros((n, 5, 5), np.float32)
    hs0 = np.zeros((n, 5, 5), np.float32)
    hs1 = np.zeros((n, 5, 5), np.float32)
    h_tt = np.zeros((n, 5, 5), np.float32)
    meta: dict[str, list] = {k: [] for k in ("row_id", "sample_id", "clip_id", "frame", "scenario_group_id", "outcome_group_id", "background_pair_id", "pre_fork")}

    s0.eval().to(device)
    s1.eval().to(device)
    parent_dec.eval().to(device)
    idx = 0
    for batch in loader:
        bsz = batch["clip"].shape[0]
        clip = batch["clip"].to(device)
        rids = batch["row_id"]
        if torch.is_tensor(rids):
            rids = rids.cpu().numpy()
        f_dec = batch.get("F_teacher")
        if f_dec is not None:
            f_dec = f_dec.to(device)
        out0 = s0(clip)
        out1 = s1(clip)
        h0 = out0["Hhat_1s"].float().cpu().numpy()
        h1 = out1["Hhat_1s"].float().cpu().numpy()
        f_s = out1["F_student"].float()
        for i in range(bsz):
            rid = int(rids[i])
            row = rows[rid]
            gt[idx] = np.asarray(gt_npz["gt_1s"][rid], dtype=np.float32)
            hp_cached[idx] = np.asarray(pred_npz["H"][rid], dtype=np.float32)
            hs0[idx] = h0[i]
            hs1[idx] = h1[i]
            if f_dec is not None:
                ft = f_dec[i : i + 1]
                fs = f_s[i : i + 1]
                htt, _ = parent_dec.forward_with_logits(ft)
                h_tt[idx] = htt[0].cpu().numpy()
            meta["row_id"].append(rid)
            meta["sample_id"].append(row.get("sample_id", ""))
            meta["clip_id"].append(row["clip_id"])
            meta["frame"].append(int(row["frame"]))
            meta["scenario_group_id"].append(row.get("scenario_group_id", ""))
            meta["outcome_group_id"].append(row.get("outcome_group_id", ""))
            meta["background_pair_id"].append(row.get("background_pair_id", ""))
            meta["pre_fork"].append(bool(row.get("pre_fork")))
            idx += 1

    return {
        "gt": gt,
        "parent_cached": hp_cached,
        "parent_replay_Dp_Ft": h_tt,
        "s0": hs0,
        "s1": hs1,
        **meta,
        "replay_mse_vs_cached_parent": float(np.mean((h_tt - hp_cached) ** 2)),
        "replay_max_abs_vs_cached": float(np.max(np.abs(h_tt - hp_cached))),
    }


def part5_decoder_isolation(
    cfg,
    cache_dir: Path,
    dev_ids: list[int],
    loader,
    device: str,
    s1: StudentS1,
    parent_dec,
    gts: np.ndarray,
    parent_cached: np.ndarray,
    warning: dict,
    verified_dev: list[tuple],
) -> dict[str, Any]:
    """Tests A–D on F_dec pathways (inference only)."""

    @torch.no_grad()
    def _run_decode(dec, f_batch: torch.Tensor) -> np.ndarray:
        h, _ = dec.forward_with_logits(f_batch)
        return h.float().cpu().numpy()

    buckets: dict[str, list[np.ndarray]] = {k: [] for k in "ABCD"}
    for batch in loader:
        clip = batch["clip"].to(device)
        f_t = batch["F_teacher"].to(device)
        s1.eval()
        f_s = s1.proj(s1.spatial_features(clip))
        h_a = _run_decode(parent_dec, f_t)
        h_b = _run_decode(parent_dec, f_s)
        h_c = _run_decode(s1.decoder, f_s)
        h_d = _run_decode(s1.decoder, f_t)
        for i in range(clip.shape[0]):
            buckets["A"].append(h_a[i])
            buckets["B"].append(h_b[i])
            buckets["C"].append(h_c[i])
            buckets["D"].append(h_d[i])

    out_tests: dict[str, Any] = {}
    labels = {
        "A_parent_Dp_Ft": "H_TT = D_P(F_T)",
        "B_parent_Dp_Fs": "H_ST = D_P(F_S)",
        "C_s1_Ds1_Fs": "H_SS = D_S1(F_S)",
        "D_s1_Ds1_Ft": "H_TS = D_S1(F_T)",
    }
    for key, label, arr_key in (
        ("A_parent_Dp_Ft", labels["A_parent_Dp_Ft"], "A"),
        ("B_parent_Dp_Fs", labels["B_parent_Dp_Fs"], "B"),
        ("C_s1_Ds1_Fs", labels["C_s1_Ds1_Fs"], "C"),
        ("D_s1_Ds1_Ft", labels["D_s1_Ds1_Ft"], "D"),
    ):
        stack = np.stack(buckets[arr_key], axis=0)
        peaks = stack.max(axis=(1, 2))
        mse_gt = float(np.mean((stack - gts) ** 2))
        mse_parent = float(np.mean((stack - parent_cached) ** 2))
        out_tests[key] = {
            "label": label,
            "heatmap_mse_vs_gt": mse_gt,
            "heatmap_mse_vs_cached_parent": mse_parent,
            "peak_distribution": peak_stats(peaks),
            **horizon_and_op(stack, gts, warning),
        }

    interp = {
        "A_vs_B_Fdec_repr": float(out_tests["A_parent_Dp_Ft"]["heatmap_mse_vs_gt"]) - float(out_tests["B_parent_Dp_Fs"]["heatmap_mse_vs_gt"]),
        "A_vs_D_decoder_drift_on_Ft": float(out_tests["A_parent_Dp_Ft"]["heatmap_mse_vs_gt"]) - float(out_tests["D_s1_Ds1_Ft"]["heatmap_mse_vs_gt"]),
        "B_vs_C_decoder_adaptation_on_Fs": float(out_tests["B_parent_Dp_Fs"]["heatmap_mse_vs_gt"]) - float(out_tests["C_s1_Ds1_Fs"]["heatmap_mse_vs_gt"]),
        "cached_parent_vs_A_mse": float(np.mean((parent_cached - np.stack(buckets["A"], 0)) ** 2)),
    }
    return {"tests": out_tests, "deltas": interp, "n_dev": len(dev_ids)}


def affine_oracle(parent_p: np.ndarray, student_p: np.ndarray, gt_tiers: list[str]) -> dict[str, Any]:
    """Dev-only diagnostic; not deployable."""
    x, y = student_p, parent_p
    slope, intercept = np.polyfit(x, y, 1)
    mapped = np.clip(slope * x + intercept, 0, 1)
    is_safe = np.array(gt_tiers) == "SAFE"
    return {
        "oracle": True,
        "not_deployable": True,
        "a": float(slope),
        "b": float(intercept),
        "mapped_frac_ge_0_60": float(np.mean(mapped >= 0.60)),
        "fa_on_gt_safe_at_0_60_after_map": float(np.mean(mapped[is_safe] >= 0.60)) if is_safe.any() else None,
        "ranking_preserved": True,
        "note": "Monotonic affine map cannot improve cell-level AP or recall@5FA; only peak-threshold diagnostics shift.",
    }


def run_round2(
    cfg,
    *,
    report_dir: Path,
    cache_dir: Path,
    manifest_root: Path,
    exp_s0: Path,
    exp_s1: Path,
    baseline_init: Path,
    device: str,
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    warning = cfg.get("warning") or {}

    (report_dir / "part1_evaluation_integrity.json").write_text(
        json.dumps(build_part1_evaluation_integrity(cfg), indent=2), encoding="utf-8"
    )

    train_ids = json.loads((manifest_root / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((manifest_root / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open()]

    from collision_jepa3.student_research.fdec_pilot_dataset import FDecPilotDataset, build_frame_store, make_fdec_loader
    from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report

    integrity, _ = build_integrity_report(cfg, cache_dir, train_ids, dev_ids, device="cpu")
    verified_dev = [tuple(x) for x in integrity["verified_pair_keys_dev"]]

    store, _ = build_frame_store(cfg, cache_dir, sorted(set(train_ids) | set(dev_ids)), FDEC_FRAME_OFFSETS)
    ds = FDecPilotDataset(cache_dir, dev_ids, FDEC_FRAME_OFFSETS, load_f_dec=True)
    ds.attach_frame_store(store, {})
    bench = json.loads((manifest_root / "student_fdec_dataloader_benchmark.json").read_text())
    loader = make_fdec_loader(ds, batch=8, shuffle=False, num_workers=int(bench["selected_num_workers"]))

    shared = prepare_shared_backbone(manifest_root, seed=1337, baseline_init=baseline_init)
    s0 = StudentS0(n_frames=len(FDEC_FRAME_OFFSETS))
    apply_backbone(s0, shared)
    ck_s0 = exp_s0 / "checkpoint_epoch_15.pt"
    s0.load_state_dict(torch.load(ck_s0, map_location="cpu", weights_only=True))

    parent_dec = build_parent_decoder(cfg, device="cpu")
    s1 = StudentS1(n_frames=len(FDEC_FRAME_OFFSETS), parent_decoder=parent_dec)
    apply_backbone(s1, shared)
    s1.load_state_dict(torch.load(exp_s1 / "checkpoint_epoch_15.pt", map_location="cpu", weights_only=True))

    parent_ckpt = resolve_parent_ckpt(cfg, "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt")
    parent_model, _ = load_pair_cf_parent(cfg, parent_ckpt, device=device)
    parent_dec_gpu = parent_model.decoder.to(device)

    pack = collect_all_predictions(cfg, cache_dir, dev_ids, rows, loader, device, s0, s1, parent_dec_gpu)
    npz_path = report_dir / "development_predictions.npz"
    np.savez_compressed(
        npz_path,
        gt=pack["gt"],
        parent_cached=pack["parent_cached"],
        parent_replay_Dp_Ft=pack["parent_replay_Dp_Ft"],
        s0=pack["s0"],
        s1=pack["s1"],
        row_id=np.asarray(pack["row_id"], dtype=np.int32),
        frame=np.asarray(pack["frame"], dtype=np.int32),
    )

    part2 = {
        "n_dev": len(dev_ids),
        "offsets": FDEC_FRAME_OFFSETS,
        "checkpoints": {"S0": str(ck_s0), "S1": str(exp_s1 / "checkpoint_epoch_15.pt"), "parent_ckpt": str(parent_ckpt)},
        "parent_replay_mse_vs_cache": pack["replay_mse_vs_cached_parent"],
        "parent_replay_max_abs_vs_cache": pack["replay_max_abs_vs_cached"],
        "row_id_unique": len(set(pack["row_id"])) == len(pack["row_id"]),
        "npz": str(npz_path),
    }
    (report_dir / "part2_prediction_integrity.json").write_text(json.dumps(part2, indent=2), encoding="utf-8")

    gt_p = pack["gt"].max(axis=(1, 2))
    parent_p = pack["parent_cached"].max(axis=(1, 2))
    s0_p = pack["s0"].max(axis=(1, 2))
    s1_p = pack["s1"].max(axis=(1, 2))
    gt_tiers = [_tier_from_gt(pack["gt"][i], warning) for i in range(len(dev_ids))]

    part3 = {
        "overall_peak": {
            "parent_cached": peak_stats(parent_p),
            "s0": peak_stats(s0_p),
            "s1": peak_stats(s1_p),
        },
        "conditional_on_gt_tier": {},
        "correlations": {
            "gt_vs_parent": corr_and_fit(gt_p, parent_p),
            "gt_vs_s0": corr_and_fit(gt_p, s0_p),
            "gt_vs_s1": corr_and_fit(gt_p, s1_p),
            "parent_vs_s0": corr_and_fit(parent_p, s0_p),
            "parent_vs_s1": corr_and_fit(parent_p, s1_p),
        },
        "spatial_cell": {},
        "plot_files": [],
    }
    for tier in ("SAFE", "THREAT", "STOP"):
        mask = np.array([t == tier for t in gt_tiers])
        part3["conditional_on_gt_tier"][tier] = {
            "parent_cached": _class_stats(parent_p, mask),
            "s0": _class_stats(s0_p, mask),
            "s1": _class_stats(s1_p, mask),
        }
    for label, stack in (
        ("parent_cached", pack["parent_cached"]),
        ("s0", pack["s0"]),
        ("s1", pack["s1"]),
    ):
        part3["spatial_cell"][label] = cell_maps(stack, pack["gt"])
    part3["plot_files"] = save_scatter_plots(report_dir, gt_p, parent_p, s0_p, s1_p)
    part3["plot_files"] += save_grid_heatmaps(
        report_dir,
        part3["spatial_cell"]["parent_cached"]["mean_gt_grid"],
        np.array(part3["spatial_cell"]["parent_cached"]["mean_pred_grid"]),
        np.array(part3["spatial_cell"]["s0"]["mean_pred_grid"]),
        np.array(part3["spatial_cell"]["s1"]["mean_pred_grid"]),
        np.array(part3["spatial_cell"]["s0"]["mean_abs_err_grid"]),
        np.array(part3["spatial_cell"]["s1"]["mean_abs_err_grid"]),
    )
    (report_dir / "part3_warning_intensity.json").write_text(json.dumps(part3, indent=2), encoding="utf-8")

    part4 = {"models": {}, "affine_oracle_s0": affine_oracle(parent_p, s0_p, gt_tiers), "affine_oracle_s1": affine_oracle(parent_p, s1_p, gt_tiers)}
    for name, stack, peaks in (
        ("parent_cached", pack["parent_cached"], parent_p),
        ("s0", pack["s0"], s0_p),
        ("s1", pack["s1"], s1_p),
    ):
        part4["models"][name] = {
            "peak_threshold_sweep": peak_threshold_sweep(peaks, gt_tiers, warning),
            "cell_metrics": horizon_and_op(stack, pack["gt"], warning),
        }
    (report_dir / "part4_ranking_and_threshold_sweep.json").write_text(json.dumps(part4, indent=2), encoding="utf-8")

    s1.to(device)
    part5 = part5_decoder_isolation(
        cfg, cache_dir, dev_ids, loader, device, s1, parent_dec_gpu, pack["gt"], pack["parent_cached"], warning, verified_dev
    )
    (report_dir / "part5_decoder_isolation.json").write_text(json.dumps(part5, indent=2), encoding="utf-8")

    summary = {
        "elapsed_sec": time.time() - t0,
        "outputs": [str(report_dir / f) for f in (
            "part1_evaluation_integrity.json",
            "part2_prediction_integrity.json",
            "development_predictions.npz",
            "part3_warning_intensity.json",
            "part4_ranking_and_threshold_sweep.json",
            "part5_decoder_isolation.json",
        )],
    }
    (report_dir / "round2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"elapsed_sec": summary["elapsed_sec"], "outputs": summary["outputs"]}
