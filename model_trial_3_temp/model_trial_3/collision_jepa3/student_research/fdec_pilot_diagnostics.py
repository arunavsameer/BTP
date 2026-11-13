"""Read-only S0 development diagnostics (no checkpoint or threshold changes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, HorizonAccumulator, classify_heatmap
from collision_jepa3.student_research.fdec_pilot_eval import evaluate_fdec_pilot
from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report


def _severity_counts(hms: np.ndarray, warning: dict) -> dict[str, int]:
    counts = {"SAFE": 0, "CAUTION": 0, "STOP": 0}
    for hm in hms:
        _, sev, _ = classify_heatmap(hm, warning)
        if sev == SAFE:
            counts["SAFE"] += 1
        elif sev == CAUTION:
            counts["CAUTION"] += 1
        else:
            counts["STOP"] += 1
    return counts


def _distribution_pctiles(peaks: np.ndarray, ps: tuple[float, ...]) -> dict[str, float | None]:
    if peaks.size == 0:
        return {f"P{int(p)}": None for p in ps}
    return {f"P{int(p)}": float(np.percentile(peaks, p)) for p in ps}


@torch.no_grad()
def collect_dev_predictions(
    model,
    loader,
    device: str,
    warning: dict,
) -> dict[str, Any]:
    model.eval()
    acc = HorizonAccumulator("1s", warning=warning)
    student_h: list[np.ndarray] = []
    parent_h: list[np.ndarray] = []
    gt_h: list[np.ndarray] = []
    row_ids: list[int] = []
    peaks_s: list[float] = []
    peaks_p: list[float] = []
    pre_fork: list[bool] = []

    for batch in loader:
        bt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(bt["clip"])
        h = out["Hhat_1s"].float()
        hp = bt["H_teacher"].float()
        gt = bt["gt_1s"]
        bsz = h.shape[0]
        h_np = h.cpu().numpy()
        hp_np = hp.cpu().numpy()
        gt_np = gt.cpu().numpy()
        pre = bt.get("pre_fork", [False] * bsz)
        pre_np = pre.detach().cpu().bool().numpy() if torch.is_tensor(pre) else np.asarray(pre, dtype=bool)
        rids = bt.get("row_id")
        if torch.is_tensor(rids):
            rids_np = rids.detach().cpu().numpy()
        else:
            rids_np = np.asarray(rids)

        for i in range(bsz):
            acc.update(
                h_np[i],
                gt_np[i],
                masked=False,
                outcome=str(bt.get("outcome_type", [""] * bsz)[i]),
                background="",
                scenario=str(bt.get("scenario_type", [""] * bsz)[i]),
                pre_fork=bool(pre_np[i]) if i < len(pre_np) else False,
            )
            student_h.append(h_np[i])
            parent_h.append(hp_np[i])
            gt_h.append(gt_np[i])
            row_ids.append(int(rids_np[i]))
            peaks_s.append(float(h_np[i].max()))
            peaks_p.append(float(hp_np[i].max()))
            pre_fork.append(bool(pre_np[i]))

    peaks_s_arr = np.asarray(peaks_s, dtype=np.float32)
    peaks_p_arr = np.asarray(peaks_p, dtype=np.float32)
    gt_stack = np.stack(gt_h, axis=0)
    summ = acc.summary()
    stop_thr = float(warning.get("stop_threshold", 0.60))
    caution_thr = float(warning.get("caution_threshold", 0.25))

    return {
        "row_ids": row_ids,
        "student_heatmaps": np.stack(student_h, axis=0).astype(np.float32),
        "parent_heatmaps": np.stack(parent_h, axis=0).astype(np.float32),
        "gt_heatmaps": gt_stack.astype(np.float32),
        "student_peaks": peaks_s_arr,
        "parent_peaks": peaks_p_arr,
        "pre_fork": np.asarray(pre_fork, dtype=bool),
        "horizon_summary": summ,
        "stop_confusion": {
            "tp": int(acc.stop_tp),
            "fp": int(acc.stop_fp),
            "fn": int(acc.stop_fn),
            "tn": int(acc.n - acc.stop_tp - acc.stop_fp - acc.stop_fn),
            "note": "STOP tier from classify_heatmap on full 5×5 grid; TN is residual non-STOP frame rows.",
        },
        "gt_severity_counts": _severity_counts(gt_stack, warning),
        "student_severity_counts": _severity_counts(np.stack(student_h, axis=0), warning),
        "parent_severity_counts": _severity_counts(np.stack(parent_h, axis=0), warning),
        "student_peak_percentiles": _distribution_pctiles(peaks_s_arr, (10, 25, 50, 75, 90, 99)),
        "parent_peak_percentiles": _distribution_pctiles(peaks_p_arr, (10, 25, 50, 75, 90, 99)),
        "student_pct_crossing_0_25": float((peaks_s_arr >= caution_thr).mean()) if peaks_s_arr.size else None,
        "student_pct_crossing_0_60": float((peaks_s_arr >= stop_thr).mean()) if peaks_s_arr.size else None,
        "parent_pct_crossing_0_25": float((peaks_p_arr >= caution_thr).mean()) if peaks_p_arr.size else None,
        "parent_pct_crossing_0_60": float((peaks_p_arr >= stop_thr).mean()) if peaks_p_arr.size else None,
        "stop_recall_pre_fork": summ.get("pre_fork_stop_recall"),
        "stop_recall_post_fork": summ.get("post_fork_stop_recall"),
        "pre_fork_stop_n": summ.get("pre_fork_stop_n"),
        "post_fork_stop_n": summ.get("post_fork_stop_n"),
        "thresholds_used": {"caution": caution_thr, "stop": stop_thr, "source": "config warning (unchanged)"},
    }


def build_s0_dev_diagnostics_report(
    cfg,
    cache_dir: Path,
    dev_ids: list[int],
    model,
    loader,
    device: str,
    *,
    verified_pair_keys_dev: list[tuple],
    checkpoint_path: Path,
    epoch: int = 15,
) -> dict[str, Any]:
    warning = cfg.get("warning") or {}
    preds = collect_dev_predictions(model, loader, device, warning)
    val_corrected = evaluate_fdec_pilot(
        model,
        loader,
        device,
        warning,
        verified_pair_keys=verified_pair_keys_dev,
        variant="S0",
    )
    legacy_note = {
        "issue": "metrics_dev_epoch_*.json used train verified_pair_keys only",
        "effect": "n_pairs_verified_evaluated=0 and pair_gap_* were NaN with misleading flip_rate 0.0",
        "dev_verified_pairs_available": len(verified_pair_keys_dev),
    }
    return {
        "checkpoint_readonly": str(checkpoint_path),
        "epoch": epoch,
        "n_dev_rows": len(dev_ids),
        "legacy_pair_metrics_bug": legacy_note,
        "dev_eval_corrected_pair_metrics": {
            k: val_corrected.get(k)
            for k in (
                "pair_gap_mean",
                "pair_gap_median",
                "pair_gap_p90",
                "pair_warning_flip_rate",
                "n_pairs_verified_evaluated",
                "n_rows_with_background_pair_id",
                "n_pair_anchor_keys_complete",
                "pair_metrics_unavailable_reason",
            )
        },
        "predictions_summary": {
            k: v
            for k, v in preds.items()
            if k not in ("row_ids",) and not k.endswith("heatmaps") and not isinstance(v, np.ndarray)
        },
        "parent_vs_student_peak_delta_mean": float(np.mean(preds["student_peaks"] - preds["parent_peaks"]))
        if preds["student_peaks"].size
        else None,
    }


def save_dev_prediction_arrays(report: dict, out_npz: Path) -> None:
    """Optional materialized arrays for downstream analysis (not in experiment dir)."""
    if "arrays" not in report:
        return
    np.savez_compressed(out_npz, **report["arrays"])
