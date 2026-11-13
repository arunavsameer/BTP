"""Aggregate S0 vs S1 comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_last_val(exp_dir: Path) -> dict[str, Any]:
    metrics = sorted(exp_dir.glob("metrics_epoch_*.json"))
    if not metrics:
        return {}
    last = json.loads(metrics[-1].read_text(encoding="utf-8"))
    return last.get("val") or {}


def write_comparison(
    report_root: Path,
    exp_s0: Path,
    exp_s1: Path,
    *,
    integrity: dict,
    budget: dict,
    grad_cal: dict | None,
) -> None:
    cfg0 = json.loads((exp_s0 / "config.json").read_text(encoding="utf-8"))
    cfg1 = json.loads((exp_s1 / "config.json").read_text(encoding="utf-8"))
    v0 = _load_last_val(exp_s0)
    v1 = _load_last_val(exp_s1)
    row = {
        "S0_standard_kd": {
            "parameters": cfg0.get("total_params"),
            "stop_recall": v0.get("stop_recall"),
            "safe_recall": v0.get("safe_recall"),
            "threat_recall": v0.get("threat_recall"),
            "recall_at_5fa": v0.get("recall_at_5fa"),
            "ap_high": v0.get("ap_high"),
            "mse_gt": v0.get("mse_gt"),
            "pre_fork_stop_recall": v0.get("pre_fork_stop_recall"),
            "post_fork_stop_recall": v0.get("post_fork_stop_recall"),
            "pair_gap_median": v0.get("pair_gap_median"),
            "pair_warning_flip_rate": v0.get("pair_warning_flip_rate"),
        },
        "S1_parent_fdec": {
            "parameters": cfg1.get("total_params"),
            "eta_fdec": cfg1.get("eta_fdec"),
            "stop_recall": v1.get("stop_recall"),
            "safe_recall": v1.get("safe_recall"),
            "threat_recall": v1.get("threat_recall"),
            "recall_at_5fa": v1.get("recall_at_5fa"),
            "ap_high": v1.get("ap_high"),
            "mse_gt": v1.get("mse_gt"),
            "pre_fork_stop_recall": v1.get("pre_fork_stop_recall"),
            "post_fork_stop_recall": v1.get("post_fork_stop_recall"),
            "pair_gap_median": v1.get("pair_gap_median"),
            "pair_warning_flip_rate": v1.get("pair_warning_flip_rate"),
            "latent_mse_mean": v1.get("latent_mse_mean"),
            "latent_cosine_mean": v1.get("latent_cosine_mean"),
        },
    }
    out = {
        "comparison_last_epoch": row,
        "integrity_summary": {
            "pair_gt_mismatch": integrity.get("pair_gt_mismatch"),
            "fdec_ok": integrity.get("ok_for_s1"),
        },
        "gradient_calibration": grad_cal,
        "parameter_budget": budget,
        "experiment_dirs": {"S0": str(exp_s0), "S1": str(exp_s1)},
    }
    (report_root / "student_fdec_pilot_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# Student F_dec pilot comparison",
        "",
        "| Metric | S0 | S1 |",
        "|---|---:|---:|",
    ]
    keys = [
        ("parameters", "Parameters"),
        ("stop_recall", "STOP recall"),
        ("safe_recall", "SAFE recall"),
        ("threat_recall", "THREAT recall"),
        ("recall_at_5fa", "Recall@5FA"),
        ("ap_high", "AP_high"),
        ("mse_gt", "MSE vs GT"),
        ("pair_gap_median", "Pair median gap"),
        ("pair_warning_flip_rate", "Pair warning flips"),
    ]
    for k, label in keys:
        a = row["S0_standard_kd"].get(k)
        b = row["S1_parent_fdec"].get(k)
        lines.append(f"| {label} | {a} | {b} |")
    lines.append("")
    lines.append("See `student_fdec_pilot_comparison.json` for full metrics and epoch histories in experiment dirs.")
    lines.append("**No winner selected.**")
    (report_root / "student_fdec_pilot_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    csv_lines = ["metric,S0,S1\n"]
    for k, label in keys:
        csv_lines.append(f"{label},{row['S0_standard_kd'].get(k)},{row['S1_parent_fdec'].get(k)}\n")
    (report_root / "student_fdec_pilot_comparison.csv").write_text("".join(csv_lines), encoding="utf-8")
    (report_root / "experiment_index.json").write_text(
        json.dumps({"student_fdec_s0": str(exp_s0), "student_fdec_s1": str(exp_s1)}, indent=2),
        encoding="utf-8",
    )
