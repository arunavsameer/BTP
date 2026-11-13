#!/usr/bin/env python3
"""S0/S1/S2 comparison for warning diagnostics (read-only)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.warning_diagnostics_round2 import peak_stats  # noqa: E402

OUT = ROOT / "reports/student_warning_diagnostics"
EXP = {k: ROOT / f"experiments/student_fdec_{k}" for k in ("s0", "s1", "s2_frozen_decoder")}


def _dev_val(exp: Path, ep: int = 15) -> dict:
    p = exp / f"metrics_dev_epoch_{ep:02d}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())["val"]


def _peak_from_npz(npz: np.ndarray) -> dict:
    peaks = np.asarray(npz).max(axis=(1, 2))
    return peak_stats(peaks)


def main() -> None:
    add_config_args(argparse.ArgumentParser())  # unused but consistent
    z = np.load(OUT / "development_predictions.npz")
    row = {}

    def add_variant(name: str, exp: Path, np_key: str | None, fdec: bool = False):
        v = _dev_val(exp)
        pk = _peak_from_npz(z[np_key]) if np_key else {}
        cfg = json.loads((exp / "config.json").read_text()) if (exp / "config.json").exists() else {}
        row[name] = {
            "parameters": cfg.get("total_params"),
            "mse_gt": v.get("mse_gt"),
            "recall_at_5fa": v.get("recall_at_5fa"),
            "ap_high": v.get("ap_high"),
            "stop_recall": v.get("stop_recall"),
            "threat_recall": v.get("threat_recall"),
            "safe_recall": v.get("safe_recall"),
            "teacher_hm_mae": v.get("teacher_hm_mae_mean"),
            "fdec_mse": v.get("latent_mse_mean") if fdec else None,
            "fdec_cosine": v.get("latent_cosine_mean") if fdec else None,
            "peak_p50": pk.get("p50"),
            "peak_p90": pk.get("p90"),
            "peak_frac_025": pk.get("frac_ge_0_25"),
            "peak_frac_060": pk.get("frac_ge_0_60"),
            "pair_gap_median": v.get("pair_gap_median"),
            "pair_warning_flips": v.get("pair_warning_flip_rate"),
        }

    add_variant("S0", EXP["s0"], "s0", False)
    add_variant("S1", EXP["s1"], "s1", True)
    if (EXP["s2_frozen_decoder"] / ".success").exists():
        z2 = None
        add_variant("S2", EXP["s2_frozen_decoder"], None, True)
        dv = _dev_val(EXP["s2_frozen_decoder"])
        if dv:
            row["S2"].update(
                {
                    "mse_gt": dv.get("mse_gt"),
                    "recall_at_5fa": dv.get("recall_at_5fa"),
                    "ap_high": dv.get("ap_high"),
                    "stop_recall": dv.get("stop_recall"),
                    "threat_recall": dv.get("threat_recall"),
                    "safe_recall": dv.get("safe_recall"),
                    "teacher_hm_mae": dv.get("teacher_hm_mae_mean"),
                    "fdec_mse": dv.get("latent_mse_mean"),
                    "fdec_cosine": dv.get("latent_cosine_mean"),
                    "peak_p50": None,
                    "peak_p90": None,
                    "peak_frac_025": None,
                    "peak_frac_060": None,
                    "pair_gap_median": dv.get("pair_gap_median"),
                    "pair_warning_flips": dv.get("pair_warning_flip_rate"),
                    "peak_note": "Run post-S2 dev npz update for peak columns",
                }
            )
    else:
        row["S2"] = {"unavailable": "S2 training not complete"}

    (OUT / "student_s0_s1_s2_comparison.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    keys = list(next(iter(row.values())).keys()) if row else []
    with (OUT / "student_s0_s1_s2_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "S0", "S1", "S2"])
        for k in keys:
            w.writerow([k, row.get("S0", {}).get(k), row.get("S1", {}).get(k), row.get("S2", {}).get(k)])

    lines = ["# S0 / S1 / S2 comparison", "", "| Metric | S0 | S1 | S2 |", "|---|---:|---:|---:|"]
    for k in keys:
        lines.append(f"| {k} | {row['S0'].get(k)} | {row['S1'].get(k)} | {row.get('S2', {}).get(k)} |")
    lines.append("\nNo architecture winner selected.")
    (OUT / "student_s0_s1_s2_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": True}, indent=2))


if __name__ == "__main__":
    main()
