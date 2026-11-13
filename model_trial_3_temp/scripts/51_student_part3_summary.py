#!/usr/bin/env python3
"""Aggregate Part 3 F1–F5 comparison table from experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRIC_KEYS = [
    "stop_recall",
    "safe_recall",
    "recall_at_5fa",
    "ap_high",
    "mse_gt",
    "pre_fork_stop_recall",
    "post_fork_stop_recall",
    "pair_gap_mean",
    "pair_warning_flip_rate",
    "train_seconds_total",
    "n_train",
    "n_dev",
    "params",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    args = parser.parse_args()
    out = Path(args.out)
    p3 = out / "part3_frame_selection.json"
    if not p3.is_file():
        raise SystemExit(f"missing {p3}")
    data = json.loads(p3.read_text(encoding="utf-8"))
    rows = []
    for fs in ("F1", "F2", "F3", "F4", "F5"):
        rec = data.get(fs, {})
        best = rec.get("best") or {k: rec.get(k) for k in METRIC_KEYS if k in rec}
        row = {"frame_set": fs, "arch": rec.get("arch", ""), "exp_id": f"part3_{fs}"}
        for k in METRIC_KEYS:
            row[k] = best.get(k, rec.get(k))
        row["complete"] = not rec.get("partial", False) and (out / "experiments" / f"part3_{fs}" / ".success").is_file()
        rows.append(row)
    json_path = out / "part3_f1_f5_comparison.json"
    json_path.write_text(json.dumps({"rows": rows, "source": str(p3)}, indent=2), encoding="utf-8")
    csv_path = out / "part3_f1_f5_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_set", "arch", "complete", *METRIC_KEYS, "exp_id"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"written": [str(json_path), str(csv_path)], "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
