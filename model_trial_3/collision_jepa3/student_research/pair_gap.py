"""Pair heatmap gap vs parent (Part 8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def part8_pair_robustness(cache_dir: Path, out_dir: Path, split: str = "val") -> dict:
    rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open()]
    pred = np.load(cache_dir / "teacher_pred.npz", mmap_mode="r")
    by_pair: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("split") != split:
            continue
        pid = r.get("background_pair_id") or ""
        if pid:
            by_pair.setdefault(pid, []).append(i)
    gaps = []
    flips = 0
    n_pairs = 0
    stop_thr = 0.60
    for ids in by_pair.values():
        if len(ids) < 2:
            continue
        a, b = ids[0], ids[1]
        ha, hb = pred["H"][a], pred["H"][b]
        gaps.append(float(np.abs(ha - hb).mean()))
        pa, pb = float(ha.max()), float(hb.max())
        ta = pa >= stop_thr
        tb = pb >= stop_thr
        if ta != tb:
            flips += 1
        n_pairs += 1
    arr = np.asarray(gaps, dtype=np.float32)
    report = {
        "split": split,
        "n_pairs": n_pairs,
        "parent_gap_mean": float(arr.mean()) if len(arr) else None,
        "parent_gap_median": float(np.median(arr)) if len(arr) else None,
        "parent_gap_p90": float(np.quantile(arr, 0.9)) if len(arr) else None,
        "parent_warning_flip_rate": flips / max(n_pairs, 1),
        "note": "Student pair gaps require trained student checkpoints; parent-only baseline here.",
    }
    (out_dir / "part8_student_pair_robustness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
