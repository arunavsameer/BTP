#!/usr/bin/env python3
"""Part 1: train/dev manifests for temporal supernet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.dev_manifest import (  # noqa: E402
    build_dev_manifest,
    build_train_manifest,
    manifest_integrity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--pairs-per-group", type=int, default=12)
    add_config_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    split = json.loads((out / "pilot_split_train_groups.json").read_text(encoding="utf-8"))
    cache_dir = out / "teacher_cache_train_pilot"
    rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open()]
    gt = np.load(cache_dir / "gt_heatmaps.npz", mmap_mode="r")
    peaks = np.nanmax(gt["gt_1s"], axis=(1, 2))

    train_ids = list(split["train_row_ids"])
    tr_g = set(split["pilot_train_groups"])
    dev_g = set(split["pilot_dev_groups"])
    train_m = build_train_manifest(rows, train_ids)
    dev_m = build_dev_manifest(
        rows,
        peaks,
        dev_row_ids=list(split["dev_row_ids"]),
        dev_groups=dev_g,
        pairs_per_group=args.pairs_per_group,
        seed=int(split.get("seed", 1337)),
    )
    integrity = manifest_integrity(rows, train_m, dev_m, train_groups=tr_g, dev_groups=dev_g)
    (out / "part1_train_manifest.json").write_text(json.dumps(train_m, indent=2), encoding="utf-8")
    (out / "part1_dev_manifest.json").write_text(json.dumps(dev_m, indent=2), encoding="utf-8")
    (out / "part1_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps({"integrity": integrity, "dev": {k: dev_m[k] for k in ("n_rows", "n_pairs", "n_groups")}}, indent=2), flush=True)
    if not integrity.get("ok"):
        raise SystemExit(integrity)


if __name__ == "__main__":
    main()
