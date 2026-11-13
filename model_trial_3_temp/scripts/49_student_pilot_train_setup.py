#!/usr/bin/env python3
"""Build train-only pilot split and dense row manifest (~800 pair-complete windows)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.pilot_rows import (  # noqa: E402
    build_train_dev_split,
    select_pilot_row_ids,
    train_partition_groups,
    verify_manifest,
    write_pilot_split_train_groups,
)
from collision_jepa3.student_research.teacher_cache import _build_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--pilot-seed", type=int, default=1337, help="pilot group shuffle seed (default 1337)")
    parser.add_argument("--target-train", type=int, default=800)
    parser.add_argument("--stride", type=int, default=6)
    add_config_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    ns = argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.pilot_seed)
    cfg = load_cfg(ns)

    pool = train_partition_groups(cfg)
    tr_g, dev_g = build_train_dev_split(pool, seed=args.pilot_seed)
    print(f"train partition groups={len(pool)} pilot_train={len(tr_g)} pilot_dev={len(dev_g)}", flush=True)

    rows, _gt, meta = _build_rows(cfg, ["train"], args.stride)
    sel = select_pilot_row_ids(
        rows,
        train_groups=set(tr_g),
        dev_groups=set(dev_g),
        target_train=args.target_train,
        seed=args.pilot_seed,
    )
    manifest = {
        "global_train_row_ids": sel["train_row_ids"],
        "global_dev_row_ids": sel["dev_row_ids"],
        **sel,
    }
    check = verify_manifest(
        rows,
        {
            **manifest,
            "pilot_train_groups": tr_g,
            "pilot_dev_groups": dev_g,
            "train_row_ids": sel["train_row_ids"],
            "dev_row_ids": sel["dev_row_ids"],
        },
    )
    split_path = out / "pilot_split_train_groups.json"
    write_pilot_split_train_groups(
        split_path,
        seed=args.pilot_seed,
        train_groups=tr_g,
        dev_groups=dev_g,
        row_manifest=manifest,
    )
    report = {
        "source_partition": "train",
        "official_val_used": False,
        "n_dense_train_rows_total": len(rows),
        "pilot_train_groups_n": len(tr_g),
        "pilot_dev_groups_n": len(dev_g),
        "group_overlap": [],
        "selection": sel,
        "verify": check,
        "stride": args.stride,
        "anchors_meta": meta,
        "note": "global_*_row_ids index the full dense train stride-6 table; compact cache ids assigned by script 50.",
    }
    (out / "part3_pilot_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not check.get("ok"):
        raise SystemExit(f"split verification failed: {check}")


if __name__ == "__main__":
    main()
