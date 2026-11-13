#!/usr/bin/env python3
"""Fill teacher cache for train-pilot rows only (does not touch validation cache)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.parent import resolve_parent_ckpt  # noqa: E402
from collision_jepa3.student_research.teacher_cache import (  # noqa: E402
    _build_rows,
    cache_integrity,
    fill_teacher_cache_rows,
    subset_rows,
)
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    add_config_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    cache_dir = out / "teacher_cache_train_pilot"
    split_path = out / "pilot_split_train_groups.json"
    if not split_path.is_file():
        raise SystemExit("run 49_student_pilot_train_setup.py first")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    g_tr = list(split["global_train_row_ids"])
    g_dev = list(split["global_dev_row_ids"])
    global_ids = g_tr + g_dev

    ns = argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=int(split.get("seed", 1337)))
    cfg = load_cfg(ns)
    device = resolve_device(args.device)
    ckpt = Path(args.ckpt) if args.ckpt else resolve_parent_ckpt(cfg, None)

    t0 = time.perf_counter()
    full_rows, full_gt, full_meta = _build_rows(cfg, ["train"], args.stride)
    rows, gt, _ = subset_rows(full_rows, full_gt, global_ids, split_label="train_pilot")
    meta_extra = {
        "global_row_ids": global_ids,
        "global_train_row_ids": g_tr,
        "global_dev_row_ids": g_dev,
        "anchors_per_clip": full_meta.get("anchors_per_clip"),
        "n_train_windows": len(g_tr),
        "n_dev_windows": len(g_dev),
    }
    fill_teacher_cache_rows(
        cfg,
        cache_dir,
        rows=rows,
        gt=gt,
        ckpt=ckpt,
        device=device,
        batch=args.batch,
        stride=args.stride,
        meta_extra=meta_extra,
    )
    elapsed = time.perf_counter() - t0
    integrity = cache_integrity(cache_dir)
    integrity["encode_seconds"] = elapsed
    integrity["val_cache_preserved"] = (out / "teacher_cache").is_dir()
    (out / "part1_train_pilot_cache_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    split["train_row_ids"] = list(range(len(g_tr)))
    split["dev_row_ids"] = list(range(len(g_tr), len(g_tr) + len(g_dev)))
    split["cache_built"] = True
    split["cache_dir"] = str(cache_dir.relative_to(ROOT) if cache_dir.is_relative_to(ROOT) else cache_dir)
    split["teacher_ckpt"] = str(ckpt)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(json.dumps(integrity, indent=2), flush=True)
    missing_h = int(integrity.get("missing_H", 0))
    missing_meta = int(integrity.get("meta", {}).get("missing", 0))
    if missing_h != 0 or missing_meta != 0:
        raise SystemExit(f"teacher cache has missing rows: missing_H={missing_h} meta_missing={missing_meta}")


if __name__ == "__main__":
    main()
