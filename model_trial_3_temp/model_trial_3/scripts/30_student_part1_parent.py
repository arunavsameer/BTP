#!/usr/bin/env python3
"""Part 1: inspect Pair-CF parent and build teacher cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.parent import resolve_parent_ckpt, write_part1_reports  # noqa: E402
from collision_jepa3.student_research.teacher_cache import cache_integrity, fill_teacher_cache  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--ckpt", default="checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args)
    ckpt = resolve_parent_ckpt(cfg, args.ckpt)
    write_part1_reports(out, cfg, ckpt)
    cache_dir = out / "teacher_cache"
    if not args.skip_cache:
        device = resolve_device(args.device)
        meta = fill_teacher_cache(
            cfg,
            cache_dir,
            ckpt=ckpt,
            device=device,
            batch=args.batch,
            stride=args.stride,
            splits=tuple(args.splits),
        )
        print(json.dumps({"teacher_cache": meta}, indent=2))
    integrity = cache_integrity(cache_dir) if (cache_dir / "meta.json").exists() else {"status": "cache_missing"}
    (out / "part1_cache_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(f"wrote Part 1 → {out}", flush=True)


if __name__ == "__main__":
    main()
