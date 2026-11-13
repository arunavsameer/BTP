#!/usr/bin/env python3
"""Part 2: RGB availability, dense sampling, grid alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.grid_align import build_grid_alignment_report  # noqa: E402
from collision_jepa3.student_research.rgb_audit import write_part2  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--student-size", type=int, default=160)
    args = parser.parse_args()
    out = Path(args.out)
    cfg = load_cfg(args)
    write_part2(out, cfg)
    build_grid_alignment_report(out, dst=args.student_size)
    print(f"wrote Part 2 → {out}", flush=True)


if __name__ == "__main__":
    main()
