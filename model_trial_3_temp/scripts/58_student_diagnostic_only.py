#!/usr/bin/env python3
"""Diagnostic-only student design study (no training, no new caches)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.diagnostic_only import run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    add_config_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    if not (out / "part1_train_manifest.json").exists():
        raise SystemExit("part1_train_manifest.json missing; use existing pilot manifests only.")
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    run_all(cfg, out, ROOT)
    print(json.dumps({"status": "ok", "report": str(out / "diagnostic_only_final_report.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
