#!/usr/bin/env python3
"""Student warning diagnostics round 2 (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.warning_diagnostics_round2 import run_round2  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402

REPORT = ROOT / "reports/student_architecture_tests"
OUT = ROOT / "reports/student_warning_diagnostics"
CACHE = REPORT / "teacher_cache_train_pilot"
BASELINE = ROOT / "experiments/student_fdec_unoptimized_baseline/shared_backbone_init.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    add_config_args(parser)
    parser.set_defaults(seed=1337)
    args = parser.parse_args()
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    device = str(resolve_device(args.device))
    out = run_round2(
        cfg,
        report_dir=OUT,
        cache_dir=CACHE,
        manifest_root=REPORT,
        exp_s0=ROOT / "experiments/student_fdec_s0",
        exp_s1=ROOT / "experiments/student_fdec_s1",
        baseline_init=BASELINE,
        device=device,
    )
    print(json.dumps({"ok": True, **out}, indent=2))


if __name__ == "__main__":
    main()
