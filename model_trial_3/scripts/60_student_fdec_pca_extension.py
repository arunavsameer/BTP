#!/usr/bin/env python3
"""F_dec cache PCA, probes, and parent-decoder interventions (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.fdec_pca_extension import run_fdec_extension  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-reports", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--out", default=str(ROOT / "reports/student_final_design_diagnostics"))
    parser.add_argument("--device", default="cuda")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    device = resolve_device(args.device)
    summary = run_fdec_extension(cfg, Path(args.out), Path(args.student_reports), device=str(device))
    print(json.dumps({"ok": True, "out": args.out, "elapsed_sec": summary.get("elapsed_sec")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
