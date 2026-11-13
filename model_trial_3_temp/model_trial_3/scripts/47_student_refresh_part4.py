#!/usr/bin/env python3
"""Regenerate part4 with 700K-budget Large and enforced param counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.cnn import MAX_TOTAL_PARAMS, assert_param_budget, preset_architectures, StudentRGBNet  # noqa: E402
from collision_jepa3.student_research.export_quant import sweep_architectures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    args = parser.parse_args()
    out = Path(args.out)
    rows = sweep_architectures(preset_architectures(), input_size=160)
    for res in (128,):
        rows += sweep_architectures(preset_architectures()[:3], input_size=128)
    checks = []
    for p in preset_architectures():
        for nf in (3, 5):
            m = StudentRGBNet(p, n_frames=nf, input_size=160)
            n = assert_param_budget(m, MAX_TOTAL_PARAMS)
            checks.append({"name": p.name, "n_frames": nf, "total_params": n, "budget": p.budget})
    report = {
        "max_total_params": MAX_TOTAL_PARAMS,
        "int8_target_bytes": 1_000_000,
        "budget_verification": checks,
        "note": "large_concat width=56 depth=5; large_shared width=52 depth=5 (replaces 2.26M design)",
        "models": rows,
    }
    (out / "part4_architecture_feasibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out / 'part4_architecture_feasibility.json'}", flush=True)


if __name__ == "__main__":
    main()
