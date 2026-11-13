#!/usr/bin/env python3
"""Validate all six overnight models before the sequential chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.gpu_bind import bind_cuda  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument("--device", default="auto")
_parser.add_argument("--batch", type=int, default=2)
_args = _parser.parse_args()
bind_cuda(str(_args.device))

from collision_jepa3.student_research.overnight6.train import pick_cuda, run_smoke_all  # noqa: E402


def main() -> None:
    device = pick_cuda(str(_args.device))
    out = run_smoke_all(device, batch=_args.batch)
    dest = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/overnight6")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "smoke.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    failed = [k for k, v in out.items() if not v.get("ok")]
    if failed:
        raise SystemExit(f"smoke failed models {failed}")
    print("ALL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
