#!/usr/bin/env python3
"""Canny student street vlog — delegates to model_trial_5/scripts/91_render_student_street_vlog.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "runs" / "student_v2_canny_hinge" / "checkpoint_best.pt"
RENDER = Path("/storage/BTP/model_trial_5/scripts/91_render_student_street_vlog.py")
PYTHON = Path("/storage/BTP/impl/venv/bin/python")


def main() -> None:
    cmd = [
        str(PYTHON),
        str(RENDER),
        "--student",
        "canny",
        "--checkpoint",
        str(DEFAULT_CKPT),
        "--device",
        "cuda",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()