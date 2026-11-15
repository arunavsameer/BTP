#!/usr/bin/env python3
"""SLURM/CLI entry for CollisionStudentV6. Implementation lives in student/v6/train.py."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(ROOT / "student" / "v6" / "train.py"), run_name="__main__")
