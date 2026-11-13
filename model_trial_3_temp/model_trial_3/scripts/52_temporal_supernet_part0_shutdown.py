#!/usr/bin/env python3
"""Part 0: graceful shutdown of cancelled F1–F5 Part 3 matrix."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/student_architecture_tests"


def _find_targets() -> list[dict]:
    out = []
    try:
        proc = subprocess.run(
            ["pgrep", "-af", "48_student_run_part3_matrix|46_student_run_experiment"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in (proc.stdout or "").strip().splitlines():
            if not line.strip():
                continue
            pid_s, cmd = line.split(" ", 1)
            if "52_temporal_supernet" in cmd:
                continue
            out.append({"pid": int(pid_s), "cmd": cmd.strip()})
    except Exception as exc:
        out.append({"error": str(exc)})
    return out


def _children(pid: int) -> list[int]:
    try:
        r = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False)
        return [int(x) for x in r.stdout.split() if x.strip()]
    except Exception:
        return []


def main() -> None:
    targets = _find_targets()
    killed = []
    all_pids = []
    for t in targets:
        if "pid" not in t:
            continue
        pid = t["pid"]
        all_pids.extend([pid, *_children(pid)])
    all_pids = sorted(set(all_pids))
    for pid in all_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append({"pid": pid, "signal": "SIGTERM"})
        except ProcessLookupError:
            killed.append({"pid": pid, "signal": "not_running"})
        except PermissionError as exc:
            killed.append({"pid": pid, "signal": "permission_denied", "error": str(exc)})
    time.sleep(5)
    for pid in all_pids:
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            killed.append({"pid": pid, "signal": "SIGKILL"})
        except (ProcessLookupError, PermissionError):
            pass
    failed = OUT / "experiments" / "part3_F1" / ".failed"
    archived = None
    if failed.is_file():
        txt = failed.read_text(encoding="utf-8")
        if "KeyError: 'lr'" in txt:
            dest = failed.with_suffix(".failed.archived_lr_keyerror")
            failed.rename(dest)
            archived = str(dest)
    report = {
        "cancelled_plan": "part3_independent_F1_F5",
        "targets_at_scan": targets,
        "pids_signaled": killed,
        "preserved": [
            "teacher_cache_train_pilot",
            "teacher_cache",
            "pilot_split_train_groups.json",
            "experiments/part3_F1/config.json",
            "part3_matrix.log",
        ],
        "archived_failed_marker": archived,
        "remaining_after_shutdown": _find_targets(),
    }
    path = OUT / "part0_previous_run_shutdown.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
