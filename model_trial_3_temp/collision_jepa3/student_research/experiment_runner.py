"""Resumable student experiments (Part 14): config, per-epoch metrics, completion markers."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch


def experiment_dir(report_root: Path, exp_id: str) -> Path:
    d = report_root / "experiments" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_complete(exp_d: Path) -> bool:
    return (exp_d / ".success").is_file()


def write_config(exp_d: Path, config: dict[str, Any]) -> None:
    (exp_d / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    sys.stdout.flush()


def write_epoch_metrics(exp_d: Path, epoch: int, payload: dict[str, Any]) -> None:
    p = exp_d / f"metrics_epoch_{epoch:02d}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.stdout.flush()


def mark_success(exp_d: Path, summary: dict[str, Any]) -> None:
    (exp_d / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (exp_d / ".success").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    sys.stdout.flush()


def mark_failure(exp_d: Path, exc: BaseException) -> None:
    (exp_d / ".failed").write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
    sys.stdout.flush()


def gpu_mem_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }


def run_wrapped(
    report_root: Path,
    exp_id: str,
    config: dict[str, Any],
    fn: Callable[[Path, dict[str, Any]], dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    exp_d = experiment_dir(report_root, exp_id)
    if is_complete(exp_d) and not force:
        return json.loads((exp_d / "result.json").read_text(encoding="utf-8"))
    write_config(exp_d, config)
    try:
        summary = fn(exp_d, config)
        summary = {**summary, "gpu_mem_end": gpu_mem_mb(), "exp_id": exp_id}
        mark_success(exp_d, summary)
        return summary
    except Exception as exc:
        mark_failure(exp_d, exc)
        raise
