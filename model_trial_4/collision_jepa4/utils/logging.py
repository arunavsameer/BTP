"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logger(
    name: str,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    *,
    also: list[str] | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_dir is not None:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path / f"{name.replace('.', '_')}.log")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.propagate = False
    for extra in also or []:
        pkg = logging.getLogger(extra)
        pkg.setLevel(level)
        if not pkg.handlers:
            for handler in logger.handlers:
                pkg.addHandler(handler)
        pkg.propagate = False
    return logger


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        payload = {"utc": datetime.now(timezone.utc).isoformat(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)
