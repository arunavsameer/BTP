"""Checkpoint save/load for the 1s-ahead trunk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def extract_teacher_state(model) -> dict[str, Any]:
    blob = {
        "trunk": model.trunk.state_dict(),
        "lora_enabled": False,
    }
    if hasattr(model, "close_head"):
        blob["close_head"] = model.close_head.state_dict()
    return blob


def load_teacher_state(model, blob: dict[str, Any], *, strict: bool = True) -> None:
    model.trunk.load_state_dict(blob["trunk"], strict=strict)
    if hasattr(model, "close_head") and "close_head" in blob:
        model.close_head.load_state_dict(blob["close_head"], strict=strict)
