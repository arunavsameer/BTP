"""Config loading helpers.

A tiny wrapper around a YAML file that also lets you address nested keys with
dotted paths (e.g. ``cfg.get("train.batch_size")``) and exposes attribute-style
access for convenience.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def resolve_device(pref: str) -> str:
    """Resolve a device preference string to a concrete torch device string."""
    if pref and pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
