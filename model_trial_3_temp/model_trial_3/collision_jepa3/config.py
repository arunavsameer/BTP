"""YAML configuration with dotted-key access and path resolution."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

_TRIAL3_RELATIVE_PATH_KEYS = {
    "paths.manifest",
    "paths.split_file",
    "paths.cache_root",
    "paths.checkpoint_root",
    "paths.logs_root",
    "paths.reports_root",
    "paths.visualizations_root",
    "paths.hf_cache_fallback",
}


class Config:
    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(data, source=path)
        if overrides:
            for key, value in overrides.items():
                cfg.set(key, value)
        cfg.validate()
        return cfg

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def path(self, dotted_key: str) -> Path:
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(dotted_key)
        p = Path(str(value)).expanduser()
        if p.is_absolute():
            return p
        if dotted_key == "paths.dataset_root":
            return (WORKSPACE_ROOT / p).resolve()
        if dotted_key in _TRIAL3_RELATIVE_PATH_KEYS or dotted_key.startswith("paths."):
            return (PACKAGE_ROOT / p).resolve()
        return (PACKAGE_ROOT / p).resolve()

    def dataset_root(self) -> Path:
        return self.path("paths.dataset_root")

    def trial_root(self) -> Path:
        return PACKAGE_ROOT

    def validate(self) -> None:
        offsets = list(self.get("sampling.frame_offsets", []))
        n = int(self.get("sampling.input_frames", 0))
        spacing = int(self.get("sampling.input_spacing_frames", 0))
        if len(offsets) != n:
            raise ValueError(f"sampling.frame_offsets length {len(offsets)} != input_frames {n}")
        if offsets[-1] != 0:
            raise ValueError("sampling.frame_offsets must end at 0 (anchor frame)")
        if any(o > 0 for o in offsets):
            raise ValueError("sampling.frame_offsets must not include future frames")
        expected = list(range(-(n - 1) * spacing, 1, spacing))
        if offsets != expected:
            raise ValueError(
                f"sampling.frame_offsets {offsets} do not match uniform spacing "
                f"(expected {expected})"
            )
        horizons = list(self.get("sampling.horizons_frames", []))
        names = list(self.get("sampling.horizon_names", []))
        if len(horizons) != 3 or len(names) != 3:
            raise ValueError("exactly three future horizons are required")
        if self.get("data.never_use_overlay_as_input") is not True:
            raise ValueError("data.never_use_overlay_as_input must remain true")
        if int(self.get("adapter.z_channels")) != 32:
            raise ValueError("adapter.z_channels must be 32 (student interface)")
        if int(self.get("adapter.feature_grid")) != 5:
            raise ValueError("adapter.feature_grid must be 5")
        decode_from = str(self.get("adapter.decode_from", "fine"))
        if decode_from not in {"fine", "z"}:
            raise ValueError(f"unsupported adapter.decode_from {decode_from}")
        heatmap_pool = str(self.get("adapter.heatmap_pool", "max"))
        if heatmap_pool not in {"max", "attn", "avg"}:
            raise ValueError(f"unsupported adapter.heatmap_pool {heatmap_pool}")
        heatmap_grid = int(self.get("adapter.heatmap_grid", 16))
        if heatmap_grid < 0 or (heatmap_grid not in {0} and heatmap_grid < 5):
            raise ValueError("adapter.heatmap_grid must be 0 (native) or >= 5")
        pool = str(self.get("adapter.temporal_pool", "softmax"))
        if pool not in {"softmax", "mean_last"}:
            raise ValueError(f"unsupported adapter.temporal_pool {pool}")
        dilations = self.get("adapter.temporal_dilations", None)
        if dilations is not None:
            n_blocks = int(self.get("adapter.temporal_blocks", 4))
            if len(list(dilations)) != n_blocks:
                raise ValueError("adapter.temporal_dilations length must match temporal_blocks")
        if int(self.get("data.target_grid")) != 5:
            raise ValueError("data.target_grid must be 5")
        hw = self.get("loss.horizon_weights", {})
        for name in names:
            if name not in hw:
                raise ValueError(f"loss.horizon_weights missing {name}")
        mode = str(self.get("preprocessing.resize_mode"))
        if mode not in {"letterbox", "checkpoint_center_crop"}:
            raise ValueError(f"unsupported resize_mode {mode}")
        if bool(self.get("preprocessing.random_horizontal_flip")):
            raise ValueError("random_horizontal_flip is disabled for the teacher baseline")
        if bool(self.get("preprocessing.random_geometric_augmentation")):
            raise ValueError("random_geometric_augmentation is disabled for the teacher baseline")
        policy = str(self.get("sampling.ambiguous_policy", "mean_target"))
        if policy not in {"mean_target", "collision_target", "mask_ambiguous_pairs", "none"}:
            raise ValueError(f"unsupported sampling.ambiguous_policy {policy}")
        anchor_mode = str(self.get("sampling.anchor_mode", "stride"))
        if anchor_mode not in {"stride", "group_random"}:
            raise ValueError(f"unsupported sampling.anchor_mode {anchor_mode}")
        if anchor_mode == "group_random":
            n_pre = int(self.get("sampling.n_pre_fork_anchors", 0))
            n_post = int(self.get("sampling.n_post_fork_anchors", 0))
            if n_pre + n_post < 1:
                raise ValueError("group_random needs n_pre_fork_anchors + n_post_fork_anchors >= 1")
            t_max = int(self.get("sampling.random_anchor_t_max", 419))
            n_frames = int(self.get("data.expected_frames", 450))
            hs = list(self.get("sampling.horizons_frames", [30, 45, 60]))
            hard_max = n_frames - 1 - min(int(x) for x in hs)
            if t_max > hard_max:
                raise ValueError(
                    f"sampling.random_anchor_t_max {t_max} > last 1s-valid anchor {hard_max}"
                )
        mode = str(self.get("heads.mode", "residual"))
        if mode not in {"residual", "direct_h1s"}:
            raise ValueError(f"unsupported heads.mode {mode}")


def heads_mode(cfg: "Config" | Any) -> str:
    return str(cfg.get("heads.mode", "residual"))


def supervised_horizon_names(cfg: "Config" | Any) -> list[str]:
    if heads_mode(cfg) == "direct_h1s":
        return ["1s"]
    return list(cfg.get("sampling.horizon_names", ["1s", "1_5s", "2s"]))


def dump_resolved(cfg: Config) -> dict[str, Any]:
    data = cfg.to_dict()
    resolved = {
        "dataset_root": str(cfg.dataset_root()),
        "manifest": str(cfg.path("paths.manifest")),
        "split_file": str(cfg.path("paths.split_file")),
        "cache_root": str(cfg.path("paths.cache_root")),
        "checkpoint_root": str(cfg.path("paths.checkpoint_root")),
        "hf_cache_dir": str(cfg.path("paths.hf_cache_dir")) if cfg.get("paths.hf_cache_dir") else None,
        "trial_root": str(PACKAGE_ROOT),
        "workspace_root": str(WORKSPACE_ROOT),
        "config_fingerprint": cfg.fingerprint(),
    }
    data["_resolved"] = resolved
    return data
