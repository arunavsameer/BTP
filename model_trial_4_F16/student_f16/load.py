"""Build CollisionStudentV2 and load trial-4 epoch-22 backbone+decoder blobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .constants import ARCH
from .decoder import HeatmapDecoder
from .model import CollisionStudentV2


def build_student() -> CollisionStudentV2:
    decoder = HeatmapDecoder(z_channels=128, hidden=48, dilated=True, out_grid=5, pool="max")
    model = CollisionStudentV2(decoder)
    model.eval()
    return model


def load_student(ckpt_path: str | Path, *, map_location="cpu") -> tuple[CollisionStudentV2, dict[str, Any]]:
    path = Path(ckpt_path)
    blob = torch.load(path, map_location=map_location, weights_only=False)
    model = build_student()
    if "backbone" in blob and "decoder" in blob:
        model.backbone.load_state_dict(blob["backbone"], strict=True)
        model.decoder.load_state_dict(blob["decoder"], strict=True)
    elif "state_dict" in blob:
        model.load_state_dict(blob["state_dict"], strict=True)
    else:
        raise KeyError(f"unrecognized checkpoint keys: {list(blob)[:20]}")
    model.eval()
    meta = {k: blob[k] for k in blob if k not in ("backbone", "decoder", "optimizer", "scheduler", "state_dict")}
    meta.setdefault("arch", blob.get("arch", ARCH))
    return model, meta
