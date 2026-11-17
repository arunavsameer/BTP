"""Build CollisionStudentV2; load fp32 blobs or packed int8 weight dumps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .constants import ARCH
from .decoder import HeatmapDecoder
from .model import CollisionStudentV2
from .quant import dequant_state, is_quant_blob


def build_student() -> CollisionStudentV2:
    decoder = HeatmapDecoder(z_channels=128, hidden=48, dilated=True, out_grid=5, pool="max")
    model = CollisionStudentV2(decoder)
    model.eval()
    return model


def load_student(
    ckpt_path: str | Path,
    *,
    map_location="cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[CollisionStudentV2, dict[str, Any]]:
    path = Path(ckpt_path)
    blob = torch.load(path, map_location=map_location, weights_only=False)
    model = build_student()
    if is_quant_blob(blob):
        bb = dequant_state(blob["backbone_q"], dtype=compute_dtype)
        dec = dequant_state(blob["decoder_q"], dtype=compute_dtype)
        model.backbone.load_state_dict(bb, strict=True)
        model.decoder.load_state_dict(dec, strict=True)
    elif "backbone" in blob and "decoder" in blob:
        model.backbone.load_state_dict(blob["backbone"], strict=True)
        model.decoder.load_state_dict(blob["decoder"], strict=True)
    else:
        raise KeyError(f"unrecognized checkpoint keys: {list(blob)[:20]}")
    if compute_dtype == torch.float16:
        model.half()
    else:
        model.float()
    model.eval()
    skip = {"backbone", "decoder", "backbone_q", "decoder_q", "optimizer", "scheduler", "state_dict"}
    meta = {k: blob[k] for k in blob if k not in skip}
    meta.setdefault("arch", blob.get("arch", ARCH))
    return model, meta
