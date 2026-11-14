"""Load the frozen Pair-CF heatmap decoder (read-only Trial-3 checkpoint)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from collision_jepa3.models.heatmap_decoder import HeatmapDecoder

from .paths import EXPECTED_DECODER_HASH, parent_ckpt


def decoder_state_hash(dec: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(dec.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_parent_decoder(ckpt: Path | None = None, *, expected_hash: str = EXPECTED_DECODER_HASH) -> HeatmapDecoder:
    path = Path(ckpt) if ckpt is not None else parent_ckpt()
    if not path.is_file():
        raise FileNotFoundError(f"Pair-CF parent checkpoint missing: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "decoder" not in blob:
        raise KeyError(f"checkpoint has no decoder key: {path} keys={list(blob)[:12]}")
    dec = HeatmapDecoder(z_channels=128, dilated=True, out_grid=5, pool="max")
    dec.load_state_dict(blob["decoder"])
    for p in dec.parameters():
        p.requires_grad_(False)
    dec.eval()
    got = decoder_state_hash(dec)
    if expected_hash and got != expected_hash:
        raise RuntimeError(f"decoder hash {got} != expected {expected_hash} from {path}")
    return dec
