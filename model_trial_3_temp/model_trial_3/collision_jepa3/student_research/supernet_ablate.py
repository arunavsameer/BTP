"""Part 7 temporal utilization ablations (inference only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from collision_jepa3.evaluation.metrics import STOP, classify_heatmap
from collision_jepa3.student_research.frames import FRAME_CANDIDATES
from collision_jepa3.student_research.supernet_metrics import eval_pattern
from collision_jepa3.student_research.temporal_supernet import MAX_FRAMES, pack_pattern_clip


def _ablate_flat(flat: torch.Tensor, pattern: str, mode: str) -> torch.Tensor:
    """flat [B, n_off, 3, H, W] in ALL_OFFSETS order; mutate copy for ablation modes."""
    from collision_jepa3.student_research.temporal_supernet import ALL_OFFSETS

    idx = {o: i for i, o in enumerate(ALL_OFFSETS)}
    offs = list(FRAME_CANDIDATES[pattern])
    b = flat.shape[0]
    out = flat.clone()
    if mode == "remove_oldest":
        if len(offs) >= 1:
            o0 = offs[0]
            out[:, idx[o0]] = 0
    elif mode == "remove_middle" and len(offs) >= 3:
        om = offs[len(offs) // 2]
        out[:, idx[om]] = 0
    elif mode == "remove_recent_hist":
        for o in offs[:-1]:
            out[:, idx[o]] = 0
    elif mode == "repeat_current":
        cur = offs[-1]
        cur_t = out[:, idx[cur] : idx[cur] + 1]
        for o in offs[:-1]:
            out[:, idx[o]] = cur_t[:, 0]
    elif mode == "permute_time" and len(offs) >= 2:
        perm = list(reversed(offs[:-1])) + [offs[-1]]
        new = out.clone()
        for i, o in enumerate(offs):
            new[:, idx[o]] = out[:, idx[perm[i]]]
        out = new
    return out


def run_temporal_ablations(model, loader, device: str, pattern: str, warning: dict) -> dict[str, Any]:
    baseline = eval_pattern(model, loader, device, pattern, warning)
    out: dict[str, Any] = {"baseline": baseline, "pattern": pattern, "ablations": {}}
    for mode in (
        "remove_oldest",
        "remove_middle",
        "remove_recent_hist",
        "repeat_current",
        "permute_time",
    ):
        # lightweight wrapper loader
        class _Abl:
            def __init__(self, base):
                self.base = base
                self.mode = mode

            def __iter__(self):
                for batch in self.base:
                    batch = dict(batch)
                    batch["flat_frames"] = _ablate_flat(batch["flat_frames"], pattern, mode)
                    yield batch

        met = eval_pattern(model, _Abl(loader), device, pattern, warning)
        out["ablations"][mode] = {
            **met,
            "delta_stop_recall": met["stop_recall"] - baseline["stop_recall"],
            "delta_safe_recall": met["safe_recall"] - baseline["safe_recall"],
        }
    return out
