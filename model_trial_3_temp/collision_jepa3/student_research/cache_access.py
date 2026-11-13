"""RGB + parent-target cache accessors for LatentMatch V3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
RGB_ROOT = TEMP / "shared_rgb_cache_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v1"
OFFSETS_5 = [-90, -66, -42, -12, 0]
MAX_OFFSETS = sorted(
    {
        *[-42, -21, 0],
        *[-90, -66, -42, -12, 0],
        *[-126, -102, -78, -54, -30, -6, -2, 0],
        *list(range(-240, 1, 16)),
    }
)


def _clip_dir(clip_id: str) -> Path:
    return RGB_ROOT / clip_id


class RGBCache:
    def __init__(self) -> None:
        self._idx_maps: dict[str, dict[int, int]] = {}
        self._mem: dict[str, np.memmap] = {}
        self._n: dict[str, int] = {}

    def has_clip(self, clip_id: str) -> bool:
        return (_clip_dir(clip_id) / "frames.npy").is_file()

    def _load_clip(self, clip_id: str) -> None:
        if clip_id in self._mem:
            return
        d = _clip_dir(clip_id)
        npy = d / "frames.npy"
        nbytes = npy.stat().st_size
        n = nbytes // (160 * 160 * 3)
        self._n[clip_id] = n
        self._mem[clip_id] = np.memmap(npy, dtype=np.uint8, mode="r", shape=(n, 160, 160, 3))
        index_path = d / "index.json"
        needed: set[int] = set()
        if index_path.is_file():
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            for a in idx:
                a_i = int(a)
                needed.update(a_i + o for o in MAX_OFFSETS if a_i + o >= 0)
        sorted_f = sorted(needed)
        if len(sorted_f) != n:
            # fall back: sequential identity if reconstruction length mismatches
            self._idx_maps[clip_id] = {i: i for i in range(n)}
            self._idx_maps[clip_id].update({f: i for i, f in enumerate(sorted_f) if i < n})
        else:
            self._idx_maps[clip_id] = {f: i for i, f in enumerate(sorted_f)}

    def get_frames(self, clip_id: str, anchor_frame: int, offsets: list[int] | None = None) -> np.ndarray | None:
        if not self.has_clip(clip_id):
            return None
        self._load_clip(clip_id)
        offs = list(offsets or OFFSETS_5)
        imap = self._idx_maps[clip_id]
        mem = self._mem[clip_id]
        out = []
        for o in offs:
            fi = max(0, int(anchor_frame) + int(o))
            pos = imap.get(fi)
            if pos is None or pos >= mem.shape[0]:
                return None
            out.append(np.asarray(mem[pos]))
        return np.stack(out, axis=0)


class ParentTargetCache:
    def __init__(self) -> None:
        mapping = json.loads((PARENT_ROOT / "mapping.json").read_text(encoding="utf-8"))
        self._map = {k: int(v) for k, v in mapping.items()}
        n = int(json.loads((PARENT_ROOT / "metadata.json").read_text())["num_training_anchors"])
        self.F = np.memmap(PARENT_ROOT / "F_dec.npy", dtype=np.float16, mode="r", shape=(n, 128, 16, 16))
        self.H = np.memmap(PARENT_ROOT / "H_teacher.npy", dtype=np.float16, mode="r", shape=(n, 5, 5))

    def get_parent_targets(self, clip_id: str, anchor_frame: int) -> dict | None:
        key = f"{clip_id}::{int(anchor_frame)}"
        i = self._map.get(key)
        if i is None:
            return None
        return {
            "F_dec": np.asarray(self.F[i], dtype=np.float32),
            "H_teacher": np.asarray(self.H[i], dtype=np.float32),
        }
