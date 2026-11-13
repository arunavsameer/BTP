"""RGB + parent-target cache accessors for LatentMatch V3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
RAID = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3")
RGB_ROOT = TEMP / "shared_rgb_cache_v1"
RGB_ROOT_V2 = RAID / "shared_rgb_cache_v2"
PARENT_ROOT = TEMP / "parent_target_cache_v1"
PARENT_ROOT_V2 = RAID / "parent_target_cache_v2"
OFFSETS_5 = [-90, -66, -42, -12, 0]
OFFSETS_16 = [
    -90, -84, -78, -72, -66, -60, -54, -48,
    -42, -36, -30, -24, -18, -12, -6, 0,
]
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
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else RGB_ROOT
        self._idx_maps: dict[str, dict[int, int]] = {}
        self._mem: dict[str, np.memmap] = {}
        self._n: dict[str, int] = {}

    def _clip_dir(self, clip_id: str) -> Path:
        return self.root / clip_id

    def has_clip(self, clip_id: str) -> bool:
        return (self._clip_dir(clip_id) / "frames.npy").is_file()

    def _load_clip(self, clip_id: str) -> None:
        if clip_id in self._mem:
            return
        d = self._clip_dir(clip_id)
        npy = d / "frames.npy"
        n = npy.stat().st_size // (160 * 160 * 3)
        self._n[clip_id] = n
        self._mem[clip_id] = np.memmap(npy, dtype=np.uint8, mode="r", shape=(n, 160, 160, 3))
        index_path = d / "index.json"
        if index_path.is_file():
            idx = [int(a) for a in json.loads(index_path.read_text(encoding="utf-8"))]
            if len(idx) == n:
                self._idx_maps[clip_id] = {f: i for i, f in enumerate(idx)}
                return
            needed: set[int] = set()
            for a_i in idx:
                needed.update(a_i + o for o in MAX_OFFSETS if a_i + o >= 0)
            sorted_f = sorted(needed)
            if len(sorted_f) == n:
                self._idx_maps[clip_id] = {f: i for i, f in enumerate(sorted_f)}
            else:
                self._idx_maps[clip_id] = {f: i for i, f in enumerate(sorted_f) if i < n}
        else:
            self._idx_maps[clip_id] = {i: i for i in range(n)}

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
            out.append(np.asarray(mem[pos], dtype=np.uint8).copy())
        return np.stack(out, axis=0)


class ParentTargetCache:
    def __init__(self, root: Path | None = None) -> None:
        root = Path(root) if root is not None else PARENT_ROOT
        mapping = json.loads((root / "mapping.json").read_text(encoding="utf-8"))
        self._map = {k: int(v) for k, v in mapping.items()}
        n = int(json.loads((root / "metadata.json").read_text())["num_training_anchors"])
        self.F = np.memmap(root / "F_dec.npy", dtype=np.float16, mode="r", shape=(n, 128, 16, 16))
        self.H = np.memmap(root / "H_teacher.npy", dtype=np.float16, mode="r", shape=(n, 5, 5))

    def get_parent_targets(self, clip_id: str, anchor_frame: int) -> dict | None:
        key = f"{clip_id}::{int(anchor_frame)}"
        i = self._map.get(key)
        if i is None:
            return None
        return {
            "F_dec": np.asarray(self.F[i], dtype=np.float32),
            "H_teacher": np.asarray(self.H[i], dtype=np.float32),
        }
