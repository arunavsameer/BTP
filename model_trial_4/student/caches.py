"""Read-only mmap accessors for the 256 RGB cache and parent F_dec cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from collision_jepa4.data.heatmap_annotations import load_heatmaps

from .paths import OFFSETS_16

SIZE = 256


class RGBCache256:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._mem: dict[str, np.ndarray] = {}
        self._idx: dict[str, dict[int, int]] = {}

    def _clip_dir(self, clip_id: str) -> Path:
        return self.root / clip_id

    def _load(self, clip_id: str) -> None:
        if clip_id in self._mem:
            return
        d = self._clip_dir(clip_id)
        npy = d / "frames.npy"
        fi = d / "frame_indices.npy"
        if not npy.is_file() or not fi.is_file():
            raise FileNotFoundError(f"uncached clip {clip_id} under {d}")
        indices = np.load(fi, mmap_mode="r")
        n = int(len(indices))
        with npy.open("rb") as f:
            magic = f.read(6)
        if magic.startswith(b"\x93NUMPY"):
            frames = np.load(npy, mmap_mode="r")
        else:
            expected = n * SIZE * SIZE * 3
            if npy.stat().st_size != expected:
                raise RuntimeError(f"raw frames size {npy.stat().st_size} != {expected} for {clip_id}")
            frames = np.memmap(npy, dtype=np.uint8, mode="r", shape=(n, SIZE, SIZE, 3))
        if frames.ndim != 4 or frames.shape[1:] != (SIZE, SIZE, 3) or frames.dtype != np.uint8:
            raise RuntimeError(f"bad frames.npy {clip_id} {frames.shape} {frames.dtype}")
        self._mem[clip_id] = frames
        self._idx[clip_id] = {int(f): i for i, f in enumerate(indices.tolist())}

    def get_clip01(self, clip_id: str, anchor_frame: int, offsets: list[int] | None = None) -> np.ndarray:
        offs = list(OFFSETS_16 if offsets is None else offsets)
        self._load(clip_id)
        imap, mem = self._idx[clip_id], self._mem[clip_id]
        out = []
        for o in offs:
            fi = max(0, int(anchor_frame) + int(o))
            pos = imap.get(fi)
            if pos is None:
                raise KeyError(f"uncached frame clip={clip_id} frame={fi}")
            out.append(np.asarray(mem[pos]))
        uint8 = np.stack(out, axis=0)
        return np.transpose(uint8.astype(np.float32), (0, 3, 1, 2)) * (1.0 / 255.0)


class ParentTargetCache:
    def __init__(self, root: str | Path):
        root = Path(root)
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


class SpatialHeatmaps:
    """Lazy per-clip 5×5 maps so G appear-weights can see h_now."""

    def __init__(self, expected_frames: int = 450, grid: int = 5):
        self.expected_frames = int(expected_frames)
        self.grid = int(grid)
        self._maps: dict[str, np.ndarray] = {}

    def _load(self, clip_id: str, video_path: str) -> np.ndarray:
        if clip_id in self._maps:
            return self._maps[clip_id]
        path = Path(video_path).parent / "spatial_annotations" / "spatial_annotations.json"
        if not path.is_file():
            raise FileNotFoundError(f"heatmap missing for {clip_id}: {path}")
        maps = load_heatmaps(path, expected_frames=self.expected_frames, grid=self.grid)
        self._maps[clip_id] = maps
        return maps

    def get_now(self, clip_id: str, video_path: str, frame: int) -> np.ndarray:
        maps = self._load(clip_id, video_path)
        t = min(max(int(frame), 0), int(maps.shape[0]) - 1)
        return np.asarray(maps[t], dtype=np.float32)
