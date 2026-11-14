"""Shared RGB 256 cache + Canny paint + parent F_dec. Pickle-safe workers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .canny import highlight_clip
from .paths import OFFSETS_16, TRIAL5

SIZE = 256
sys.path.insert(0, str(TRIAL5))
from collision_jepa3.data.preprocessing import uint8_to_nchw01  # noqa: E402


def rgb_clip_exists(root: str | Path, clip_id: str) -> bool:
    return (Path(root) / str(clip_id) / "frames.npy").is_file()


class RGBCache256:
    def __init__(self, root: str | Path, *, canny: bool = True):
        self.root = Path(root)
        self.canny = bool(canny)
        self._mem: dict[str, np.ndarray] = {}
        self._idx: dict[str, dict[int, int]] = {}

    def __getstate__(self):
        return {"root": self.root, "canny": self.canny}

    def __setstate__(self, state):
        self.root = Path(state["root"])
        self.canny = bool(state.get("canny", True))
        self._mem = {}
        self._idx = {}

    def has_clip(self, clip_id: str) -> bool:
        return rgb_clip_exists(self.root, clip_id)

    def _load_shared(self, clip_id: str) -> tuple[np.ndarray, dict[int, int]]:
        if clip_id in self._mem and clip_id in self._idx:
            return self._mem[clip_id], self._idx[clip_id]
        d = self.root / str(clip_id)
        npy, fi = d / "frames.npy", d / "frame_indices.npy"
        if not npy.is_file() or not fi.is_file():
            raise FileNotFoundError(f"uncached clip {clip_id} under {d}")
        indices = np.load(fi, mmap_mode="r")
        with npy.open("rb") as f:
            magic = f.read(6)
        if magic.startswith(b"\x93NUMPY"):
            frames = np.load(npy, mmap_mode="r")
        else:
            n = int(len(indices))
            frames = np.memmap(npy, dtype=np.uint8, mode="r", shape=(n, SIZE, SIZE, 3))
        imap = {int(fr): i for i, fr in enumerate(indices.tolist())}
        self._mem[clip_id] = frames
        self._idx[clip_id] = imap
        return frames, imap

    def get_clip01(self, clip_id: str, frame: int) -> np.ndarray:
        mem, imap = self._load_shared(clip_id)
        out = []
        for o in OFFSETS_16:
            fi = max(0, int(frame) + int(o))
            pos = imap.get(fi)
            if pos is None:
                raise KeyError(f"uncached frame clip={clip_id} frame={fi} (anchor={frame})")
            out.append(np.asarray(mem[pos]))
        clip = np.stack(out, axis=0)
        if clip.dtype != np.uint8:
            clip = clip.astype(np.uint8)
        if self.canny:
            clip = highlight_clip(clip)
        return uint8_to_nchw01(clip)


class ParentTargetCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._open()

    def _open(self) -> None:
        root = self.root
        self._map = {k: int(v) for k, v in json.loads((root / "mapping.json").read_text()).items()}
        n = int(json.loads((root / "metadata.json").read_text())["num_training_anchors"])
        self.n = n
        self.F = np.memmap(root / "F_dec.npy", dtype=np.float16, mode="r", shape=(n, 128, 16, 16))
        self.H = np.memmap(root / "H_teacher.npy", dtype=np.float16, mode="r", shape=(n, 5, 5))

    def __getstate__(self):
        return {"root": self.root}

    def __setstate__(self, state):
        self.root = Path(state["root"])
        self._open()

    def index_of(self, clip_id: str, anchor_frame: int) -> int | None:
        return self._map.get(f"{clip_id}::{int(anchor_frame)}")

    def get_parent_targets(self, clip_id: str, anchor_frame: int) -> dict | None:
        i = self.index_of(clip_id, anchor_frame)
        if i is None:
            return None
        return {
            "F_dec": np.asarray(self.F[i], dtype=np.float32),
            "H_teacher": np.asarray(self.H[i], dtype=np.float32),
            "index": i,
        }


class WinDS(Dataset):
    def __init__(self, rows, ids, gt, rgb, parent):
        self.rows = rows
        self.gt = gt
        self.rgb = rgb
        self.parent = parent
        self.ids = [int(i) for i in ids if rgb.has_clip(rows[int(i)]["clip_id"])]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rid = int(self.ids[i])
        row = self.rows[rid]
        cid, t = row["clip_id"], int(row["frame"])
        pt = self.parent.get_parent_targets(cid, t)
        if pt is None:
            raise KeyError(f"missing parent {cid} {t}")
        return {
            "clip": torch.from_numpy(np.ascontiguousarray(self.rgb.get_clip01(cid, t))),
            "F_teacher": torch.from_numpy(pt["F_dec"]),
            "H_teacher": torch.from_numpy(pt["H_teacher"]),
            "gt_1s": torch.from_numpy(np.asarray(self.gt[rid], dtype=np.float32)),
            "amb_1s": bool(row.get("amb_1s")),
            "pre_fork": bool(row.get("pre_fork")),
            "sample_id": str(row.get("sample_id", "")),
            "clip_id": cid,
            "frame": t,
            "background_id": str(row.get("background_id", "")),
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "F_teacher": torch.stack([b["F_teacher"] for b in batch]),
        "H_teacher": torch.stack([b["H_teacher"] for b in batch]),
        "gt_1s": torch.stack([b["gt_1s"] for b in batch]),
        "amb_1s": torch.tensor([b["amb_1s"] for b in batch], dtype=torch.bool),
        "pre_fork": torch.tensor([b["pre_fork"] for b in batch], dtype=torch.bool),
        "sample_id": [b["sample_id"] for b in batch],
        "clip_id": [b["clip_id"] for b in batch],
        "frame": torch.tensor([b["frame"] for b in batch], dtype=torch.int64),
        "background_id": [b["background_id"] for b in batch],
    }
