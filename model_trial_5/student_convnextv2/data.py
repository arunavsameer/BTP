"""RGB 256 cache + parent F_dec cache + optional h_now for appear weights."""

from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from collision_jepa3.data.heatmap_annotations import load_heatmaps
from collision_jepa3.data.preprocessing import uint8_to_nchw01

from .paths import OFFSETS_16, LOCAL_DATASET_ROOTS, resolve_clip_dir


def _clip_frames_dir(root: Path, clip_id: str) -> Path:
    slug = str(clip_id).replace("/", "__")
    if root.name == "frames":
        return root / slug
    return root / "frames" / slug


class RGBCache256:
    """Shared 256 cache (loader.py) or Trial-3 letterboxed frames_uint8.npy."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        loader = self.root / "loader.py"
        self.inner = None
        if loader.is_file():
            mod = SourceFileLoader("rgb256_t5", str(loader)).load_module()
            self.inner = mod.RGBCache256(self.root)
        self._mem: dict[str, np.ndarray] = {}

    def _load_frames(self, clip_id: str) -> np.ndarray:
        if clip_id in self._mem:
            return self._mem[clip_id]
        d = _clip_frames_dir(self.root, clip_id)
        npy = d / "frames_uint8.npy"
        if not npy.is_file():
            raise FileNotFoundError(f"RGB frames missing for {clip_id}: {npy}")
        arr = np.load(npy, mmap_mode="r")
        self._mem[clip_id] = arr
        return arr

    def get_clip01(self, clip_id: str, frame: int) -> np.ndarray:
        if self.inner is not None:
            return self.inner.get_frames(clip_id, frame, OFFSETS_16, input_size=256, normalize=False)
        frames = self._load_frames(clip_id)
        n = int(frames.shape[0])
        idxs = [min(max(int(frame) + int(o), 0), n - 1) for o in OFFSETS_16]
        clip = np.stack([np.asarray(frames[i]) for i in idxs], axis=0)
        if clip.dtype != np.uint8:
            clip = clip.astype(np.uint8)
        if clip.ndim != 4 or clip.shape[1:] != (256, 256, 3):
            raise ValueError(f"expected [16,256,256,3] uint8, got {clip.shape} {clip.dtype} for {clip_id}")
        return uint8_to_nchw01(clip)


class ParentTargetCache:
    def __init__(self, root: str | Path):
        root = Path(root)
        mapping = json.loads((root / "mapping.json").read_text(encoding="utf-8"))
        self._map = {k: int(v) for k, v in mapping.items()}
        n = int(json.loads((root / "metadata.json").read_text())["num_training_anchors"])
        self.n = n
        self.F = np.memmap(root / "F_dec.npy", dtype=np.float16, mode="r", shape=(n, 128, 16, 16))
        self.H = np.memmap(root / "H_teacher.npy", dtype=np.float16, mode="r", shape=(n, 5, 5))

    def index_of(self, clip_id: str, anchor_frame: int) -> int | None:
        key = f"{clip_id}::{int(anchor_frame)}"
        return self._map.get(key)

    def get_parent_targets(self, clip_id: str, anchor_frame: int) -> dict | None:
        i = self.index_of(clip_id, anchor_frame)
        if i is None:
            return None
        return {
            "F_dec": np.asarray(self.F[i], dtype=np.float32),
            "H_teacher": np.asarray(self.H[i], dtype=np.float32),
            "index": i,
        }


def _spatial_json(clip_id: str, video_path: str) -> Path:
    cands = []
    if video_path:
        cands.append(Path(video_path).parent / "spatial_annotations" / "spatial_annotations.json")
    clip_dir = resolve_clip_dir(clip_id)
    if clip_dir is not None:
        cands.append(clip_dir / "spatial_annotations" / "spatial_annotations.json")
    for root in LOCAL_DATASET_ROOTS:
        cands.append(root / str(clip_id) / "spatial_annotations" / "spatial_annotations.json")
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError(f"heatmap missing for {clip_id}: tried {cands[:4]}")


class SpatialHeatmaps:
    def __init__(self, expected_frames: int = 450, grid: int = 5):
        self.expected_frames = int(expected_frames)
        self.grid = int(grid)
        self._maps: dict[str, np.ndarray] = {}

    def _load(self, clip_id: str, video_path: str) -> np.ndarray:
        if clip_id in self._maps:
            return self._maps[clip_id]
        path = _spatial_json(clip_id, video_path)
        maps = load_heatmaps(path, expected_frames=self.expected_frames, grid=self.grid)
        self._maps[clip_id] = maps
        return maps

    def get_now(self, clip_id: str, video_path: str, frame: int) -> np.ndarray:
        maps = self._load(clip_id, video_path)
        t = min(max(int(frame), 0), int(maps.shape[0]) - 1)
        return np.asarray(maps[t], dtype=np.float32)


class WinDS(Dataset):
    def __init__(self, rows, ids, gt, rgb, parent, heatmaps: SpatialHeatmaps | None = None):
        self.rows = rows
        self.ids = list(ids)
        self.gt = gt
        self.rgb = rgb
        self.parent = parent
        self.heatmaps = heatmaps

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rid = int(self.ids[i])
        row = self.rows[rid]
        cid, t = row["clip_id"], int(row["frame"])
        clip = self.rgb.get_clip01(cid, t)
        pt = self.parent.get_parent_targets(cid, t)
        if pt is None:
            raise KeyError(f"missing parent {cid} {t}")
        video_path = str(row.get("video_path", ""))
        if self.heatmaps is not None:
            h_now = self.heatmaps.get_now(cid, video_path, t)
        else:
            h_now = np.zeros((5, 5), dtype=np.float32)
        return {
            "clip": torch.from_numpy(np.ascontiguousarray(clip)),
            "F_teacher": torch.from_numpy(pt["F_dec"]),
            "H_teacher": torch.from_numpy(pt["H_teacher"]),
            "gt_1s": torch.from_numpy(np.asarray(self.gt[rid], dtype=np.float32)),
            "h_now": torch.from_numpy(np.ascontiguousarray(h_now)),
            "pre_fork": bool(row.get("pre_fork")),
            "amb_1s": bool(row.get("amb_1s")),
            "sample_id": str(row.get("sample_id", "")),
            "clip_id": cid,
            "frame": t,
            "background_id": str(row.get("background_id", "")),
            "outcome": str(row.get("outcome", "")),
            "video_path": video_path,
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "F_teacher": torch.stack([b["F_teacher"] for b in batch]),
        "H_teacher": torch.stack([b["H_teacher"] for b in batch]),
        "gt_1s": torch.stack([b["gt_1s"] for b in batch]),
        "h_now": torch.stack([b["h_now"] for b in batch]),
        "pre_fork": torch.tensor([b["pre_fork"] for b in batch], dtype=torch.bool),
        "amb_1s": torch.tensor([b["amb_1s"] for b in batch], dtype=torch.bool),
        "sample_id": [b["sample_id"] for b in batch],
        "clip_id": [b["clip_id"] for b in batch],
        "frame": torch.tensor([b["frame"] for b in batch], dtype=torch.int64),
        "background_id": [b["background_id"] for b in batch],
        "outcome": [b["outcome"] for b in batch],
        "video_path": [b["video_path"] for b in batch],
    }


def compute_fdec_stats(parent: ParentTargetCache, rows, ids, out_path: Path) -> tuple[np.ndarray, np.ndarray]:
    idxs = []
    for rid in ids:
        row = rows[int(rid)]
        i = parent.index_of(row["clip_id"], int(row["frame"]))
        if i is not None:
            idxs.append(int(i))
    idxs = np.asarray(idxs, dtype=np.int64)
    acc = np.zeros(128, dtype=np.float64)
    acc2 = np.zeros(128, dtype=np.float64)
    n = 0
    chunk = 256
    for s in range(0, len(idxs), chunk):
        sl = idxs[s : s + chunk]
        x = np.asarray(parent.F[sl], dtype=np.float32)
        acc += x.mean(axis=(0, 2, 3)) * x.shape[0]
        acc2 += (x.astype(np.float64) ** 2).mean(axis=(0, 2, 3)) * x.shape[0]
        n += int(x.shape[0])
    mu = (acc / max(n, 1)).astype(np.float32)
    second = acc2 / max(n, 1)
    var = np.maximum(second - mu.astype(np.float64) ** 2, 1e-8)
    sigma = np.sqrt(var).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mu=mu.reshape(128, 1, 1), sigma=sigma.reshape(128, 1, 1), n=n)
    return mu.reshape(128, 1, 1), sigma.reshape(128, 1, 1)
