"""Persistent uint8 RGB cache for all supernet anchor frames."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from collision_jepa3.data.preprocessing import letterbox_rgb, signature_from_cfg, uint8_to_nchw01
from collision_jepa3.student_research.temporal_supernet import ALL_OFFSETS


def abs_frame_indices(anchor_t: int) -> list[int]:
    return [int(anchor_t) + int(o) for o in ALL_OFFSETS]


class RGBSupernetCache:
    """Per-clip packed uint8 frames indexed by absolute frame index."""

    def __init__(self, root: Path, meta: dict):
        self.root = Path(root)
        self.meta = meta
        self._mmaps: dict[str, tuple[np.memmap, dict[int, int]]] = {}

    def clip_path(self, clip_id: str) -> Path:
        return self.root / "clips" / f"{clip_id}.bin"

    def _load_clip(self, clip_id: str) -> tuple[np.memmap, dict[int, int]]:
        if clip_id in self._mmaps:
            return self._mmaps[clip_id]
        path = self.clip_path(clip_id)
        mapping = self.meta["clips"][clip_id]["frame_to_slot"]
        mapping = {int(k): int(v) for k, v in mapping.items()}
        n = len(mapping)
        size = int(self.meta["input_size"])
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, size, size, 3))
        self._mmaps[clip_id] = (mm, mapping)
        return mm, mapping

    def get_uint8(self, clip_id: str, frame_index: int) -> np.ndarray:
        mm, mapping = self._load_clip(clip_id)
        slot = mapping[int(frame_index)]
        return np.asarray(mm[slot])

    def get_anchor_stack(self, clip_id: str, anchor_t: int, *, normalize_01: bool = True) -> np.ndarray:
        """Return [len(ALL_OFFSETS), 3, H, W] float or uint8."""
        chans = []
        for fi in abs_frame_indices(anchor_t):
            u = self.get_uint8(clip_id, fi)
            if normalize_01:
                chans.append(uint8_to_nchw01(u))
            else:
                chans.append(np.transpose(u, (2, 0, 1)).astype(np.float32) / 255.0)
        return np.stack(chans, axis=0)


def benchmark_decode(rec, cfg, sig, anchor_t: int, *, n_repeat: int = 3) -> dict[str, float]:
    from collision_jepa3.data.cache import _clip_frames_uint8

    t0 = time.perf_counter()
    frames = _clip_frames_uint8(rec, cfg, sig)
    t_decode = time.perf_counter() - t0
    size = int(cfg.get("student_research.input_size", 160))
    times = []
    arrs = []
    for fi in abs_frame_indices(anchor_t):
        fi = max(0, min(fi, len(frames) - 1))
        fr = np.asarray(frames[fi])
        t1 = time.perf_counter()
        for _ in range(n_repeat):
            out = letterbox_rgb(fr, size)
        times.append((time.perf_counter() - t1) / n_repeat)
        arrs.append(out)
    return {
        "decode_clip_s": t_decode,
        "mean_letterbox_s": float(np.mean(times)),
        "n_frames": len(ALL_OFFSETS),
    }


def verify_equivalence(rec, cfg, sig, anchor_t: int, cached: np.ndarray) -> float:
    """Max abs diff vs on-the-fly letterbox path."""
    from collision_jepa3.data.cache import _clip_frames_uint8

    size = int(cfg.get("student_research.input_size", 160))
    frames = _clip_frames_uint8(rec, cfg, sig)
    live = []
    for fi in abs_frame_indices(anchor_t):
        fi = max(0, min(fi, len(frames) - 1))
        fr = letterbox_rgb(np.asarray(frames[fi]), size)
        live.append(uint8_to_nchw01(fr))
    live = np.stack(live, axis=0)
    return float(np.max(np.abs(live - cached)))


def write_clip_bin(path: Path, frames: dict[int, np.ndarray]) -> dict[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(frames.keys())
    mapping = {int(k): i for i, k in enumerate(keys)}
    if not keys:
        return mapping
    sample = frames[keys[0]]
    n, h, w, c = len(keys), sample.shape[0], sample.shape[1], sample.shape[2]
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(n, h, w, c))
    for k, i in mapping.items():
        mm[i] = frames[k]
    mm.flush()
    del mm
    return mapping


def build_rgb_cache(
    cfg,
    *,
    out_dir: Path,
    clip_ids: set[str],
    row_anchors: list[tuple[str, int]],
    records: dict,
    input_size: int = 160,
    benchmark_clips: int = 8,
) -> dict[str, Any]:
    sig = signature_from_cfg(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(exist_ok=True)

    needed: dict[str, set[int]] = {cid: set() for cid in clip_ids}
    for cid, t in row_anchors:
        for fi in abs_frame_indices(t):
            needed.setdefault(cid, set()).add(fi)

    benchmarks = []
    meta_clips: dict[str, Any] = {}
    bytes_total = 0
    for ci, cid in enumerate(sorted(needed.keys()), start=1):
        rec = records[cid]
        indices = sorted(needed[cid])
        if ci <= benchmark_clips:
            benchmarks.append({"clip_id": cid, **benchmark_decode(rec, cfg, sig, row_anchors[0][1] if row_anchors else 0)})

        from collision_jepa3.data.cache import _clip_frames_uint8

        frames_u8 = _clip_frames_uint8(rec, cfg, sig)
        packed: dict[int, np.ndarray] = {}
        for fi in indices:
            fi_clamped = max(0, min(int(fi), len(frames_u8) - 1))
            fr = letterbox_rgb(np.asarray(frames_u8[fi_clamped]), input_size)
            packed[int(fi)] = np.asarray(fr, dtype=np.uint8)
        mapping = write_clip_bin(clip_dir / f"{cid}.bin", packed)
        nbytes = len(indices) * input_size * input_size * 3
        bytes_total += nbytes
        meta_clips[cid] = {"frame_to_slot": mapping, "n_frames": len(indices), "bytes": nbytes}
        if ci % 32 == 0 or ci == len(needed):
            print(f"rgb_cache clips {ci}/{len(needed)}", flush=True)

    manifest = {
        "input_size": input_size,
        "offsets_union": ALL_OFFSETS,
        "n_clips": len(meta_clips),
        "n_unique_frames": sum(c["n_frames"] for c in meta_clips.values()),
        "bytes_estimated": bytes_total,
        "preprocess": sig.as_dict(),
        "clips": meta_clips,
        "benchmark_samples": benchmarks,
    }
    return manifest


def save_cache_meta(report_root: Path, manifest: dict, cache_dir: Path) -> Path:
    out = report_root / "shared_rgb_supernet_cache.json"
    blob = {**manifest, "cache_dir": str(cache_dir)}
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    (cache_dir / "meta.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
