"""Read-only dataset access: manifest, letterboxed frames, GT heatmaps.

Does not import a student class or checkpoint. Cache is never written.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    DATASET_ROOT,
    FRAME_CACHE_ROOT,
    FRAME_OFFSETS,
    HORIZON_1S_FRAMES,
    INPUT_SIZE,
    MANIFEST_PATH,
    N_CLIPS,
    N_FRAMES,
    PREFERRED_SPLIT,
    SAFE_PEAK,
    SPLITS_PATH,
    STOP_PEAK,
    T_MIN,
    TRIAL3,
)

if str(TRIAL3) not in sys.path:
    sys.path.insert(0, str(TRIAL3))

# Test-split clips: mix collision (STOP-heavy) and miss (SAFE-heavier), people + cars.
PREFERRED_CLIP_IDS = [
    "sample_0003_jaywalker_turn_toward/collision/bg_00",
    "sample_0003_jaywalker_turn_toward/miss_left/bg_00",
    "sample_0016_child_darting/collision/bg_00",
    "sample_0016_child_darting/miss_left/bg_00",
    "sample_0098_car_cross_critical/collision/bg_00",
    "sample_0098_car_cross_critical/miss_right/bg_00",
    "sample_0117_car_cut_in/collision/bg_00",
    "sample_0125_periph_child_cut/collision/bg_00",
    "sample_0139_cyclist_head_on/collision/bg_00",
    "sample_0042_jaywalker/miss_left/bg_00",
    "sample_0101_crossing_car_side/miss_right/bg_00",
    "sample_0026_cube_head_on/miss_right/bg_00",
]


def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(path or MANIFEST_PATH)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_splits(path: Path | None = None) -> dict[str, list[str]]:
    p = Path(path or SPLITS_PATH)
    blob = json.loads(p.read_text(encoding="utf-8"))
    return blob.get("splits") or blob


def cache_dir(clip_id: str, cache_root: Path | None = None) -> Path:
    root = Path(cache_root or FRAME_CACHE_ROOT)
    return root / "frames" / clip_id.replace("/", "__")


def frame_cache_path(clip_id: str, cache_root: Path | None = None) -> Path:
    return cache_dir(clip_id, cache_root) / "frames_uint8.npy"


def load_gt_heatmaps(rec: dict[str, Any]) -> np.ndarray:
    from collision_jepa3.data.heatmap_annotations import load_heatmaps

    path = rec.get("heatmap_json_path")
    if not path:
        path = (
            DATASET_ROOT
            / rec["clip_id"]
            / "spatial_annotations"
            / "spatial_annotations.json"
        )
    return load_heatmaps(path, expected_frames=int(rec.get("frame_count") or N_FRAMES))


def letterbox_uint8(frame: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    from collision_jepa3.data.preprocessing import letterbox_rgb

    return letterbox_rgb(frame, int(size), 0)


def uint8_to_nchw01(clip_thwc: np.ndarray) -> np.ndarray:
    x = np.transpose(np.asarray(clip_thwc, dtype=np.float32), (0, 3, 1, 2))
    x *= 1.0 / 255.0
    return x


def input_indices(anchor: int, offsets=FRAME_OFFSETS) -> list[int]:
    idx = [int(anchor) + int(o) for o in offsets]
    if any(i > int(anchor) for i in idx):
        raise ValueError(f"future leakage at t={anchor}: {idx}")
    return idx


def _decode_preview_ffmpeg(path: Path, width: int, height: int) -> np.ndarray:
    import cv2

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    frame_bytes = int(width) * int(height) * 3
    frames = []
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        proc.stdout.close()
        proc.wait()
    if not frames:
        raise RuntimeError(f"ffmpeg returned no frames for {path}")
    return np.stack(frames, axis=0)


def load_clip_frames(rec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Prefer the letterboxed 256 cache. If missing, decode preview.mp4 and letterbox."""
    clip_id = rec["clip_id"]
    npy = frame_cache_path(clip_id)
    meta: dict[str, Any] = {
        "clip_id": clip_id,
        "source": "cache" if npy.is_file() else "ffmpeg",
        "source_width": int(rec.get("width") or 1920),
        "source_height": int(rec.get("height") or 1080),
    }
    meta_path = cache_dir(clip_id) / "meta.json"
    if meta_path.is_file():
        try:
            extra = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["source_width"] = int(extra.get("source_width") or meta["source_width"])
            meta["source_height"] = int(extra.get("source_height") or meta["source_height"])
        except json.JSONDecodeError:
            pass
    if npy.is_file():
        frames = np.load(npy, mmap_mode="r")
        meta["n_frames"] = int(frames.shape[0])
        return frames, meta

    video = Path(rec.get("video_path") or (DATASET_ROOT / clip_id / "preview.mp4"))
    if not video.is_file():
        raise FileNotFoundError(f"no frame cache and no preview.mp4 for {clip_id}")
    raw = _decode_preview_ffmpeg(video, meta["source_width"], meta["source_height"])
    letterboxed = np.stack([letterbox_uint8(raw[i]) for i in range(raw.shape[0])], axis=0)
    meta["n_frames"] = int(letterboxed.shape[0])
    return letterboxed, meta


def gt_class_counts(heat: np.ndarray) -> dict[str, int]:
    n = heat.shape[0]
    n_safe = n_stop = n_threat = 0
    for t in range(T_MIN, n - HORIZON_1S_FRAMES):
        peak = float(heat[t + HORIZON_1S_FRAMES].max())
        if peak >= STOP_PEAK:
            n_stop += 1
        elif peak >= SAFE_PEAK:
            n_threat += 1
        else:
            n_safe += 1
    return {"n_safe": n_safe, "n_stop": n_stop, "n_threat": n_threat}


def pick_clips(
    records: list[dict[str, Any]] | None = None,
    splits: dict[str, list[str]] | None = None,
    n: int = N_CLIPS,
) -> list[dict[str, Any]]:
    records = records if records is not None else load_manifest()
    splits = splits if splits is not None else load_splits()
    test_groups = set(splits.get(PREFERRED_SPLIT) or [])
    by_id = {r["clip_id"]: r for r in records if r.get("valid", True)}
    picked: list[dict[str, Any]] = []
    used = set()

    def _add(cid: str) -> None:
        rec = by_id.get(cid)
        if rec is None or cid in used:
            return
        group = rec.get("scenario_group_id") or rec.get("sample_id")
        if test_groups and group not in test_groups:
            return
        picked.append(rec)
        used.add(cid)

    for cid in PREFERRED_CLIP_IDS:
        _add(cid)
        if len(picked) >= n:
            return picked[:n]

    rest = [
        r
        for r in records
        if r.get("valid", True)
        and r["clip_id"] not in used
        and (not test_groups or (r.get("scenario_group_id") or r.get("sample_id")) in test_groups)
        and r.get("background_id") == "bg_00"
    ]
    rest.sort(key=lambda r: r["clip_id"])
    used_scen = {r.get("scenario_type") for r in picked}
    used_out = {r.get("outcome_type") for r in picked}
    for rec in rest:
        scen, out = rec.get("scenario_type"), rec.get("outcome_type")
        if scen in used_scen and out in used_out:
            continue
        picked.append(rec)
        used.add(rec["clip_id"])
        used_scen.add(scen)
        used_out.add(out)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for rec in rest:
            if rec["clip_id"] not in used:
                picked.append(rec)
                used.add(rec["clip_id"])
            if len(picked) >= n:
                break
    if len(picked) < n:
        raise RuntimeError(f"only found {len(picked)} clips, need {n}")
    return picked[:n]


def clip_stem(index: int, rec: dict[str, Any]) -> str:
    cid = rec["clip_id"].replace("/", "_")
    return f"clip_{index:02d}_{cid}"
