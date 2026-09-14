"""Video decode, heatmap load, and on-disk frame cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def decode_all_frames(video_path: str | Path, size: int | tuple[int, int]) -> np.ndarray:
    """Decode every frame of ``video_path`` and resize to square ``size``.

    Returns uint8 RGB ``[N, H, W, 3]``.
    """
    import cv2

    if isinstance(size, int):
        out_h, out_w = size, size
    else:
        out_h, out_w = size

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Decoded 0 frames from {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def decode_video_full(video_path: str | Path) -> tuple[np.ndarray, float]:
    """Decode RGB at native resolution. Returns ``[N,H,W,3]`` uint8 and fps."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Decoded 0 frames from {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8), fps


def load_heatmaps(annotations_path: str | Path) -> np.ndarray:
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = sorted(data["frames"], key=lambda fr: int(fr["frame_id"]))
    mats = [np.asarray(fr["matrix"], dtype=np.float32) for fr in frames]
    return np.stack(mats, axis=0)


def upsample_heatmap(hm: np.ndarray, grid: int = 5, mode: str = "nearest") -> np.ndarray:
    """Resize a K×K threat matrix to ``grid``×``grid``.

    mixed/side annotations are native 3×3; pack is native 5×5. If ``k == grid``
    this is a no-op. ``nearest`` is used when upsampling 3×3 → 5×5 so a hot cell
    stays a block instead of bilinear-smearing (which diluted STOP / direction).
    """
    if hm.ndim != 3:
        raise ValueError(f"expected [N,K,K], got {hm.shape}")
    k = hm.shape[-1]
    if k == grid:
        return hm.astype(np.float32, copy=False)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv is required to upsample 3x3 heatmaps") from exc
    interp = cv2.INTER_NEAREST if mode == "nearest" else cv2.INTER_LINEAR
    out = np.empty((hm.shape[0], grid, grid), dtype=np.float32)
    for i in range(hm.shape[0]):
        out[i] = cv2.resize(hm[i], (grid, grid), interpolation=interp)
    return out


def cache_episode(
    episode_dir: str | Path,
    cache_dir: str | Path,
    student_size: int,
    target_grid: int = 5,
    cache_key: str | None = None,
    upsample_mode: str = "nearest",
    frames_only_if_missing: bool = True,
) -> dict:
    """Decode + cache one episode under ``cache_dir / cache_key``."""
    episode_dir = Path(episode_dir)
    cache_dir = Path(cache_dir)
    key = cache_key if cache_key is not None else episode_dir.name
    out_dir = cache_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_path = out_dir / f"frames_{student_size}.npy"
    heatmaps_path = out_dir / "heatmaps.npy"

    heatmaps = load_heatmaps(episode_dir / "spatial_annotations" / "spatial_annotations.json")
    heatmaps = upsample_heatmap(heatmaps, grid=target_grid, mode=upsample_mode)
    np.save(heatmaps_path, heatmaps)

    if not frames_path.exists() or not frames_only_if_missing:
        video = episode_dir / "preview.mp4"
        frames = decode_all_frames(video, student_size)
        np.save(frames_path, frames)
        n_frames = len(frames)
    else:
        n_frames = int(np.load(frames_path, mmap_mode="r").shape[0])

    return {
        "key": key,
        "episode": episode_dir.name,
        "frames_path": str(frames_path),
        "heatmaps_path": str(heatmaps_path),
        "n_frames": n_frames,
        "n_heatmaps": int(heatmaps.shape[0]),
        "grid": int(heatmaps.shape[-1]),
    }
