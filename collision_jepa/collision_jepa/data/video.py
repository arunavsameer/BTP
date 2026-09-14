"""Video decoding, resizing and frame/heatmap caching.

The raw clips are 1920x1080 H.264. We never decode 1080p inside the training loop;
instead we decode once, resize, and cache to ``.npy`` per episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def decode_all_frames(video_path: str | Path, size: int | tuple[int, int]) -> np.ndarray:
    """Decode every frame of ``video_path`` and resize to ``size``.

    Returns a ``uint8`` array of shape ``[N, H, W, 3]`` in RGB order.
    ``size`` may be an int (square) or ``(H, W)``.
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


def load_heatmaps(annotations_path: str | Path) -> np.ndarray:
    """Load the per-frame 5x5 threat matrices.

    Returns a ``float32`` array of shape ``[N, grid, grid]`` ordered by frame_id.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]
    frames_sorted = sorted(frames, key=lambda fr: int(fr["frame_id"]))
    mats = [np.asarray(fr["matrix"], dtype=np.float32) for fr in frames_sorted]
    return np.stack(mats, axis=0)


def ensure_frame_cache(
    episode_dir: str | Path,
    cache_dir: str | Path,
    size: int,
) -> Path:
    """Ensure ``frames_<size>.npy`` exists for an episode; decode+save if missing.

    Returns the path to the cached array. Used by the teacher (256px) which needs a
    different resolution than the student (128px). Kept separate so it can be deleted
    after Z caching to reclaim disk.
    """
    episode_dir = Path(episode_dir)
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / episode_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / f"frames_{size}.npy"
    if not frames_path.exists():
        frames = decode_all_frames(episode_dir / "preview.mp4", size)
        np.save(frames_path, frames)
    return frames_path


def cache_episode(
    episode_dir: str | Path,
    cache_dir: str | Path,
    student_size: int,
    frame_indices: list[int] | None = None,
) -> dict:
    """Decode + cache one episode.

    Writes ``<cache>/<episode>/frames_<size>.npy`` and ``heatmaps.npy``. When
    ``frame_indices`` is given, only those frames are stored (others zeroed) to save
    disk; otherwise all frames are cached.

    Returns a small manifest dict describing what was written.
    """
    episode_dir = Path(episode_dir)
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / episode_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_path = out_dir / f"frames_{student_size}.npy"
    heatmaps_path = out_dir / "heatmaps.npy"

    heatmaps = load_heatmaps(episode_dir / "spatial_annotations" / "spatial_annotations.json")
    np.save(heatmaps_path, heatmaps)

    if not frames_path.exists():
        frames = decode_all_frames(episode_dir / "preview.mp4", student_size)
        if frame_indices is not None:
            keep = np.zeros_like(frames)
            for i in frame_indices:
                if 0 <= i < len(frames):
                    keep[i] = frames[i]
            frames = keep
        np.save(frames_path, frames)
        n_frames = len(frames)
    else:
        n_frames = int(np.load(frames_path, mmap_mode="r").shape[0])

    return {
        "episode": episode_dir.name,
        "frames_path": str(frames_path),
        "heatmaps_path": str(heatmaps_path),
        "n_frames": n_frames,
        "n_heatmaps": int(heatmaps.shape[0]),
    }
