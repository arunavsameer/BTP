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

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = np.empty((n_hint, out_h, out_w, 3), dtype=np.uint8) if n_hint > 0 else None
    extra: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if out is not None and i < n_hint:
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB, dst=out[i])
        else:
            extra.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()

    if out is None:
        if not extra:
            raise RuntimeError(f"Decoded 0 frames from {video_path}")
        return np.stack(extra, axis=0)
    if i == 0:
        raise RuntimeError(f"Decoded 0 frames from {video_path}")
    if extra:
        return np.concatenate([out, np.stack(extra, axis=0)], axis=0)
    return out if i == n_hint else out[:i]


def load_heatmaps(annotations_path: str | Path, grid: int = 5) -> tuple[np.ndarray, int]:
    """Load per-frame threat matrices and resize to ``grid x grid`` if needed.

    Native k=3 packs are upsampled with nearest-neighbor (no smear into leaf cells).
    Returns ``(float32 [N, grid, grid], native_k)``.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    native_k = int(data.get("k") or len(data["frames"][0]["matrix"]))
    frames = data["frames"]
    frames_sorted = sorted(frames, key=lambda fr: int(fr["frame_id"]))
    mats = [np.asarray(fr["matrix"], dtype=np.float32) for fr in frames_sorted]
    stacked = np.stack(mats, axis=0)
    if stacked.shape[-1] != grid or stacked.shape[-2] != grid:
        import cv2

        resized = np.empty((stacked.shape[0], grid, grid), dtype=np.float32)
        for i, m in enumerate(stacked):
            resized[i] = cv2.resize(m, (grid, grid), interpolation=cv2.INTER_NEAREST)
        stacked = np.clip(resized, 0.0, 1.0)
    return stacked, native_k


def read_cached_native_k(cache_dir: str | Path, episode: str, default: int = 5) -> int:
    path = Path(cache_dir) / episode / "meta.json"
    if not path.exists():
        return default
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("native_k", default))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def load_or_cache_frames(
    episode_dir: str | Path,
    cache_dir: str | Path,
    size: int,
    cache_name: str | None = None,
) -> np.ndarray:
    """Load ``frames_<size>.npy`` into RAM, decoding and saving on a cache miss.

    Returns the uint8 array so callers can pack clips without a second disk read.
    """
    episode_dir = Path(episode_dir)
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / (cache_name or episode_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / f"frames_{size}.npy"
    if frames_path.exists():
        return np.load(frames_path)
    frames = decode_all_frames(episode_dir / "preview.mp4", size)
    np.save(frames_path, frames)
    return frames


def ensure_frame_cache(
    episode_dir: str | Path,
    cache_dir: str | Path,
    size: int,
    cache_name: str | None = None,
) -> Path:
    """Ensure ``frames_<size>.npy`` exists for an episode; decode+save if missing.

    Returns the path to the cached array. Used by the teacher (256px) which needs a
    different resolution than the student (128px). Kept separate so it can be deleted
    after Z caching to reclaim disk.
    """
    episode_dir = Path(episode_dir)
    cache_dir = Path(cache_dir)
    name = cache_name or episode_dir.name
    frames_path = cache_dir / name / f"frames_{size}.npy"
    if not frames_path.exists():
        load_or_cache_frames(episode_dir, cache_dir, size, cache_name=name)
    return frames_path


def ensure_heatmap_cache(
    episode_dir: str | Path,
    cache_dir: str | Path,
    cache_name: str | None = None,
    grid: int = 5,
) -> dict:
    """Write heatmaps.npy + meta.json without decoding RGB (teacher can decode 256 later)."""
    episode_dir = Path(episode_dir)
    cache_dir = Path(cache_dir)
    name = cache_name or episode_dir.name
    out_dir = cache_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_path = out_dir / "heatmaps.npy"
    heatmaps, native_k = load_heatmaps(
        episode_dir / "spatial_annotations" / "spatial_annotations.json", grid=grid
    )
    np.save(heatmaps_path, heatmaps)
    (out_dir / "meta.json").write_text(
        json.dumps({"native_k": int(native_k), "heatmap_grid": int(heatmaps.shape[-1])}, indent=2),
        encoding="utf-8",
    )
    return {
        "episode": name,
        "n_heatmaps": int(heatmaps.shape[0]),
        "native_k": int(native_k),
    }


def cache_episode(
    episode_dir: str | Path,
    cache_dir: str | Path,
    student_size: int,
    frame_indices: list[int] | None = None,
    cache_name: str | None = None,
    grid: int = 5,
) -> dict:
    """Decode + cache one episode.

    Writes ``<cache>/<episode>/frames_<size>.npy`` and ``heatmaps.npy``. When
    ``frame_indices`` is given, only those frames are stored (others zeroed) to save
    disk; otherwise all frames are cached.

    Returns a small manifest dict describing what was written.
    """
    episode_dir = Path(episode_dir)
    cache_dir = Path(cache_dir)
    name = cache_name or episode_dir.name
    out_dir = cache_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_path = out_dir / f"frames_{student_size}.npy"
    heatmaps_path = out_dir / "heatmaps.npy"

    heatmaps, native_k = load_heatmaps(
        episode_dir / "spatial_annotations" / "spatial_annotations.json", grid=grid
    )
    np.save(heatmaps_path, heatmaps)
    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"native_k": int(native_k), "heatmap_grid": int(heatmaps.shape[-1])}, indent=2),
        encoding="utf-8",
    )

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
        "episode": name,
        "src_dir": str(episode_dir),
        "frames_path": str(frames_path),
        "heatmaps_path": str(heatmaps_path),
        "n_frames": n_frames,
        "n_heatmaps": int(heatmaps.shape[0]),
        "heatmap_grid": int(heatmaps.shape[-1]),
        "native_k": int(native_k),
    }
