"""OpenCV video reader. Converts BGR→RGB and refuses silent frame substitution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class VideoReadError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    reported_frames: int
    decoded_frames: int | None = None


def probe_video(path: str | Path) -> VideoInfo:
    import cv2

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return VideoInfo(path=path, width=width, height=height, fps=fps, reported_frames=reported)


def decode_all_rgb(path: str | Path) -> tuple[np.ndarray, VideoInfo]:
    import cv2

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"could not open video: {path}")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        reported_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise VideoReadError(f"decoded 0 frames from {path}")
    arr = np.stack(frames, axis=0).astype(np.uint8)
    info.decoded_frames = int(arr.shape[0])
    return arr, info


def decode_indices_rgb(path: str | Path, indices: list[int]) -> tuple[np.ndarray, VideoInfo]:
    """Decode requested frames by sequential read. Missing indices are errors."""
    if not indices:
        raise VideoReadError("empty index list")
    needed = sorted(set(int(i) for i in indices))
    if any(i < 0 for i in needed):
        raise VideoReadError(f"negative frame indices {needed}")
    max_i = needed[-1]
    frames, info = _decode_until(path, max_i)
    missing = [i for i in needed if i >= frames.shape[0]]
    if missing:
        raise VideoReadError(
            f"{path}: requested frames {missing} missing (decoded {frames.shape[0]})"
        )
    stacked = np.stack([frames[i] for i in indices], axis=0)
    return stacked, info


def _decode_until(path: str | Path, last_index: int) -> tuple[np.ndarray, VideoInfo]:
    import cv2

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"could not open video: {path}")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        reported_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    frames: list[np.ndarray] = []
    try:
        while len(frames) <= last_index:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise VideoReadError(f"decoded 0 frames from {path}")
    arr = np.stack(frames, axis=0).astype(np.uint8)
    info.decoded_frames = int(arr.shape[0])
    return arr, info
