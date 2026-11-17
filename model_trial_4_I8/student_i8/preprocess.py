"""Letterbox 256 + 16-frame window sampling. No overlay / dataset code."""

from __future__ import annotations

import numpy as np

from .constants import INPUT_SIZE, OFFSETS_16


def letterbox_rgb(frame: np.ndarray, size: int = INPUT_SIZE, pad_value: int = 0) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"empty frame shape {frame.shape}")
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def window_indices(anchor: int, n_frames: int, offsets=None) -> list[int]:
    offs = list(OFFSETS_16 if offsets is None else offsets)
    return [min(max(0, int(anchor) + o), int(n_frames) - 1) for o in offs]


def clip_thwc_to_btchw01(frames_thwc: np.ndarray) -> np.ndarray:
    x = np.transpose(frames_thwc.astype(np.float32), (0, 3, 1, 2))
    x *= 1.0 / 255.0
    return x
