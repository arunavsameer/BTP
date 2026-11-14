"""Canny cyan overlay on already-letterboxed 256 RGB (street-vlog order a).

Gaussian 5×5 (sigma=0 → OpenCV 1.1), Canny 40/120, paint RGB (0,255,255).
Stays 3-channel: the photograph is kept; edge pixels are overwritten. No 4th
channel. Pad bars on the 256 cache are typically already black; Canny on a
constant region does not fire, so the tensor remains [T,H,W,3] / [B,16,3,256,256].
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EdgeHighlightConfig:
    blur_kernel: int = 5
    canny_low: float = 40.0
    canny_high: float = 120.0
    edge_color: tuple[int, int, int] = (0, 255, 255)


def highlight_edges(frame: np.ndarray, cfg: EdgeHighlightConfig | None = None) -> np.ndarray:
    """uint8 HxWx3 RGB in, same shape out; cyan on Canny edges, photo otherwise."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 RGB, got {frame.shape} {frame.dtype}")
    cfg = EdgeHighlightConfig() if cfg is None else cfg
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if cfg.blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)
    mask = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    out = frame.copy()
    out[mask > 0] = np.asarray(cfg.edge_color, dtype=np.uint8)
    return out


def highlight_clip(clip_u8: np.ndarray) -> np.ndarray:
    """clip_u8: [T,H,W,3] uint8 -> same with per-frame Canny paint."""
    return np.stack([highlight_edges(clip_u8[t]) for t in range(clip_u8.shape[0])], axis=0)
