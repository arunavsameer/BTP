"""Drop graph-segmentation boxes that look like the road.

Ulrich and Nourbakhsh (AAAI 2000) treat a trapezoid at the bottom of a
forward camera as ground, and reject image parts whose color matches that
sample. The boxes still come from Felzenszwalb. A box is removed when
enough of its own pixels fall in the common hue/intensity bins of the
trapezoid. Gray pavement has no reliable hue, so low-saturation pixels
share one extra hue bin.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import PavementConfig
from felzenszwalb import boxes_from_labels, segment_labels, working_image
from regions import Region


def trapezoid_polygon(width: int, height: int, top: float, top_width: float) -> np.ndarray:
    """Four corners, top edge first, then the two bottom corners."""
    y0 = int(round(float(top) * (height - 1)))
    half = 0.5 * float(top_width) * (width - 1)
    center = (width - 1) / 2.0
    return np.array(
        [
            [center - half, y0],
            [center + half, y0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.int32,
    )


def pavement_match_mask(image: np.ndarray, cfg: PavementConfig) -> np.ndarray:
    """True where a pixel's hue/value bin is common inside the road trapezoid."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2].astype(np.int32)
    bins = int(cfg.hist_bins)
    hue_bin = np.clip((hue * bins) // 180, 0, bins - 1)
    hue_bin = np.where(saturation < cfg.gray_saturation, bins, hue_bin)
    value_bin = np.clip((value * bins) // 256, 0, bins - 1)

    height, width = image.shape[:2]
    trap = np.zeros((height, width), np.uint8)
    polygon = trapezoid_polygon(width, height, cfg.trap_top, cfg.trap_top_width)
    cv2.fillConvexPoly(trap, polygon.reshape(-1, 1, 2), 1)
    inside = trap.astype(bool)
    if int(inside.sum()) < 8:
        return np.zeros((height, width), dtype=bool)

    hist = np.zeros((bins + 1, bins), dtype=np.float64)
    np.add.at(hist, (hue_bin[inside], value_bin[inside]), 1)
    probability = hist / hist.sum()
    return probability[hue_bin, value_bin] >= float(cfg.match_threshold)


def find_regions(frame: np.ndarray, cfg: PavementConfig) -> list[Region]:
    small, scale_x, scale_y = working_image(frame, cfg.process_width)
    labels = segment_labels(small, cfg.k, cfg.min_size, cfg.sigma)
    if labels.shape[:2] != small.shape[:2]:
        raise RuntimeError(f"segmentation returned {labels.shape}, expected {small.shape[:2]}")
    mask = pavement_match_mask(small, cfg)
    return boxes_from_labels(
        labels,
        scale_x,
        scale_y,
        frame.shape[1],
        frame.shape[0],
        cfg,
        reject_mask=mask,
        reject_fraction=cfg.reject_fraction,
        name="seg",
    )
