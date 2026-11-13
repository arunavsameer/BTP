"""Letterbox the street frame into FOMO's square, and map cells back out."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Fit image inside size×size. Returns canvas, scale, pad_x, pad_y."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def frame_point_to_cell(cx: float, cy: float, scale: float, pad_x: int, pad_y: int, stride: int) -> tuple[int, int]:
    """Original-frame centroid → (column, row) on the grid. May lie outside."""
    x = cx * scale + pad_x
    y = cy * scale + pad_y
    return int(x // stride), int(y // stride)


def cell_center_to_frame(
    col: float, row: float, scale: float, pad_x: int, pad_y: int, stride: int
) -> tuple[float, float]:
    """Grid coordinate (column, row), in cells, → original-frame pixel."""
    x = (col + 0.5) * stride
    y = (row + 0.5) * stride
    return (x - pad_x) / scale, (y - pad_y) / scale


def cell_box_to_frame(
    col0: int, row0: int, col1: int, row1: int, scale: float, pad_x: int, pad_y: int, stride: int
) -> tuple[float, float, float, float]:
    """Inclusive cell box → original-frame x1, y1, x2, y2.

    col1/row1 are the last lit cell, so the far edge is the next grid line.
    """
    x1 = (col0 * stride - pad_x) / scale
    y1 = (row0 * stride - pad_y) / scale
    x2 = ((col1 + 1) * stride - pad_x) / scale
    y2 = ((row1 + 1) * stride - pad_y) / scale
    return x1, y1, x2, y2
