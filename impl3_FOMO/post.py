"""Turn a class heatmap into centroids and approximate boxes.

Neighboring cells of the same class are one object. The centroid is the
probability-weighted center of those cells. The box is their outer grid
lines, mapped back to the original frame. One lit cell is the smallest box.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from geom import cell_box_to_frame, cell_center_to_frame


class HeatmapGate:
    """Exponential average of the class probabilities, with hysteresis.

    momentum is the weight on the previous frame (0.7 means 30% new).
    A cell turns on at `on` and stays on while it stays at or above `off`
    and keeps the same class.
    """

    def __init__(self, momentum: float = 0.7, on: float = 0.55, off: float = 0.35):
        self.momentum = float(momentum)
        self.on = float(on)
        self.off = float(off)
        self.state: np.ndarray | None = None
        self.active: np.ndarray | None = None
        self.cls_map: np.ndarray | None = None

    def update(self, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prob = np.asarray(prob, dtype=np.float32)
        if self.state is None:
            self.state = prob.copy()
        else:
            self.state = self.momentum * self.state + (1.0 - self.momentum) * prob
        background = self.state[0]
        objects = self.state[1:]
        cls = objects.argmax(axis=0).astype(np.int32)
        conf = objects.max(axis=0)
        hot = (conf >= self.on) & (conf >= background)
        warm = (conf >= self.off) & (conf >= background)
        if self.active is None:
            self.active = hot.copy()
            self.cls_map = np.where(hot, cls, -1).astype(np.int32)
        else:
            stay = self.active & warm & (cls == self.cls_map)
            self.active = hot | stay
            shown = np.where(hot, cls, self.cls_map)
            self.cls_map = np.where(self.active, shown, -1).astype(np.int32)
        return self.state, self.active, self.cls_map


def class_prob(logits: torch.Tensor) -> np.ndarray:
    """Softmax over the class channels only. logits [1+C, G, G], size channels excluded."""
    return F.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float32)


def heatmap(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """logits [1+C, G, G] → background prob, best object class, that class's prob."""
    prob = F.softmax(logits, dim=0).detach().cpu().numpy()
    background = prob[0]
    objects = prob[1:]
    cls = objects.argmax(axis=0).astype(np.int32)
    conf = objects.max(axis=0)
    return background, cls, conf


def _components(active: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean grid. Each cell is (row, col)."""
    height, width = active.shape
    seen = np.zeros_like(active, dtype=bool)
    found = []
    for row in range(height):
        for col in range(width):
            if not active[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if not active[ny, nx] or seen[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            found.append(cells)
    return found


def detections_from_logits(
    logits: torch.Tensor,
    class_names: list[str],
    threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    stride: int,
    has_size: bool = False,
    active: np.ndarray | None = None,
    cls_map: np.ndarray | None = None,
) -> list[dict]:
    """One dict per object: class_name, class_id, confidence, cx, cy, x1, y1, x2, y2.

    When has_size is set, the last two channels are log(width) and log(height)
    in original-frame pixels, and the box is that size centered on the centroid.
    """
    size_log = None
    class_logits = logits
    if has_size:
        size_log = logits[-2:].detach().cpu().numpy()
        class_logits = logits[:-2]
    background, cls, conf = heatmap(class_logits)
    if active is None:
        active = (conf >= threshold) & (conf >= background)
    if cls_map is not None:
        cls = cls_map
    dets = []
    for class_index, name in enumerate(class_names):
        cells_of_class = active & (cls == class_index)
        for cells in _components(cells_of_class):
            weights = np.array([float(conf[r, c]) for r, c in cells], dtype=np.float64)
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                continue
            rows = np.array([r for r, _c in cells], dtype=np.float64)
            cols = np.array([c for _r, c in cells], dtype=np.float64)
            row_c = float((rows * weights).sum() / weight_sum)
            col_c = float((cols * weights).sum() / weight_sum)
            cx, cy = cell_center_to_frame(col_c, row_c, scale, pad_x, pad_y, stride)
            if size_log is not None:
                log_w = float(sum(float(size_log[0, r, c]) * float(conf[r, c]) for r, c in cells) / weight_sum)
                log_h = float(sum(float(size_log[1, r, c]) * float(conf[r, c]) for r, c in cells) / weight_sum)
                width = float(np.clip(np.exp(log_w), 8.0, 900.0))
                height = float(np.clip(np.exp(log_h), 8.0, 900.0))
                x1, y1, x2, y2 = cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height
            else:
                row0, row1 = int(rows.min()), int(rows.max())
                col0, col1 = int(cols.min()), int(cols.max())
                x1, y1, x2, y2 = cell_box_to_frame(col0, row0, col1, row1, scale, pad_x, pad_y, stride)
            dets.append(
                {
                    "class_name": name,
                    "class_id": class_index,
                    "confidence": float(weights.max()),
                    "cx": cx,
                    "cy": cy,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return dets
