"""Boxes around corners that expand after the camera motion is removed.

Track a few hundred corners with pyramidal Lucas-Kanade, fit one partial
affine (translation, rotation, and scale) to the cloud, and treat the
residual as motion the camera model did not explain. Neighboring points
that are moving apart belong to something getting closer. One box goes
around each such cluster. A frame with too few tracks produces no boxes.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import FlowConfig
from regions import Region


_LK_STOP = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)


class FlowBoxes:
    def __init__(self, cfg: FlowConfig):
        self.cfg = cfg
        self._previous: np.ndarray | None = None

    def boxes(self, frame: np.ndarray) -> list[Region]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous = self._previous
        self._previous = gray
        if previous is None:
            return []
        return _boxes_from_flow(previous, gray, frame.shape[1], frame.shape[0], self.cfg)


def _boxes_from_flow(
    previous: np.ndarray,
    gray: np.ndarray,
    width: int,
    height: int,
    cfg: FlowConfig,
) -> list[Region]:
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=int(cfg.max_corners),
        qualityLevel=float(cfg.quality),
        minDistance=float(cfg.min_distance),
        blockSize=int(cfg.corner_block),
    )
    if corners is None or len(corners) < cfg.min_tracks:
        return []

    try:
        nxt, status, error = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            corners,
            None,
            winSize=(int(cfg.lk_window), int(cfg.lk_window)),
            maxLevel=int(cfg.lk_levels),
            criteria=_LK_STOP,
        )
    except cv2.error:
        return []
    if nxt is None or status is None:
        return []

    ok = status.reshape(-1) == 1
    if error is not None:
        ok = ok & (error.reshape(-1) <= float(cfg.max_lk_error))
    start = corners.reshape(-1, 2)[ok].astype(np.float32)
    finish = nxt.reshape(-1, 2)[ok].astype(np.float32)
    if len(start) < cfg.min_tracks:
        return []

    try:
        model, _inliers = cv2.estimateAffinePartial2D(
            start,
            finish,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(cfg.ransac_px),
        )
    except cv2.error:
        return []
    if model is None:
        return []

    predicted = cv2.transform(start.reshape(-1, 1, 2), model).reshape(-1, 2)
    residual = finish - predicted
    expanding = _expanding_points(
        finish,
        residual,
        float(cfg.cluster_radius),
        float(cfg.expand_threshold),
        int(cfg.min_neighbors),
    )
    chosen = finish[expanding]
    if len(chosen) < cfg.min_points:
        return []

    frame_area = float(width * height)
    regions: list[Region] = []
    for cluster in _clusters(chosen, float(cfg.cluster_radius)):
        if len(cluster) < cfg.min_points:
            continue
        pad = float(cfg.box_pad)
        x1 = max(0.0, float(cluster[:, 0].min()) - pad)
        y1 = max(0.0, float(cluster[:, 1].min()) - pad)
        x2 = min(float(width), float(cluster[:, 0].max()) + pad)
        y2 = min(float(height), float(cluster[:, 1].max()) + pad)
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < cfg.min_area or area > cfg.max_area_frac * frame_area:
            continue
        regions.append(Region(x1=x1, y1=y1, x2=x2, y2=y2, area=area, name="flow"))

    regions.sort(key=lambda region: region.area, reverse=True)
    return regions[: cfg.max_boxes]


def _expanding_points(
    points: np.ndarray,
    residual: np.ndarray,
    radius: float,
    threshold: float,
    min_neighbors: int,
) -> np.ndarray:
    """True where nearby points are moving apart in the residual.

    The score is the average, over neighbors inside `radius`, of how fast
    the two residuals increase the distance between those points.
    """
    count = len(points)
    keep = np.zeros(count, dtype=bool)
    if count < 2:
        return keep

    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt((delta * delta).sum(axis=2))
    neighbors = (distance > 1.0) & (distance <= radius)
    residual_delta = residual[:, None, :] - residual[None, :, :]
    unit = delta / np.maximum(distance[:, :, None], 1e-6)
    separating = (residual_delta * unit).sum(axis=2)

    neighbor_count = neighbors.sum(axis=1)
    total = np.where(neighbors, separating, 0.0).sum(axis=1)
    mean = np.zeros(count, dtype=np.float64)
    np.divide(total, neighbor_count, out=mean, where=neighbor_count > 0)
    keep[(neighbor_count >= min_neighbors) & (mean >= threshold)] = True
    return keep


def _clusters(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Groups of points. Two points join when they are within `radius`."""
    count = len(points)
    used = np.zeros(count, dtype=bool)
    limit = radius * radius
    groups = []
    for start in range(count):
        if used[start]:
            continue
        used[start] = True
        stack = [start]
        members = [start]
        while stack:
            current = stack.pop()
            offset_x = points[:, 0] - points[current, 0]
            offset_y = points[:, 1] - points[current, 1]
            nearby = np.flatnonzero(offset_x * offset_x + offset_y * offset_y <= limit)
            for other in nearby.tolist():
                if used[other]:
                    continue
                used[other] = True
                stack.append(other)
                members.append(other)
        groups.append(points[members])
    return groups
