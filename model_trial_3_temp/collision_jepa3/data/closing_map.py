"""5×5 approaching maps from annotations.json kinematics. Train-only aux."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DYNAMIC_CLASSES = frozenset({"person", "vehicle"})
TTC_NOT_CONVERGING = 100.0
RVZ_APPROACH = -0.10


def object_approaching(obj: dict[str, Any]) -> bool:
    """True if the object is closing on the camera (negative camera-Z relative vel)."""
    kin = obj.get("kinematics") or {}
    try:
        ttc = float(kin.get("ttc", 9999.0))
    except (TypeError, ValueError):
        ttc = 9999.0
    rv = kin.get("relative_velocity") or [0.0, 0.0, 0.0]
    try:
        rvz = float(rv[2]) if len(rv) > 2 else 0.0
    except (TypeError, ValueError, IndexError):
        rvz = 0.0
    if ttc >= TTC_NOT_CONVERGING:
        return rvz < RVZ_APPROACH
    return rvz < RVZ_APPROACH or ttc < 8.0


def splat_bbox(
    grid: np.ndarray,
    bbox: dict[str, Any],
    value: float,
    *,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Paint a 2D box onto a G×G grid. Row 0 is image top (far)."""
    try:
        xmin = float(bbox["xmin"])
        ymin = float(bbox["ymin"])
        xmax = float(bbox["xmax"])
        ymax = float(bbox["ymax"])
    except (KeyError, TypeError, ValueError):
        return
    g = int(grid.shape[0])
    w = max(int(width), 1)
    h = max(int(height), 1)
    c0 = max(0, int(np.floor(xmin / w * g)))
    c1 = min(g, int(np.ceil(xmax / w * g)))
    r0 = max(0, int(np.floor(ymin / h * g)))
    r1 = min(g, int(np.ceil(ymax / h * g)))
    if c1 <= c0:
        c1 = min(g, c0 + 1)
    if r1 <= r0:
        r1 = min(g, r0 + 1)
    patch = grid[r0:r1, c0:c1]
    np.maximum(patch, value, out=patch)


def maps_from_frames(
    frames: list[dict[str, Any]],
    *,
    grid: int = 5,
    width: int = 1920,
    height: int = 1080,
    expected_frames: int | None = 450,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (close, occupied) float32 [T, G, G] for person/vehicle boxes."""
    n = int(expected_frames) if expected_frames is not None else len(frames)
    close = np.zeros((n, grid, grid), dtype=np.float32)
    occ = np.zeros((n, grid, grid), dtype=np.float32)
    for item in frames:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("frame_id", item.get("id", 0))
        try:
            fid = int(str(raw_id).lstrip("0") or "0") if not isinstance(raw_id, int) else int(raw_id)
        except (TypeError, ValueError):
            continue
        if fid < 0 or fid >= n:
            continue
        for obj in item.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("class_name", "")).lower() not in DYNAMIC_CLASSES:
                continue
            bbox = obj.get("bounding_box_2d") or {}
            splat_bbox(occ[fid], bbox, 1.0, width=width, height=height)
            if object_approaching(obj):
                splat_bbox(close[fid], bbox, 1.0, width=width, height=height)
    return close, occ


def load_closing_maps(
    path: str | Path | None,
    *,
    grid: int = 5,
    width: int = 1920,
    height: int = 1080,
    expected_frames: int = 450,
) -> tuple[np.ndarray, np.ndarray]:
    """Load annotations.json. Missing/corrupt files return zeros, never crash train."""
    zeros = (
        np.zeros((int(expected_frames), int(grid), int(grid)), dtype=np.float32),
        np.zeros((int(expected_frames), int(grid), int(grid)), dtype=np.float32),
    )
    if not path:
        return zeros
    p = Path(path)
    if not p.is_file():
        return zeros
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return zeros
    frames = data.get("frames") if isinstance(data, dict) else data
    if not isinstance(frames, list):
        return zeros
    return maps_from_frames(
        frames,
        grid=int(grid),
        width=int(width),
        height=int(height),
        expected_frames=int(expected_frames),
    )
