"""Letterbox geometry: 1920×1080 → student input and 5×5 GT cell mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collision_jepa3.data.closing_map import splat_bbox
from collision_jepa3.data.preprocessing import letterbox_rgb


@dataclass(frozen=True)
class LetterboxGeom:
    src_w: int
    src_h: int
    dst: int
    scale: float
    resized_w: int
    resized_h: int
    pad_x: int
    pad_y: int

    def src_to_dst(self, x: float, y: float) -> tuple[float, float]:
        """Image coords (x right, y down) → letterboxed canvas."""
        return self.pad_x + x * self.scale, self.pad_y + y * self.scale

    def cell_bounds_dst(self, grid: int = 5) -> list[list[tuple[int, int, int, int]]]:
        """Per-cell pixel bounds on dst×dst canvas [row][col] = (y0,y1,x0,x1)."""
        g = int(grid)
        s = int(self.dst)
        out: list[list[tuple[int, int, int, int]]] = []
        for r in range(g):
            row: list[tuple[int, int, int, int]] = []
            for c in range(g):
                x0 = (c * s) // g
                x1 = ((c + 1) * s) // g
                y0 = (r * s) // g
                y1 = ((r + 1) * s) // g
                row.append((y0, y1, x0, x1))
            out.append(row)
        return out

    def cell_bounds_src(self, grid: int = 5) -> list[list[tuple[float, float, float, float]]]:
        """Map dst cell corners back to source image coords (may clip outside valid image)."""
        inv = 1.0 / max(self.scale, 1e-8)
        bounds = self.cell_bounds_dst(grid)
        out: list[list[tuple[float, float, float, float]]] = []
        for row in bounds:
            src_row: list[tuple[float, float, float, float]] = []
            for y0, y1, x0, x1 in row:
                sx0 = (x0 - self.pad_x) * inv
                sx1 = (x1 - self.pad_x) * inv
                sy0 = (y0 - self.pad_y) * inv
                sy1 = (y1 - self.pad_y) * inv
                src_row.append((sx0, sy0, sx1, sy1))
            out.append(src_row)
        return out


def letterbox_geom(src_w: int = 1920, src_h: int = 1080, dst: int = 160) -> LetterboxGeom:
    scale = min(dst / float(src_w), dst / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    pad_x = (dst - resized_w) // 2
    pad_y = (dst - resized_h) // 2
    return LetterboxGeom(src_w, src_h, dst, scale, resized_w, resized_h, pad_x, pad_y)


def gt_cell_fraction_in_content(geom: LetterboxGeom, grid: int = 5) -> np.ndarray:
    """[G,G] fraction of each cell that lies inside the letterboxed content (not padding)."""
    g = int(grid)
    s = int(geom.dst)
    frac = np.zeros((g, g), dtype=np.float32)
    for r in range(g):
        for c in range(g):
            y0 = (r * s) // g
            y1 = ((r + 1) * s) // g
            x0 = (c * s) // g
            x1 = ((c + 1) * s) // g
            cx0 = geom.pad_x
            cy0 = geom.pad_y
            cx1 = geom.pad_x + geom.resized_w
            cy1 = geom.pad_y + geom.resized_h
            ix0 = max(x0, cx0)
            iy0 = max(y0, cy0)
            ix1 = min(x1, cx1)
            iy1 = min(y1, cy1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            area = max(1, (x1 - x0) * (y1 - y0))
            frac[r, c] = inter / area
    return frac


def native_gt_vs_dst_grid(
    width: int = 1920, height: int = 1080, dst: int = 160, grid: int = 5, value: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compare annotation splat on native vs splat mapped through letterbox cells."""
    native = np.zeros((grid, grid), dtype=np.float32)
    bbox = {"xmin": width * 0.45, "ymin": height * 0.35, "xmax": width * 0.55, "ymax": height * 0.65}
    splat_bbox(native, bbox, value, width=width, height=height)
    geom = letterbox_geom(width, height, dst)
    mapped = np.zeros((grid, grid), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            sx0, sy0, sx1, sy1 = geom.cell_bounds_src(grid)[r][c]
            cell_bbox = {
                "xmin": max(0, sx0),
                "ymin": max(0, sy0),
                "xmax": min(width, sx1),
                "ymax": min(height, sy1),
            }
            splat_bbox(mapped, cell_bbox, value, width=width, height=height)
    return native, mapped


def build_grid_alignment_report(out_dir: Path, *, src_w: int = 1920, src_h: int = 1080, dst: int = 160) -> dict[str, Any]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    ex_dir = out_dir / "grid_alignment_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    geom = letterbox_geom(src_w, src_h, dst)
    native, mapped = native_gt_vs_dst_grid(src_w, src_h, dst)
    canvas = np.zeros((dst, dst, 3), dtype=np.uint8)
    content = np.full((geom.resized_h, geom.resized_w, 3), 200, dtype=np.uint8)
    canvas[geom.pad_y : geom.pad_y + geom.resized_h, geom.pad_x : geom.pad_x + geom.resized_w] = content
    for r in range(5):
        for c in range(5):
            y0, y1, x0, x1 = geom.cell_bounds_dst()[r][c]
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
    cv2.imwrite(str(ex_dir / "letterbox_grid.png"), canvas)
    report = {
        "src_resolution": [src_w, src_h],
        "student_input_size": dst,
        "letterbox": {
            "scale": geom.scale,
            "resized_w": geom.resized_w,
            "resized_h": geom.resized_h,
            "pad_x": geom.pad_x,
            "pad_y": geom.pad_y,
        },
        "cell_bounds_dst": geom.cell_bounds_dst(),
        "cell_bounds_src": geom.cell_bounds_src(),
        "content_coverage_per_cell": gt_cell_fraction_in_content(geom).tolist(),
        "example_bbox_native_vs_mapped_max_diff": float(np.abs(native - mapped).max()),
        "notes": (
            "GT heatmaps are defined on native 1920×1080 projection. "
            "Student 160×160 uses the same row/col semantics; letterbox padding cells "
            "may have reduced effective FOV."
        ),
    }
    (out_dir / "part2_grid_alignment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
