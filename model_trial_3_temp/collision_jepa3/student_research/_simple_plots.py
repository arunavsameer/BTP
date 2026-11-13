"""Minimal scatter/grid plots without matplotlib (stdlib SVG)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _svg_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlab: str, ylab: str, size: int = 480) -> None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return
    pad = 40
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin:
        xmax = xmin + 1e-6
    if ymax <= ymin:
        ymax = ymin + 1e-6
    dots = []
    for xi, yi in zip(x, y):
        px = pad + (xi - xmin) / (xmax - xmin) * (size - 2 * pad)
        py = size - pad - (yi - ymin) / (ymax - ymin) * (size - 2 * pad)
        dots.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2" fill="#3366cc" opacity="0.35"/>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
<title>{title}</title>
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="20" font-size="12">{title}</text>
<text x="{size//2}" y="{size-8}" text-anchor="middle" font-size="11">{xlab}</text>
<text x="12" y="{size//2}" transform="rotate(-90 12 {size//2})" font-size="11">{ylab}</text>
{"".join(dots)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _svg_grid(path: Path, grid: np.ndarray, title: str, vmax: float | None = None) -> None:
    g = np.asarray(grid, dtype=float)
    vmax = vmax or max(0.6, float(g.max()))
    cell = 48
    n = g.shape[0]
    rects = []
    for i in range(n):
        for j in range(n):
            v = g[i, j] / vmax if vmax > 0 else 0
            v = min(max(v, 0), 1)
            c = int(255 * (1 - v))
            b = int(255 * v)
            rects.append(
                f'<rect x="{j*cell}" y="{i*cell}" width="{cell}" height="{cell}" fill="rgb({c},{c},{b})"/>'
            )
    w, h = n * cell, n * cell
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h+24}">
<text x="4" y="16" font-size="12">{title}</text>
<g transform="translate(0,20)">{"".join(rects)}</g>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def save_scatter_plots(out_dir: Path, gt_p, parent_p, s0_p, s1_p) -> list[str]:
    pairs = [
        ("gt_vs_parent.svg", gt_p, parent_p, "GT peak", "Parent peak"),
        ("gt_vs_s0.svg", gt_p, s0_p, "GT peak", "S0 peak"),
        ("gt_vs_s1.svg", gt_p, s1_p, "GT peak", "S1 peak"),
        ("parent_vs_s0.svg", parent_p, s0_p, "Parent peak", "S0 peak"),
        ("parent_vs_s1.svg", parent_p, s1_p, "Parent peak", "S1 peak"),
    ]
    paths = []
    for name, x, y, xl, yl in pairs:
        p = out_dir / name
        _svg_scatter(p, x, y, name.replace(".svg", ""), xl, yl)
        paths.append(str(p))
    return paths


def save_grid_heatmaps(out_dir: Path, gt_m, parent_m, s0_m, s1_m, err0, err1) -> list[str]:
    items = [
        ("mean_gt.svg", gt_m, "Mean GT"),
        ("mean_parent.svg", parent_m, "Mean parent"),
        ("mean_s0.svg", s0_m, "Mean S0"),
        ("mean_s1.svg", s1_m, "Mean S1"),
        ("mae_s0.svg", err0, "S0 |error|"),
        ("mae_s1.svg", err1, "S1 |error|"),
    ]
    paths = []
    for name, grid, title in items:
        p = out_dir / name
        _svg_grid(p, np.asarray(grid), title)
        paths.append(str(p))
    return paths
