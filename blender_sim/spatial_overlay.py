"""Visualize the k×k spatial threat matrix on top of the rendered RGB.

Pure Python (no bpy). Writes a small PPM heat sequence, then ffmpeg scales
it onto a hazy copy of the original frames: blue at 0, bright red at 1.

``python spatial_overlay.py`` runs the numeric / PPM self-test.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence


# 3×5 bitmap for the score printed in each cell.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ".": ("000", "000", "000", "000", "010"),
}

# Heat canvas: ~16:9 cells so ffmpeg's scale to 1920×1080 does not squash text.
CELL_W = 160
CELL_H = 90
GRID_LINE = 3
GLYPH_SCALE = 4


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    u = _clamp01(t)
    return (
        int(round(a[0] + (b[0] - a[0]) * u)),
        int(round(a[1] + (b[1] - a[1]) * u)),
        int(round(a[2] + (b[2] - a[2]) * u)),
    )


def threat_to_rgb(score: float) -> tuple[int, int, int]:
    """Blue (0) → amber → bright red (1). Skips the green HSV valley."""
    t = _clamp01(score)
    blue = (22, 72, 220)
    amber = (255, 186, 18)
    red = (255, 18, 10)
    if t <= 0.55:
        return _lerp_rgb(blue, amber, t / 0.55)
    return _lerp_rgb(amber, red, (t - 0.55) / 0.45)


def overlay_filter_complex(k: int, res_x: int, res_y: int, heat_alpha: float = 0.52) -> str:
    """Hazy original + nearest-neighbour heat + cell grid. Map ``[out]``."""
    a = max(0.05, min(0.85, float(heat_alpha)))
    inv = 1.0 - a
    kk = max(1, int(k))
    return (
        f"[0:v]eq=saturation=0.38:brightness=-0.10:contrast=0.88,format=gbrp[hazy];"
        f"[1:v]scale={int(res_x)}:{int(res_y)}:flags=neighbor,format=gbrp[heat];"
        f"[hazy][heat]blend=all_expr='A*{inv:.3f}+B*{a:.3f}'[mix];"
        f"[mix]drawgrid=w=iw/{kk}:h=ih/{kk}:t=2:c=white@0.40[out]"
    )


def _fill_rect(buf: bytearray, w: int, x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]) -> None:
    r, g, b = rgb
    x0 = max(0, x0)
    y0 = max(0, y0)
    row_span = (x1 - x0) * 3
    if row_span <= 0 or y1 <= y0:
        return
    line = bytearray([r, g, b] * (x1 - x0))
    for y in range(y0, y1):
        i = (y * w + x0) * 3
        buf[i : i + row_span] = line


def _plot(buf: bytearray, w: int, h: int, x: int, y: int, rgb: tuple[int, int, int]) -> None:
    if 0 <= x < w and 0 <= y < h:
        i = (y * w + x) * 3
        buf[i] = rgb[0]
        buf[i + 1] = rgb[1]
        buf[i + 2] = rgb[2]


def _draw_glyph(
    buf: bytearray,
    w: int,
    h: int,
    x: int,
    y: int,
    ch: str,
    scale: int,
    fg: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    rows = _GLYPHS.get(ch)
    if rows is None:
        return
    for gy, row in enumerate(rows):
        for gx, bit in enumerate(row):
            if bit != "1":
                continue
            for oy in range(scale):
                for ox in range(scale):
                    px = x + gx * scale + ox
                    py = y + gy * scale + oy
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx or dy:
                                _plot(buf, w, h, px + dx, py + dy, outline)
                    _plot(buf, w, h, px, py, fg)


def _text_size(text: str, scale: int) -> tuple[int, int]:
    # 3-wide glyphs, 1 px gap; '.' is still 3 wide so the gap stays even.
    n = len(text)
    return n * (3 * scale + scale) - scale, 5 * scale


def _draw_text_centered(
    buf: bytearray,
    w: int,
    h: int,
    cx: int,
    cy: int,
    text: str,
    scale: int,
) -> None:
    tw, th = _text_size(text, scale)
    x = cx - tw // 2
    y = cy - th // 2
    step = 3 * scale + scale
    white = (255, 255, 255)
    ink = (8, 8, 8)
    for i, ch in enumerate(text):
        _draw_glyph(buf, w, h, x + i * step, y, ch, scale, white, ink)


def render_heat_frame(matrix: Sequence[Sequence[float]], *, cell_w: int = CELL_W, cell_h: int = CELL_H) -> tuple[int, int, bytes]:
    """RGB888 image, row 0 = top of the frame (same as the annotation matrix)."""
    k = len(matrix)
    if k <= 0:
        raise ValueError("empty threat matrix")
    cw = max(16, int(cell_w))
    ch = max(16, int(cell_h))
    w = k * cw
    h = k * ch
    buf = bytearray(w * h * 3)
    line = max(1, int(GRID_LINE))
    scale = max(2, int(GLYPH_SCALE))
    if min(cw, ch) < 48:
        scale = 2
    for i, row in enumerate(matrix):
        if len(row) != k:
            raise ValueError(f"matrix is not {k}×{k}")
        y0 = i * ch
        y1 = y0 + ch
        for j, val in enumerate(row):
            s = _clamp01(float(val))
            x0 = j * cw
            x1 = x0 + cw
            _fill_rect(buf, w, x0, y0, x1, y1, threat_to_rgb(s))
            _draw_text_centered(buf, w, h, (x0 + x1) // 2, (y0 + y1) // 2, f"{s:.2f}", scale)
    # Grid on top so labels stay inside the cell.
    black = (12, 12, 16)
    white = (230, 230, 230)
    for i in range(k + 1):
        y = min(h, i * ch)
        y0 = max(0, y - line // 2)
        y1 = min(h, y0 + line)
        _fill_rect(buf, w, 0, y0, w, y1, black)
        if y1 + 1 <= h:
            _fill_rect(buf, w, 0, y1, w, min(h, y1 + 1), white)
    for j in range(k + 1):
        x = min(w, j * cw)
        x0 = max(0, x - line // 2)
        x1 = min(w, x0 + line)
        _fill_rect(buf, w, x0, 0, x1, h, black)
        if x1 + 1 <= w:
            _fill_rect(buf, w, x1, 0, min(w, x1 + 1), h, white)
    return w, h, bytes(buf)


def write_ppm(path: Path, width: int, height: int, rgb: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{int(width)} {int(height)}\n255\n".encode("ascii")
    path.write_bytes(header + rgb)


def write_heat_sequence(
    frames: Iterable[dict],
    out_dir: Path,
    *,
    k: int,
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
) -> int:
    """Write ``000000.ppm`` … for each entry. Returns the number of files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    expect_k = max(1, int(k))
    for entry in frames:
        matrix = entry["matrix"]
        if len(matrix) != expect_k or any(len(r) != expect_k for r in matrix):
            raise ValueError(f"frame {entry.get('frame_id')} is not {expect_k}×{expect_k}")
        fid = str(entry.get("frame_id", f"{n:06d}"))
        w, h, rgb = render_heat_frame(matrix, cell_w=cell_w, cell_h=cell_h)
        write_ppm(out_dir / f"{fid}.ppm", w, h, rgb)
        n += 1
    return n


def _self_test() -> None:
    b0 = threat_to_rgb(0.0)
    b1 = threat_to_rgb(1.0)
    assert b0[2] > b0[0] + 40, b0  # blue
    assert b1[0] > b1[2] + 40, b1  # red
    w, h, rgb = render_heat_frame([[0.0, 1.0], [0.5, 0.2]])
    assert w == 2 * CELL_W and h == 2 * CELL_H
    assert len(rgb) == w * h * 3
    # Interior of top-left (cold) vs top-right (hot), away from the grid lines.
    def _px(x: int, y: int) -> bytes:
        i = (y * w + x) * 3
        return rgb[i : i + 3]

    cold = _px(CELL_W // 2, CELL_H // 2)
    hot = _px(CELL_W + CELL_W // 2, CELL_H // 2)
    assert cold[2] > cold[0] + 20, cold
    assert hot[0] > hot[2] + 20, hot
    filt = overlay_filter_complex(3, 1920, 1080)
    assert "drawgrid" in filt and "scale=1920:1080" in filt
    print("spatial_overlay self-test: OK")


if __name__ == "__main__":
    _self_test()
