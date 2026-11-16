#!/usr/bin/env python3
"""Recolor first-15min teacher overlay without re-running Giant.

Reads the existing wide video (left overlay + right 5×5 with printed values)
and the source RGB. Reconstructs the 5×5 from the right panel, paints the
left overlay as STOP-red / else-blue only, and keeps the original right panel.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.preprocessing import letterbox_rgb  # noqa: E402
from scripts._common import add_config_args  # noqa: E402

import importlib.util

_spec = importlib.util.spec_from_file_location("r20", ROOT / "scripts" / "20_render_street_first15.py")
_r20 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_r20)

SOURCE = Path("/storage/BTP/impl/input/street_walking_480p.mp4")
OVERLAY = ROOT / "visualizations" / "street_first15min" / "street_walking_first15min.mp4"
LEFT_W, HEIGHT, FPS = 854, 480, 30.0
RIGHT_W = 480
GRID = 5
HISTORY = 90
STOP_THRESHOLD = 0.60
SIZE = 256


def open_rgb_decoder(path: Path, pix: str, start: float, duration: float) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{duration:.3f}",
            "-f", "rawvideo", "-pix_fmt", pix, "-an", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def hm_from_right_panel(right: np.ndarray) -> np.ndarray:
    """Invert the red/blue cell fill on the 5×5 panel (avoid the printed text)."""
    h, w = right.shape[:2]
    cell_h, cell_w = h // GRID, w // GRID
    hm = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(GRID):
        for j in range(GRID):
            y0, x0 = i * cell_h, j * cell_w
            y1, x1 = y0 + cell_h, x0 + cell_w
            patch = right[y0 + cell_h // 2 : y1 - 6, x0 + cell_w // 2 : x1 - 6]
            r = float(np.median(patch[:, :, 0])) / 255.0
            b = float(np.median(patch[:, :, 2])) / 255.0
            hm[i, j] = np.clip(0.5 * (r + (1.0 - b)), 0.0, 1.0)
    return hm


def main() -> None:
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument("--overlay", default=str(OVERLAY))
    p.add_argument("--out", default="", help="default: overwrite overlay after backing up _continuous.mp4")
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--stop-threshold", type=float, default=STOP_THRESHOLD)
    args = p.parse_args()

    overlay_path = Path(args.overlay)
    out_path = Path(args.out) if args.out else overlay_path
    backup = overlay_path.with_name(overlay_path.stem + "_continuous.mp4")
    if out_path.resolve() == overlay_path.resolve() and overlay_path.is_file() and not backup.is_file():
        overlay_path.rename(backup)
        print(f"kept continuous overlay as {backup}", flush=True)
        src_overlay = backup
    else:
        src_overlay = overlay_path if overlay_path.is_file() else backup

    src_dec = open_rgb_decoder(Path(args.source), "bgr24", 0.0, args.duration)
    ov_dec = open_rgb_decoder(src_overlay, "rgb24", 0.0, args.duration)
    assert src_dec.stdout is not None and ov_dec.stdout is not None
    src_bytes = LEFT_W * HEIGHT * 3
    ov_bytes = (LEFT_W + RIGHT_W) * HEIGHT * 3
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = _r20.open_writer(tmp, FPS, LEFT_W + RIGHT_W, HEIGHT)
    assert writer.stdin is not None
    t = 0
    n_stop_cells = 0
    try:
        while True:
            sraw = src_dec.stdout.read(src_bytes)
            oraw = ov_dec.stdout.read(ov_bytes)
            if len(sraw) < src_bytes or len(oraw) < ov_bytes:
                break
            rgb = cv2.cvtColor(np.frombuffer(sraw, dtype=np.uint8).reshape((HEIGHT, LEFT_W, 3)), cv2.COLOR_BGR2RGB)
            canvas = np.frombuffer(oraw, dtype=np.uint8).reshape((HEIGHT, LEFT_W + RIGHT_W, 3)).copy()
            right = canvas[:, LEFT_W:]
            if t >= HISTORY:
                hm = hm_from_right_panel(right)
                n_stop_cells += int((hm >= args.stop_threshold).sum())
                left = _r20.overlay_on_frame_binary(rgb, hm, SIZE, args.stop_threshold)
                canvas[:, :LEFT_W] = left
                peak = float(hm.max())
                label = (
                    f"first15min  t={t}  teacher ep08 H+1s  peak {peak:.2f}  "
                    f"overlay STOP>={args.stop_threshold:.2f}"
                )
                cv2.putText(
                    canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )
            writer.stdin.write(np.ascontiguousarray(canvas, dtype=np.uint8).tobytes())
            t += 1
            if t % 900 == 0:
                print(f"  {t} frames ({t / FPS:.1f}s)  STOP cells so far {n_stop_cells}", flush=True)
    finally:
        src_dec.stdout.close()
        ov_dec.stdout.close()
        src_dec.wait()
        ov_dec.wait()
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {tmp}")
    tmp.replace(out_path)
    print(f"wrote {out_path}  frames={t}  STOP-cell hits={n_stop_cells}", flush=True)


if __name__ == "__main__":
    main()
