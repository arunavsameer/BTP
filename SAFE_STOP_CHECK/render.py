"""Side-by-side GT vs student video. HUD reads metrics only — no student class."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    HORIZON_1S_FRAMES,
    HUD_HEIGHT,
    OVERLAY_ALPHA,
    T_MIN,
    TRIAL3,
    VIDEO_FPS,
)
from metrics import fmt_score

if str(TRIAL3) not in sys.path:
    sys.path.insert(0, str(TRIAL3))


def content_box(src_w: int, src_h: int, size: int) -> tuple[int, int, int, int]:
    scale = min(size / float(src_w), size / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    return x0, y0, new_w, new_h


def overlay_on_letterbox(
    square: np.ndarray,
    hm: np.ndarray | None,
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Paint 5×5 on the 256 letterbox, crop black bars, resize to display size."""
    from collision_jepa3.evaluation.visualizations import overlay_heatmap

    if hm is None:
        painted = np.asarray(square)
    else:
        painted = overlay_heatmap(np.asarray(square), hm, alpha=OVERLAY_ALPHA)
    x0, y0, new_w, new_h = content_box(src_w, src_h, painted.shape[0])
    crop = painted[y0 : y0 + new_h, x0 : x0 + new_w]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def numeric_grid(hm: np.ndarray | None, height: int) -> np.ndarray:
    from collision_jepa3.evaluation.visualizations import heatmap_to_image

    if hm is None:
        canvas = np.full((height, height, 3), 24, dtype=np.uint8)
        cv2.putText(
            canvas, "needs 3s", (24, height // 2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "history", (40, height // 2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA,
        )
        return canvas
    cell = max(1, height // int(hm.shape[0]))
    im = heatmap_to_image(hm, cell=cell)
    im = im.resize((height, height), Image.Resampling.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def _panel(rgb_sq: np.ndarray, hm: np.ndarray | None, src_w: int, src_h: int, title: str) -> np.ndarray:
    left = overlay_on_letterbox(rgb_sq, hm, src_w, src_h, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    right = numeric_grid(hm, DISPLAY_HEIGHT)
    canvas = np.concatenate([left, right], axis=1)
    cv2.putText(
        canvas, title, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


def _hud(width: int, lines: list[str]) -> np.ndarray:
    img = Image.new("RGB", (width, HUD_HEIGHT), (12, 12, 18))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font
    draw.text((10, 6), lines[0], fill=(240, 240, 240), font=font)
    if len(lines) > 1:
        draw.text((10, 34), lines[1], fill=(200, 210, 230), font=font_sm)
    return np.asarray(img, dtype=np.uint8)


def compose_frame(
    rgb_sq: np.ndarray,
    gt_hm: np.ndarray | None,
    pred_hm: np.ndarray | None,
    *,
    src_w: int,
    src_h: int,
    hud_lines: list[str],
) -> np.ndarray:
    left = _panel(rgb_sq, gt_hm, src_w, src_h, "LEFT  RGB@t + GT occupancy at t+1s")
    right = _panel(rgb_sq, pred_hm, src_w, src_h, "RIGHT  RGB@t + student Hhat_1s (t+1s)")
    body = np.concatenate([left, right], axis=1)
    hud = _hud(body.shape[1], hud_lines)
    frame = np.concatenate([hud, body], axis=0)
    h, w = frame.shape[:2]
    if h % 2 or w % 2:
        frame = frame[: h - (h % 2), : w - (w % 2)]
    return frame


def hud_lines_for(
    rec: dict,
    t: int,
    n_frames: int,
    row: dict | None,
    clip: dict,
) -> list[str]:
    cid = rec["clip_id"]
    if row is None:
        return [
            f"{cid}  t={t}/{n_frames - 1}  waiting for 3s history (no Hhat_1s)",
            f"clip SAFE={fmt_score(clip.get('safe_score'))}  "
            f"STOP={fmt_score(clip.get('stop_score'))}  "
            f"n_safe={clip.get('n_safe_frames', 0)}  n_stop={clip.get('n_stop_frames', 0)}",
        ]
    q_label = "n/a" if row["gt_class"] == "THREAT" else fmt_score(row["quality"])
    return [
        f"{cid}  t={t}  RGB@t  GT_h@t+1s={row['gt_class']}({row['gt_peak']:.2f})  "
        f"Pred={row['pred_class']}({row['pred_peak']:.2f})  frame_q={q_label}",
        f"clip SAFE={fmt_score(clip.get('safe_score'))}  STOP={fmt_score(clip.get('stop_score'))}  "
        f"spatial={row['spatial']:.2f} (cov={row['coverage']:.2f} leak={row['leak']:.2f})  "
        f"intent={row['intent']:.2f}  n_safe={clip.get('n_safe_frames', 0)} n_stop={clip.get('n_stop_frames', 0)}",
    ]


def write_mp4(path: Path, frames: list[np.ndarray], fps: float = VIDEO_FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18", "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def render_clip_frames(
    rec: dict,
    letterboxed: np.ndarray,
    pred_by_t: dict[int, np.ndarray],
    gt: np.ndarray,
    rows_by_t: dict[int, dict],
    clip: dict,
    src_w: int,
    src_h: int,
) -> list[np.ndarray]:
    n = int(letterboxed.shape[0])
    out = []
    for t in range(n):
        rgb = np.asarray(letterboxed[t])
        row = rows_by_t.get(t)
        if t < T_MIN:
            gt_hm = None
            pred_hm = None
        else:
            tgt = t + HORIZON_1S_FRAMES
            gt_hm = gt[tgt] if tgt < n else None
            pred_hm = pred_by_t.get(t)
        lines = hud_lines_for(rec, t, n, row, clip)
        out.append(compose_frame(rgb, gt_hm, pred_hm, src_w=src_w, src_h=src_h, hud_lines=lines))
    return out
