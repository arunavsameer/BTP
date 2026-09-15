"""Blender-style spatial overlay: hazy RGB + blue→red k×k threat grid.

Colour map matches ``blender_sim/spatial_overlay.py`` (blue / amber / red).
Compositing is OpenCV so we do not need ffmpeg filter_complex.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("opencv is required for overlay videos") from exc


def threat_to_rgb(score: float) -> tuple[int, int, int]:
    t = 0.0 if score < 0.0 else (1.0 if score > 1.0 else float(score))
    blue, amber, red = (22, 72, 220), (255, 186, 18), (255, 18, 10)
    if t <= 0.55:
        u = t / 0.55
        a, b = blue, amber
    else:
        u = (t - 0.55) / 0.45
        a, b = amber, red
    return (
        int(round(a[0] + (b[0] - a[0]) * u)),
        int(round(a[1] + (b[1] - a[1]) * u)),
        int(round(a[2] + (b[2] - a[2]) * u)),
    )


def haze_rgb(rgb: np.ndarray) -> np.ndarray:
    """Approx ffmpeg ``eq=saturation=0.38:brightness=-0.10:contrast=0.88``."""
    img = rgb.astype(np.float32) / 255.0
    img = (img - 0.5) * 0.88 + 0.5 - 0.10
    gray = img.mean(axis=2, keepdims=True)
    img = gray + 0.38 * (img - gray)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def heat_image(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour k×k colour field, RGB uint8."""
    hm = np.asarray(matrix, dtype=np.float32)
    k = hm.shape[0]
    lut = np.zeros((k, k, 3), dtype=np.uint8)
    for i in range(k):
        for j in range(k):
            lut[i, j] = threat_to_rgb(float(hm[i, j]))
    return cv2.resize(lut, (width, height), interpolation=cv2.INTER_NEAREST)


def blend_heat(rgb: np.ndarray, matrix: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    hazy = haze_rgb(rgb)
    heat = heat_image(matrix, rgb.shape[1], rgb.shape[0])
    a = float(np.clip(alpha, 0.05, 0.85))
    mix = (hazy.astype(np.float32) * (1.0 - a) + heat.astype(np.float32) * a).clip(0, 255)
    out = mix.astype(np.uint8)
    k = int(np.asarray(matrix).shape[0])
    gh, gw = rgb.shape[0] / k, rgb.shape[1] / k
    for i in range(k + 1):
        y = int(round(i * gh))
        cv2.line(out, (0, y), (rgb.shape[1] - 1, y), (230, 230, 230), 1, cv2.LINE_AA)
    for j in range(k + 1):
        x = int(round(j * gw))
        cv2.line(out, (x, 0), (x, rgb.shape[0] - 1), (230, 230, 230), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(rgb.shape[1] / k / 140.0, 0.7))
    for i in range(k):
        for j in range(k):
            tx = int((j + 0.5) * gw)
            ty = int((i + 0.5) * gh)
            text = f"{float(matrix[i, j]):.2f}"
            (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
            cv2.putText(
                out, text, (tx - tw // 2, ty + th // 2), font, scale, (8, 8, 8), 3, cv2.LINE_AA
            )
            cv2.putText(
                out, text, (tx - tw // 2, ty + th // 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA
            )
    return out


_SEV_RGB = {
    "SAFE": (60, 180, 60),
    "CAUTION": (255, 186, 18),
    "STOP": (255, 18, 10),
}


def banner(
    width: int,
    title: str,
    direction: str,
    severity: str,
    extra: str = "",
    height: int = 48,
) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    color = _SEV_RGB.get(severity, (180, 180, 180))
    bar[:] = (18, 18, 22)
    cv2.rectangle(bar, (0, 0), (12, height - 1), color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"{title}  {direction} {severity}"
    if extra:
        label = f"{label}   {extra}"
    cv2.putText(bar, label, (22, int(height * 0.68)), font, 0.55, color, 1, cv2.LINE_AA)
    return bar


def stack_panel(rgb: np.ndarray, matrix: np.ndarray, title: str, direction: str, severity: str, extra: str = "") -> np.ndarray:
    vis = blend_heat(rgb, matrix)
    cap = banner(vis.shape[1], title, direction, severity, extra)
    return np.concatenate([cap, vis], axis=0)


def write_mp4(path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    path = str(path)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
