"""Side-by-side RGB + heatmap grids using PIL (no matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame)
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        if x.max() <= 1.5:
            x = np.clip(x * 255.0, 0, 255)
        x = x.astype(np.uint8)
    return x


def heatmap_to_image(hm: np.ndarray, cell: int = 48) -> Image.Image:
    hm = np.clip(np.asarray(hm, dtype=np.float32), 0.0, 1.0)
    h, w = hm.shape
    img = Image.new("RGB", (w * cell, h * cell), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(h):
        for j in range(w):
            v = float(hm[i, j])
            r = int(255 * v)
            g = int(255 * (0.4 + 0.2 * (1.0 - v)))
            b = int(255 * (1.0 - v))
            x0, y0 = j * cell, i * cell
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(r, g, b))
            draw.text((x0 + 4, y0 + 4), f"{v:.2f}", fill=(255, 255, 255))
    return img


def render_example(
    rgb_anchor: np.ndarray,
    maps: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "",
) -> Path:
    """maps keys: gt_now, pred_now, gt_1s, pred_1s, gt_1_5s, pred_1_5s, gt_2s, pred_2s."""
    rgb = Image.fromarray(_to_uint8_rgb(rgb_anchor)).resize((256, 256))
    order = [
        ("gt_now", "GT now"),
        ("pred_now", "Pred now"),
        ("gt_1s", "GT 1.0s"),
        ("pred_1s", "Pred 1.0s"),
        ("gt_1_5s", "GT 1.5s"),
        ("pred_1_5s", "Pred 1.5s"),
        ("gt_2s", "GT 2.0s"),
        ("pred_2s", "Pred 2.0s"),
    ]
    tiles = [("RGB t", rgb)] + [(lab, heatmap_to_image(maps[k])) for k, lab in order if k in maps]
    cell_w, cell_h = tiles[1][1].size if len(tiles) > 1 else (256, 256)
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    pad_top = 22
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + pad_top) + 28), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((8, 4), title, fill=(240, 240, 240))
    for idx, (lab, im) in enumerate(tiles):
        r, c = divmod(idx, cols)
        im = im.resize((cell_w, cell_h))
        y = 28 + r * (cell_h + pad_top)
        x = c * cell_w
        canvas.paste(im, (x, y + pad_top))
        draw.text((x + 6, y + 2), lab, fill=(230, 230, 230))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def render_now_1s_strip(
    rgb: np.ndarray,
    gt_now: np.ndarray,
    pred_now: np.ndarray,
    gt_1s: np.ndarray,
    pred_1s: np.ndarray,
    *,
    title: str = "",
    panel: int = 320,
) -> np.ndarray:
    """RGB | GT now | Pred now | GT 1s | Pred 1s as a uint8 HxWx3 frame."""
    rgb_im = Image.fromarray(_to_uint8_rgb(rgb)).resize((panel, panel), Image.BILINEAR)
    tiles = [
        ("RGB (t)", rgb_im),
        ("GT H_now", heatmap_to_image(gt_now).resize((panel, panel), Image.NEAREST)),
        ("Pred H_now", heatmap_to_image(pred_now).resize((panel, panel), Image.NEAREST)),
        ("GT H_1s", heatmap_to_image(gt_1s).resize((panel, panel), Image.NEAREST)),
        ("Pred H_1s", heatmap_to_image(pred_1s).resize((panel, panel), Image.NEAREST)),
    ]
    header = 36
    label_h = 22
    canvas = Image.new("RGB", (panel * 5, panel + header + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((8, 8), title, fill=(240, 240, 240))
    y = header
    for i, (lab, im) in enumerate(tiles):
        x = i * panel
        draw.text((x + 8, y + 2), lab, fill=(230, 230, 230))
        canvas.paste(im, (x, y + label_h))
    return np.asarray(canvas, dtype=np.uint8)


def overlay_heatmap(rgb: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = _to_uint8_rgb(rgb)
    h, w = base.shape[:2]
    heat = np.clip(np.asarray(hm, dtype=np.float32), 0.0, 1.0)
    heat_img = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1)
    heat_img = (heat_img * 255).astype(np.uint8)
    heat_img = np.array(Image.fromarray(heat_img).resize((w, h), Image.NEAREST))
    out = (base.astype(np.float32) * (1 - alpha) + heat_img.astype(np.float32) * alpha).clip(0, 255)
    return out.astype(np.uint8)
