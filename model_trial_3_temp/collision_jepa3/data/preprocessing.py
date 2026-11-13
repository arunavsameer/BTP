"""RGB resize/pad. Letterbox is the default so left/right FOV is preserved."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessSignature:
    resize_mode: str
    input_size: int
    pad_value: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    preserve_fov: bool

    def as_dict(self) -> dict:
        return asdict(self)


def signature_from_cfg(cfg) -> PreprocessSignature:
    mean = tuple(float(x) for x in cfg.get("preprocessing.mean", IMAGENET_MEAN))
    std = tuple(float(x) for x in cfg.get("preprocessing.std", IMAGENET_STD))
    return PreprocessSignature(
        resize_mode=str(cfg.get("preprocessing.resize_mode", "letterbox")),
        input_size=int(cfg.get("preprocessing.input_size", 256)),
        pad_value=int(cfg.get("preprocessing.pad_value", 0)),
        mean=(mean[0], mean[1], mean[2]),
        std=(std[0], std[1], std[2]),
        preserve_fov=bool(cfg.get("preprocessing.preserve_fov", True)),
    )


def letterbox_rgb(frame: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    """Aspect-preserving resize with symmetric padding to a square."""
    import cv2

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"empty frame shape {frame.shape}")
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def checkpoint_center_crop_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    """Official V-JEPA processor: shortest-edge to size*256/224, then center crop."""
    import cv2

    h, w = frame.shape[:2]
    short = min(h, w)
    resize_short = int(round(size * 256 / 224))
    scale = resize_short / float(short)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    y0 = max(0, (new_h - size) // 2)
    x0 = max(0, (new_w - size) // 2)
    crop = resized[y0 : y0 + size, x0 : x0 + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop


def preprocess_frame(frame: np.ndarray, sig: PreprocessSignature) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {frame.shape}")
    if sig.resize_mode == "letterbox":
        return letterbox_rgb(frame, sig.input_size, sig.pad_value)
    if sig.resize_mode == "checkpoint_center_crop":
        return checkpoint_center_crop_rgb(frame, sig.input_size)
    raise ValueError(f"unknown resize_mode {sig.resize_mode}")


def preprocess_clip(frames: np.ndarray, sig: PreprocessSignature) -> np.ndarray:
    """frames [T,H,W,3] uint8 RGB → [T,H',W',3] uint8."""
    out = [preprocess_frame(frames[i], sig) for i in range(frames.shape[0])]
    return np.stack(out, axis=0)


def uint8_to_nchw01(clip_thwc: np.ndarray) -> np.ndarray:
    """[T,H,W,3] uint8 → [T,3,H,W] float32 in [0,1]. Does not normalize."""
    x = np.transpose(clip_thwc.astype(np.float32), (0, 3, 1, 2))
    x *= 1.0 / 255.0
    return x
