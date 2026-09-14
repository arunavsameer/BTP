"""RGB augmentations applied consistently across a clip's frames."""

from __future__ import annotations

import numpy as np

try:
    import cv2 as _cv2

    _cv2.setNumThreads(1)
except Exception:
    pass


class ClipAugmentor:
    def __init__(
        self,
        brightness: float = 0.25,
        contrast: float = 0.25,
        blur_prob: float = 0.2,
        noise_std: float = 6.0,
        flip_prob: float = 0.5,
        seed: int | None = None,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.blur_prob = blur_prob
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, frames: np.ndarray, heatmaps: list[np.ndarray]):
        """Augment ``frames`` [n, H, W, 3] uint8 and a list of k×k heatmaps."""
        frames = frames.astype(np.float32)

        b = 1.0 + self.rng.uniform(-self.brightness, self.brightness)
        c = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
        mean = frames.mean(axis=(0, 1, 2), keepdims=True)
        frames = (frames - mean) * c + mean * b

        if self.noise_std > 0:
            frames = frames + self.rng.normal(0.0, self.noise_std, size=frames.shape)

        frames = np.clip(frames, 0, 255)

        if self.blur_prob > 0 and self.rng.random() < self.blur_prob:
            frames = self._blur(frames)

        if self.flip_prob > 0 and self.rng.random() < self.flip_prob:
            frames = frames[:, :, ::-1, :].copy()
            heatmaps = [hm[:, ::-1].copy() for hm in heatmaps]

        return frames.astype(np.uint8), heatmaps

    @staticmethod
    def _blur(frames: np.ndarray) -> np.ndarray:
        try:
            import cv2

            out = np.empty_like(frames)
            k = 3
            for i in range(frames.shape[0]):
                out[i] = cv2.GaussianBlur(frames[i], (k, k), 0)
            return out
        except Exception:
            return frames
