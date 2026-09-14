"""RGB augmentations applied consistently across a clip's frames.

The dataset spans noon / dawn / dusk / night and clear / fog / smog, so photometric
augmentation (brightness, contrast, blur, noise) is genuinely useful. Geometric
augmentation is restricted to horizontal flip, which MUST also flip the 5x5 heatmap
columns (left <-> right) or the label becomes wrong.
"""

from __future__ import annotations

import numpy as np

# OpenCV defaults to using every core, which thrashes against the training loop
# (num_workers=0) and GPU. One thread per blur call is plenty for 128x128 frames.
try:  # pragma: no cover - environment dependent
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

    def __call__(
        self,
        frames: np.ndarray,
        heatmaps: list[np.ndarray],
        latents: list[np.ndarray] | None = None,
    ):
        """Augment ``frames`` [n, H, W, 3] uint8 and a list of 5x5 heatmaps.

        Optional ``latents`` are tensors with a width axis last (e.g. Z as
        [C, grid, grid]); they are flipped left-right with the RGB/heatmap so
        JEPA targets stay aligned.

        The same photometric params are used for every frame in the clip so temporal
        differences stay meaningful.
        """
        frames = frames.astype(np.float32)

        # Photometric (shared across frames).
        b = 1.0 + self.rng.uniform(-self.brightness, self.brightness)
        c = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
        mean = frames.mean(axis=(0, 1, 2), keepdims=True)
        frames = (frames - mean) * c + mean * b

        if self.noise_std > 0:
            frames = frames + self.rng.normal(0.0, self.noise_std, size=frames.shape)

        frames = np.clip(frames, 0, 255)

        if self.blur_prob > 0 and self.rng.random() < self.blur_prob:
            frames = self._blur(frames)

        # Horizontal flip (also flips heatmap columns and latent width).
        if self.flip_prob > 0 and self.rng.random() < self.flip_prob:
            frames = frames[:, :, ::-1, :].copy()
            heatmaps = [hm[:, ::-1].copy() for hm in heatmaps]
            if latents is not None:
                latents = [z[..., ::-1].copy() for z in latents]

        out_frames = frames.astype(np.uint8)
        if latents is None:
            return out_frames, heatmaps
        return out_frames, heatmaps, latents

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
