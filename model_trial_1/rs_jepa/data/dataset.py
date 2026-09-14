"""Torch dataset on cached frames + k×k heatmaps.

Each item also carries the *future* clip (same offsets, anchored at t+τ) so the
EMA target encoder can run V-JEPA-style future-latent prediction on the
identically shuffled mosaic, plus a mid-horizon heatmap at t+τ/2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore
        pass

from .augment import ClipAugmentor  # CPU path kept for tests; training uses GPU preprocess

_QUIET_EPISODE_MARKERS = (
    "empty_street",
    "safe_walk",
    "car_pass_far",
    "parallel_pedestrian",
    "periph_empty",
    "parked_car",
)


def _quiet_episode(name: str) -> bool:
    key = name.lower()
    return any(m in key for m in _QUIET_EPISODE_MARKERS)


def _build_index(
    n_episodes: int,
    n_frames: int,
    frame_offsets: list[int],
    tau_frames: int,
    stride: int,
) -> list[tuple[int, int]]:
    min_off = min(frame_offsets)
    t_lo = -min_off
    t_hi = n_frames - 1 - tau_frames
    index: list[tuple[int, int]] = []
    for ep_idx in range(n_episodes):
        for t in range(t_lo, t_hi + 1, stride):
            index.append((ep_idx, t))
    return index


class CollisionDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        episodes: list[str],
        student_size: int,
        frame_offsets: list[int],
        tau_frames: int,
        n_frames: int = 150,
        stride: int = 3,
        train: bool = False,
        augmentor: ClipAugmentor | None = None,
        mid_tau_frames: int | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.episodes = list(episodes)
        self.student_size = student_size
        self.frame_offsets = list(frame_offsets)
        self.tau_frames = tau_frames
        self.mid_tau_frames = int(tau_frames // 2 if mid_tau_frames is None else mid_tau_frames)
        self.n_frames = n_frames
        self.stride = stride
        self.train = train
        self.augmentor = augmentor if (train and augmentor is not None) else None
        self.index = _build_index(len(self.episodes), n_frames, frame_offsets, tau_frames, stride)
        self._frames_cache: dict[int, np.ndarray] = {}
        self._heatmaps_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _frames(self, ep_idx: int) -> np.ndarray:
        if ep_idx not in self._frames_cache:
            path = self.cache_dir / self.episodes[ep_idx] / f"frames_{self.student_size}.npy"
            self._frames_cache[ep_idx] = np.load(path, mmap_mode="r")
        return self._frames_cache[ep_idx]

    def _heatmaps(self, ep_idx: int) -> np.ndarray:
        if ep_idx not in self._heatmaps_cache:
            path = self.cache_dir / self.episodes[ep_idx] / "heatmaps.npy"
            self._heatmaps_cache[ep_idx] = np.load(path, mmap_mode="r")
        return self._heatmaps_cache[ep_idx]

    def sample_weights(
        self,
        has_rare: dict | None = None,
        hot_mult: float = 5.0,
        rare_mult: float = 2.5,
        change_mult: float = 3.0,
        quiet_mult: float = 1.0,
        quiet_peak: float = 0.25,
        quiet_ep_mult: float = 1.0,
    ) -> np.ndarray:
        """Upsample collisions, changing frames, and (optionally) quiet / empty-road clips."""
        has_rare = has_rare or {}
        w = np.ones(len(self.index), dtype=np.float64)
        for i, (ep_idx, t) in enumerate(self.index):
            hm = self._heatmaps(ep_idx)
            h_now = np.asarray(hm[t], dtype=np.float32)
            h_fut = np.asarray(hm[t + self.tau_frames], dtype=np.float32)
            peak = float(h_fut.max())
            change = float(np.abs(h_fut - h_now).max())
            w[i] = 1.0 + hot_mult * peak + change_mult * change
            if has_rare.get(self.episodes[ep_idx], False):
                w[i] *= rare_mult
            if quiet_mult > 1.0 and peak < quiet_peak:
                w[i] *= quiet_mult
            if quiet_ep_mult > 1.0 and _quiet_episode(self.episodes[ep_idx]):
                w[i] *= quiet_ep_mult
        return w

    def __getitem__(self, i: int):
        ep_idx, t = self.index[i]
        frames_arr = self._frames(ep_idx)
        heatmaps_arr = self._heatmaps(ep_idx)

        now_ids = [t + off for off in self.frame_offsets]
        fut_ids = [t + self.tau_frames + off for off in self.frame_offsets]
        # uint8 NHWC — no /255, no photometric aug (those run on GPU).
        clip_now = np.stack([frames_arr[fid] for fid in now_ids], axis=0)
        clip_fut = np.stack([frames_arr[fid] for fid in fut_ids], axis=0)
        h_now = np.array(heatmaps_arr[t], dtype=np.float32)
        h_future = np.array(heatmaps_arr[t + self.tau_frames], dtype=np.float32)
        h_mid = np.array(heatmaps_arr[t + self.mid_tau_frames], dtype=np.float32)

        if torch is None:
            return {
                "frames": clip_now,
                "frames_future": clip_fut,
                "h_now": h_now,
                "h_future": h_future,
                "h_mid": h_mid,
                "ep_idx": ep_idx,
                "t": t,
            }
        return {
            "frames": torch.from_numpy(np.ascontiguousarray(clip_now)),
            "frames_future": torch.from_numpy(np.ascontiguousarray(clip_fut)),
            "h_now": torch.from_numpy(np.ascontiguousarray(h_now)),
            "h_future": torch.from_numpy(np.ascontiguousarray(h_future)),
            "h_mid": torch.from_numpy(np.ascontiguousarray(h_mid)),
            "ep_idx": torch.tensor(ep_idx, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }
