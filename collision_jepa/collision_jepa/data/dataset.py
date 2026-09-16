"""Torch datasets built on the cached frames + 5x5 heatmaps.

Sampling rule (per plan): a training sample is anchored at time ``t`` where the
student history frames ``t + offsets`` and the future target ``t + tau_frames`` all
exist. With offsets ``[-10, -5, 0]`` and ``tau_frames = 30`` this gives
``t in [10, n_frames - 1 - tau_frames]`` (i.e. [10, 119] for 150-frame clips).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # torch not installed yet; keep module importable for tooling.
    torch = None

    class Dataset:  # type: ignore
        pass

from .augment import ClipAugmentor


def teacher_clip_indices(t_end: int, clip_frames: int, stride: int) -> list[int]:
    """Frame indices for a teacher clip ENDING at ``t_end``.

    e.g. clip_frames=8, stride=2, t_end=t -> [t-14, t-12, ..., t].
    """
    start = t_end - (clip_frames - 1) * stride
    return [start + i * stride for i in range(clip_frames)]


def pack_teacher_clips(
    frames: np.ndarray,
    heatmaps: np.ndarray | None = None,
    clip_frames: int = 8,
    clip_stride: int = 2,
    sample_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray] | None:
    """Vectorized gather of teacher clips.

    Returns uint8 clips ``[B, T, 3, H, W]``, float32 targets ``[B, G, G]``, and
    clip-end frame indices, or None if nothing is valid.
    """
    n = int(frames.shape[0])
    t_lo = (clip_frames - 1) * clip_stride
    if t_lo >= n:
        return None
    ends = np.arange(t_lo, n, sample_stride, dtype=np.int64)
    offs = np.arange(clip_frames, dtype=np.int64) * clip_stride
    idx = ends[:, None] - (clip_frames - 1) * clip_stride + offs[None, :]
    if int(idx.min()) < 0 or int(idx.max()) >= n:
        keep = (idx.min(axis=1) >= 0) & (idx.max(axis=1) < n)
        ends, idx = ends[keep], idx[keep]
        if ends.size == 0:
            return None
    clips = np.asarray(frames)[idx]  # [B, T, H, W, 3]
    clips = np.ascontiguousarray(np.transpose(clips, (0, 1, 4, 2, 3)))
    if heatmaps is None:
        return clips, None, ends
    targets = np.ascontiguousarray(heatmaps[ends], dtype=np.float32)
    return clips, targets, ends


def _build_index(
    episodes: list[str],
    n_frames: int,
    frame_offsets: list[int],
    tau_frames: int,
    stride: int,
) -> list[tuple[int, int]]:
    min_off = min(frame_offsets)
    t_lo = -min_off  # smallest t so that t + min_off >= 0
    t_hi = n_frames - 1 - tau_frames  # largest t so that t + tau exists
    index: list[tuple[int, int]] = []
    for ep_idx in range(len(episodes)):
        for t in range(t_lo, t_hi + 1, stride):
            index.append((ep_idx, t))
    return index


class CollisionDataset(Dataset):
    """Yields student history frames + current/future heatmaps for one anchor ``t``.

    Each item is a dict of tensors:
      - ``frames``:   float32 [n_frames, 3, H, W] in [0, 1]
      - ``h_now``:    float32 [grid, grid]
      - ``h_future``: float32 [grid, grid]
      - ``ep_idx``:   int64 scalar (index into ``self.episodes``)
      - ``t``:        int64 scalar (anchor frame)
    Cached teacher targets ``Z_t`` / ``Z+`` are attached later by the student
    training script using ``(ep_idx, t)`` as the key.
    """

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
    ):
        self.cache_dir = Path(cache_dir)
        self.episodes = list(episodes)
        self.student_size = student_size
        self.frame_offsets = list(frame_offsets)
        self.tau_frames = tau_frames
        self.n_frames = n_frames
        self.stride = stride
        self.train = train
        self.augmentor = augmentor if (train and augmentor is not None) else None

        self.index = _build_index(self.episodes, n_frames, frame_offsets, tau_frames, stride)

        # Lazily memory-map the per-episode arrays.
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

    def _spatial_latents(self, ep_idx: int, t: int) -> list[np.ndarray] | None:
        """Optional extra spatial maps (e.g. teacher Z) flipped with the heatmap."""
        return None

    def __getitem__(self, i: int):
        ep_idx, t = self.index[i]
        frames_arr = self._frames(ep_idx)
        heatmaps_arr = self._heatmaps(ep_idx)

        frame_ids = [t + off for off in self.frame_offsets]
        clip = np.stack([np.array(frames_arr[fid]) for fid in frame_ids], axis=0)  # [n,H,W,3]
        h_now = np.array(heatmaps_arr[t], dtype=np.float32)
        h_future = np.array(heatmaps_arr[t + self.tau_frames], dtype=np.float32)
        latents = self._spatial_latents(ep_idx, t)

        if self.augmentor is not None:
            if latents is not None:
                clip, (h_now, h_future), latents = self.augmentor(clip, [h_now, h_future], latents)
            else:
                clip, (h_now, h_future) = self.augmentor(clip, [h_now, h_future])

        clip = clip.astype(np.float32) / 255.0
        clip = np.transpose(clip, (0, 3, 1, 2))  # [n, 3, H, W]

        if torch is None:
            item = {
                "frames": clip,
                "h_now": h_now,
                "h_future": h_future,
                "ep_idx": ep_idx,
                "t": t,
            }
            if latents is not None:
                item["z_now"] = latents[0]
                item["z_plus"] = latents[1]
            return item
        item = {
            "frames": torch.from_numpy(np.ascontiguousarray(clip)).float(),
            "h_now": torch.from_numpy(np.ascontiguousarray(h_now)).float(),
            "h_future": torch.from_numpy(np.ascontiguousarray(h_future)).float(),
            "ep_idx": torch.tensor(ep_idx, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }
        if latents is not None:
            item["z_now"] = torch.from_numpy(np.ascontiguousarray(latents[0])).float()
            item["z_plus"] = torch.from_numpy(np.ascontiguousarray(latents[1])).float()
        return item


class StudentDataset(CollisionDataset):
    """CollisionDataset + cached teacher Z targets (Z_t and Z+).

    Reads ``teacher_z.npz`` (produced by scripts/04_cache_teacher_z.py) per episode.
    Anchor times are additionally restricted so the teacher clip ending at ``t``
    exists (``t >= (clip_frames-1)*stride``).
    """

    def __init__(
        self,
        *args,
        teacher_clip_frames: int = 8,
        teacher_stride: int = 2,
        **kwargs,
    ):
        self.teacher_clip_frames = teacher_clip_frames
        self.teacher_stride = teacher_stride
        super().__init__(*args, **kwargs)
        # Rebuild the index with the teacher-availability constraint on t.
        t_lo = max(-min(self.frame_offsets), (teacher_clip_frames - 1) * teacher_stride)
        t_hi = self.n_frames - 1 - self.tau_frames
        self.index = [
            (ep, t)
            for ep in range(len(self.episodes))
            for t in range(t_lo, t_hi + 1, self.stride)
        ]
        self._z_cache: dict[int, np.ndarray] = {}

    def _z_end(self, ep_idx: int) -> np.ndarray:
        if ep_idx not in self._z_cache:
            path = self.cache_dir / self.episodes[ep_idx] / "teacher_z.npz"
            with np.load(path) as data:
                self._z_cache[ep_idx] = np.asarray(data["z_end"], dtype=np.float32)
        return self._z_cache[ep_idx]

    def _spatial_latents(self, ep_idx: int, t: int) -> list[np.ndarray]:
        z_end = self._z_end(ep_idx)
        z_now = np.array(z_end[t], dtype=np.float32, copy=True)
        z_plus = np.array(z_end[t + self.tau_frames], dtype=np.float32, copy=True)
        return [z_now, z_plus]
