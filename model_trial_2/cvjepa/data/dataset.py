"""Datasets: teacher clips (256px) and student clips (160px) + optional cached Z."""

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

_QUIET_EPISODE_MARKERS = (
    "empty_street",
    "safe_walk",
    "car_pass_far",
    "parallel_pedestrian",
    "periph_empty",
    "parked_car",
)


def teacher_clip_indices(t_end: int, clip_frames: int, stride: int) -> list[int]:
    """Frame indices for a teacher clip ENDING at ``t_end``."""
    start = t_end - (clip_frames - 1) * stride
    return [start + i * stride for i in range(clip_frames)]


def _quiet_episode(name: str) -> bool:
    key = name.lower()
    return any(m in key for m in _QUIET_EPISODE_MARKERS)


def _build_index(
    n_episodes: int,
    n_frames: int,
    t_lo: int,
    tau_frames: int,
    stride: int,
) -> list[tuple[int, int]]:
    t_hi = n_frames - 1 - tau_frames
    index: list[tuple[int, int]] = []
    for ep_idx in range(n_episodes):
        for t in range(t_lo, t_hi + 1, stride):
            index.append((ep_idx, t))
    return index


class StudentDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        episodes: list[str],
        student_size: int,
        frame_offsets: list[int],
        tau_frames: int,
        n_frames: int = 150,
        stride: int = 2,
        train: bool = False,
        mid_tau_frames: int | None = None,
        load_teacher_z: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.episodes = list(episodes)
        self.student_size = student_size
        self.frame_offsets = list(frame_offsets)
        self.tau_frames = tau_frames
        self.mid_tau_frames = int(tau_frames // 2 if mid_tau_frames is None else mid_tau_frames)
        self.n_frames = n_frames
        self.train = train
        self.load_teacher_z = load_teacher_z
        t_lo = -min(self.frame_offsets)
        self.index = _build_index(len(self.episodes), n_frames, t_lo, tau_frames, stride)
        self._frames: dict[int, np.ndarray] = {}
        self._heatmaps: dict[int, np.ndarray] = {}
        self._z: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _frames_arr(self, ep_idx: int) -> np.ndarray:
        if ep_idx not in self._frames:
            path = self.cache_dir / self.episodes[ep_idx] / f"frames_{self.student_size}.npy"
            self._frames[ep_idx] = np.load(path, mmap_mode="r")
        return self._frames[ep_idx]

    def _hm(self, ep_idx: int) -> np.ndarray:
        if ep_idx not in self._heatmaps:
            self._heatmaps[ep_idx] = np.load(self.cache_dir / self.episodes[ep_idx] / "heatmaps.npy")
        return self._heatmaps[ep_idx]

    def _z_end(self, ep_idx: int) -> np.ndarray | None:
        if not self.load_teacher_z:
            return None
        if ep_idx not in self._z:
            path = self.cache_dir / self.episodes[ep_idx] / "teacher_z.npz"
            if not path.exists():
                self._z[ep_idx] = None  # type: ignore
            else:
                self._z[ep_idx] = np.load(path)["z_end"]
        return self._z[ep_idx]

    def sample_weights(
        self,
        has_rare: dict | None = None,
        hot_mult: float = 1.5,
        rare_mult: float = 2.0,
        change_mult: float = 3.0,
        quiet_mult: float = 4.0,
        quiet_peak: float = 0.25,
        quiet_ep_mult: float = 2.5,
        center_cool_mult: float = 5.5,
        center_cool_peak: float = 0.28,
        center_cool_mode: str = "inner",
    ) -> np.ndarray:
        has_rare = has_rare or {}
        w = np.ones(len(self.index), dtype=np.float64)
        for i, (ep_idx, t) in enumerate(self.index):
            hm = self._hm(ep_idx)
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
            if center_cool_mult > 1.0:
                g = h_fut.shape[0] // 2
                if center_cool_mode == "cell":
                    center_heat = float(h_fut[g, g])
                else:
                    lo, hi = max(0, g - 1), min(h_fut.shape[0], g + 2)
                    center_heat = float(h_fut[lo:hi, lo:hi].mean())
                if center_heat < center_cool_peak:
                    w[i] *= center_cool_mult
        return w

    def __getitem__(self, i: int):
        ep_idx, t = self.index[i]
        frames_arr = self._frames_arr(ep_idx)
        heatmaps_arr = self._hm(ep_idx)
        now_ids = [t + off for off in self.frame_offsets]
        clip = np.stack([frames_arr[fid] for fid in now_ids], axis=0)
        h_now = np.array(heatmaps_arr[t], dtype=np.float32)
        h_future = np.array(heatmaps_arr[t + self.tau_frames], dtype=np.float32)
        h_mid = np.array(heatmaps_arr[t + self.mid_tau_frames], dtype=np.float32)
        item = {
            "frames": torch.from_numpy(np.ascontiguousarray(clip)),
            "h_now": torch.from_numpy(np.ascontiguousarray(h_now)),
            "h_future": torch.from_numpy(np.ascontiguousarray(h_future)),
            "h_mid": torch.from_numpy(np.ascontiguousarray(h_mid)),
            "ep_idx": torch.tensor(ep_idx, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }
        z_end = self._z_end(ep_idx)
        if z_end is not None:
            item["z_t"] = torch.from_numpy(np.array(z_end[t], dtype=np.float32))
            item["z_plus"] = torch.from_numpy(np.array(z_end[t + self.tau_frames], dtype=np.float32))
        return item
