"""Teacher windows plus RGB frames. Token cache still feeds the frozen teacher."""

from __future__ import annotations

import numpy as np

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader, collate_teacher_batch

HARD_SCENARIOS = ("shape", "cube", "child")


class StudentDataset(TeacherDataset):
    """Same index as TeacherDataset, but always returns letterboxed RGB."""

    def __getitem__(self, i: int) -> dict:
        sample = super().__getitem__(i)
        if "frames" not in sample:
            clip_id, t = self.index[i]
            rec = self.records_by_id[clip_id]
            frames = self._load_input_clip(rec, t)
            sample["frames"] = frames.astype(np.float32, copy=False)
        return sample


def upsample_small_appear(
    dataset: StudentDataset,
    *,
    repeats: int = 3,
    occ_thr: float = 0.50,
    stop_thr: float = 0.60,
    now_thr: float = 0.25,
    max_occ: int = 2,
) -> dict[str, int]:
    """Repeat small STOP / appear-now-empty train windows so they are not drowned."""
    h1 = int(dataset.horizons[0]) if dataset.horizons else 30
    extra: list[tuple[str, int]] = []
    n_small = 0
    n_appear = 0
    n_hard = 0
    for clip_id, t in list(dataset.index):
        hm = dataset._heatmaps.get(clip_id)
        if hm is None:
            continue
        t = int(t)
        t1 = t + h1
        if t < 0 or t1 >= len(hm):
            continue
        now = np.asarray(hm[t], dtype=np.float32)
        fut = np.asarray(hm[t1], dtype=np.float32)
        n_occ = int((fut >= occ_thr).sum())
        is_stop = float(fut.max()) >= stop_thr
        if not is_stop:
            continue
        empty_now = float(now.max()) < now_thr
        small = n_occ <= int(max_occ)
        if not (small or empty_now):
            continue
        rec = dataset.records_by_id.get(clip_id)
        scen = str(getattr(rec, "scenario_type", "") or "").lower()
        hard = any(k in scen for k in HARD_SCENARIOS)
        fork = int(getattr(rec, "fork_frame", None) or dataset.fork_frame)
        pre = t < fork
        n_rep = int(repeats)
        if hard and pre:
            n_rep += 2
            n_hard += 1
        if small:
            n_small += 1
        if empty_now:
            n_appear += 1
        extra.extend([(clip_id, t)] * n_rep)
    n0 = len(dataset.index)
    dataset.index.extend(extra)
    return {
        "base": n0,
        "added": len(extra),
        "total": len(dataset.index),
        "small": n_small,
        "appear": n_appear,
        "hard_prefork": n_hard,
        "repeats": int(repeats),
    }


def build_student_loader(dataset: StudentDataset, cfg, *, shuffle: bool, drop_last: bool = False):
    return build_dataloader(dataset, cfg, shuffle=shuffle, drop_last=drop_last)


__all__ = [
    "StudentDataset",
    "build_student_loader",
    "collate_teacher_batch",
    "upsample_small_appear",
]
