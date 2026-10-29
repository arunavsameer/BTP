"""Teacher windows plus RGB frames. Token cache still feeds the frozen teacher."""

from __future__ import annotations

import numpy as np

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader, collate_teacher_batch


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


def build_student_loader(dataset: StudentDataset, cfg, *, shuffle: bool, drop_last: bool = False):
    return build_dataloader(dataset, cfg, shuffle=shuffle, drop_last=drop_last)


__all__ = ["StudentDataset", "build_student_loader", "collate_teacher_batch"]
