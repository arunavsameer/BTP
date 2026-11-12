"""Windows from parent index + 256 RGB cache + parent F_dec."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WinDS(Dataset):
    def __init__(self, rows, ids, gt, rgb, parent):
        self.rows = rows
        self.ids = list(ids)
        self.gt = gt
        self.rgb = rgb
        self.parent = parent

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rid = int(self.ids[i])
        row = self.rows[rid]
        cid, t = row["clip_id"], int(row["frame"])
        clip = self.rgb.get_clip01(cid, t)
        pt = self.parent.get_parent_targets(cid, t)
        if pt is None:
            raise KeyError(f"missing parent {cid} {t}")
        return {
            "clip": torch.from_numpy(np.ascontiguousarray(clip)),
            "F_teacher": torch.from_numpy(pt["F_dec"]),
            "H_teacher": torch.from_numpy(pt["H_teacher"]),
            "gt_1s": torch.from_numpy(np.asarray(self.gt[rid], dtype=np.float32)),
            "pre_fork": bool(row.get("pre_fork")),
            "amb_1s": bool(row.get("amb_1s")),
            "sample_id": str(row.get("sample_id", "")),
            "clip_id": cid,
            "frame": t,
            "background_id": str(row.get("background_id", "")),
            "outcome": str(row.get("outcome", "")),
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "F_teacher": torch.stack([b["F_teacher"] for b in batch]),
        "H_teacher": torch.stack([b["H_teacher"] for b in batch]),
        "gt_1s": torch.stack([b["gt_1s"] for b in batch]),
        "pre_fork": torch.tensor([b["pre_fork"] for b in batch], dtype=torch.bool),
        "amb_1s": torch.tensor([b["amb_1s"] for b in batch], dtype=torch.bool),
        "sample_id": [b["sample_id"] for b in batch],
        "clip_id": [b["clip_id"] for b in batch],
        "frame": torch.tensor([b["frame"] for b in batch], dtype=torch.int64),
        "background_id": [b["background_id"] for b in batch],
        "outcome": [b["outcome"] for b in batch],
    }
