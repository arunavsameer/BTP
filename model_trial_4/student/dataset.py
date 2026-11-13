"""Windows from parent index + 256 RGB cache + parent F_dec + h_now."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WinDS(Dataset):
    def __init__(self, rows, ids, gt, rgb, parent, heatmaps):
        self.rows = rows
        self.ids = list(ids)
        self.gt = gt
        self.rgb = rgb
        self.parent = parent
        self.heatmaps = heatmaps

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
        video_path = str(row.get("video_path", ""))
        h_now = self.heatmaps.get_now(cid, video_path, t)
        h_1s = np.asarray(self.gt[rid], dtype=np.float32)
        z16 = pt.get("Z_16")
        if z16 is None:
            z16 = np.zeros((1, 16, 16), dtype=np.float32)
        return {
            "clip": torch.from_numpy(np.ascontiguousarray(clip)),
            "F_teacher": torch.from_numpy(pt["F_dec"]),
            "H_teacher": torch.from_numpy(pt["H_teacher"]),
            "P_teacher_5": torch.from_numpy(pt["H_teacher"]),
            "Z_teacher_16": torch.from_numpy(np.ascontiguousarray(z16)),
            "gt_1s": torch.from_numpy(h_1s),
            "h_1s": torch.from_numpy(h_1s),
            "h_now": torch.from_numpy(np.ascontiguousarray(h_now)),
            "pre_fork": bool(row.get("pre_fork")),
            "amb_1s": bool(row.get("amb_1s")),
            "sample_id": str(row.get("sample_id", "")),
            "clip_id": cid,
            "frame": t,
            "background_id": str(row.get("background_id", "")),
            "outcome": str(row.get("outcome", "")),
            "video_path": video_path,
        }


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "F_teacher": torch.stack([b["F_teacher"] for b in batch]),
        "H_teacher": torch.stack([b["H_teacher"] for b in batch]),
        "P_teacher_5": torch.stack([b["P_teacher_5"] for b in batch]),
        "Z_teacher_16": torch.stack([b["Z_teacher_16"] for b in batch]),
        "gt_1s": torch.stack([b["gt_1s"] for b in batch]),
        "h_1s": torch.stack([b["h_1s"] for b in batch]),
        "h_now": torch.stack([b["h_now"] for b in batch]),
        "pre_fork": torch.tensor([b["pre_fork"] for b in batch], dtype=torch.bool),
        "amb_1s": torch.tensor([b["amb_1s"] for b in batch], dtype=torch.bool),
        "sample_id": [b["sample_id"] for b in batch],
        "clip_id": [b["clip_id"] for b in batch],
        "frame": torch.tensor([b["frame"] for b in batch], dtype=torch.int64),
        "background_id": [b["background_id"] for b in batch],
        "outcome": [b["outcome"] for b in batch],
        "video_path": [b["video_path"] for b in batch],
    }
