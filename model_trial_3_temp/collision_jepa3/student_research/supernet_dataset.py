"""RGB supernet datasets (flat offset stack per anchor)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from collision_jepa3.student_research.dev_manifest import PATTERN_IDS
from collision_jepa3.student_research.rgb_supernet_cache import RGBSupernetCache


def pair_pattern_index(background_pair_id: str, epoch: int) -> int:
    h = int(hashlib.sha256(background_pair_id.encode()).hexdigest()[:8], 16)
    return (h + int(epoch)) % 5


def pattern_name_for_row(row: dict, epoch: int) -> str:
    pid = str(row.get("background_pair_id") or row["clip_id"])
    return PATTERN_IDS[pair_pattern_index(pid, epoch)]


class SupernetAnchorDataset(Dataset):
    def __init__(
        self,
        teacher_cache: Path,
        rgb_cache: RGBSupernetCache,
        row_ids: list[int],
        *,
        epoch: int = 0,
    ):
        self.rows = [json.loads(l) for l in (Path(teacher_cache) / "index.jsonl").open()]
        self.pred = np.load(teacher_cache / "teacher_pred.npz", mmap_mode="r")
        self.gt = np.load(teacher_cache / "gt_heatmaps.npz", mmap_mode="r")
        self.rgb = rgb_cache
        self.row_ids = list(row_ids)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, i: int) -> dict:
        rid = int(self.row_ids[i])
        row = self.rows[rid]
        cid = row["clip_id"]
        t = int(row["frame"])
        flat = self.rgb.get_anchor_stack(cid, t, normalize_01=True)
        pattern = pattern_name_for_row(row, self.epoch)
        return {
            "row_id": rid,
            "clip_id": cid,
            "frame": t,
            "flat_frames": torch.from_numpy(flat.astype(np.float32)),
            "gt_1s": torch.from_numpy(np.asarray(self.gt["gt_1s"][rid], dtype=np.float32)),
            "H_teacher": torch.from_numpy(np.asarray(self.pred["H"][rid], dtype=np.float32)),
            "pre_fork": bool(row.get("pre_fork")),
            "background_pair_id": str(row.get("background_pair_id") or ""),
            "outcome_type": row.get("outcome", row.get("outcome_type", "collision")),
            "train_pattern": pattern,
            "scenario_group_id": row.get("scenario_group_id", ""),
        }


def collate_supernet(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        if k == "flat_frames":
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif torch.is_tensor(batch[0][k]):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif k in ("clip_id", "background_pair_id", "outcome_type", "train_pattern", "scenario_group_id"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = [b[k] for b in batch]
    return out


def make_loader(ds: Dataset, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, collate_fn=collate_supernet, drop_last=False)
