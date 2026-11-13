"""Pilot dataset: RGB windows at student resolution + teacher cache targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from collision_jepa3.data.cache import _clip_frames_uint8
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.preprocessing import letterbox_rgb, signature_from_cfg, uint8_to_nchw01

from .frames import indices_at_anchor


class PilotAnchorDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        *,
        row_ids: list[int],
        frame_offsets: list[int],
        input_size: int = 160,
        normalize_01: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.rows = [json.loads(l) for l in (self.cache_dir / "index.jsonl").open()]
        self.pred = np.load(self.cache_dir / "teacher_pred.npz", mmap_mode="r")
        self.gt = np.load(self.cache_dir / "gt_heatmaps.npz", mmap_mode="r")
        self.row_ids = list(row_ids)
        self.offsets = list(frame_offsets)
        self.input_size = int(input_size)
        self.normalize_01 = bool(normalize_01)
        cfg_path = self.cache_dir / "meta.json"
        self._cfg = None
        self._records: dict | None = None
        self._sig = None
        self._frame_mem: dict[str, np.ndarray] = {}

    def set_cache_root(self, cfg) -> None:
        self._cfg = cfg
        self._records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
        self._sig = signature_from_cfg(cfg)

    def preload_clips(self) -> None:
        """Decode each clip once before training (faster than lazy load per batch)."""
        if self._records is None:
            raise RuntimeError("call set_cache_root(cfg) first")
        for cid in sorted({self.rows[int(rid)]["clip_id"] for rid in self.row_ids}):
            if cid in self._frame_mem:
                continue
            rec = self._records[cid]
            assert self._cfg is not None and self._sig is not None
            self._frame_mem[cid] = _clip_frames_uint8(rec, self._cfg, self._sig)

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, i: int) -> dict:
        rid = int(self.row_ids[i])
        row = self.rows[rid]
        cid = row["clip_id"]
        t = int(row["frame"])
        if self._records is None:
            raise RuntimeError("call set_cache_root(cfg)")
        if cid not in self._frame_mem:
            rec = self._records[cid]
            assert self._cfg is not None and self._sig is not None
            self._frame_mem[cid] = _clip_frames_uint8(rec, self._cfg, self._sig)
        frames = self._frame_mem[cid]
        idxs = indices_at_anchor(t, self.offsets)
        stack = []
        for fi in idxs:
            fi = max(0, min(fi, len(frames) - 1))
            fr = np.asarray(frames[fi])
            if fr.shape[0] != self.input_size:
                fr = letterbox_rgb(fr, self.input_size)
            stack.append(fr)
        clip = uint8_to_nchw01(np.stack(stack, axis=0))
        if not self.normalize_01:
            clip = clip  # already [0,1]
        sample = {
            "clip_id": cid,
            "frame": t,
            "row_id": rid,
            "clip": torch.from_numpy(clip),
            "gt_1s": torch.from_numpy(np.asarray(self.gt["gt_1s"][rid], dtype=np.float32)),
            "H_teacher": torch.from_numpy(np.asarray(self.pred["H"][rid], dtype=np.float32)),
            "Z_teacher": torch.from_numpy(np.asarray(self.pred["Z"][rid], dtype=np.float32)),
            "F_teacher": torch.from_numpy(np.asarray(self.pred["F_dec"][rid], dtype=np.float32)),
            "pre_fork": bool(row.get("pre_fork")),
            "background_pair_id": row.get("background_pair_id") or "",
            "outcome_type": row.get("outcome", row.get("outcome_type", "")),
            "scenario_type": row.get("scenario_type", ""),
        }
        return sample


def collate_pilot(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        if torch.is_tensor(batch[0][k]):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif k in ("outcome_type", "scenario_type", "clip_id", "background_pair_id"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = [b[k] for b in batch]
    return out


def filter_rows_by_groups(cache_dir: Path, groups: set[str], split: str | None = None) -> list[int]:
    rows = [json.loads(l) for l in (Path(cache_dir) / "index.jsonl").open()]
    out = []
    for i, r in enumerate(rows):
        if split and r.get("split") != split:
            continue
        if r.get("scenario_group_id") in groups:
            out.append(i)
    return out


def make_loader(ds: PilotAnchorDataset, batch: int, shuffle: bool, *, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pilot,
        pin_memory=torch.cuda.is_available(),
    )
