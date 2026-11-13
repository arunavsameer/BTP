"""Dataset with group-mean targets and optional horizontal flip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from collision_jepa3.data.preprocessing import uint8_to_nchw01
from collision_jepa3.student_research.fdec_pilot_dataset import SparseClipFrameStore, STUDENT_INPUT_SIZE
from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS
from collision_jepa3.student_research.frames import indices_at_anchor


def flip_heatmap_columns(h: np.ndarray) -> np.ndarray:
  return np.asarray(h)[..., ::-1].copy()


class StudentFinalDataset(Dataset):
  def __init__(
    self,
    cache_dir: Path,
    row_ids: list[int],
    gt_group_mean: np.ndarray,
    *,
    frame_store: SparseClipFrameStore,
    frame_offsets: list[int] | None = None,
    flip_prob: float = 0.0,
    train: bool = False,
  ):
    self.cache_dir = Path(cache_dir)
    self.rows = [json.loads(l) for l in (self.cache_dir / "index.jsonl").open(encoding="utf-8")]
    self.row_ids = list(row_ids)
    self.gt = np.asarray(gt_group_mean, dtype=np.float32)
    self._store = frame_store
    self.offsets = list(frame_offsets or FDEC_FRAME_OFFSETS)
    self.flip_prob = float(flip_prob) if train else 0.0
    self.train = bool(train)
    self._H_arr: np.ndarray | None = None

  def materialize_teacher_cache(self) -> None:
    if self._H_arr is not None:
      return
    with np.load(self.cache_dir / "teacher_pred.npz") as z:
      self._H_arr = np.asarray(z["H"], dtype=np.float32)

  def __len__(self) -> int:
    return len(self.row_ids)

  def __getitem__(self, i: int) -> dict:
    rid = int(self.row_ids[i])
    row = self.rows[rid]
    cid = row["clip_id"]
    t = int(row["frame"])
    idxs = indices_at_anchor(t, self.offsets)
    stack = [np.asarray(self._store.get(cid, fi)) for fi in idxs]
    clip_u8 = np.stack(stack, axis=0)
    gt = np.asarray(self.gt[rid], dtype=np.float32)
    h_teacher = np.asarray(self._H_arr[rid], dtype=np.float32)
    do_flip = self.train and self.flip_prob > 0 and float(torch.rand(()).item()) < self.flip_prob
    if do_flip:
      clip_u8 = clip_u8[..., ::-1, :].copy()
      gt = flip_heatmap_columns(gt)
    clip = uint8_to_nchw01(clip_u8)
    peak = float(gt.max())
    sw = 1.0  # scenario weight applied in collate
    sample = {
      "clip_id": cid,
      "frame": t,
      "row_id": rid,
      "sample_id": row.get("sample_id", ""),
      "clip": torch.from_numpy(clip),
      "gt_1s": torch.from_numpy(gt),
      "pre_fork": bool(row.get("pre_fork")),
      "amb_1s": bool(row.get("amb_1s")),
      "p_gt": peak,
      "scenario_weight": sw,
      "do_kd": not do_flip,
      "background_pair_id": row.get("background_pair_id") or "",
      "outcome_type": row.get("outcome", ""),
      "true_side": row.get("true_side", ""),
      "flipped": do_flip,
    }
    sample["H_teacher"] = torch.from_numpy(h_teacher)
    return sample


def collate_student_final(batch: list[dict]) -> dict:
  keys = batch[0].keys()
  out: dict = {}
  for k in keys:
    if k == "clip":
      out[k] = torch.stack([b[k] for b in batch], dim=0)
    elif k in ("gt_1s", "H_teacher"):
      out[k] = torch.stack([b[k] for b in batch], dim=0)
    elif k in ("pre_fork", "amb_1s", "do_kd", "flipped"):
      out[k] = torch.tensor([bool(b[k]) for b in batch], dtype=torch.bool)
    elif k in ("p_gt", "scenario_weight"):
      out[k] = torch.tensor([float(b[k]) for b in batch], dtype=torch.float32)
    elif k in ("row_id", "frame"):
      out[k] = torch.tensor([int(b[k]) for b in batch], dtype=torch.long)
    else:
      out[k] = [b[k] for b in batch]
  # equalize per-sample group weight within batch
  sids = [str(b["sample_id"]) for b in batch]
  counts: dict[str, int] = {}
  for sid in sids:
    counts[sid] = counts.get(sid, 0) + 1
  weights = torch.tensor([1.0 / counts[sid] for sid in sids], dtype=torch.float32)
  weights = weights * (len(batch) / weights.sum().clamp_min(1e-6))
  out["group_weight"] = weights
  return out
