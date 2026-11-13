"""At most one window per sample_id per batch; cycle anchors across epochs."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler


class SampleGroupedBatchSampler(Sampler[list[int]]):
  def __init__(
    self,
    dataset,
    row_ids: list[int],
    sample_ids: list[str],
    *,
    batch_size: int,
    seed: int,
    epoch: int = 0,
  ):
    self.dataset = dataset
    self.batch_size = int(batch_size)
    self.seed = int(seed)
    self.epoch = int(epoch)
    by_sample: dict[str, list[int]] = defaultdict(list)
    for idx, (rid, sid) in enumerate(zip(row_ids, sample_ids)):
      by_sample[str(sid)].append(idx)
    self._by_sample = {k: sorted(v) for k, v in by_sample.items()}
    self._samples = sorted(self._by_sample.keys())

  def set_epoch(self, epoch: int) -> None:
    self.epoch = int(epoch)

  def __iter__(self) -> Iterator[list[int]]:
    rng = random.Random(self.seed + self.epoch * 9973)
    order = list(self._samples)
    rng.shuffle(order)
    ptr: dict[str, int] = {s: (self.epoch + rng.randint(0, 999)) % len(idxs) for s, idxs in self._by_sample.items()}
    batch: list[int] = []
    active = list(order)
    while active:
      rng.shuffle(active)
      next_active: list[str] = []
      for sid in active:
        idxs = self._by_sample[sid]
        if not idxs:
          continue
        i = idxs[ptr[sid] % len(idxs)]
        ptr[sid] = (ptr[sid] + 1) % len(idxs)
        batch.append(i)
        if len(batch) >= self.batch_size:
          yield batch
          batch = []
        if ptr[sid] != 0 or len(idxs) > 1:
          next_active.append(sid)
      active = next_active
      if len(active) == 0 and batch:
        yield batch
        batch = []

  def __len__(self) -> int:
    return max(1, (len(self._samples) + self.batch_size - 1) // self.batch_size)
