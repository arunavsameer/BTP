"""Construct matched-background pair batches (canonical row + mate row)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

from ..training.pair_signals import is_canonical_background


def _index_positions(index: Sequence[tuple[str, int]]) -> dict[tuple[str, int], int]:
    return {k: i for i, k in enumerate(index)}


def build_pair_batch_indices(
    dataset,
) -> tuple[list[tuple[int, int]], list[int], np.ndarray]:
    """Return (list of (i_canon, i_mate), unpaired indices, per-row gt_valid bool).

    ``gt_valid[i]`` is True when mate exists at same anchor and max |Δ h_1s| < 1e-3.
    """
    index = dataset.index
    pos = _index_positions(index)
    tol = float(dataset.bg_tol)
    pairs: list[tuple[int, int]] = []
    covered: set[int] = set()

    for i, (clip_id, t) in enumerate(index):
        if i in covered:
            continue
        rec = dataset.records_by_id[clip_id]
        mate_id = dataset._mate_clip_id(rec)
        if not mate_id:
            continue
        j = pos.get((mate_id, int(t)))
        if j is None:
            continue
        # Ensure one row is bg_00 / canonical; swap if the mate is canonical.
        if is_canonical_background(rec.background_id or "", clip_id):
            i_c, i_m = int(i), int(j)
        elif is_canonical_background(
            dataset.records_by_id[mate_id].background_id or "", mate_id
        ):
            i_c, i_m = int(j), int(i)
        else:
            continue
        tga, _ = dataset._training_targets(index[i_c][0], int(index[i_c][1]))
        tgb, _ = dataset._training_targets(index[i_m][0], int(index[i_m][1]))
        pairs.append((i_c, i_m))
        covered.add(i_c)
        covered.add(i_m)

    unpaired = [k for k in range(len(index)) if k not in covered]
    gt_valid = np.zeros(len(index), dtype=np.bool_)
    for i_c, i_m in pairs:
        tga, _ = dataset._training_targets(index[i_c][0], int(index[i_c][1]))
        tgb, _ = dataset._training_targets(index[i_m][0], int(index[i_m][1]))
        ok = bool(
            np.abs(np.asarray(tga["h_1s"], dtype=np.float64) - np.asarray(tgb["h_1s"], dtype=np.float64)).max()
            < tol
        )
        if ok:
            gt_valid[i_c] = True
            gt_valid[i_m] = True
    return pairs, unpaired, gt_valid


class PairAwareBatchSampler(Sampler[list[int]]):
    """Yields lists of dataset indices: ``[canon, mate, …]`` then shuffled unpaired."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        unpaired: list[int],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool,
    ):
        if batch_size % 2 != 0:
            raise ValueError("pair batching requires even micro_batch_size")
        self.pairs = pairs
        self.unpaired = unpaired
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._n_batches = self._count_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _count_batches(self) -> int:
        bs = self.batch_size
        pair_linear_len = 2 * len(self.pairs)
        n = 0
        i = 0
        while i + bs <= pair_linear_len:
            n += 1
            i += bs
        i = 0
        while i + bs <= len(self.unpaired):
            n += 1
            i += bs
        if not self.drop_last:
            if i < len(self.unpaired) and self.unpaired:
                n += 1
        return n

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order_p = list(self.pairs)
        order_u = list(self.unpaired)
        if self.shuffle:
            rng.shuffle(order_p)
            rng.shuffle(order_u)
        pair_linear: list[int] = []
        for a, b in order_p:
            pair_linear.extend([a, b])
        bs = self.batch_size
        chunks: list[list[int]] = []
        # Homogeneous batches: full pair minibatches or full unpaired minibatches only.
        i = 0
        while i + bs <= len(pair_linear):
            chunks.append(pair_linear[i : i + bs])
            i += bs

        i = 0
        while i + bs <= len(order_u):
            chunks.append(order_u[i : i + bs])
            i += bs
        if not self.drop_last and i < len(order_u):
            rest = order_u[i:]
            if rest:
                chunks.append(rest)
        if self.shuffle:
            rng.shuffle(chunks)
        return iter(chunks)
