"""Pair-aware index helpers. Matching is by IDs, never by list position."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Sequence

from .manifest import ClipRecord


def index_by_clip_id(records: Iterable[ClipRecord]) -> dict[str, ClipRecord]:
    return {r.clip_id: r for r in records}


def background_partner_id(rec: ClipRecord, records: list[ClipRecord]) -> str | None:
    mates = [
        r
        for r in records
        if r.valid
        and r.background_pair_id == rec.background_pair_id
        and r.clip_id != rec.clip_id
    ]
    if len(mates) != 1:
        return mates[0].clip_id if mates else None
    return mates[0].clip_id


def outcome_partner_ids(rec: ClipRecord, records: list[ClipRecord]) -> list[str]:
    return [
        r.clip_id
        for r in records
        if r.valid
        and r.outcome_group_id == rec.outcome_group_id
        and r.clip_id != rec.clip_id
    ]


def group_indices_by_clip(index: list[tuple[str, int]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for clip_id, t in index:
        grouped[clip_id].append(t)
    return grouped


def group_row_indices_by_clip(index: Sequence[tuple[str, int]]) -> dict[str, list[int]]:
    """Map clip_id → dataset row indices, preserving first-seen clip order."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, (clip_id, _t) in enumerate(index):
        grouped[clip_id].append(i)
    return grouped


class ClipContiguousSampler:
    """Yield all anchors of a clip before moving to the next clip.

    Shuffle permutes clip order each epoch; order inside a clip stays time-sorted.
    """

    def __init__(self, index: Sequence[tuple[str, int]], *, shuffle: bool, seed: int = 1337):
        grouped = group_row_indices_by_clip(index)
        self.groups = [grouped[cid] for cid in grouped]
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.n = len(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        groups = list(self.groups)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(groups)
        for rows in groups:
            yield from rows
