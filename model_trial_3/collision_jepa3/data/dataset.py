"""Teacher dataset: 16-frame clips ending at t, plus current and future heatmaps."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore
        pass

from .cache import TokenCache, frame_cache_path, load_cached_frames, token_fingerprint
from .heatmap_annotations import load_heatmaps, targets_for_anchor
from .manifest import ClipRecord
from .pairing import (
    apply_collision_future_targets,
    apply_mean_future_targets,
    background_pair_valid,
    collision_clip_by_outcome_group,
    compute_ambiguity_table,
    outcome_clip_ids_by_group,
)
from .preprocessing import preprocess_clip, signature_from_cfg, uint8_to_nchw01
from .paired_sampler import ClipContiguousSampler
from .sampling import FRAME_OFFSETS, HORIZONS_FRAMES, available_horizons, input_indices, iter_anchors
from .video_reader import decode_indices_rgb


class TeacherDataset(Dataset):
    def __init__(
        self,
        cfg,
        records: list[ClipRecord],
        split: str,
        splits: dict[str, list[str]],
        *,
        use_frame_cache: bool | None = None,
        dummy_backbone: bool = False,
        use_token_cache: bool | None = None,
    ):
        self.cfg = cfg
        self.split = split
        self.sig = signature_from_cfg(cfg)
        self.offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
        self.horizons = list(cfg.get("sampling.horizons_frames", HORIZONS_FRAMES))
        self.horizon_names = list(cfg.get("sampling.horizon_names", ["1s", "1_5s", "2s"]))
        self.fork_frame = int(cfg.get("sampling.fork_frame", 365))
        self.cache_root = cfg.path("paths.cache_root")
        self.use_frame_cache = (
            bool(cfg.get("cache.cache_frames", True)) if use_frame_cache is None else use_frame_cache
        )
        groups = set(splits.get(split, []))
        self.records = [r for r in records if r.valid and r.scenario_group_id in groups]
        self.records_by_id = {r.clip_id: r for r in self.records}
        n_frames = int(cfg.get("data.expected_frames", 450))
        stride_key = {
            "train": "sampling.train_anchor_stride",
            "val": "sampling.val_anchor_stride",
            "test": "sampling.test_anchor_stride",
        }[split]
        self.stride = int(cfg.get(stride_key, 15))
        self.allow_partial_horizons = bool(cfg.get("sampling.allow_partial_horizons", True))
        self.n_frames = n_frames
        self.anchors = iter_anchors(
            n_frames,
            self.stride,
            self.offsets,
            self.horizons,
            allow_partial_horizons=self.allow_partial_horizons,
        )

        self._heatmaps: dict[str, np.ndarray] = {}
        for rec in self.records:
            self._heatmaps[rec.clip_id] = load_heatmaps(
                rec.heatmap_json_path,
                expected_frames=n_frames,
                grid=int(cfg.get("data.target_grid", 5)),
            )

        self.ambiguous_policy = str(cfg.get("sampling.ambiguous_policy", "mean_target"))
        detect_ambiguous = self.ambiguous_policy in {
            "mean_target",
            "collision_target",
            "mask_ambiguous_pairs",
        } or bool(cfg.get("sampling.mask_ambiguous_pairs", True))
        self.ambiguity = compute_ambiguity_table(
            self._heatmaps,
            self.records,
            fork_frame=self.fork_frame,
            n_frames=n_frames,
            stride=self.stride,
            offsets=self.offsets,
            horizons=self.horizons,
            horizon_names=self.horizon_names,
            heatmap_tol=float(cfg.get("sampling.heatmap_disagree_tol", 0.02)),
            mask_ambiguous=detect_ambiguous,
            allow_partial_horizons=self.allow_partial_horizons,
        )
        self._collision_ids = collision_clip_by_outcome_group(self.records)
        self._outcome_ids = outcome_clip_ids_by_group(self.records)

        self.index: list[tuple[str, int]] = []
        for rec in self.records:
            for t in self.anchors:
                self.index.append((rec.clip_id, int(t)))

        self._lru: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lru_limit = int(cfg.get("cache.lru_clips_in_ram", 4))
        self.bg_tol = float(cfg.get("sampling.background_target_tolerance", 0.001))
        self.use_token_cache = (
            bool(cfg.get("cache.cache_encoder_tokens", False)) if use_token_cache is None else bool(use_token_cache)
        )
        self._token_cache: TokenCache | None = None
        self._token_lru: OrderedDict[str, tuple[np.ndarray, dict[int, int]]] = OrderedDict()
        if self.use_token_cache:
            self._token_cache = TokenCache(
                self.cache_root, token_fingerprint(cfg, dummy=bool(dummy_backbone))
            )

    def _horizon_masked(
        self,
        name: str,
        amb: dict[str, bool],
        avail: dict[str, bool],
        replaced: set[str],
    ) -> bool:
        if not avail.get(name, True):
            return True
        if name in replaced:
            return False
        return bool(amb.get(name, False))

    def __len__(self) -> int:
        return len(self.index)

    def clip_ids(self) -> list[str]:
        seen: list[str] = []
        got: set[str] = set()
        for cid, _t in self.index:
            if cid not in got:
                got.add(cid)
                seen.append(cid)
        return seen

    def subsample_clips(self, fraction: float, seed: int) -> None:
        """Keep a seeded random subset of clips (all anchors of each kept clip)."""
        frac = float(fraction)
        if frac >= 1.0:
            return
        if frac <= 0.0:
            raise ValueError("val_subset_fraction must be > 0")
        ids = self.clip_ids()
        n = max(1, int(round(len(ids) * frac)))
        n = min(n, len(ids))
        rng = np.random.RandomState(int(seed))
        chosen = set(rng.choice(np.array(ids, dtype=object), size=n, replace=False).tolist())
        self.index = [(c, t) for c, t in self.index if c in chosen]

    def ambiguity_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.horizon_names}
        total = 0
        for flags in self.ambiguity.values():
            total += 1
            for name in self.horizon_names:
                if flags.get(name):
                    counts[name] += 1
        counts["n_pairs"] = total
        return counts

    def _frames_for_clip(self, rec: ClipRecord) -> np.ndarray:
        if rec.clip_id in self._lru:
            self._lru.move_to_end(rec.clip_id)
            return self._lru[rec.clip_id]
        cached = frame_cache_path(self.cache_root, rec.clip_id)
        if self.use_frame_cache and cached.exists():
            arr = load_cached_frames(self.cache_root, rec.clip_id, mmap=True)
        else:
            arr = None
        if arr is None:
            # Fall back: decode on demand is handled per-index, not whole clip.
            raise FileNotFoundError
        self._lru[rec.clip_id] = arr
        while len(self._lru) > self._lru_limit:
            self._lru.popitem(last=False)
        return arr

    def _load_input_clip(self, rec: ClipRecord, t: int) -> np.ndarray:
        idx = input_indices(t, self.offsets)
        try:
            frames = self._frames_for_clip(rec)
            clip = np.stack([np.asarray(frames[i]) for i in idx], axis=0)
            if clip.dtype != np.uint8:
                clip = clip.astype(np.uint8)
            if clip.shape[1] != self.sig.input_size:
                clip = preprocess_clip(clip, self.sig)
        except FileNotFoundError:
            raw, _ = decode_indices_rgb(rec.video_path, idx)
            clip = preprocess_clip(raw, self.sig)
        return uint8_to_nchw01(clip)

    def _tokens_for_clip(self, clip_id: str) -> tuple[np.ndarray, dict[int, int]]:
        if clip_id in self._token_lru:
            self._token_lru.move_to_end(clip_id)
            return self._token_lru[clip_id]
        if self._token_cache is None:
            raise FileNotFoundError(clip_id)
        arr, mapping = self._token_cache.load_clip(clip_id, mmap=True)
        self._token_lru[clip_id] = (arr, mapping)
        while len(self._token_lru) > self._lru_limit:
            self._token_lru.popitem(last=False)
        return arr, mapping

    def _load_tokens(self, clip_id: str, t: int) -> np.ndarray:
        arr, mapping = self._tokens_for_clip(clip_id)
        row = mapping[int(t)]
        return np.asarray(arr[row], dtype=np.float32)

    def _mate_clip_id(self, rec: ClipRecord) -> str | None:
        mates = [
            r.clip_id
            for r in self.records
            if r.background_pair_id == rec.background_pair_id and r.clip_id != rec.clip_id
        ]
        return mates[0] if len(mates) == 1 else None

    def __getitem__(self, i: int) -> dict[str, Any]:
        clip_id, t = self.index[i]
        rec = self.records_by_id[clip_id]
        tokens = None
        frames = None
        if self.use_token_cache and self._token_cache is not None:
            try:
                tokens = self._load_tokens(clip_id, t)
            except (FileNotFoundError, KeyError, OSError):
                tokens = None
        if tokens is None:
            frames = self._load_input_clip(rec, t)
        tg = targets_for_anchor(
            self._heatmaps[clip_id],
            t,
            self.horizons,
            allow_missing=self.allow_partial_horizons,
        )
        amb = self.ambiguity.get((clip_id, t), {n: False for n in self.horizon_names})
        avail = available_horizons(t, self.n_frames, self.horizons)
        replaced: set[str] = set()
        if self.ambiguous_policy == "mean_target":
            tg, replaced = apply_mean_future_targets(
                tg,
                rec=rec,
                anchor=t,
                amb=amb,
                clip_heatmaps=self._heatmaps,
                outcome_ids=self._outcome_ids,
                horizons=self.horizons,
                horizon_names=self.horizon_names,
                available=avail,
            )
        elif self.ambiguous_policy == "collision_target":
            tg, replaced = apply_collision_future_targets(
                tg,
                rec=rec,
                anchor=t,
                amb=amb,
                clip_heatmaps=self._heatmaps,
                collision_ids=self._collision_ids,
                horizons=self.horizons,
                horizon_names=self.horizon_names,
                available=avail,
            )
        mate_id = self._mate_clip_id(rec)
        bg_ok = False
        if mate_id is not None and mate_id in self._heatmaps:
            bg_ok = background_pair_valid(
                rec,
                self.records_by_id[mate_id],
                self._heatmaps[clip_id],
                self._heatmaps[mate_id],
                t,
                self.horizons,
                self.bg_tol,
            )
        fork = int(rec.fork_frame or self.fork_frame)
        sample = {
            "h_now": tg["h_now"],
            "h_1s": tg["h_1s"],
            "h_1_5s": tg["h_1_5s"],
            "h_2s": tg["h_2s"],
            "anchor": np.int64(t),
            "clip_id": clip_id,
            "sample_id": rec.sample_id,
            "scenario_group_id": rec.scenario_group_id,
            "scenario_type": rec.scenario_type,
            "outcome_type": rec.outcome_type,
            "background_id": rec.background_id,
            "outcome_group_id": rec.outcome_group_id or "",
            "background_pair_id": rec.background_pair_id or "",
            "mate_clip_id": mate_id or "",
            "bg_pair_valid": np.bool_(bg_ok),
            "pre_fork": np.bool_(t < fork),
            "fork_frame": np.int64(fork),
            "amb_1s": np.bool_(self._horizon_masked("1s", amb, avail, replaced)),
            "amb_1_5s": np.bool_(self._horizon_masked("1_5s", amb, avail, replaced)),
            "amb_2s": np.bool_(self._horizon_masked("2s", amb, avail, replaced)),
        }
        if tokens is not None:
            sample["tokens"] = tokens
        else:
            sample["frames"] = frames.astype(np.float32, copy=False)
        return sample


def build_dataloader(dataset: TeacherDataset, cfg, *, shuffle: bool, drop_last: bool = False):
    from torch.utils.data import DataLoader

    workers = int(cfg.get("training.num_workers", 0))
    kwargs = {
        "batch_size": int(cfg.get("training.micro_batch_size", 1)),
        "num_workers": workers,
        "pin_memory": bool(cfg.get("training.pin_memory", True)),
        "collate_fn": collate_teacher_batch,
        "drop_last": drop_last,
    }
    if bool(cfg.get("training.clip_contiguous_sampling", True)) and getattr(dataset, "index", None) is not None:
        kwargs["sampler"] = ClipContiguousSampler(
            dataset.index,
            shuffle=shuffle,
            seed=int(cfg.get("project.seed", 1337)),
        )
    else:
        kwargs["shuffle"] = shuffle
    if workers > 0:
        kwargs["prefetch_factor"] = int(cfg.get("training.prefetch_factor", 2))
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def collate_teacher_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    def stack(key: str):
        return torch.from_numpy(np.stack([s[key] for s in samples], axis=0))

    out = {
        "h_now": stack("h_now"),
        "h_1s": stack("h_1s"),
        "h_1_5s": stack("h_1_5s"),
        "h_2s": stack("h_2s"),
        "anchor": stack("anchor"),
        "bg_pair_valid": stack("bg_pair_valid"),
        "pre_fork": stack("pre_fork"),
        "fork_frame": stack("fork_frame"),
        "amb_1s": stack("amb_1s"),
        "amb_1_5s": stack("amb_1_5s"),
        "amb_2s": stack("amb_2s"),
        "clip_id": [s["clip_id"] for s in samples],
        "sample_id": [s["sample_id"] for s in samples],
        "scenario_group_id": [s["scenario_group_id"] for s in samples],
        "scenario_type": [s["scenario_type"] for s in samples],
        "outcome_type": [s["outcome_type"] for s in samples],
        "background_id": [s["background_id"] for s in samples],
        "outcome_group_id": [s["outcome_group_id"] for s in samples],
        "background_pair_id": [s["background_pair_id"] for s in samples],
        "mate_clip_id": [s["mate_clip_id"] for s in samples],
    }
    if "tokens" in samples[0]:
        out["tokens"] = stack("tokens")
    if "frames" in samples[0]:
        out["frames"] = stack("frames")
    return out
