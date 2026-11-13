"""In-memory sparse RGB frames for F_dec pilot (no disk RGB cache)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from collision_jepa3.data.manifest import ClipRecord, load_manifest
from collision_jepa3.data.preprocessing import letterbox_rgb, preprocess_clip, signature_from_cfg, uint8_to_nchw01
from collision_jepa3.data.video_reader import _decode_until

from .dataset_pilot import collate_pilot
from .frames import indices_at_anchor

STUDENT_INPUT_SIZE = 160


def _decode_clip_frames(job: tuple[str, list[int], str, dict, int]) -> list[tuple[str, int, np.ndarray]]:
    cid, idxs, vpath, sig_dict, student_size = job
    from collision_jepa3.data.preprocessing import PreprocessSignature, preprocess_clip

    s = PreprocessSignature(**sig_dict)
    raw, _ = _decode_until(vpath, max(idxs))
    proc = preprocess_clip(raw, s)
    out = []
    for fi in idxs:
        fi_c = min(fi, proc.shape[0] - 1)
        fr = proc[fi_c]
        if fr.shape[0] != student_size or fr.shape[1] != student_size:
            fr = letterbox_rgb(fr, student_size)
        out.append((cid, fi, np.asarray(fr, dtype=np.uint8).copy()))
    return out


def letterbox_stored_frame(fr: np.ndarray, student_size: int = STUDENT_INPUT_SIZE) -> np.ndarray:
    """Legacy per-sample path: 256×256 (parent preprocess) → student_size."""
    fr = np.asarray(fr)
    if fr.shape[0] == student_size and fr.shape[1] == student_size:
        return fr
    return letterbox_rgb(fr, student_size)


class SparseClipFrameStore:
    """Per-clip decode of only the frame indices required by the pilot manifest."""

    def __init__(self, cfg, *, student_input_size: int = STUDENT_INPUT_SIZE):
        self.cfg = cfg
        self.student_input_size = int(student_input_size)
        self._records: dict[str, ClipRecord] = {
            r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid
        }
        self._sig = signature_from_cfg(cfg)
        self._frames: dict[tuple[str, int], np.ndarray] = {}
        self._frames_raw: dict[tuple[str, int], np.ndarray] = {}
        self.decode_time_sec = 0.0
        self.letterbox_time_sec = 0.0
        self.n_clips_loaded = 0

    def prepare(
        self,
        rows: list[dict],
        row_ids: list[int],
        offsets: list[int],
        *,
        workers: int = 8,
        store_letterboxed: bool = True,
    ) -> dict:
        needed: dict[str, set[int]] = defaultdict(set)
        for rid in row_ids:
            row = rows[rid]
            cid = row["clip_id"]
            t = int(row["frame"])
            for fi in indices_at_anchor(t, offsets):
                needed[cid].add(max(0, fi))
        t0 = time.perf_counter()
        jobs: list[tuple[str, list[int], str]] = []
        for cid, idxs in sorted(needed.items()):
            rec = self._records.get(cid)
            if rec is None:
                continue
            jobs.append((cid, sorted(idxs), str(rec.video_path)))

        sig = self._sig.as_dict()
        if not store_letterboxed:
            raise ValueError("store_letterboxed=False only for parity harness (use _frames_raw)")
        packed = [(cid, idxs, vpath, sig, self.student_input_size) for cid, idxs, vpath in jobs]

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_decode_clip_frames, j) for j in packed]
                for fut in as_completed(futs):
                    for cid, fi, fr in fut.result():
                        self._frames[(cid, fi)] = fr
                    self.n_clips_loaded += 1
                    if self.n_clips_loaded % 50 == 0:
                        print(f"rgb_prepare clips {self.n_clips_loaded}/{len(packed)}", flush=True)
        else:
            for job in packed:
                for cid, fi, fr in _decode_clip_frames(job):
                    self._frames[(cid, fi)] = fr
                self.n_clips_loaded += 1
        self.decode_time_sec = time.perf_counter() - t0
        return {
            "n_clips_decoded": self.n_clips_loaded,
            "n_frames_stored": len(self._frames),
            "rgb_prepare_sec": self.decode_time_sec,
            "letterbox_at_prepare": True,
            "stored_shape_sample": list(next(iter(self._frames.values())).shape) if self._frames else None,
            "student_input_size": self.student_input_size,
        }

    def get(self, cid: str, fi: int) -> np.ndarray:
        fi = max(0, fi)
        key = (cid, fi)
        if key not in self._frames:
            raise KeyError(f"missing frame {key}")
        return self._frames[key]

    def get_raw_preprocess(self, cid: str, fi: int) -> np.ndarray:
        fi = max(0, fi)
        key = (cid, fi)
        if key in self._frames_raw:
            return self._frames_raw[key]
        fr = self._frames[key]
        if self._frames_raw:
            raise KeyError(f"missing raw frame {key}")
        return fr


class FDecPilotDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        row_ids: list[int],
        frame_offsets: list[int],
        *,
        input_size: int = STUDENT_INPUT_SIZE,
        frame_store: SparseClipFrameStore | None = None,
        load_f_dec: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.rows = [json.loads(l) for l in (self.cache_dir / "index.jsonl").open()]
        self.pred = np.load(self.cache_dir / "teacher_pred.npz", mmap_mode="r")
        self.gt = np.load(self.cache_dir / "gt_heatmaps.npz", mmap_mode="r")
        self.row_ids = list(row_ids)
        self.offsets = list(frame_offsets)
        self.input_size = int(input_size)
        self._store = frame_store
        self.load_f_dec = bool(load_f_dec)
        self._prepare_stats: dict | None = None
        self._H_arr: np.ndarray | None = None
        self._F_dec_arr: np.ndarray | None = None
        self._gt_arr: np.ndarray | None = None

    def materialize_teacher_cache(self) -> None:
        """Load teacher targets into RAM (mmap npz is unsafe with forked DataLoader workers)."""
        if self._H_arr is not None:
            return
        with np.load(self.cache_dir / "teacher_pred.npz") as z:
            self._H_arr = np.asarray(z["H"], dtype=np.float32)
            if self.load_f_dec:
                self._F_dec_arr = np.asarray(z["F_dec"], dtype=np.float32)
        with np.load(self.cache_dir / "gt_heatmaps.npz") as z:
            self._gt_arr = np.asarray(z["gt_1s"], dtype=np.float32)
        if self.pred is not None:
            self.pred.close()
            self.pred = None  # type: ignore[assignment]
        if self.gt is not None:
            self.gt.close()
            self.gt = None  # type: ignore[assignment]

    def attach_frame_store(self, store: SparseClipFrameStore, stats: dict) -> None:
        self._store = store
        self._prepare_stats = stats

    def stats(self) -> dict:
        base = dict(self._prepare_stats or {})
        if self._store is not None:
            base["n_frames_stored"] = len(self._store._frames)
        return base

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, i: int) -> dict:
        rid = int(self.row_ids[i])
        row = self.rows[rid]
        cid = row["clip_id"]
        t = int(row["frame"])
        if self._store is None:
            raise RuntimeError("frame store not prepared")
        idxs = indices_at_anchor(t, self.offsets)
        stack = [np.asarray(self._store.get(cid, fi)) for fi in idxs]
        if stack[0].shape[0] != self.input_size:
            raise ValueError(f"expected {self.input_size}px frames at prepare time, got {stack[0].shape}")
        clip = uint8_to_nchw01(np.stack(stack, axis=0))
        sample = {
            "clip_id": cid,
            "frame": t,
            "row_id": rid,
            "clip": torch.from_numpy(clip),
            "gt_1s": torch.from_numpy(
                self._gt_arr[rid] if self._gt_arr is not None else np.asarray(self.gt["gt_1s"][rid], dtype=np.float32)
            ),
            "H_teacher": torch.from_numpy(
                self._H_arr[rid] if self._H_arr is not None else np.asarray(self.pred["H"][rid], dtype=np.float32)
            ),
            "pre_fork": bool(row.get("pre_fork")),
            "background_pair_id": row.get("background_pair_id") or "",
            "outcome_type": row.get("outcome", row.get("outcome_type", "")),
            "scenario_type": row.get("scenario_type", ""),
            "scenario_group_id": row.get("scenario_group_id", ""),
        }
        if self.load_f_dec:
            if self._F_dec_arr is not None:
                sample["F_teacher"] = torch.from_numpy(self._F_dec_arr[rid])
            else:
                sample["F_teacher"] = torch.from_numpy(np.asarray(self.pred["F_dec"][rid], dtype=np.float32))
        return sample


def rgb_letterbox_parity_test(
    cfg,
    cache_dir: Path,
    row_ids: list[int],
    offsets: list[int],
    *,
    n: int = 128,
    seed: int = 1337,
) -> dict:
    """Compare legacy __getitem__ letterbox vs prepare-time letterbox on stored 256 frames."""
    rows = [json.loads(l) for l in (Path(cache_dir) / "index.jsonl").open()]
    rng = np.random.default_rng(seed)
    pick = row_ids if len(row_ids) <= n else list(rng.choice(row_ids, size=n, replace=False))
    raw_store = SparseClipFrameStore(cfg)
    jobs_needed: dict[str, set[int]] = defaultdict(set)
    for rid in pick:
        row = rows[rid]
        cid = row["clip_id"]
        t = int(row["frame"])
        for fi in indices_at_anchor(t, offsets):
            jobs_needed[cid].add(max(0, fi))
    sig = raw_store._sig.as_dict()
    from collision_jepa3.data.preprocessing import PreprocessSignature

    for cid, idxs in jobs_needed.items():
        rec = raw_store._records.get(cid)
        if rec is None:
            continue
        raw, _ = _decode_until(str(rec.video_path), max(idxs))
        proc = preprocess_clip(raw, PreprocessSignature(**sig))
        for fi in idxs:
            fi_c = min(fi, proc.shape[0] - 1)
            raw_store._frames_raw[(cid, fi)] = proc[fi_c].copy()

    opt_store = SparseClipFrameStore(cfg)
    opt_store.prepare(rows, pick, offsets, workers=1, store_letterboxed=True)

    max_abs, mean_abs, n_pix = 0.0, 0.0, 0
    for rid in pick:
        row = rows[rid]
        cid, t = row["clip_id"], int(row["frame"])
        for fi in indices_at_anchor(t, offsets):
            leg = letterbox_stored_frame(raw_store.get_raw_preprocess(cid, fi), STUDENT_INPUT_SIZE)
            new = opt_store.get(cid, fi)
            diff = np.abs(leg.astype(np.int16) - new.astype(np.int16))
            max_abs = max(max_abs, float(diff.max()))
            mean_abs += float(diff.mean())
            n_pix += 1
    mean_abs /= max(n_pix, 1)
    return {
        "n_frames_checked": n_pix,
        "n_rows": len(pick),
        "max_abs_pixel_diff": max_abs,
        "mean_abs_pixel_diff": mean_abs,
        "exact_match": max_abs == 0.0,
        "stored_shape": list(opt_store.get(rows[pick[0]]["clip_id"], indices_at_anchor(int(rows[pick[0]]["frame"]), offsets)[0]).shape),
    }


def build_frame_store(cfg, cache_dir: Path, row_ids: list[int], offsets: list[int], *, prepare_workers: int = 8) -> tuple[SparseClipFrameStore, dict]:
    rows = [json.loads(l) for l in (Path(cache_dir) / "index.jsonl").open()]
    store = SparseClipFrameStore(cfg)
    stats = store.prepare(rows, row_ids, offsets, workers=prepare_workers, store_letterboxed=True)
    return store, stats


def make_fdec_loader(
    ds: FDecPilotDataset,
    batch: int,
    shuffle: bool,
    *,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    kw: dict = {
        "batch_size": batch,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_pilot,
        "pin_memory": pin,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    if num_workers > 0:
        ds.materialize_teacher_cache()
    return DataLoader(ds, **kw)
