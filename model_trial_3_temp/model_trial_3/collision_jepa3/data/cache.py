"""Optional resized-frame and frozen-encoder token caches."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .manifest import ClipRecord
from .preprocessing import PreprocessSignature, preprocess_clip, signature_from_cfg, uint8_to_nchw01
from .sampling import FRAME_OFFSETS, anchors_from_cfg, input_indices
from .video_reader import decode_all_rgb

logger = logging.getLogger("collision_jepa3.cache")


def clip_cache_dir(cache_root: Path, clip_id: str) -> Path:
    return Path(cache_root) / "frames" / clip_id.replace("/", "__")


def frame_cache_path(cache_root: Path, clip_id: str) -> Path:
    return clip_cache_dir(cache_root, clip_id) / "frames_uint8.npy"


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def cache_signature_ok(meta: dict, sig: PreprocessSignature, rec: ClipRecord) -> bool:
    if meta.get("resize_mode") != sig.resize_mode:
        return False
    if int(meta.get("input_size", -1)) != int(sig.input_size):
        return False
    if meta.get("video_path") != rec.video_path:
        return False
    return True


def estimate_frame_cache_gb(n_clips: int, n_frames: int, size: int) -> float:
    bytes_ = n_clips * n_frames * size * size * 3
    return bytes_ / (1024**3)


def prepare_frame_cache(
    records: list[ClipRecord],
    cfg,
    *,
    max_gb: float | None = None,
) -> dict:
    sig = signature_from_cfg(cfg)
    cache_root = cfg.path("paths.cache_root")
    valid = [r for r in records if r.valid]
    estimate = estimate_frame_cache_gb(
        len(valid),
        int(cfg.get("data.expected_frames", 450)),
        int(sig.input_size),
    )
    logger.info("frame-cache estimate %.2f GB for %d clips", estimate, len(valid))
    if max_gb is not None and estimate > float(max_gb):
        raise RuntimeError(
            f"frame cache estimate {estimate:.1f} GB exceeds --max-gb {float(max_gb):.1f}"
        )
    done = 0
    skipped = 0
    for rec in valid:
        out_dir = clip_cache_dir(cache_root, rec.clip_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        npy = out_dir / "frames_uint8.npy"
        meta_path = out_dir / "meta.json"
        if npy.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = int(cfg.get("data.expected_frames", 450)) * int(sig.input_size) ** 2 * 3
            if cache_signature_ok(meta, sig, rec) and npy.stat().st_size >= expected:
                skipped += 1
                continue
            logger.info("invalidating stale frame cache for %s", rec.clip_id)
        frames, info = decode_all_rgb(rec.video_path)
        processed = preprocess_clip(frames, sig)
        np.save(npy, processed)
        write_meta(
            meta_path,
            {
                **sig.as_dict(),
                "clip_id": rec.clip_id,
                "video_path": rec.video_path,
                "n_frames": int(processed.shape[0]),
                "shape": list(processed.shape),
                "reported_frames": info.reported_frames,
                "decoded_frames": info.decoded_frames,
                "fps": info.fps,
                "source_width": info.width,
                "source_height": info.height,
            },
        )
        done += 1
        logger.info("cached %s (%d frames)", rec.clip_id, processed.shape[0])
    return {"written": done, "reused": skipped, "estimate_gb": estimate}


def load_cached_frames(cache_root: Path, clip_id: str, mmap: bool = True) -> np.ndarray:
    path = frame_cache_path(cache_root, clip_id)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r" if mmap else None)


def token_fingerprint(cfg, *, dummy: bool = False) -> str:
    blob = {
        "model_id": "dummy" if dummy else cfg.get("backbone.model_id"),
        "input_frames": cfg.get("sampling.input_frames"),
        "frame_offsets": cfg.get("sampling.frame_offsets"),
        "resize_mode": cfg.get("preprocessing.resize_mode"),
        "input_size": cfg.get("preprocessing.input_size"),
        "torch_dtype": cfg.get("backbone.torch_dtype"),
        "skip_predictor": cfg.get("backbone.skip_predictor"),
        "use_get_vision_features": cfg.get("backbone.use_get_vision_features"),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def token_cache_id(cfg, *, dummy: bool = False, stride: int | None = None) -> str:
    """Directory id for encoder tokens. Group-random and non-15 stride get a suffix."""
    fp = token_fingerprint(cfg, dummy=dummy)
    mode = str(cfg.get("sampling.anchor_mode", "stride"))
    if mode == "group_random":
        policy = {
            "mode": "group_random",
            "n_pre": int(cfg.get("sampling.n_pre_fork_anchors", 26)),
            "n_post": int(cfg.get("sampling.n_post_fork_anchors", 14)),
            "t_min": int(cfg.get("sampling.random_anchor_t_min", 90)),
            "t_max": int(cfg.get("sampling.random_anchor_t_max", 419)),
            "fork": int(cfg.get("sampling.fork_frame", 365)),
            "seed": int(cfg.get("project.seed", 1337)),
        }
        ph = hashlib.sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        return f"{fp}_gr{ph}"
    s = int(cfg.get("sampling.train_anchor_stride", 15) if stride is None else stride)
    if s != 15:
        return f"{fp}_s{s}"
    return fp


def estimate_token_cache_gb(n_windows: int, n_tokens: int, hidden: int, itemsize: int = 2) -> float:
    return n_windows * n_tokens * hidden * itemsize / (1024**3)


class TokenCache:
    """Per-clip FP16 encoder-token arrays. Invalid if backbone/preprocess change."""

    def __init__(self, root: Path, fingerprint: str):
        self.root = Path(root) / "tokens" / fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def clip_dir(self, clip_id: str) -> Path:
        return self.root / clip_id.replace("/", "__")

    def npy_path(self, clip_id: str) -> Path:
        return self.clip_dir(clip_id) / "tokens_fp16.npy"

    def meta_path(self, clip_id: str) -> Path:
        return self.clip_dir(clip_id) / "meta.json"

    def has_clip(self, clip_id: str, anchors: list[int], n_tokens: int, hidden: int) -> bool:
        npy = self.npy_path(clip_id)
        meta_p = self.meta_path(clip_id)
        if not npy.exists() or not meta_p.exists():
            return False
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if list(meta.get("anchors", [])) != [int(t) for t in anchors]:
            return False
        if list(meta.get("shape", [])) != [len(anchors), int(n_tokens), int(hidden)]:
            return False
        expected = len(anchors) * int(n_tokens) * int(hidden) * 2
        return npy.stat().st_size >= expected

    def save_clip(self, clip_id: str, anchors: list[int], tokens: np.ndarray) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"expected [n_anchors, n_tokens, C], got {tokens.shape}")
        if tokens.shape[0] != len(anchors):
            raise ValueError(f"anchor count {len(anchors)} != token rows {tokens.shape[0]}")
        d = self.clip_dir(clip_id)
        d.mkdir(parents=True, exist_ok=True)
        arr = np.ascontiguousarray(tokens.astype(np.float16, copy=False))
        np.save(self.npy_path(clip_id), arr)
        write_meta(
            self.meta_path(clip_id),
            {
                "clip_id": clip_id,
                "anchors": [int(t) for t in anchors],
                "shape": list(arr.shape),
                "dtype": "float16",
                "fingerprint": self.fingerprint,
            },
        )

    def load_clip(self, clip_id: str, mmap: bool = True) -> tuple[np.ndarray, dict[int, int]]:
        npy = self.npy_path(clip_id)
        meta = json.loads(self.meta_path(clip_id).read_text(encoding="utf-8"))
        arr = np.load(npy, mmap_mode="r" if mmap else None)
        mapping = {int(t): i for i, t in enumerate(meta["anchors"])}
        return arr, mapping

    def key_path(self, clip_id: str, anchor: int) -> Path:
        return self.npy_path(clip_id)

    def has(self, clip_id: str, anchor: int) -> bool:
        return self.npy_path(clip_id).exists() and self.meta_path(clip_id).exists()

    def save(self, clip_id: str, anchor: int, tokens: np.ndarray) -> None:
        raise RuntimeError("use save_clip() for per-clip token arrays")

    def load(self, clip_id: str, anchor: int) -> np.ndarray:
        arr, mapping = self.load_clip(clip_id, mmap=True)
        return np.asarray(arr[mapping[int(anchor)]])


def _clip_frames_uint8(rec: ClipRecord, cfg, sig: PreprocessSignature) -> np.ndarray:
    """Load preprocessed uint8 frames. Decode preview.mp4 when the frame cache is absent."""
    cache_root = cfg.path("paths.cache_root")
    npy = frame_cache_path(cache_root, rec.clip_id)
    if bool(cfg.get("cache.cache_frames", False)) and npy.exists():
        return load_cached_frames(cache_root, rec.clip_id, mmap=True)
    frames, _info = decode_all_rgb(rec.video_path)
    return preprocess_clip(frames, sig)


def prepare_token_cache(records: list[ClipRecord], cfg, model, device: str) -> dict:
    """Encode each valid clip's windows once. Resumable. No size cap. No frame files."""
    import torch

    valid = [r for r in records if r.valid]
    dummy = str(getattr(model.backbone, "model_id", "")) == "dummy"
    cache = TokenCache(cfg.path("paths.cache_root"), token_cache_id(cfg, dummy=dummy))
    n_frames = int(cfg.get("data.expected_frames", 450))
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    sig = signature_from_cfg(cfg)
    num_frames = int(cfg.get("sampling.input_frames", 16))
    if hasattr(model.backbone, "expected_tokens"):
        _t, _s, n_tokens = model.backbone.expected_tokens(num_frames)
    else:
        n_tokens = int(model.backbone.temporal_groups) * int(model.backbone.spatial) ** 2
    hidden = int(model.backbone.hidden_size)
    group_anchors: dict[str, list[int]] = {}
    clip_anchors: dict[str, list[int]] = {}
    for rec in valid:
        gid = rec.scenario_group_id
        if gid not in group_anchors:
            group_anchors[gid] = anchors_from_cfg(cfg, gid, n_frames=n_frames)
        clip_anchors[rec.clip_id] = group_anchors[gid]
    n_windows = sum(len(a) for a in clip_anchors.values())
    estimate = estimate_token_cache_gb(n_windows, n_tokens, hidden)
    logger.info(
        "token-cache estimate %.2f GB for %d clips (%d windows, no size cap) fp=%s",
        estimate,
        len(valid),
        n_windows,
        cache.fingerprint,
    )
    batch_size = max(1, int(cfg.get("cache.token_encode_batch_size", cfg.get("training.micro_batch_size", 8))))
    encode = model.encode_tokens if hasattr(model, "encode_tokens") else model
    written = 0
    skipped = 0
    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for rec in valid:
            anchors = clip_anchors[rec.clip_id]
            if cache.has_clip(rec.clip_id, anchors, n_tokens, hidden):
                skipped += 1
                continue
            frames_u8 = _clip_frames_uint8(rec, cfg, sig)
            rows = []
            for start in range(0, len(anchors), batch_size):
                chunk = anchors[start : start + batch_size]
                clips = []
                for t in chunk:
                    idx = input_indices(t, offsets)
                    window = np.stack([np.asarray(frames_u8[i]) for i in idx], axis=0)
                    clips.append(uint8_to_nchw01(window))
                x = torch.from_numpy(np.stack(clips, axis=0)).to(device)
                with torch.no_grad():
                    if device.startswith("cuda"):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            tok = encode(x)
                    else:
                        tok = encode(x)
                rows.append(tok.float().cpu().numpy())
            stacked = np.concatenate(rows, axis=0)
            if tuple(stacked.shape) != (len(anchors), n_tokens, hidden):
                raise RuntimeError(
                    f"{rec.clip_id}: token shape {stacked.shape} != "
                    f"({len(anchors)}, {n_tokens}, {hidden})"
                )
            cache.save_clip(rec.clip_id, anchors, stacked)
            written += 1
            logger.info("token-cached %s %s n=%d", rec.clip_id, tuple(stacked.shape), written + skipped)
    finally:
        if model_was_training:
            model.train()
            if hasattr(model, "backbone"):
                model.backbone.eval()
    return {
        "written": written,
        "reused": skipped,
        "estimate_gb": estimate,
        "fingerprint": cache.fingerprint,
        "n_clips": len(valid),
        "n_windows": n_windows,
        "token_shape": [n_tokens, hidden],
    }
