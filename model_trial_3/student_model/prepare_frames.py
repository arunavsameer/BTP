"""Build the 256² uint8 frame cache in parallel. Preview.mp4 only, never overlay."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from collision_jepa3.data.cache import (
    cache_signature_ok,
    clip_cache_dir,
    estimate_frame_cache_gb,
    write_meta,
)
from collision_jepa3.data.preprocessing import PreprocessSignature, preprocess_clip, signature_from_cfg
from collision_jepa3.data.video_reader import decode_all_rgb

logger = logging.getLogger("student_model.prepare_frames")


def _cache_one(payload: dict) -> tuple[str, str]:
    cache_root = Path(payload["cache_root"])
    clip_id = payload["clip_id"]
    video_path = payload["video_path"]
    if "spatial_overlay" in video_path:
        raise RuntimeError(f"refusing overlay as input: {video_path}")
    sig = PreprocessSignature(**payload["sig"])
    expected = int(payload["expected_bytes"])
    out_dir = clip_cache_dir(cache_root, clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / "frames_uint8.npy"
    meta_path = out_dir / "meta.json"
    rec_video = video_path
    if npy.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        class _Rec:
            video_path = rec_video

        if cache_signature_ok(meta, sig, _Rec()) and npy.stat().st_size >= expected:
            return clip_id, "skip"
    frames, info = decode_all_rgb(video_path)
    processed = preprocess_clip(frames, sig)
    tmp = npy.with_name(npy.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, processed)
    tmp.replace(npy)
    write_meta(
        meta_path,
        {
            **sig.as_dict(),
            "clip_id": clip_id,
            "video_path": video_path,
            "n_frames": int(processed.shape[0]),
            "shape": list(processed.shape),
            "reported_frames": info.reported_frames,
            "decoded_frames": info.decoded_frames,
            "fps": info.fps,
            "source_width": info.width,
            "source_height": info.height,
        },
    )
    return clip_id, "write"


def prepare_frame_cache_parallel(records, cfg, *, workers: int = 8, max_gb: float | None = 80.0) -> dict:
    sig = signature_from_cfg(cfg)
    cache_root = cfg.path("paths.cache_root")
    valid = [r for r in records if r.valid]
    estimate = estimate_frame_cache_gb(
        len(valid),
        int(cfg.get("data.expected_frames", 450)),
        int(sig.input_size),
    )
    logger.info("frame-cache estimate %.2f GB for %d clips workers=%d", estimate, len(valid), workers)
    if max_gb is not None and estimate > float(max_gb):
        raise RuntimeError(f"frame cache estimate {estimate:.1f} GB exceeds --max-gb {float(max_gb):.1f}")
    expected = int(cfg.get("data.expected_frames", 450)) * int(sig.input_size) ** 2 * 3
    payloads = [
        {
            "cache_root": str(cache_root),
            "clip_id": rec.clip_id,
            "video_path": rec.video_path,
            "sig": sig.as_dict(),
            "expected_bytes": expected,
        }
        for rec in valid
    ]
    written = 0
    skipped = 0
    n = max(1, int(workers))
    with ProcessPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(_cache_one, p) for p in payloads]
        for i, fut in enumerate(as_completed(futs), 1):
            clip_id, status = fut.result()
            if status == "skip":
                skipped += 1
            else:
                written += 1
            if i % 25 == 0 or i == len(futs):
                logger.info("frame cache %d/%d written=%d reused=%d last=%s", i, len(futs), written, skipped, clip_id)
    return {"written": written, "reused": skipped, "estimate_gb": estimate, "workers": n}
