"""Stage B: cache teacher Z_end[e] for every valid frame (student distillation)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config, resolve_device  # noqa: E402
from cvjepa.data import load_json  # noqa: E402
from cvjepa.data.dataset import teacher_clip_indices  # noqa: E402
from cvjepa.models.teacher import CollisionTeacher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("teacher.device", "auto"))
    hf_cache = Path(cfg.get("teacher.hf_cache_dir", ROOT / "hf_cache"))
    os.environ.setdefault("HF_HOME", str(hf_cache))
    split = load_json(cfg.get("data.split_file"))
    all_eps = sorted(set(split["train"]) | set(split["val"]) | set(split.get("test") or []))
    size = int(cfg.get("teacher.img_size"))
    clip_frames = int(cfg.get("teacher.clip_frames"))
    stride = int(cfg.get("teacher.clip_stride"))
    zc = int(cfg.get("teacher.z_channels"))
    grid = int(cfg.get("student.feature_grid"))
    cache_dir = Path(cfg.get("data.cache_dir"))
    n_frames = int(cfg.get("data.n_frames", 150))
    mb = int(cfg.get("teacher.micro_batch", 1))
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.get("paths.ckpt_dir")) / "teacher_best.pt"

    model = CollisionTeacher(
        hf_model_id=cfg.get("teacher.hf_model_id"),
        z_channels=zc,
        feature_grid=grid,
        pool_hidden=int(cfg.get("teacher.pool_hidden", 128)),
        cache_dir=hf_cache,
        torch_dtype=str(cfg.get("teacher.torch_dtype", "float16")),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if state.get("lora"):
        model.backbone.enable_lora(
            last_n=int(cfg.get("teacher.lora_last_blocks", 6)),
            rank=int(cfg.get("teacher.lora_rank", 16)),
            alpha=float(cfg.get("teacher.lora_alpha", 16.0)),
        )
        model.backbone.set_backbone_grad(False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    print(f"[cacheZ] loaded {ckpt_path} epoch={state.get('epoch')} phase={state.get('phase')}")

    e_lo = (clip_frames - 1) * stride
    for ei, ep in enumerate(all_eps):
        frames = np.load(cache_dir / ep / f"frames_{size}.npy", mmap_mode="r")
        z_end = np.zeros((n_frames, zc, grid, grid), dtype=np.float32)
        mask = np.zeros((n_frames,), dtype=bool)
        ends = list(range(e_lo, min(n_frames, len(frames))))
        clips, clip_ends = [], []
        for e in ends:
            idx = teacher_clip_indices(e, clip_frames, stride)
            if idx[0] < 0 or idx[-1] >= len(frames):
                continue
            clip = np.stack([np.asarray(frames[i]) for i in idx], axis=0)
            clip = np.transpose(clip.astype(np.float32) / 255.0, (0, 3, 1, 2))
            clips.append(clip)
            clip_ends.append(e)
        with torch.no_grad():
            for s in range(0, len(clips), mb):
                cb = torch.from_numpy(np.stack(clips[s : s + mb])).float().to(device)
                z = model.encode(cb).float().cpu().numpy()
                for j, e in enumerate(clip_ends[s : s + mb]):
                    z_end[e] = z[j]
                    mask[e] = True
        np.savez_compressed(cache_dir / ep / "teacher_z.npz", z_end=z_end, mask=mask)
        if (ei + 1) % 10 == 0 or ei == 0 or ei + 1 == len(all_eps):
            print(f"[cacheZ] {ei + 1}/{len(all_eps)}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    print("[cacheZ] done")


if __name__ == "__main__":
    main()
