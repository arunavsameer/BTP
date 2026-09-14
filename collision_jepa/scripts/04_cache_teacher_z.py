"""Stage B: freeze the teacher and cache its collision state Z for every frame.

For each episode we compute the teacher Z from the clip ENDING at time ``e`` for all
valid ``e``, and store ``z_end[e]``. Student training then reads:
  - Z_t   = z_end[t]
  - Z+    = z_end[t + tau]
so a single cache serves any horizon. Cheap on disk (~240 KB/episode).

Run:  python scripts/04_cache_teacher_z.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.config import Config, resolve_device  # noqa: E402
from collision_jepa.data.dataset import teacher_clip_indices  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.data.unzip import find_episode_dirs  # noqa: E402
from collision_jepa.data.video import ensure_frame_cache  # noqa: E402
from collision_jepa.models.teacher import CollisionTeacher  # noqa: E402


def episode_dir_map(cfg: Config) -> dict[str, Path]:
    raw_dir = Path(cfg.get("data.raw_dir"))
    roots = [p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("dataset")]
    root = roots[0] if roots else raw_dir
    return {p.name: p for p in find_episode_dirs(root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--micro-batch", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("teacher.device", "auto"))
    print(f"[cacheZ] device={device}")

    split = load_split(cfg.get("data.split_file"))
    all_eps = sorted(set(split["train"]) | set(split["val"]))
    dir_map = episode_dir_map(cfg)

    size = int(cfg.get("teacher.img_size", 256))
    clip_frames = int(cfg.get("teacher.clip_frames", 8))
    stride = int(cfg.get("teacher.clip_stride", 2))
    zc = int(cfg.get("teacher.z_channels", 16))
    grid = int(cfg.get("student.feature_grid", 5))
    cache_dir = Path(cfg.get("data.cache_dir"))
    n_frames = int(cfg.get("data.n_frames", 150))
    mb = args.micro_batch

    model = CollisionTeacher(
        hf_model_id=cfg.get("teacher.hf_model_id"),
        z_channels=zc,
        feature_grid=grid,
        fp16=bool(cfg.get("teacher.fp16", True)),
    ).to(device)
    ckpt_path = Path(cfg.get("paths.ckpt_dir")) / "teacher.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    if device == "cuda" and bool(cfg.get("teacher.fp16", True)):
        model.backbone.model.half()
    model.eval()
    print(
        f"[cacheZ] loaded teacher from {ckpt_path}  "
        f"epoch={state.get('epoch')}  "
        f"HT-recall={(state.get('val') or {}).get('overall', {}).get('high_threat_recall')}  "
        f"STOP-F1={(state.get('val') or {}).get('overall', {}).get('stop_f1')}"
    )

    e_lo = (clip_frames - 1) * stride
    for ei, ep in enumerate(all_eps):
        if ep not in dir_map:
            continue
        frames_path = ensure_frame_cache(dir_map[ep], cache_dir, size)
        frames = np.load(frames_path, mmap_mode="r")

        z_end = np.zeros((n_frames, zc, grid, grid), dtype=np.float32)
        mask = np.zeros((n_frames,), dtype=bool)

        ends = list(range(e_lo, min(n_frames, len(frames))))
        clips = []
        clip_ends = []
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
                z = model.encode(cb).float().cpu().numpy()  # [b, zc, g, g]
                for j, e in enumerate(clip_ends[s : s + mb]):
                    z_end[e] = z[j]
                    mask[e] = True

        del clips
        if device == "cuda":
            torch.cuda.empty_cache()
        out_path = cache_dir / ep / "teacher_z.npz"
        np.savez_compressed(out_path, z_end=z_end, mask=mask)
        if (ei + 1) % 5 == 0 or ei == 0 or ei == len(all_eps) - 1:
            print(f"[cacheZ] {ei + 1}/{len(all_eps)} episodes cached", flush=True)

    print("[cacheZ] done. Student training can now read teacher_z.npz per episode.")


if __name__ == "__main__":
    main()
