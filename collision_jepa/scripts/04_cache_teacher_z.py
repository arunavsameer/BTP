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
from collision_jepa.data.dataset import pack_teacher_clips  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.data.unzip import episode_dir_map, resolve_dataset_root  # noqa: E402
from collision_jepa.data.video import load_or_cache_frames  # noqa: E402
from collision_jepa.engine import configure_runtime, prefetch_items, uint8_clips_to_device  # noqa: E402
from collision_jepa.models.teacher import CollisionTeacher  # noqa: E402


def _dir_map(cfg: Config) -> dict[str, Path]:
    return episode_dir_map(resolve_dataset_root(cfg.get("data.raw_dir")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--micro-batch", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    configure_runtime()
    device = resolve_device(cfg.get("teacher.device", "auto"))
    print(f"[cacheZ] device={device}")

    split = load_split(cfg.get("data.split_file"))
    all_eps = sorted(set(split["train"]) | set(split["val"]) | set(split.get("test") or []))
    dir_map = _dir_map(cfg)

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
        occ_channels=int(cfg.get("student.occ_channels", 4)),
        fp16=bool(cfg.get("teacher.fp16", True)),
    ).to(device)
    ckpt_path = Path(cfg.get("paths.ckpt_dir")) / "teacher.pt"
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    missing = [k for k in missing if ".predictor." not in k]
    unexpected = [k for k in unexpected if ".predictor." not in k]
    if missing or unexpected:
        raise SystemExit(
            f"[cacheZ] teacher.pt does not match this teacher code "
            f"(missing={missing[:8]} unexpected={unexpected[:8]})"
        )
    if device == "cuda" and bool(cfg.get("teacher.fp16", True)):
        model.backbone.model.half()
    model.eval()
    print(
        f"[cacheZ] loaded teacher from {ckpt_path}  "
        f"epoch={state.get('epoch')}  "
        f"HT-recall={(state.get('val') or {}).get('overall', {}).get('high_threat_recall')}  "
        f"STOP-F1={(state.get('val') or {}).get('overall', {}).get('stop_f1')}"
    )

    def _load(ep):
        if ep not in dir_map:
            return None
        frames = load_or_cache_frames(dir_map[ep], cache_dir, size, cache_name=ep)
        packed = pack_teacher_clips(frames, None, clip_frames, stride, sample_stride=1)
        if packed is None:
            return None
        clips_u8, _targets, clip_ends = packed
        return clips_u8, clip_ends, ep

    for ei, (clips_u8, clip_ends, ep) in enumerate(prefetch_items(all_eps, _load)):
        z_end = np.zeros((n_frames, zc, grid, grid), dtype=np.float32)
        mask = np.zeros((n_frames,), dtype=bool)
        clip_ends = clip_ends.tolist()

        with torch.inference_mode():
            for s in range(0, clips_u8.shape[0], mb):
                cb = uint8_clips_to_device(clips_u8[s : s + mb], device)
                z = model.encode(cb).float().cpu().numpy()
                for j, e in enumerate(clip_ends[s : s + mb]):
                    z_end[e] = z[j]
                    mask[e] = True

        del clips_u8
        out_path = cache_dir / ep / "teacher_z.npz"
        np.savez_compressed(out_path, z_end=z_end, mask=mask)
        if (ei + 1) % 5 == 0 or ei == 0 or ei == len(all_eps) - 1:
            print(f"[cacheZ] {ei + 1}/{len(all_eps)} episodes cached", flush=True)

    print("[cacheZ] done. Student training can now read teacher_z.npz per episode.")


if __name__ == "__main__":
    main()
