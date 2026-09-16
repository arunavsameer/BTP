"""Overlay videos: GT | student prediction on the original dataset RGB.

Same layout as model_trial_1: native preview.mp4 underneath the 5×5 threat grid,
Blender blue/amber/red. The student still infers from cached 160px frames.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config, resolve_device  # noqa: E402
from cvjepa.data import load_json  # noqa: E402
from cvjepa.data.video import decode_video_full  # noqa: E402
from cvjepa.engine import configure_runtime, predict_episode_heatmaps  # noqa: E402
from cvjepa.models.student import CollisionStudent  # noqa: E402
from cvjepa.viz.overlay import stack_panel, write_mp4  # noqa: E402
from cvjepa.warning import HitLatch, classify, severity_name  # noqa: E402


def _pick_episodes(keys: list[str], split: dict, limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(keys):
        return list(keys)
    by_fam: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_fam[split["families"].get(k, "other")].append(k)
    rng = random.Random(seed)
    for v in by_fam.values():
        rng.shuffle(v)
    picked: list[str] = []
    fams = sorted(by_fam)
    i = 0
    while len(picked) < limit and fams:
        fam = fams[i % len(fams)]
        if by_fam[fam]:
            picked.append(by_fam[fam].pop())
        i += 1
        if i > limit * 8:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--latch", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    configure_runtime()
    device = resolve_device(cfg.get("train.device", "auto"))
    split = load_json(cfg.get("data.split_file"))
    pool = list(split.get(args.split) or split["val"])
    keys = _pick_episodes(pool, split, args.limit, args.seed)
    offsets = list(cfg.get("student.frame_offsets"))
    tau = int(cfg.get("horizon.tau_frames"))
    size = int(cfg.get("student.img_size"))
    warn = cfg.get("warning")
    cache = Path(cfg.get("data.cache_dir"))
    out_dir = Path(cfg.get("paths.overlay_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CollisionStudent(
        width=int(cfg.get("student.cnn_width", 32)),
        feature_grid=int(cfg.get("student.feature_grid", 5)),
        z_channels=int(cfg.get("student.z_channels", 32)),
        n_frames=len(offsets),
        use_loom=bool(cfg.get("student.use_loom", True)),
        copy_residual=bool(cfg.get("student.copy_residual", True)),
    ).to(device)
    ckpt = Path(args.ckpt) if args.ckpt else Path(cfg.get("paths.ckpt_dir")) / "student_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    print(f"[overlay] device={device}  ckpt={ckpt}  split={args.split}  n={len(keys)}", flush=True)

    t_lo = -min(offsets)
    for i, key in enumerate(keys, start=1):
        small = np.load(cache / key / f"frames_{size}.npy")
        gt = np.load(cache / key / "heatmaps.npy")
        video = Path(split["paths"][key]) / "preview.mp4"
        if video.is_file():
            display, fps = decode_video_full(video)
        else:
            display, fps = small, float(cfg.get("data.fps", 30))
        n = min(len(display), len(small), len(gt))
        display, small, gt = display[:n], small[:n], gt[:n]
        pred = predict_episode_heatmaps(model, small, offsets, device)
        latch = HitLatch(int(warn["confirm_frames"]), int(warn["release_frames"])) if args.latch else None
        vis = []
        for t in range(t_lo, n - tau):
            p_dir, p_raw, _ = classify(pred[t], warn)
            g_dir, g_sev, _ = classify(gt[t + tau], warn)
            if latch is not None:
                p_dir, p_sev = latch.update(p_raw, p_dir)
            else:
                p_sev = p_raw
            left = stack_panel(display[t], gt[t + tau], "GT t+1s", g_dir, severity_name(g_sev))
            right = stack_panel(display[t], pred[t], "PRED t+1s", p_dir, severity_name(p_sev))
            gap = np.zeros((left.shape[0], 8, 3), dtype=np.uint8)
            vis.append(np.concatenate([left, gap, right], axis=1))
        name = key.replace("/", "__")
        path = out_dir / f"{name}.mp4"
        write_mp4(path, vis, fps=fps or 30.0)
        print(f"[overlay] {i}/{len(keys)}  {path}  frames={len(vis)}", flush=True)

    print(f"[overlay] wrote {len(keys)} videos → {out_dir}")


if __name__ == "__main__":
    main()
