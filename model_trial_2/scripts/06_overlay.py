"""Overlay videos: GT | student prediction (Blender blue/amber/red)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config, resolve_device  # noqa: E402
from cvjepa.data import load_json  # noqa: E402
from cvjepa.engine import predict_episode_heatmaps  # noqa: E402
from cvjepa.models.student import CollisionStudent  # noqa: E402
from cvjepa.viz.overlay import stack_panel, write_mp4  # noqa: E402
from cvjepa.warning import HitLatch, classify, severity_name  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--latch", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("train.device", "auto"))
    split = load_json(cfg.get("data.split_file"))
    keys = list(split.get(args.split) or split["val"])[: args.limit]
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

    t_lo = -min(offsets)
    for key in keys:
        frames = np.load(cache / key / f"frames_{size}.npy")
        gt = np.load(cache / key / "heatmaps.npy")
        pred = predict_episode_heatmaps(model, frames, offsets, device)
        latch = HitLatch(int(warn["confirm_frames"]), int(warn["release_frames"])) if args.latch else None
        vis = []
        for t in range(t_lo, frames.shape[0] - tau):
            p_dir, p_raw, _ = classify(pred[t], warn)
            g_dir, g_sev, _ = classify(gt[t + tau], warn)
            if latch is not None:
                p_dir, p_sev = latch.update(p_raw, p_dir)
            else:
                p_sev = p_raw
            left = stack_panel(frames[t], gt[t + tau], "GT t+1s", g_dir, severity_name(g_sev))
            right = stack_panel(frames[t], pred[t], "PRED t+1s", p_dir, severity_name(p_sev))
            vis.append(np.concatenate([left, right], axis=1))
        name = key.replace("/", "_")
        path = out_dir / f"{name}.mp4"
        write_mp4(path, vis, fps=float(cfg.get("data.fps", 30)))
        print(f"[overlay] {path}")


if __name__ == "__main__":
    main()
