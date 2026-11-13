#!/usr/bin/env python3
"""Render RGB + GT/pred heatmap grids for review. Never used as training input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg, resolve_checkpoint  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.visualizations import overlay_heatmap, render_example  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_cfg(args)
    device = resolve_device(args.device)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds = TeacherDataset(cfg, records, args.split, splits, use_token_cache=False)
    model = CollisionTeacher3(cfg, dummy_backbone=bool(args.dummy))
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    blob = load_checkpoint(ckpt, map_location=device)
    if blob.get("lora_enabled") and not getattr(model.backbone, "lora_enabled", False):
        model.enable_lora()
    load_teacher_state(model, blob)
    model.to(device).eval()
    out_dir = cfg.path("paths.visualizations_root")
    n = 0
    seen = {"collision": 0, "miss_left": 0, "miss_right": 0, "pre": 0, "post": 0}
    with torch.no_grad():
        for i in range(len(ds)):
            if n >= args.limit:
                break
            sample = ds[i]
            outcome = sample["outcome_type"]
            tag = "pre" if bool(sample["pre_fork"]) else "post"
            if seen.get(outcome, 0) >= max(2, args.limit // 4) and seen[tag] >= max(2, args.limit // 4):
                continue
            frames = torch.from_numpy(sample["frames"][None]).to(device)
            out = model(clip=frames)
            rgb = np.transpose(sample["frames"][-1], (1, 2, 0))
            maps = {
                "gt_now": sample["h_now"],
                "pred_now": out["Hhat_now"][0].cpu().numpy(),
                "gt_1s": sample["h_1s"],
                "pred_1s": out["Hhat_1s"][0].cpu().numpy(),
                "gt_1_5s": sample["h_1_5s"],
                "pred_1_5s": out["Hhat_1_5s"][0].cpu().numpy(),
                "gt_2s": sample["h_2s"],
                "pred_2s": out["Hhat_2s"][0].cpu().numpy(),
            }
            title = f"{sample['clip_id']} t={int(sample['anchor'])} {outcome} {tag}"
            render_example(rgb, maps, out_dir / f"{n:03d}_{outcome}_{tag}.png", title=title)
            Image_path = out_dir / f"{n:03d}_{outcome}_{tag}_overlay.png"
            from PIL import Image

            Image.fromarray(overlay_heatmap(rgb, maps["pred_1s"])).save(Image_path)
            seen[outcome] = seen.get(outcome, 0) + 1
            seen[tag] += 1
            n += 1
    print(f"wrote {n} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
