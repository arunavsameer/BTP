#!/usr/bin/env python3
"""Render preview.mp4 + student/GT 1s heatmap overlay. CPU by default (no training GPU)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa4.data.heatmap_annotations import load_heatmaps  # noqa: E402
from collision_jepa4.data.preprocessing import letterbox_rgb  # noqa: E402
from student.parent import load_parent_decoder  # noqa: E402
from student.paths import OFFSETS_16, parent_ckpt  # noqa: E402
from student.v2.model import CollisionStudentV2  # noqa: E402

DATASET = Path("/home/asrivastava/VJEPA2_BTP/BTP/blender_sim/datasets/groups15_n400_s67_n400")
TRAIN_RUN = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_4_decay/student_v2_featdecay")
OUT_DIR = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4/viz")
FORK = 365
SIZE = 256
PANEL = 512
DEFAULT_CLIPS = [
    "sample_0169_jaywalker_from_left/collision/bg_01",
    "sample_0149_jaywalker_from_right/collision/bg_00",
    "sample_0019_periph_ped_cut/collision/bg_01",
    "sample_0059_crossing_car_side/miss_left/bg_00",
    "sample_0109_swerve_vehicle/miss_right/bg_00",
]


def latest_ckpt(run: Path) -> Path:
    cks = sorted(run.glob("checkpoint_epoch_*.pt"))
    if not cks:
        raise SystemExit(f"no ckpts in {run}")
    return cks[-1]


def peak_class(h: np.ndarray) -> tuple[float, str]:
    p = float(np.max(h))
    if p >= 0.60:
        return p, "STOP"
    if p >= 0.25:
        return p, "THREAT"
    return p, "SAFE"


def overlay(rgb: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = rgb.astype(np.float32)
    h, w = base.shape[:2]
    heat = np.clip(np.asarray(hm, dtype=np.float32), 0.0, 1.0)
    heat_img = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1)
    heat_img = (heat_img * 255.0).astype(np.uint8)
    heat_img = cv2.resize(heat_img, (w, h), interpolation=cv2.INTER_NEAREST)
    out = base * (1.0 - alpha) + heat_img.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def up(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def banner(img: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (PANEL, 36), (10, 10, 10), -1)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return out


def render_clip(model, clip_id: str, epoch: int, batch: int, stride: int) -> Path:
    video = DATASET / clip_id / "preview.mp4"
    ann = video.parent / "spatial_annotations" / "spatial_annotations.json"
    if not video.is_file():
        raise FileNotFoundError(video)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    raw = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        raw.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack([letterbox_rgb(f, SIZE) for f in raw], axis=0)
    n = int(frames.shape[0])
    gt_now = load_heatmaps(ann, expected_frames=n, grid=5)
    slug = clip_id.replace("/", "_")
    out_path = OUT_DIR / f"overlay_ep{epoch:02d}_{slug}.mp4"
    print(f"clip={clip_id} frames={n} -> {out_path.name}", flush=True)

    preds = np.zeros((n, 5, 5), dtype=np.float32)
    ts = list(range(0, n, max(int(stride), 1)))
    with torch.no_grad():
        for i0 in range(0, len(ts), batch):
            chunk = ts[i0 : i0 + batch]
            clips = []
            for t in chunk:
                idx = [max(0, t + o) for o in OFFSETS_16]
                clip = np.transpose(frames[idx], (0, 3, 1, 2)).astype(np.float32) / 255.0
                clips.append(clip)
            x = torch.from_numpy(np.stack(clips, axis=0))
            h = model(x)["Hhat_1s"].float().cpu().numpy()
            for j, t in enumerate(chunk):
                preds[t] = h[j]
            if i0 % (batch * 10) == 0:
                print(f"  infer {i0}/{len(ts)}", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / max(int(stride), 1), (PANEL * 2, PANEL))
    for t in ts:
        rgb = frames[t]
        t1 = min(t + 30, n - 1)
        gt = np.asarray(gt_now[t1], dtype=np.float32)
        ps, cs = peak_class(preds[t])
        pg, cg = peak_class(gt)
        fork = "POST" if t >= FORK else "pre"
        left = banner(
            overlay(up(rgb), preds[t]),
            f"STUDENT  t={t} {fork}  {cs} {ps:.2f}  (1s ahead)",
            (80, 220, 255) if cs != "SAFE" else (220, 220, 220),
        )
        right = banner(
            overlay(up(rgb), gt),
            f"GT  t+30={t1}  {cg} {pg:.2f}",
            (80, 255, 120) if cg != "SAFE" else (220, 220, 220),
        )
        canvas = np.concatenate([left, right], axis=1)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"wrote {out_path}", flush=True)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ckpt", default="")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--clip", action="append", default=[])
    args = p.parse_args()
    if str(args.device).startswith("cuda"):
        raise SystemExit("Use --device cpu so this does not share a GPU with training.")

    ckpt_path = Path(args.ckpt) if args.ckpt else latest_ckpt(TRAIN_RUN)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = int(blob["epoch"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decoder = load_parent_decoder(parent_ckpt())
    model = CollisionStudentV2(decoder)
    model.backbone.load_state_dict(blob["backbone"])
    model.eval().to("cpu")
    print(f"ckpt={ckpt_path} epoch={epoch}", flush=True)
    clips = list(args.clip) if args.clip else list(DEFAULT_CLIPS)
    for clip_id in clips:
        render_clip(model, clip_id, epoch, args.batch, args.stride)


if __name__ == "__main__":
    main()
