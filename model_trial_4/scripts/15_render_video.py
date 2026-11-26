#!/usr/bin/env python3
"""Render a student-V2 (trial 4) prediction overlay on an arbitrary video.
Left: binary overlay (red if H>=0.60 else blue). Right: 5x5 real values."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from collision_jepa4.data.preprocessing import letterbox_rgb  # noqa: E402
from student.parent import load_parent_decoder  # noqa: E402
from student.paths import OFFSETS_16, parent_ckpt  # noqa: E402
from student.v4.model import CollisionStudentV4 as CollisionStudentV2  # noqa: E402

SIZE, PANEL, THR = 256, 512, 0.60


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    from student.decoder import HeatmapDecoder
    dec = HeatmapDecoder(z_channels=128, dilated=True, out_grid=5, pool="max")
    dec.load_state_dict(blob["decoder"])
    model = CollisionStudentV2(dec)
    model.backbone.load_state_dict(blob["backbone"])
    if "decoder" in blob:
        model.decoder.load_state_dict(blob["decoder"])
    model.eval().to(a.device)

    cap = cv2.VideoCapture(a.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    raw = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        raw.append(letterbox_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), SIZE))
    cap.release()
    frames = np.stack(raw)
    n = len(frames)
    print(f"frames={n} fps={fps}", flush=True)

    preds = np.zeros((n, 5, 5), np.float32)
    with torch.no_grad():
        for i0 in range(0, n, a.batch):
            ts = list(range(i0, min(n, i0 + a.batch)))
            clips = [np.transpose(frames[[max(0, t + o) for o in OFFSETS_16]], (0, 3, 1, 2)).astype(np.float32) / 255.0 for t in ts]
            x = torch.from_numpy(np.stack(clips)).to(a.device)
            h = model(x)["Hhat_1s"].float().cpu().numpy()
            for j, t in enumerate(ts):
                preds[t] = h[j]
            if i0 % (a.batch * 10) == 0:
                print(f"  infer {i0}/{n}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (PANEL * 2, PANEL))
    cell = PANEL // 5
    for t in range(n):
        base = cv2.resize(frames[t], (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        hm = preds[t]
        binm = np.where(hm >= THR, 1.0, 0.0).astype(np.float32)
        col = np.stack([binm, np.zeros_like(binm), 1.0 - binm], -1)
        col = cv2.resize((col * 255).astype(np.uint8), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        left = (base * 0.55 + col * 0.45).astype(np.uint8)
        pk = float(hm.max())
        label = "STOP" if pk >= THR else ("THREAT" if pk >= 0.25 else "SAFE")
        cv2.rectangle(left, (0, 0), (PANEL, 36), (10, 10, 10), -1)
        cv2.putText(left, f"t={t} {label} peak={pk:.2f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 80, 80) if label == "STOP" else (220, 220, 220), 2, cv2.LINE_AA)
        right = base.copy()
        for r in range(5):
            for c in range(5):
                v = float(hm[r, c])
                x0, y0 = c * cell, r * cell
                ov = right[y0:y0 + cell, x0:x0 + cell].astype(np.float32)
                tint = np.array([255 * v, 0, 255 * (1 - v)], np.float32)
                right[y0:y0 + cell, x0:x0 + cell] = (ov * 0.5 + tint * 0.5).astype(np.uint8)
                cv2.putText(right, f"{v:.2f}", (x0 + 18, y0 + cell // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
        w.write(cv2.cvtColor(np.concatenate([left, right], 1), cv2.COLOR_RGB2BGR))
    w.release()
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
