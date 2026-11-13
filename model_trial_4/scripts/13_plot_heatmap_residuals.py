#!/usr/bin/env python3
"""Histogram residual plots from a frozen student ckpt.

Reads the latest feat-decay checkpoint, writes plots to a *separate* raid
directory. CPU by default so it does not share a GPU with job 10258.
Does not write into student_v2_featdecay/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student.caches import ParentTargetCache, RGBCache256, SpatialHeatmaps  # noqa: E402
from student.dataset import WinDS, collate  # noqa: E402
from student.parent import load_parent_decoder  # noqa: E402
from student.paths import parent_cache, parent_ckpt, rgb_cache, z16_cache  # noqa: E402
from student.v2.model import CollisionStudentV2  # noqa: E402

TRAIN_RUN = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_4_decay/student_v2_featdecay")
OUT_ROOT = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_4_decay/heatmap_residuals")


def latest_ckpt(run: Path) -> Path:
    cks = sorted(run.glob("checkpoint_epoch_*.pt"))
    if not cks:
        raise SystemExit(f"no checkpoints in {run}")
    return cks[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="")
    p.add_argument("--split", default="selection", choices=("selection", "train", "test"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()
    if str(args.device).startswith("cuda"):
        raise SystemExit("Refusing GPU so this does not collide with the training job. Use --device cpu.")

    ckpt_path = Path(args.ckpt) if args.ckpt else latest_ckpt(TRAIN_RUN)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = int(blob["epoch"])
    out = Path(args.out) if args.out else OUT_ROOT / f"ep{epoch:02d}_{args.split}"
    if TRAIN_RUN in out.resolve().parents or out.resolve() == TRAIN_RUN:
        raise SystemExit(f"refusing to write into training run dir {TRAIN_RUN}")
    out.mkdir(parents=True, exist_ok=True)

    par = parent_cache()
    rows = [json.loads(l) for l in (par / "index.jsonl").open(encoding="utf-8")]
    gt = np.load(par / "gt_1s.npy").astype(np.float32)
    ids = [i for i, r in enumerate(rows) if r["split"] == args.split]
    if args.max_windows:
        ids = ids[: int(args.max_windows)]
    parent = ParentTargetCache(par, z16_root=z16_cache())
    ds = WinDS(rows, ids, gt, RGBCache256(rgb_cache()), parent, SpatialHeatmaps())
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=False,
    )

    decoder = load_parent_decoder(parent_ckpt())
    model = CollisionStudentV2(decoder)
    model.backbone.load_state_dict(blob["backbone"])
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    preds, gts, teachers = [], [], []
    print(f"ckpt={ckpt_path} epoch={epoch} split={args.split} n={len(ds)} out={out}", flush=True)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            clip = batch["clip"].to(device)
            h = model(clip)["Hhat_1s"].float().cpu().numpy()
            preds.append(h)
            gts.append(batch["gt_1s"].numpy())
            teachers.append(batch["H_teacher"].numpy())
            if (bi + 1) % 20 == 0 or bi == 0:
                print(f"  batch {bi+1}/{len(loader)}", flush=True)

    pred = np.concatenate(preds, axis=0)
    y = np.concatenate(gts, axis=0)
    ht = np.concatenate(teachers, axis=0)
    d_gt = pred - y
    d_par = pred - ht
    np.savez_compressed(out / "residuals.npz", pred=pred, gt=y, parent=ht, d_gt=d_gt, d_par=d_par)

    peak_pred, peak_gt = pred.max(axis=(1, 2)), y.max(axis=(1, 2))
    stats = {
        "ckpt": str(ckpt_path),
        "epoch": epoch,
        "split": args.split,
        "n": int(pred.shape[0]),
        "cell_pred_minus_gt": {
            "mean": float(d_gt.mean()),
            "std": float(d_gt.std()),
            "mae": float(np.abs(d_gt).mean()),
            "mse": float((d_gt**2).mean()),
            "p5": float(np.percentile(d_gt, 5)),
            "p50": float(np.percentile(d_gt, 50)),
            "p95": float(np.percentile(d_gt, 95)),
        },
        "peak_pred_minus_gt": {
            "mean": float((peak_pred - peak_gt).mean()),
            "mae": float(np.abs(peak_pred - peak_gt).mean()),
        },
        "cell_pred_minus_parent": {
            "mean": float(d_par.mean()),
            "mae": float(np.abs(d_par).mean()),
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    fig.suptitle(f"Student vs GT heatmap  ·  epoch {epoch}  ·  {args.split} n={pred.shape[0]}", fontsize=12)

    ax = axes[0, 0]
    ax.hist(d_gt.ravel(), bins=80, range=(-1, 1), color="#3b6ea8", alpha=0.9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Per-cell  pred − GT")
    ax.set_xlabel("residual")
    ax.set_ylabel("count")

    ax = axes[0, 1]
    ax.hist(np.abs(d_gt).ravel(), bins=60, range=(0, 1), color="#c45c26", alpha=0.9)
    ax.set_title("Per-cell  |pred − GT|")
    ax.set_xlabel("|residual|")

    ax = axes[0, 2]
    im = ax.imshow(d_gt.mean(axis=0), vmin=-0.25, vmax=0.25, cmap="RdBu_r")
    ax.set_title("Mean signed residual (5×5)")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    ax.hist(peak_pred - peak_gt, bins=60, range=(-1, 1), color="#4a7c59", alpha=0.9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Peak  pred − GT")
    ax.set_xlabel("peak residual")
    ax.set_ylabel("windows")

    ax = axes[1, 1]
    cls = np.where(peak_gt >= 0.60, "STOP", np.where(peak_gt >= 0.25, "THREAT", "SAFE"))
    for name, c in [("SAFE", "#6aa84f"), ("THREAT", "#f1c232"), ("STOP", "#cc0000")]:
        m = cls == name
        if m.any():
            ax.hist(d_gt[m].ravel(), bins=50, range=(-1, 1), histtype="step", lw=1.6, label=name, color=c, density=True)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Per-cell residual by GT class")
    ax.set_xlabel("pred − GT")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    im = ax.imshow(np.abs(d_gt).mean(axis=0), vmin=0, vmax=0.25, cmap="magma")
    ax.set_title("Mean |residual| (5×5)")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    png = out / "residual_histograms.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
