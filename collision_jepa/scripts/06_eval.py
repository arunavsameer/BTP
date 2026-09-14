"""Full evaluation: copy baseline vs Stage-0 vs student, plus hysteresis STOP-F1.

Emphasizes what matters for a blind user: high-threat recall and STOP precision/
recall/F1, both overall and on rare (NEAR_MISS/CRITICAL) episodes.

Run:  python scripts/06_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.config import Config, resolve_device  # noqa: E402
from collision_jepa.data.dataset import CollisionDataset  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.engine import evaluate_future_heatmap  # noqa: E402
from collision_jepa.metrics import MetricAccumulator, format_summary  # noqa: E402
from collision_jepa.models.baseline import Stage0Model  # noqa: E402
from collision_jepa.models.student import Student  # noqa: E402
from collision_jepa.warning import STOP, HitLatch, classify  # noqa: E402


def copy_baseline(cfg, episodes, has_rare):
    cache_dir = Path(cfg.get("data.cache_dir"))
    tau = int(cfg.get("horizon.tau_frames"))
    offsets = cfg.get("student.frame_offsets")
    recall_threshold = float(cfg.get("heatmap.recall_threshold", 0.5))
    warn_cfg = cfg.get("warning")
    overall = MetricAccumulator(recall_threshold, warn_cfg)
    rare = MetricAccumulator(recall_threshold, warn_cfg)
    t_lo = -min(offsets)
    for ep in episodes:
        hm_path = cache_dir / ep / "heatmaps.npy"
        if not hm_path.exists():
            continue
        hms = np.load(hm_path)
        for t in range(t_lo, hms.shape[0] - tau, 3):
            overall.update(hms[t], hms[t + tau])
            if has_rare.get(ep, False):
                rare.update(hms[t], hms[t + tau])
    return {"overall": overall.summary(), "rare": rare.summary()}


def hysteresis_stop_f1(cfg, model, device, episodes):
    """Sequential per-episode STOP-F1 with HitLatch applied to student predictions."""
    cache_dir = Path(cfg.get("data.cache_dir"))
    size = int(cfg.get("student.img_size"))
    offsets = cfg.get("student.frame_offsets")
    tau = int(cfg.get("horizon.tau_frames"))
    warn_cfg = cfg.get("warning")
    t_lo = -min(offsets)

    tp = fp = fn = 0
    model.eval()
    for ep in episodes:
        fpath = cache_dir / ep / f"frames_{size}.npy"
        hpath = cache_dir / ep / "heatmaps.npy"
        if not fpath.exists() or not hpath.exists():
            continue
        frames = np.load(fpath, mmap_mode="r")
        hms = np.load(hpath)
        latch = HitLatch(int(warn_cfg["confirm_frames"]), int(warn_cfg["release_frames"]))
        for t in range(t_lo, hms.shape[0] - tau):
            clip = np.stack([np.asarray(frames[t + o]) for o in offsets], axis=0)
            clip = np.transpose(clip.astype(np.float32) / 255.0, (0, 3, 1, 2))
            x = torch.from_numpy(clip).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred = model.predict_future_heatmap(x)[0].cpu().numpy()
            _, raw_sev, _ = classify(pred, warn_cfg)
            direction = classify(pred, warn_cfg)[0]
            _, stable_sev = latch.update(raw_sev, direction)
            _, true_sev, _ = classify(hms[t + tau], warn_cfg)
            p_stop = stable_sev >= STOP
            t_stop = true_sev >= STOP
            tp += int(p_stop and t_stop)
            fp += int(p_stop and not t_stop)
            fn += int(not p_stop and t_stop)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"stop_precision": p, "stop_recall": r, "stop_f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("train.device", "auto"))
    split = load_split(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    ckpt_dir = Path(cfg.get("paths.ckpt_dir"))

    size = int(cfg.get("student.img_size"))
    offsets = cfg.get("student.frame_offsets")
    tau = int(cfg.get("horizon.tau_frames"))
    n_frames = int(cfg.get("data.n_frames", 150))
    bs = int(cfg.get("train.batch_size"))
    val_ds = CollisionDataset(cfg.get("data.cache_dir"), split["val"], size, offsets, tau, n_frames, stride=3)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    print("=" * 78)
    print("COLLISION-JEPA EVALUATION (val)")
    print("=" * 78)

    res = copy_baseline(cfg, split["val"], has_rare)
    print(format_summary("copy      [all]", res["overall"]))
    if res["rare"]["n"]:
        print(format_summary("copy      [rare]", res["rare"]))

    stage0_ckpt = ckpt_dir / "stage0.pt"
    if stage0_ckpt.exists():
        m = Stage0Model(int(cfg.get("student.cnn_width", 32)), int(cfg.get("student.feature_grid", 5))).to(device)
        m.load_state_dict(torch.load(stage0_ckpt, map_location=device)["model"])
        m.eval()
        res = evaluate_future_heatmap(lambda b: m(b["frames"]), val_loader, cfg, device, val_ds.episodes, has_rare)
        print(format_summary("stage0    [all]", res["overall"]))
        if res["rare"]["n"]:
            print(format_summary("stage0    [rare]", res["rare"]))

    student_ckpt = ckpt_dir / "student.pt"
    if student_ckpt.exists():
        m = Student(
            int(cfg.get("student.cnn_width", 32)),
            int(cfg.get("student.feature_grid", 5)),
            int(cfg.get("student.z_channels", 16)),
        ).to(device)
        m.load_state_dict(torch.load(student_ckpt, map_location=device)["model"])
        m.eval()
        res = evaluate_future_heatmap(
            lambda b: m.predict_future_heatmap(b["frames"]), val_loader, cfg, device, val_ds.episodes, has_rare
        )
        print(format_summary("student   [all]", res["overall"]))
        if res["rare"]["n"]:
            print(format_summary("student   [rare]", res["rare"]))
        hyst = hysteresis_stop_f1(cfg, m, device, split["val"])
        print(
            f"student   [hysteresis]  STOP-F1={hyst['stop_f1']:.3f} "
            f"(P={hyst['stop_precision']:.3f} R={hyst['stop_recall']:.3f})"
        )
    else:
        print("(student.pt not found; run scripts 03-05 to train the full pipeline.)")


if __name__ == "__main__":
    main()
