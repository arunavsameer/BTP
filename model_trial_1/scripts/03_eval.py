"""Evaluate copy baseline vs trained RS-JEPA student (no shuffle at eval)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.config import Config, resolve_device  # noqa: E402
from rs_jepa.data import load_json  # noqa: E402
from rs_jepa.data.dataset import CollisionDataset  # noqa: E402
from rs_jepa.engine import evaluate_future_heatmap, evaluate_wearable  # noqa: E402
from rs_jepa.metrics import MetricAccumulator, format_summary, format_wearable  # noqa: E402
from rs_jepa.models.student import RSJEPA  # noqa: E402
from rs_jepa.warning import STOP, HitLatch, classify  # noqa: E402


def copy_baseline(cfg, episodes, has_rare):
    cache_dir = Path(cfg.get("data.cache_dir"))
    tau = int(cfg.get("horizon.tau_frames"))
    offsets = cfg.get("student.frame_offsets")
    overall = MetricAccumulator(float(cfg.get("heatmap.recall_threshold", 0.5)), cfg.get("warning"))
    rare = MetricAccumulator(float(cfg.get("heatmap.recall_threshold", 0.5)), cfg.get("warning"))
    t_lo = -min(offsets)
    for ep in episodes:
        hms = np.load(cache_dir / ep / "heatmaps.npy")
        for t in range(t_lo, hms.shape[0] - tau, 3):
            overall.update(hms[t], hms[t + tau])
            if has_rare.get(ep, False):
                rare.update(hms[t], hms[t + tau])
    return {"overall": overall.summary(), "rare": rare.summary()}


def hysteresis_stop_f1(cfg, model, device, episodes):
    cache_dir = Path(cfg.get("data.cache_dir"))
    size = int(cfg.get("student.img_size"))
    offsets = cfg.get("student.frame_offsets")
    tau = int(cfg.get("horizon.tau_frames"))
    warn_cfg = cfg.get("warning")
    t_lo = -min(offsets)
    tp = fp = fn = 0
    model.eval()
    for ep in episodes:
        frames = np.load(cache_dir / ep / f"frames_{size}.npy", mmap_mode="r")
        hms = np.load(cache_dir / ep / "heatmaps.npy")
        latch = HitLatch(int(warn_cfg["confirm_frames"]), int(warn_cfg["release_frames"]))
        for t in range(t_lo, hms.shape[0] - tau):
            clip = np.stack([np.asarray(frames[t + o]) for o in offsets], axis=0)
            clip = np.transpose(clip.astype(np.float32) / 255.0, (0, 3, 1, 2))
            x = torch.from_numpy(clip).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred = model.predict_future_heatmap(x)[0].cpu().numpy()
            direction, raw_sev, _ = classify(pred, warn_cfg)
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
    parser.add_argument(
        "--split",
        default="both",
        choices=("val", "test", "both"),
        help="Which holdout to score. 'both' runs val then test if present.",
    )
    parser.add_argument("--ckpt", default="", help="Override checkpoint path.")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    device = resolve_device(cfg.get("train.device", "auto"))
    split = load_json(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})

    names = []
    if args.split in ("val", "both"):
        names.append("val")
    if args.split in ("test", "both") and split.get("test"):
        names.append("test")
    if args.split == "test" and not split.get("test"):
        raise SystemExit("split.json has no test list. Run scripts/05_make_test_split.py")

    ckpt_dir = Path(cfg.get("paths.ckpt_dir"))
    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        best = ckpt_dir / "student_best.pt"
        ckpt = best if best.exists() else ckpt_dir / "student.pt"
    model = None
    if ckpt.exists():
        model = RSJEPA.from_config(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        model.eval()
    else:
        print(f"(student checkpoint not found: {ckpt})")

    for split_name in names:
        episodes = split[split_name]
        ds = CollisionDataset(
            cfg.get("data.cache_dir"),
            episodes,
            int(cfg.get("student.img_size")),
            cfg.get("student.frame_offsets"),
            int(cfg.get("horizon.tau_frames")),
            int(cfg.get("data.n_frames", 150)),
            stride=3,
        )
        loader = DataLoader(ds, batch_size=int(cfg.get("train.batch_size")), shuffle=False)
        print("=" * 78)
        print(f"RS-JEPA EVALUATION ({split_name}, no shuffle)  n_episodes={len(episodes)}")
        print("=" * 78)
        res = copy_baseline(cfg, episodes, has_rare)
        tag = f"copy      [{split_name} all]"
        print(format_summary(tag, res["overall"]))
        if res["rare"]["n"]:
            print(format_summary(f"copy      [{split_name} rare]", res["rare"]))
        copy_w = evaluate_wearable(
            cfg, episodes, has_rare, device, copy_baseline=True
        )
        print(format_wearable(f"copy      [{split_name} all]", copy_w["overall"]))
        if copy_w["rare"]["n"]:
            print(format_wearable(f"copy      [{split_name} rare]", copy_w["rare"]))
        if model is None:
            continue
        res = evaluate_future_heatmap(
            lambda b: model.predict_future_heatmap(b["frames"]),
            loader,
            cfg,
            device,
            ds.episodes,
            has_rare,
        )
        print(format_summary(f"student   [{split_name} all]", res["overall"]))
        if res["rare"]["n"]:
            print(format_summary(f"student   [{split_name} rare]", res["rare"]))
        stud_w = evaluate_wearable(cfg, episodes, has_rare, device, model=model)
        print(format_wearable(f"student   [{split_name} all]", stud_w["overall"]))
        if stud_w["rare"]["n"]:
            print(format_wearable(f"student   [{split_name} rare]", stud_w["rare"]))
        hyst = hysteresis_stop_f1(cfg, model, device, episodes)
        print(
            f"student   [{split_name} hysteresis]  STOP-F1={hyst['stop_f1']:.3f} "
            f"(P={hyst['stop_precision']:.3f} R={hyst['stop_recall']:.3f})"
        )


if __name__ == "__main__":
    main()
