"""Stage A: train the frozen-V-JEPA-2 teacher to reconstruct the CURRENT 5x5 heatmap.

Only the small pool/bottleneck and the heatmap decoder are trained; the V-JEPA-2
backbone stays frozen. We decode teacher-resolution (256px) frames per episode and
cache them to disk so repeated epochs are fast.

Gate: the teacher must beat the copy baseline on high-threat / STOP metrics, else the
bottleneck is not capturing collision-specific information and we should stop.

Run:  python scripts/03_train_teacher.py
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
from collision_jepa.engine import configure_runtime, save_checkpoint, set_seed  # noqa: E402
from collision_jepa.losses import weighted_focal_mse  # noqa: E402
from collision_jepa.metrics import MetricAccumulator, format_summary  # noqa: E402
from collision_jepa.models.teacher import CollisionTeacher  # noqa: E402


def episode_dir_map(cfg: Config) -> dict[str, Path]:
    raw_dir = Path(cfg.get("data.raw_dir"))
    roots = [p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("dataset")]
    root = roots[0] if roots else raw_dir
    return {p.name: p for p in find_episode_dirs(root)}


def iter_episode_samples(cfg, ep_names, dir_map, size, clip_frames, stride, tau, sample_stride):
    """Yield (clips_tensor[B,T,3,H,W], H_now[B,5,5]) batched per episode."""
    cache_dir = Path(cfg.get("data.cache_dir"))
    n_frames = int(cfg.get("data.n_frames", 150))
    t_lo = (clip_frames - 1) * stride
    t_hi = n_frames - 1
    for ep in ep_names:
        if ep not in dir_map:
            continue
        frames_path = ensure_frame_cache(dir_map[ep], cache_dir, size)
        frames = np.load(frames_path, mmap_mode="r")
        heatmaps = np.load(cache_dir / ep / "heatmaps.npy")
        clips = []
        targets = []
        for t in range(t_lo, t_hi + 1, sample_stride):
            idx = teacher_clip_indices(t, clip_frames, stride)
            if idx[0] < 0 or idx[-1] >= len(frames):
                continue
            clip = np.stack([np.asarray(frames[i]) for i in idx], axis=0)  # [T,H,W,3]
            clip = np.transpose(clip.astype(np.float32) / 255.0, (0, 3, 1, 2))
            clips.append(clip)
            targets.append(np.asarray(heatmaps[t], dtype=np.float32))
        if clips:
            yield (
                torch.from_numpy(np.stack(clips)).float(),
                torch.from_numpy(np.stack(targets)).float(),
                ep,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=8, help="Clips per forward pass (VRAM).")
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoints/teacher.pt.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 1337)))
    configure_runtime()
    device = resolve_device(cfg.get("teacher.device", "auto"))
    print(f"[teacherA] device={device}")

    split = load_split(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    dir_map = episode_dir_map(cfg)

    size = int(cfg.get("teacher.img_size", 256))
    clip_frames = int(cfg.get("teacher.clip_frames", 8))
    stride = int(cfg.get("teacher.clip_stride", 2))
    tau = int(cfg.get("horizon.tau_frames", 30))
    sample_stride = 5

    print(f"[teacherA] loading frozen V-JEPA-2 ({cfg.get('teacher.hf_model_id')}) ...")
    model = CollisionTeacher(
        hf_model_id=cfg.get("teacher.hf_model_id"),
        z_channels=int(cfg.get("teacher.z_channels", 16)),
        feature_grid=int(cfg.get("student.feature_grid", 5)),
        fp16=bool(cfg.get("teacher.fp16", True)),
    ).to(device)
    if device == "cuda" and bool(cfg.get("teacher.fp16", True)):
        model.backbone.model.half()
    n_train = sum(p.numel() for p in model.trainable_parameters())
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[teacherA] trainable adapter+decoder params={n_train:,} (V-JEPA-2 frozen)  VRAM={mem:.2f} GB")
    else:
        print(f"[teacherA] trainable adapter+decoder params={n_train:,} (V-JEPA-2 frozen)")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.get("train.lr", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    focal_w = float(cfg.get("train.focal_weight", 1.0))
    gamma = float(cfg.get("train.focal_gamma", 2.0))
    mb = args.micro_batch

    def run_eval() -> dict:
        model.eval()
        recall_threshold = float(cfg.get("heatmap.recall_threshold", 0.5))
        warn_cfg = cfg.get("warning")
        overall = MetricAccumulator(recall_threshold, warn_cfg)
        rare = MetricAccumulator(recall_threshold, warn_cfg)
        with torch.no_grad():
            for clips, targets, ep in iter_episode_samples(
                cfg, split["val"], dir_map, size, clip_frames, stride, tau, sample_stride
            ):
                for s in range(0, clips.shape[0], mb):
                    cb = clips[s : s + mb].to(device)
                    _, h_hat = model(cb)
                    h_hat = h_hat.float().cpu().numpy()
                    tb = targets[s : s + mb].numpy()
                    for b in range(h_hat.shape[0]):
                        overall.update(h_hat[b], tb[b])
                        if has_rare.get(ep, False):
                            rare.update(h_hat[b], tb[b])
        return {"overall": overall.summary(), "rare": rare.summary()}

    epochs = args.epochs or int(cfg.get("train.teacher_epochs", 40))
    ckpt_path = Path(cfg.get("paths.ckpt_dir")) / "teacher.pt"
    start_epoch = 1
    best = -1.0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        val = ckpt.get("val") or {}
        ov = val.get("overall") or {}
        best = float(ov.get("high_threat_recall", 0.0)) + float(ov.get("stop_f1", 0.0))
        print(f"[teacherA] resumed from {ckpt_path}  last_saved_epoch={ckpt.get('epoch')}  best={best:.4f}")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        # Frozen backbone must stay in eval so dropout/BN stats do not drift and
        # we do not accumulate extra CUDA graph state.
        model.backbone.eval()
        total, n = 0.0, 0
        ep_order = list(split["train"])
        np.random.shuffle(ep_order)
        for ei, (clips, targets, ep) in enumerate(
            iter_episode_samples(cfg, ep_order, dir_map, size, clip_frames, stride, tau, sample_stride)
        ):
            perm = np.random.permutation(clips.shape[0])
            for s in range(0, len(perm), mb):
                sel = perm[s : s + mb]
                cb = clips[sel].to(device, non_blocking=True)
                tb = targets[sel].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                _, h_hat = model(cb)
                loss = weighted_focal_mse(h_hat, tb, focal_w, gamma)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(sel)
                n += len(sel)
                del cb, tb, h_hat, loss
            del clips, targets
            if device == "cuda" and (ei + 1) % 5 == 0:
                torch.cuda.empty_cache()
            if (ei + 1) % 10 == 0 or ei == 0:
                vram = ""
                if device == "cuda":
                    vram = f"  VRAM={torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                print(
                    f"[teacherA] epoch {epoch:03d}  ep {ei + 1}/{len(ep_order)}  "
                    f"loss={total / max(n, 1):.4f}{vram}",
                    flush=True,
                )
        if device == "cuda":
            torch.cuda.empty_cache()
        res = run_eval()
        if device == "cuda":
            torch.cuda.empty_cache()
        sel_metric = res["overall"]["high_threat_recall"] + res["overall"]["stop_f1"]
        tag = ""
        if sel_metric > best:
            best = sel_metric
            save_checkpoint(
                {"model": model.state_dict(), "cfg": cfg.raw, "epoch": epoch, "val": res},
                ckpt_path,
            )
            tag = "  *saved"
        print(
            f"[teacherA] epoch {epoch:03d}  loss={total / max(n, 1):.4f}  "
            + format_summary("val", res["overall"])
            + tag
        )

    print(f"[teacherA] best selection metric={best:.4f}  checkpoint={ckpt_path}")
    print("[teacherA] GATE: compare the above against scripts/01_eval_copy_baseline.py.")


if __name__ == "__main__":
    main()
