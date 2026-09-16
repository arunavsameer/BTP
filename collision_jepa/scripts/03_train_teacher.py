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
from collision_jepa.data.dataset import pack_teacher_clips  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.data.unzip import episode_dir_map, resolve_dataset_root  # noqa: E402
from collision_jepa.data.video import load_or_cache_frames, read_cached_native_k  # noqa: E402
from collision_jepa.engine import (  # noqa: E402
    configure_runtime,
    prefetch_items,
    save_checkpoint,
    set_seed,
    uint8_clips_to_device,
)
from collision_jepa.losses import collision_heatmap_loss, occupancy_bce  # noqa: E402
from collision_jepa.metrics import MetricAccumulator, format_summary  # noqa: E402
from collision_jepa.models.teacher import CollisionTeacher  # noqa: E402


def _dir_map(cfg: Config) -> dict[str, Path]:
    return episode_dir_map(resolve_dataset_root(cfg.get("data.raw_dir")))


def load_episode_pack(ep, dir_map, cache_dir, size, clip_frames, stride, sample_stride):
    if ep not in dir_map:
        return None
    hm_path = cache_dir / ep / "heatmaps.npy"
    if not hm_path.exists():
        return None
    frames = load_or_cache_frames(dir_map[ep], cache_dir, size, cache_name=ep)
    heatmaps = np.load(hm_path)
    packed = pack_teacher_clips(frames, heatmaps, clip_frames, stride, sample_stride)
    if packed is None:
        return None
    clips, targets, _ends = packed
    return clips, targets, ep


def expand_train_episodes(ep_names, has_rare, cache_dir, rare_factor: int, k5_factor: int) -> list[str]:
    """Repeat NEAR_MISS/CRITICAL and native k=5 episodes in the teacher epoch order."""
    out: list[str] = []
    for ep in ep_names:
        out.append(ep)
        extra = 0
        if has_rare.get(ep, False):
            extra += max(rare_factor, 1) - 1
        if read_cached_native_k(cache_dir, ep) == 5:
            extra += max(k5_factor, 1) - 1
        out.extend([ep] * extra)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=8, help="Clips per forward pass (VRAM).")
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoints/teacher.pt.")
    parser.add_argument("--rare-oversample", type=int, default=None)
    parser.add_argument("--k5-oversample", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 1337)))
    configure_runtime()
    device = resolve_device(cfg.get("teacher.device", "auto"))
    print(f"[teacherA] device={device}")

    split = load_split(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    dir_map = _dir_map(cfg)
    print(
        f"[teacherA] split train={len(split['train'])}  val={len(split['val'])}  "
        f"test={len(split.get('test') or [])}  mapped={len(dir_map)}"
    )

    size = int(cfg.get("teacher.img_size", 256))
    clip_frames = int(cfg.get("teacher.clip_frames", 8))
    stride = int(cfg.get("teacher.clip_stride", 2))
    tau = int(cfg.get("horizon.tau_frames", 30))
    del tau
    sample_stride = 5

    print(f"[teacherA] loading frozen V-JEPA-2 ({cfg.get('teacher.hf_model_id')}) ...")
    model = CollisionTeacher(
        hf_model_id=cfg.get("teacher.hf_model_id"),
        z_channels=int(cfg.get("teacher.z_channels", 16)),
        feature_grid=int(cfg.get("student.feature_grid", 5)),
        occ_channels=int(cfg.get("student.occ_channels", 4)),
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
    hm_kw = dict(
        focal_weight=float(cfg.get("train.focal_weight", 4.0)),
        gamma=float(cfg.get("train.focal_gamma", 2.0)),
        bg_weight=float(cfg.get("train.bg_weight", 0.05)),
        ignore_below=float(cfg.get("train.ignore_below", 0.15)),
        fa_weight=float(cfg.get("train.fa_weight", 1.0)),
        fa_pred_thr=float(cfg.get("train.fa_pred_thr", 0.45)),
        fa_true_thr=float(cfg.get("train.fa_true_thr", 0.2)),
        beta=float(cfg.get("train.smooth_l1_beta", 0.1)),
    )
    occ_w = float(cfg.get("train.occupancy_weight", 0.5))
    occ_thr = float(cfg.get("train.occupancy_threshold", 0.3))
    occ_pos = float(cfg.get("train.occupancy_pos_weight", 4.0))
    rare_factor = int(args.rare_oversample if args.rare_oversample is not None else cfg.get("train.rare_oversample", 3))
    k5_factor = int(args.k5_oversample if args.k5_oversample is not None else cfg.get("train.k5_oversample", 2))
    cache_dir = Path(cfg.get("data.cache_dir"))
    mb = args.micro_batch
    print(f"[teacherA] micro-batch={mb}  (raise this until VRAM is ~3 GB; 1 underuses a 4 GB card)")

    def _load(ep):
        return load_episode_pack(ep, dir_map, cache_dir, size, clip_frames, stride, sample_stride)

    def run_eval(ep_names: list[str]) -> dict:
        model.eval()
        recall_threshold = float(cfg.get("heatmap.recall_threshold", 0.5))
        warn_cfg = cfg.get("warning")
        overall = MetricAccumulator(recall_threshold, warn_cfg)
        rare = MetricAccumulator(recall_threshold, warn_cfg)
        with torch.inference_mode():
            n_eps = len(ep_names)
            for ei, (clips_u8, targets_np, ep) in enumerate(prefetch_items(ep_names, _load)):
                for s in range(0, clips_u8.shape[0], mb):
                    cb, tb = uint8_clips_to_device(
                        clips_u8[s : s + mb], device, targets_np[s : s + mb]
                    )
                    _, h_hat, _occ = model(cb)
                    h_hat = h_hat.float().cpu().numpy()
                    tb_np = tb.float().cpu().numpy()
                    for b in range(h_hat.shape[0]):
                        overall.update(h_hat[b], tb_np[b])
                        if has_rare.get(ep, False):
                            rare.update(h_hat[b], tb_np[b])
                if (ei + 1) % 10 == 0 or ei == 0:
                    print(f"[teacherA] eval {ei + 1}/{n_eps}", flush=True)
        return {"overall": overall.summary(), "rare": rare.summary()}

    epochs = args.epochs or int(cfg.get("train.teacher_epochs", 40))
    ckpt_path = Path(cfg.get("paths.ckpt_dir")) / "teacher.pt"
    start_epoch = 1
    best = -1.0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        except RuntimeError as exc:
            raise SystemExit(
                "[teacherA] teacher.pt does not match the new adapter "
                "(attention pool / 3x3 decoder / occupancy). Train from scratch without --resume."
            ) from exc
        missing = [k for k in missing if ".predictor." not in k]
        unexpected = [k for k in unexpected if ".predictor." not in k]
        if missing:
            raise SystemExit(
                "[teacherA] teacher.pt does not match the new adapter "
                "(attention pool / 3x3 decoder / occupancy). Train from scratch without --resume."
            )
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
        ep_order = expand_train_episodes(
            list(split["train"]), has_rare, cache_dir, rare_factor, k5_factor
        )
        np.random.shuffle(ep_order)
        print(
            f"[teacherA] epoch {epoch:03d} train clips from {len(ep_order)} episode draws "
            f"(rare x{rare_factor}, k5 x{k5_factor})",
            flush=True,
        )
        for ei, (clips_u8, targets_np, ep) in enumerate(prefetch_items(ep_order, _load)):
            perm = np.random.permutation(clips_u8.shape[0])
            for s in range(0, len(perm), mb):
                sel = perm[s : s + mb]
                cb, tb = uint8_clips_to_device(clips_u8[sel], device, targets_np[sel])
                optimizer.zero_grad(set_to_none=True)
                _, h_hat, occ_logits = model(cb)
                loss = collision_heatmap_loss(h_hat, tb, **hm_kw)
                loss = loss + occ_w * occupancy_bce(
                    occ_logits, tb, occ_threshold=occ_thr, pos_weight=occ_pos
                )
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(sel)
                n += len(sel)
                del cb, tb, h_hat, occ_logits, loss
            if (ei + 1) % 5 == 0 or ei == 0:
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
        val_res = run_eval(list(split["val"]))
        test_names = list(split.get("test") or [])
        test_res = run_eval(test_names) if test_names else None
        sel_metric = val_res["overall"]["high_threat_recall"] + val_res["overall"]["stop_f1"]
        tag = ""
        if sel_metric > best:
            best = sel_metric
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.raw,
                    "epoch": epoch,
                    "val": val_res,
                    "test": test_res,
                },
                ckpt_path,
            )
            tag = "  *saved"
        print(
            f"[teacherA] epoch {epoch:03d}  loss={total / max(n, 1):.4f}  "
            + format_summary("val", val_res["overall"])
            + tag,
            flush=True,
        )
        if val_res["rare"]["n"]:
            print("[teacherA]          " + format_summary("val-rare", val_res["rare"]), flush=True)
        if test_res is not None:
            print("[teacherA]          " + format_summary("test", test_res["overall"]), flush=True)
            if test_res["rare"]["n"]:
                print("[teacherA]          " + format_summary("test-rare", test_res["rare"]), flush=True)

    print(f"[teacherA] best selection metric={best:.4f}  checkpoint={ckpt_path}")
    print("[teacherA] GATE: compare the above against scripts/01_eval_copy_baseline.py.")


if __name__ == "__main__":
    main()
