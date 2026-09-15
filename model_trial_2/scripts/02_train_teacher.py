"""Stage A: fine-tune V-JEPA 2 into a collision teacher.

Phase 1 — freeze the ViT, train spatial adapter + residual predictor + decoder.
Phase 2 — LoRA on the last encoder blocks so the latent itself is collision-specific.

Selection is latched wearable score (minimize nuisance AND miss). Red = obstacle
(STOP), not-red on empty road (SAFE).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config, resolve_device  # noqa: E402
from cvjepa.data import load_json  # noqa: E402
from cvjepa.data.dataset import teacher_clip_indices  # noqa: E402
from cvjepa.engine import (  # noqa: E402
    configure_runtime,
    evaluate_wearable,
    save_checkpoint,
    set_seed,
    warning_index_tensors,
    wearable_better,
)
from cvjepa.losses import (  # noqa: E402
    false_positive_penalty,
    spatial_heatmap_weights,
    warning_alignment_loss,
    weighted_focal_mse,
)
from cvjepa.metrics import MetricAccumulator, format_summary, format_wearable  # noqa: E402
from cvjepa.models.teacher import CollisionTeacher  # noqa: E402
from cvjepa.models.tiny_cnn import count_params  # noqa: E402


def _clip_tensor(frames: np.ndarray, idx: list[int]) -> np.ndarray:
    clip = np.stack([np.asarray(frames[i]) for i in idx], axis=0)
    return np.transpose(clip.astype(np.float32) / 255.0, (0, 3, 1, 2))


def iter_episode_samples(cfg, ep_names, size, clip_frames, stride, tau, sample_stride):
    cache_dir = Path(cfg.get("data.cache_dir"))
    n_frames = int(cfg.get("data.n_frames", 150))
    t_lo = (clip_frames - 1) * stride
    t_hi = n_frames - 1 - tau
    for ep in ep_names:
        frames = np.load(cache_dir / ep / f"frames_{size}.npy", mmap_mode="r")
        heatmaps = np.load(cache_dir / ep / "heatmaps.npy")
        clips, h_now, h_fut = [], [], []
        for t in range(t_lo, t_hi + 1, sample_stride):
            idx = teacher_clip_indices(t, clip_frames, stride)
            if idx[0] < 0 or idx[-1] >= len(frames):
                continue
            clips.append(_clip_tensor(frames, idx))
            h_now.append(np.asarray(heatmaps[t], dtype=np.float32))
            h_fut.append(np.asarray(heatmaps[t + tau], dtype=np.float32))
        if clips:
            yield (
                torch.from_numpy(np.stack(clips)).float(),
                torch.from_numpy(np.stack(h_now)).float(),
                torch.from_numpy(np.stack(h_fut)).float(),
                ep,
            )


def heatmap_eval(model, cfg, ep_names, device, has_rare) -> dict:
    model.eval()
    size = int(cfg.get("teacher.img_size"))
    clip_frames = int(cfg.get("teacher.clip_frames"))
    stride = int(cfg.get("teacher.clip_stride"))
    tau = int(cfg.get("horizon.tau_frames"))
    sample_stride = int(cfg.get("teacher.sample_stride", 4))
    mb = int(cfg.get("teacher.micro_batch", 1))
    thr = float(cfg.get("heatmap.recall_threshold", 0.5))
    warn = cfg.get("warning")
    overall = MetricAccumulator(thr, warn)
    rare = MetricAccumulator(thr, warn)
    with torch.no_grad():
        for clips, _h_now, h_fut, ep in iter_episode_samples(
            cfg, ep_names, size, clip_frames, stride, tau, sample_stride
        ):
            for s in range(0, clips.shape[0], mb):
                cb = clips[s : s + mb].to(device)
                pred = model.predict_future_heatmap(cb).float().cpu().numpy()
                tb = h_fut[s : s + mb].numpy()
                for b in range(pred.shape[0]):
                    overall.update(pred[b], tb[b])
                    if has_rare.get(ep, False):
                        rare.update(pred[b], tb[b])
    return {"overall": overall.summary(), "rare": rare.summary()}


def teacher_loss(out, h_now, h_fut, cfg, spatial_w, row_w, left, center, right, caution, fp_idx, device):
    change = (h_fut - h_now).abs()
    focal_w = float(cfg.get("train.focal_weight", 1.0))
    gamma = float(cfg.get("train.focal_gamma", 2.0))
    change_w = float(cfg.get("train.change_weight", 2.5))
    fn_w = float(cfg.get("train.fn_weight", 0.6))
    l_h = weighted_focal_mse(
        out["h_plus_hat"], h_fut, focal_w, gamma, change, change_w, fn_w, cell_weights=spatial_w
    )
    l_now = weighted_focal_mse(
        out["h_now_hat"], h_now, focal_w, gamma, fn_weight=fn_w, cell_weights=spatial_w
    )
    l_fp = false_positive_penalty(
        out["h_plus_hat"],
        h_fut,
        float(cfg.get("jepa.fp_safe_threshold", 0.25)),
        center_cols=fp_idx,
        center_mult=float(cfg.get("jepa.fp_center_mult", 3.5)),
    )
    l_dir = warning_alignment_loss(
        out["h_plus_hat"], h_fut, row_w, left, center, right, caution
    )
    l_delta = torch.nn.functional.mse_loss(
        out["delta"].float(), (h_fut - h_now).float().clamp(-1.0, 1.0)
    )
    loss = (
        l_h
        + float(cfg.get("jepa.now_weight", 0.35)) * l_now
        + float(cfg.get("jepa.dir_weight", 0.30)) * l_dir
        + float(cfg.get("jepa.fp_weight", 0.45)) * l_fp
        + float(cfg.get("jepa.delta_weight", 0.50)) * l_delta
    )
    return loss, float(l_h.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", choices=["adapter", "lora", "both"], default="both")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 1337)))
    configure_runtime()
    device = resolve_device(cfg.get("teacher.device", "auto"))
    hf_cache = Path(cfg.get("teacher.hf_cache_dir", ROOT / "hf_cache"))
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    print(f"[teacher] device={device}  hf_cache={hf_cache}")

    split = load_json(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    ckpt_dir = Path(cfg.get("paths.ckpt_dir"))
    best_path = ckpt_dir / "teacher_best.pt"
    last_path = ckpt_dir / "teacher_last.pt"

    print(f"[teacher] loading {cfg.get('teacher.hf_model_id')} ...")
    model = CollisionTeacher(
        hf_model_id=cfg.get("teacher.hf_model_id"),
        z_channels=int(cfg.get("teacher.z_channels", 32)),
        feature_grid=int(cfg.get("student.feature_grid", 5)),
        pool_hidden=int(cfg.get("teacher.pool_hidden", 128)),
        cache_dir=hf_cache,
        torch_dtype=str(cfg.get("teacher.torch_dtype", "float16")),
    ).to(device)
    print(
        f"[teacher] hidden={model.backbone.hidden_size}  spatial={model.backbone.spatial}  "
        f"adapter_params={count_params(model.adapter)+count_params(model.predictor)+count_params(model.decoder)+count_params(model.delta_head):,}"
    )

    grid = int(cfg.get("student.feature_grid", 5))
    spatial_w = None
    if cfg.get("train.spatial_loss", True):
        spatial_w = spatial_heatmap_weights(
            grid,
            center=float(cfg.get("train.spatial_center", 2.2)),
            ring=float(cfg.get("train.spatial_ring", 1.5)),
            edge=float(cfg.get("train.spatial_edge", 1.0)),
            corner=float(cfg.get("train.spatial_corner", 0.65)),
        ).to(device)
    row_w, left, center, right, caution = warning_index_tensors(cfg, device)
    fp_idx = torch.tensor(list(cfg.get("jepa.fp_center_cols")), dtype=torch.long, device=device)
    size = int(cfg.get("teacher.img_size"))
    clip_frames = int(cfg.get("teacher.clip_frames"))
    stride = int(cfg.get("teacher.clip_stride"))
    tau = int(cfg.get("horizon.tau_frames"))
    sample_stride = int(cfg.get("teacher.sample_stride", 4))
    mb = int(cfg.get("teacher.micro_batch", 1))
    clip = float(cfg.get("train.grad_clip", 1.0))
    wd = float(cfg.get("train.weight_decay", 1e-4))

    lora_applied = False
    best_wear = None
    start_epoch = 1

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt.get("lora"):
            n = model.backbone.enable_lora(
                last_n=int(cfg.get("teacher.lora_last_blocks", 6)),
                rank=int(cfg.get("teacher.lora_rank", 16)),
                alpha=float(cfg.get("teacher.lora_alpha", 16.0)),
            )
            lora_applied = True
            print(f"[teacher] resumed LoRA on {n} Linear layers")
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if best_path.exists():
            best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
            best_wear = (best_ck.get("val") or {}).get("wearable")
        print(f"[teacher] resumed epoch={ckpt.get('epoch')} best={best_wear}")

    def run_epochs(phase: str, n_epochs: int, include_lora: bool, lr: float, epoch_offset: int) -> int:
        nonlocal best_wear, lora_applied
        if include_lora and not lora_applied:
            n = model.backbone.enable_lora(
                last_n=int(cfg.get("teacher.lora_last_blocks", 6)),
                rank=int(cfg.get("teacher.lora_rank", 16)),
                alpha=float(cfg.get("teacher.lora_alpha", 16.0)),
            )
            lora_applied = True
            print(f"[teacher] LoRA wrapped {n} Linear layers in last blocks")
        model.backbone.set_backbone_grad(include_lora)
        opt = torch.optim.AdamW(model.trainable_parameters(include_lora=include_lora), lr=lr, weight_decay=wd)
        use_amp = device.startswith("cuda")
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        last = epoch_offset
        for e in range(1, n_epochs + 1):
            epoch = epoch_offset + e
            last = epoch
            model.train()
            model.backbone.model.eval()
            tot, n = 0.0, 0
            order = list(split["train"])
            np.random.shuffle(order)
            print(f"[teacher] {phase} epoch {epoch:03d} start", flush=True)
            for ei, (clips, h_now, h_fut, _ep) in enumerate(
                iter_episode_samples(cfg, order, size, clip_frames, stride, tau, sample_stride)
            ):
                perm = np.random.permutation(clips.shape[0])
                for s in range(0, len(perm), mb):
                    sel = perm[s : s + mb]
                    cb = clips[sel].to(device, non_blocking=True)
                    hn = h_now[sel].to(device, non_blocking=True)
                    hf = h_fut[sel].to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        out = model(cb)
                        loss, _ = teacher_loss(
                            out, hn, hf, cfg, spatial_w, row_w, left, center, right, caution, fp_idx, device
                        )
                    scaler.scale(loss).backward()
                    if clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(include_lora=include_lora), clip)
                    scaler.step(opt)
                    scaler.update()
                    tot += float(loss.detach()) * len(sel)
                    n += len(sel)
                if (ei + 1) % 20 == 0 or ei == 0:
                    vram = ""
                    if device == "cuda":
                        vram = f"  VRAM={torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                    print(
                        f"[teacher] {phase} epoch {epoch:03d}  ep {ei + 1}/{len(order)}  "
                        f"loss={tot / max(n, 1):.4f}{vram}",
                        flush=True,
                    )
            hm = heatmap_eval(model, cfg, split["val"], device, has_rare)
            wear = evaluate_wearable(cfg, split["val"], has_rare, device, model=model, teacher=True)
            payload = {
                "model": model.state_dict(),
                "cfg": cfg.raw,
                "epoch": epoch,
                "phase": phase,
                "lora": include_lora,
                "val": {"heatmap": hm, "wearable": wear["overall"]},
            }
            save_checkpoint(payload, last_path)
            tag = "  *last"
            if wearable_better(wear["overall"], best_wear):
                best_wear = wear["overall"]
                save_checkpoint(payload, best_path)
                tag = "  *best+last"
            print(
                f"[teacher] {phase} epoch {epoch:03d}  loss={tot / max(n, 1):.4f}  "
                + format_summary("val", hm["overall"])
                + "  "
                + format_wearable("wear", wear["overall"])
                + tag,
                flush=True,
            )
            if device == "cuda":
                torch.cuda.empty_cache()
        return last

    epoch_at = start_epoch - 1
    if args.phase in ("adapter", "both") and epoch_at < int(cfg.get("teacher.adapter_epochs", 6)):
        done = int(cfg.get("teacher.adapter_epochs", 6)) - epoch_at
        if done > 0 and not lora_applied:
            epoch_at = run_epochs(
                "adapter",
                done,
                False,
                float(cfg.get("teacher.adapter_lr", 1e-3)),
                epoch_at,
            )
    if args.phase in ("lora", "both"):
        remaining = int(cfg.get("teacher.adapter_epochs", 6)) + int(cfg.get("teacher.lora_epochs", 14)) - epoch_at
        if remaining > 0:
            run_epochs(
                "lora",
                remaining,
                True,
                float(cfg.get("teacher.lora_lr", 2e-4)),
                epoch_at,
            )

    best_txt = "none"
    if best_wear is not None:
        best_txt = (
            f"score={best_wear['score']:.4f}  nuisance={best_wear['nuisance']:.3f}  "
            f"miss={best_wear['miss']:.3f}  eligible={bool(best_wear['eligible'])}"
        )
    print(f"[teacher] best wearable {best_txt}  best={best_path}")


if __name__ == "__main__":
    main()
