"""Stage C: train the tiny student against cached teacher Z and future heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config, resolve_device  # noqa: E402
from cvjepa.data import load_json  # noqa: E402
from cvjepa.data.dataset import StudentDataset  # noqa: E402
from cvjepa.data.gpu_preprocess import prepare_batch  # noqa: E402
from cvjepa.engine import (  # noqa: E402
    configure_runtime,
    evaluate_future_heatmap,
    evaluate_wearable,
    save_checkpoint,
    set_seed,
    warning_index_tensors,
    wearable_better,
)
from cvjepa.losses import (  # noqa: E402
    false_positive_penalty,
    jepa_distance,
    spatial_heatmap_weights,
    warning_alignment_loss,
    weighted_focal_mse,
)
from cvjepa.metrics import format_summary, format_wearable  # noqa: E402
from cvjepa.models.student import CollisionStudent  # noqa: E402
from cvjepa.models.tiny_cnn import count_params  # noqa: E402


def _loader_kwargs(workers: int, prefetch: int) -> dict:
    kw: dict = {"num_workers": workers, "pin_memory": True}
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, prefetch)
    return kw


def build_loaders(cfg: Config, split: dict):
    cache_dir = cfg.get("data.cache_dir")
    size = int(cfg.get("student.img_size"))
    offsets = cfg.get("student.frame_offsets")
    tau = int(cfg.get("horizon.tau_frames"))
    mid = int(cfg.get("horizon.mid_tau_frames", tau // 2))
    n_frames = int(cfg.get("data.n_frames", 150))
    bs = int(cfg.get("train.batch_size"))
    workers = int(cfg.get("train.num_workers", 0))
    train_ds = StudentDataset(
        cache_dir, split["train"], size, offsets, tau, n_frames, stride=2, train=True, mid_tau_frames=mid
    )
    val_ds = StudentDataset(
        cache_dir, split["val"], size, offsets, tau, n_frames, stride=3, train=False, mid_tau_frames=mid
    )
    weights = train_ds.sample_weights(
        split.get("has_rare", {}),
        hot_mult=float(cfg.get("train.hot_mult", 1.5)),
        rare_mult=float(cfg.get("train.rare_mult", 2.0)),
        change_mult=float(cfg.get("train.change_mult", 3.0)),
        quiet_mult=float(cfg.get("train.quiet_mult", 4.0)),
        quiet_peak=float(cfg.get("train.quiet_peak", 0.25)),
        quiet_ep_mult=float(cfg.get("train.quiet_ep_mult", 2.5)),
        center_cool_mult=float(cfg.get("train.center_cool_mult", 5.5)),
        center_cool_peak=float(cfg.get("train.center_cool_peak", 0.28)),
        center_cool_mode=str(cfg.get("train.center_cool_mode", "inner")),
    )
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    kw = _loader_kwargs(workers, int(cfg.get("train.prefetch_factor", 4)))
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, **kw)
    return train_ds, val_ds, train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 1337)))
    configure_runtime()
    device = resolve_device(cfg.get("train.device", "auto"))
    print(f"[student] device={device}")

    split = load_json(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg, split)
    print(f"[student] train={len(train_ds)}  val={len(val_ds)}")

    offsets = list(cfg.get("student.frame_offsets"))
    model = CollisionStudent(
        width=int(cfg.get("student.cnn_width", 32)),
        feature_grid=int(cfg.get("student.feature_grid", 5)),
        z_channels=int(cfg.get("student.z_channels", 32)),
        n_frames=len(offsets),
        use_loom=bool(cfg.get("student.use_loom", True)),
        copy_residual=bool(cfg.get("student.copy_residual", True)),
    ).to(device)

    teacher_ckpt = Path(cfg.get("paths.ckpt_dir")) / "teacher_best.pt"
    if teacher_ckpt.exists():
        tstate = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model"]
        model.load_frozen_heads(tstate)
        print(f"[student] froze teacher decoder/delta from {teacher_ckpt}")
    else:
        print("[student] WARNING: teacher_best.pt missing; decoder trains jointly")

    print(f"[student] params={count_params(model):,}  n_frames={model.n_frames}")

    ckpt_dir = Path(cfg.get("paths.ckpt_dir"))
    best_path = ckpt_dir / "student_best.pt"
    last_path = ckpt_dir / "student_last.pt"
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.get("train.lr", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    start_epoch = 1
    best_wear = None
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if best_path.exists():
            best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
            best_wear = (best_ck.get("val") or {}).get("wearable")

    epochs = args.epochs or int(cfg.get("train.student_epochs", 40))
    scheduler = None
    if cfg.get("train.cosine", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - start_epoch + 1, 1), eta_min=float(cfg.get("train.lr", 1e-3)) * 0.05
        )
    use_cuda = device.startswith("cuda")
    use_amp = bool(cfg.get("train.amp", True)) and use_cuda
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    grid = int(cfg.get("student.feature_grid", 5))
    spatial_w = None
    if cfg.get("train.spatial_loss", True):
        spatial_w = spatial_heatmap_weights(
            grid,
            float(cfg.get("train.spatial_center", 2.2)),
            float(cfg.get("train.spatial_ring", 1.5)),
            float(cfg.get("train.spatial_edge", 1.0)),
            float(cfg.get("train.spatial_corner", 0.65)),
        ).to(device)
    row_w, left, center, right, caution = warning_index_tensors(cfg, device)
    fp_idx = torch.tensor(list(cfg.get("jepa.fp_center_cols")), dtype=torch.long, device=device)
    lam = float(cfg.get("jepa.lambda", 0.25))
    clip = float(cfg.get("train.grad_clip", 1.0))
    n_batches = len(train_loader)

    def predict_fn(batch):
        return model.predict_future_heatmap(batch["frames"])

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tot = tot_h = tot_j = 0.0
        n = 0
        print(f"[student] epoch {epoch:03d}/{epochs} start  batches={n_batches}", flush=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = prepare_batch(batch, device, augment=True)
            frames = batch["frames"]
            h_now, h_fut, h_mid = batch["h_now"], batch["h_future"], batch["h_mid"]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(frames)
                change = (h_fut - h_now).abs()
                l_h = weighted_focal_mse(
                    out["h_plus_hat"],
                    h_fut,
                    float(cfg.get("train.focal_weight", 1.0)),
                    float(cfg.get("train.focal_gamma", 2.0)),
                    change,
                    float(cfg.get("train.change_weight", 2.5)),
                    float(cfg.get("train.fn_weight", 0.6)),
                    cell_weights=spatial_w,
                )
                l_now = weighted_focal_mse(
                    out["h_now_hat"],
                    h_now,
                    float(cfg.get("train.focal_weight", 1.0)),
                    float(cfg.get("train.focal_gamma", 2.0)),
                    fn_weight=float(cfg.get("train.fn_weight", 0.6)),
                    cell_weights=spatial_w,
                )
                l_mid = weighted_focal_mse(
                    out["h_mid_hat"],
                    h_mid,
                    float(cfg.get("train.focal_weight", 1.0)),
                    float(cfg.get("train.focal_gamma", 2.0)),
                    fn_weight=float(cfg.get("train.fn_weight", 0.6)),
                    cell_weights=spatial_w,
                )
                l_fp = false_positive_penalty(
                    out["h_plus_hat"],
                    h_fut,
                    float(cfg.get("jepa.fp_safe_threshold", 0.25)),
                    fp_idx,
                    float(cfg.get("jepa.fp_center_mult", 3.5)),
                )
                l_dir = warning_alignment_loss(
                    out["h_plus_hat"], h_fut, row_w, left, center, right, caution
                )
                l_j = out["z_plus_hat"].new_zeros(())
                if lam > 0 and "z_plus" in batch:
                    l_j = jepa_distance(out["z_plus_hat"], batch["z_plus"])
                l_delta = out["h_plus_hat"].new_zeros(())
                if out.get("delta") is not None:
                    l_delta = torch.nn.functional.mse_loss(
                        out["delta"].float(), (h_fut - h_now).float().clamp(-1.0, 1.0)
                    )
                loss = (
                    l_h
                    + float(cfg.get("jepa.now_weight", 0.35)) * l_now
                    + float(cfg.get("jepa.mid_weight", 0.20)) * l_mid
                    + float(cfg.get("jepa.dir_weight", 0.30)) * l_dir
                    + float(cfg.get("jepa.fp_weight", 0.45)) * l_fp
                    + lam * l_j
                    + float(cfg.get("jepa.delta_weight", 0.50)) * l_delta
                )
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            bs = h_fut.shape[0]
            tot += float(loss.detach()) * bs
            tot_h += float(l_h.detach()) * bs
            tot_j += float(l_j.detach()) * bs
            n += bs
            if step % 200 == 0 or step == n_batches:
                print(
                    f"[student] epoch {epoch:03d}  step {step}/{n_batches}  "
                    f"batch_loss={float(loss.detach()):.4f}",
                    flush=True,
                )
        if scheduler is not None:
            scheduler.step()
        model.eval()
        res = evaluate_future_heatmap(predict_fn, val_loader, cfg, device, val_ds.episodes, has_rare)
        wear = evaluate_wearable(cfg, val_ds.episodes, has_rare, device, model=model)
        payload = {"model": model.state_dict(), "cfg": cfg.raw, "epoch": epoch, "val": {**res, "wearable": wear["overall"]}}
        save_checkpoint(payload, last_path)
        tag = "  *last"
        if wearable_better(wear["overall"], best_wear):
            best_wear = wear["overall"]
            save_checkpoint(payload, best_path)
            tag = "  *best+last"
        print(
            f"[student] epoch {epoch:03d}  loss={tot / max(n, 1):.4f}  L_H={tot_h / max(n, 1):.4f}  "
            f"L_JEPA={tot_j / max(n, 1):.4f}  "
            + format_summary("val", res["overall"])
            + "  "
            + format_wearable("wear", wear["overall"])
            + tag,
            flush=True,
        )

    best_txt = "none"
    if best_wear is not None:
        best_txt = (
            f"score={best_wear['score']:.4f}  nuisance={best_wear['nuisance']:.3f}  "
            f"miss={best_wear['miss']:.3f}  eligible={bool(best_wear['eligible'])}"
        )
    print(f"[student] best wearable {best_txt}  best={best_path}")


if __name__ == "__main__":
    main()
