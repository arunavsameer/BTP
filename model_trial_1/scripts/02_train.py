"""Train RS-JEPA with region shuffle + 180° opposite-cell rotation.

Training view: shuffled mosaic (same permutation on now-clip and future-clip).
Z is unpermuted before the predictor so deploy layout is what P/H see.
Eval / deploy: no shuffle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.config import Config, resolve_device  # noqa: E402
from rs_jepa.data import load_json  # noqa: E402
from rs_jepa.data.dataset import CollisionDataset  # noqa: E402
from rs_jepa.data.gpu_preprocess import prepare_batch  # noqa: E402
from rs_jepa.data.shuffle import apply_shuffle_pair, sample_perm  # noqa: E402
from rs_jepa.engine import (  # noqa: E402
    configure_runtime,
    evaluate_future_heatmap,
    evaluate_wearable,
    save_checkpoint,
    set_seed,
    wearable_better,
)
from rs_jepa.losses import (  # noqa: E402
    false_positive_penalty,
    jepa_distance,
    warning_alignment_loss,
    weighted_focal_mse,
)
from rs_jepa.metrics import format_summary, format_wearable  # noqa: E402
from rs_jepa.models.student import RSJEPA  # noqa: E402
from rs_jepa.models.tiny_cnn import count_params  # noqa: E402


def shuffle_prob_for_epoch(epoch: int, epochs: int, cfg: Config) -> float:
    p1 = float(cfg.get("shuffle.prob", 0.8))
    if not cfg.get("shuffle.curriculum", False):
        return p1
    p0 = float(cfg.get("shuffle.prob_start", p1))
    t = (epoch - 1) / max(epochs - 1, 1)
    return p0 + (p1 - p0) * t


def _loader_kwargs(workers: int, prefetch: int, pin: bool = True) -> dict:
    kw: dict = {"num_workers": workers, "pin_memory": pin}
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
    prefetch = int(cfg.get("train.prefetch_factor", 3))
    train_ds = CollisionDataset(
        cache_dir,
        split["train"],
        size,
        offsets,
        tau,
        n_frames,
        stride=2,
        train=True,
        mid_tau_frames=mid,
    )
    val_ds = CollisionDataset(
        cache_dir,
        split["val"],
        size,
        offsets,
        tau,
        n_frames,
        stride=3,
        train=False,
        mid_tau_frames=mid,
    )
    has_rare = split.get("has_rare", {})
    weights = train_ds.sample_weights(
        has_rare,
        hot_mult=float(cfg.get("train.hot_mult", 5.0)),
        rare_mult=float(cfg.get("train.rare_mult", 2.5)),
        change_mult=float(cfg.get("train.change_mult", 3.0)),
        quiet_mult=float(cfg.get("train.quiet_mult", 1.0)),
        quiet_peak=float(cfg.get("train.quiet_peak", 0.25)),
        quiet_ep_mult=float(cfg.get("train.quiet_ep_mult", 1.0)),
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    kw = _loader_kwargs(workers, prefetch)
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, drop_last=True, **kw
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, **kw)
    return train_ds, val_ds, train_loader, val_loader


def warning_index_tensors(cfg: Config, device: str):
    warn = cfg.get("warning")
    row_w = torch.tensor(warn["row_weights"], dtype=torch.float32, device=device)
    left = torch.tensor(warn["left_cols"], dtype=torch.long, device=device)
    center = torch.tensor(warn["center_cols"], dtype=torch.long, device=device)
    right = torch.tensor(warn["right_cols"], dtype=torch.long, device=device)
    return row_w, left, center, right, float(warn["caution_threshold"])


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
    print(f"[rsjepa] device={device}")

    split = load_json(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    print("[rsjepa] building weighted sampler (hot / rare / changing frames)...")
    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg, split)
    print(f"[rsjepa] train samples={len(train_ds)}  val samples={len(val_ds)}")
    probe = train_ds[0]["h_future"]
    g = int(cfg.get("student.feature_grid"))
    if tuple(probe.shape[-2:]) != (g, g):
        raise SystemExit(f"cached heatmap {tuple(probe.shape)} != {g}x{g}. Recache with this config.")
    warn_rows = len(cfg.get("warning.row_weights"))
    if warn_rows != g:
        raise SystemExit(f"warning.row_weights length {warn_rows} != grid {g}")

    grid = int(cfg.get("student.feature_grid", 5))
    grid_data = int(cfg.get("data.target_grid", grid))
    if grid != grid_data:
        raise SystemExit(f"student.feature_grid={grid} != data.target_grid={grid_data}")
    model = RSJEPA.from_config(cfg).to(device)
    print(
        f"[rsjepa] params={count_params(model):,}  n_frames={model.n_frames}  "
        f"grid={grid}x{grid}  cells={grid * grid}  "
        f"z={int(cfg.get('student.z_channels'))}  loom={cfg.get('student.use_loom')}  "
        f"copy_residual={bool(cfg.get('student.copy_residual', False))}"
    )

    ckpt_dir = Path(cfg.get("paths.ckpt_dir"))
    best_path = ckpt_dir / "student_best.pt"
    last_path = ckpt_dir / "student_last.pt"
    alias_path = ckpt_dir / "student.pt"
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("train.lr", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    start_epoch = 1
    best_wear: dict | None = None
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if best_path.exists():
            best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
            best_wear = (best_ck.get("val") or {}).get("wearable")
        print(f"[rsjepa] resumed epoch={ckpt.get('epoch')} best_wear={best_wear}")

    epochs = args.epochs or int(cfg.get("train.epochs", 40))
    scheduler = None
    if cfg.get("train.cosine", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - start_epoch + 1, 1), eta_min=float(cfg.get("train.lr", 1e-3)) * 0.05
        )

    use_cuda = device.startswith("cuda")
    use_amp = bool(cfg.get("train.amp", True)) and use_cuda
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    print(f"[rsjepa] amp={use_amp} dtype={amp_dtype}")
    print(
        f"[rsjepa] shuffle curriculum={cfg.get('shuffle.curriculum')}  "
        f"prob={cfg.get('shuffle.prob')}  start={cfg.get('shuffle.prob_start')}"
    )
    print(
        "[rsjepa] wearable score=1-0.5*nuisance-0.5*miss  "
        f"gates nuis<={cfg.get('selection.max_nuisance', 0.20)} "
        f"miss<={cfg.get('selection.max_miss', 0.25)}  "
        f"latch={cfg.get('selection.use_latch', True)}"
    )

    focal_w = float(cfg.get("train.focal_weight", 1.0))
    gamma = float(cfg.get("train.focal_gamma", 2.0))
    change_w = float(cfg.get("train.change_weight", 0.0))
    fn_w = float(cfg.get("train.fn_weight", 0.0))
    lam = float(cfg.get("jepa.lambda", 0.15))
    now_w = float(cfg.get("jepa.now_weight", 0.25))
    mid_w = float(cfg.get("jepa.mid_weight", 0.0))
    dir_w = float(cfg.get("jepa.dir_weight", 0.0))
    fp_w = float(cfg.get("jepa.fp_weight", 0.35))
    fp_thr = float(cfg.get("jepa.fp_safe_threshold", 0.20))
    fp_center = float(cfg.get("jepa.fp_center_mult", 1.0))
    delta_w = float(cfg.get("jepa.delta_weight", 0.0))
    clip = float(cfg.get("train.grad_clip", 1.0))
    row_w, left, center, right, caution = warning_index_tensors(cfg, device)

    def predict_fn(batch):
        return model.predict_future_heatmap(batch["frames"])

    Path(cfg.get("paths.log_dir")).mkdir(parents=True, exist_ok=True)
    n_batches = len(train_loader)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        model.target.eval()
        shuf_p = shuffle_prob_for_epoch(epoch, epochs, cfg)
        tot = tot_h = tot_j = tot_fp = tot_now = tot_mid = tot_dir = 0.0
        n = 0
        print(
            f"[rsjepa] epoch {epoch:03d}/{epochs} start  shuf={shuf_p:.2f}  batches={n_batches}",
            flush=True,
        )
        for step, batch in enumerate(train_loader, start=1):
            batch = prepare_batch(batch, device, augment=True)
            frames = batch["frames"]
            frames_f = batch["frames_future"]
            h_now = batch["h_now"]
            h_fut = batch["h_future"]
            h_mid = batch["h_mid"]
            if shuf_p > 0:
                perm, rot = sample_perm(frames.shape[0], grid, shuf_p, frames.device)
                frames_s, frames_fs = apply_shuffle_pair(frames, frames_f, perm, rot, grid)
            else:
                perm = None
                frames_s, frames_fs = frames, frames_f

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(frames_s, frames_fs, perm=perm)
                h_pred = out["h_plus_hat"]
                h_now_pred = out["h_now_hat"]
                change = (h_fut - h_now).abs()
                l_h = weighted_focal_mse(h_pred, h_fut, focal_w, gamma, change, change_w, fn_w)
                l_now = weighted_focal_mse(h_now_pred, h_now, focal_w, gamma, fn_weight=fn_w)
                l_mid = weighted_focal_mse(out["h_mid_hat"], h_mid, focal_w, gamma, fn_weight=fn_w)
                l_fp = false_positive_penalty(
                    h_pred, h_fut, fp_thr, center_cols=center, center_mult=fp_center
                )
                l_j = jepa_distance(out["z_plus_hat"], out["z_plus"])
                l_dir = warning_alignment_loss(
                    h_pred, h_fut, row_w, left, center, right, caution
                )
                l_delta = h_pred.new_zeros(())
                if delta_w > 0 and out.get("delta") is not None:
                    l_delta = torch.nn.functional.mse_loss(
                        out["delta"].float(), (h_fut - h_now).float().clamp(-1.0, 1.0)
                    )
                loss = (
                    l_h
                    + now_w * l_now
                    + mid_w * l_mid
                    + dir_w * l_dir
                    + fp_w * l_fp
                    + lam * l_j
                    + delta_w * l_delta
                )

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            model.update_ema()

            bs = h_fut.shape[0]
            tot += float(loss.detach()) * bs
            tot_h += float(l_h.detach()) * bs
            tot_j += float(l_j.detach()) * bs
            tot_fp += float(l_fp.detach()) * bs
            tot_now += float(l_now.detach()) * bs
            tot_mid += float(l_mid.detach()) * bs
            tot_dir += float(l_dir.detach()) * bs
            n += bs
            if step % 200 == 0 or step == n_batches:
                print(
                    f"[rsjepa] epoch {epoch:03d}  step {step}/{n_batches}  "
                    f"batch_loss={float(loss.detach()):.4f}",
                    flush=True,
                )

        if scheduler is not None:
            scheduler.step()

        model.eval()
        res = evaluate_future_heatmap(predict_fn, val_loader, cfg, device, val_ds.episodes, has_rare)
        wear = evaluate_wearable(cfg, val_ds.episodes, has_rare, device, model=model)
        res["wearable"] = wear["overall"]
        res["wearable_rare"] = wear["rare"]
        payload = {"model": model.state_dict(), "cfg": cfg.raw, "epoch": epoch, "val": res}
        save_checkpoint(payload, last_path)
        tag = "  *last"
        if wearable_better(wear["overall"], best_wear):
            best_wear = wear["overall"]
            save_checkpoint(payload, best_path)
            save_checkpoint(payload, alias_path)
            tag = "  *best+last"
        den = max(n, 1)
        print(
            f"[rsjepa] epoch {epoch:03d}  shuf={shuf_p:.2f}  loss={tot / den:.4f}  "
            f"L_H={tot_h / den:.4f}  L_now={tot_now / den:.4f}  L_mid={tot_mid / den:.4f}  "
            f"L_dir={tot_dir / den:.4f}  L_FP={tot_fp / den:.4f}  L_JEPA={tot_j / den:.4f}  "
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
    print(f"[rsjepa] best wearable {best_txt}  best={best_path}  last={last_path}")


if __name__ == "__main__":
    main()
