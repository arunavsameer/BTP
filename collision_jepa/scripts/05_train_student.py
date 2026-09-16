"""Stage C: train the tiny student.

Heatmap path: Ĥ+ = clamp(sg(Ĥ_t) + head(Zhat+ − Z_t), 0, 1). decoder(Zhat+) is a
balanced auxiliary heatmap so the JEPA predictor is on the product path.
JEPA is on by default at train.jepa_frac of L(H+). Pass --jepa-lambda 0 to disable.

Run:  python scripts/05_train_student.py --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.config import Config, resolve_device  # noqa: E402
from collision_jepa.data.augment import ClipAugmentor  # noqa: E402
from collision_jepa.data.dataset import StudentDataset  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.engine import (  # noqa: E402
    configure_runtime,
    evaluate_future_heatmap,
    move_batch,
    save_checkpoint,
    set_seed,
)
from collision_jepa.losses import student_loss  # noqa: E402
from collision_jepa.metrics import format_summary  # noqa: E402
from collision_jepa.models.student import Student  # noqa: E402
from collision_jepa.models.tiny_cnn import count_params  # noqa: E402


def build_loaders(cfg: Config, split: dict):
    cache_dir = cfg.get("data.cache_dir")
    size = int(cfg.get("student.img_size"))
    offsets = cfg.get("student.frame_offsets")
    tau = int(cfg.get("horizon.tau_frames"))
    n_frames = int(cfg.get("data.n_frames", 150))
    bs = int(cfg.get("train.batch_size"))
    workers = int(cfg.get("train.num_workers", 0))
    # Windows DataLoader processes each import torch/CUDA and blow the page file
    # on a 4 GB laptop. The student reads mmap'd npy; workers=0 is fine.
    if sys.platform == "win32":
        workers = 0
    tcf = int(cfg.get("teacher.clip_frames", 8))
    tst = int(cfg.get("teacher.clip_stride", 2))

    aug = ClipAugmentor(seed=int(cfg.get("seed", 1337)))
    train_ds = StudentDataset(
        cache_dir, split["train"], size, offsets, tau, n_frames, stride=2, train=True, augmentor=aug,
        teacher_clip_frames=tcf, teacher_stride=tst,
    )
    val_ds = StudentDataset(
        cache_dir, split["val"], size, offsets, tau, n_frames, stride=3, train=False,
        teacher_clip_frames=tcf, teacher_stride=tst,
    )
    persist = workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=workers,
        drop_last=True, pin_memory=True, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=persist,
    )
    return train_ds, val_ds, train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument(
        "--jepa-lambda",
        type=float,
        default=None,
        help="Override train.jepa_lambda. 0 = JEPA off; >0 enables JEPA at jepa_frac of L(H+).",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 1337)))
    configure_runtime()
    device = resolve_device(cfg.get("train.device", "auto"))
    print(f"[studentC] device={device}")

    split = load_split(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg, split)
    print(f"[studentC] train samples={len(train_ds)}  val samples={len(val_ds)}")

    model = Student(
        width=int(cfg.get("student.cnn_width", 32)),
        feature_grid=int(cfg.get("student.feature_grid", 5)),
        z_channels=int(cfg.get("student.z_channels", 16)),
        occ_channels=int(cfg.get("student.occ_channels", 4)),
    ).to(device)

    # Load + freeze the teacher decoder from Stage A.
    teacher_ckpt = Path(cfg.get("paths.ckpt_dir")) / "teacher.pt"
    dec_state = None
    if teacher_ckpt.exists():
        tstate = torch.load(teacher_ckpt, map_location=device)["model"]
        dec_state = {k[len("decoder.") :]: v for k, v in tstate.items() if k.startswith("decoder.")}
        model.load_frozen_decoder(dec_state)
        occ_state = {k[len("occupancy.") :]: v for k, v in tstate.items() if k.startswith("occupancy.")}
        if occ_state:
            model.load_frozen_occupancy(occ_state)
            print(f"[studentC] loaded + froze teacher decoder + occupancy from {teacher_ckpt}")
        else:
            print(f"[studentC] loaded + froze teacher decoder from {teacher_ckpt} (no occupancy keys)")
    else:
        print("[studentC] WARNING: teacher.pt not found; training decoder jointly (Stage-0-like).")

    print(f"[studentC] total params={count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.get("train.lr", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    lam = float(cfg.get("train.jepa_lambda", 0.0) if args.jepa_lambda is None else args.jepa_lambda)
    warning = cfg.raw["warning"]
    loss_kw = dict(
        focal_weight=float(cfg.get("train.focal_weight", 4.0)),
        focal_gamma=float(cfg.get("train.focal_gamma", 2.0)),
        change_weight=float(cfg.get("train.change_weight", 2.0)),
        now_heatmap_weight=float(cfg.get("train.now_heatmap_weight", 0.5)),
        smooth_l1_beta=float(cfg.get("train.smooth_l1_beta", 0.1)),
        readout_frac=float(cfg.get("train.readout_frac", 0.40)),
        stop_frac=float(cfg.get("train.stop_frac", 0.15)),
        stop_fn_weight=float(cfg.get("train.stop_fn_weight", 2.5)),
        stop_temperature=float(cfg.get("train.stop_temperature", 0.05)),
        stop_focal_gamma=float(cfg.get("train.stop_focal_gamma", 2.0)),
        latent_heatmap_frac=float(cfg.get("train.latent_heatmap_frac", 0.30)),
        occupancy_frac=float(cfg.get("train.occupancy_frac", 0.20)),
        occupancy_threshold=float(cfg.get("train.occupancy_threshold", 0.3)),
        occupancy_pos_weight=float(cfg.get("train.occupancy_pos_weight", 4.0)),
        bg_weight=float(cfg.get("train.bg_weight", 0.05)),
        ignore_below=float(cfg.get("train.ignore_below", 0.15)),
        fa_weight=float(cfg.get("train.fa_weight", 1.0)),
        fa_pred_thr=float(cfg.get("train.fa_pred_thr", 0.45)),
        fa_true_thr=float(cfg.get("train.fa_true_thr", 0.2)),
        jepa_frac=float(cfg.get("train.jepa_frac", 0.10)),
        jepa_lambda=lam,
        balance_max_boost=float(cfg.get("train.balance_max_boost", 5.0)),
    )
    print(
        f"[studentC] readout_frac={loss_kw['readout_frac']}  stop_frac={loss_kw['stop_frac']}  "
        f"latent_frac={loss_kw['latent_heatmap_frac']}  "
        f"change_weight={loss_kw['change_weight']}  jepa_lambda={lam}  jepa_frac={loss_kw['jepa_frac']}"
    )

    def predict_fn(batch):
        return model.predict_future_heatmap(batch["frames"])

    epochs = args.epochs or int(cfg.get("train.student_epochs", 60))
    ckpt_path = Path(cfg.get("paths.ckpt_dir")) / "student.pt"
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        acc = {k: 0.0 for k in ("total", "h_plus", "h_now", "readout", "stop", "jepa", "latent", "occ")}
        n = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            out = model(batch["frames"])
            loss, parts = student_loss(out, batch, warning, **loss_kw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = batch["h_future"].shape[0]
            n += bs
            for k in acc:
                acc[k] += float(parts[k].detach()) * bs
        model.eval()
        res = evaluate_future_heatmap(predict_fn, val_loader, cfg, device, val_ds.episodes, has_rare)
        sel = res["overall"]["high_threat_recall"] + res["overall"]["stop_f1"]
        tag = ""
        if sel > best:
            best = sel
            save_checkpoint(
                {"model": model.state_dict(), "cfg": cfg.raw, "epoch": epoch, "val": res, "jepa_lambda": lam},
                ckpt_path,
            )
            tag = "  *saved"
        d = max(n, 1)
        print(
            f"[studentC] epoch {epoch:03d}  loss={acc['total'] / d:.4f}  "
            f"L_H+={acc['h_plus'] / d:.4f}  L_Ht={acc['h_now'] / d:.4f}  "
            f"L_read={acc['readout'] / d:.4f}  L_stop={acc['stop'] / d:.4f}  "
            f"L_lat={acc['latent'] / d:.4f}  L_occ={acc['occ'] / d:.4f}  L_JEPA={acc['jepa'] / d:.4f}  "
            + format_summary("val", res["overall"])
            + tag
        )

    print(f"[studentC] best selection metric={best:.4f}  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
