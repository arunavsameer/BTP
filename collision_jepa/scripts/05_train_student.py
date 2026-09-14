"""Stage C: train the tiny student.

Experiment A (current default): L = L_H only (jepa_lambda=0), with Z flipped
together with RGB/H. Architecture, 3-frame input, 128px, and L_H are unchanged.

When bringing JEPA back later: --jepa-lambda 0.01 then 0.03 (not 0.1).
  L_H    = weighted focal MSE between Hhat+ and the REAL future heatmap H(t+tau)
  L_JEPA = distance between Zhat+ and the frozen teacher's Z+ (std-normalized)

The teacher decoder is loaded from the Stage A checkpoint and FROZEN.

Run:  python scripts/05_train_student.py
      python scripts/05_train_student.py --epochs 30 --jepa-lambda 0
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
from collision_jepa.losses import jepa_distance, weighted_focal_mse  # noqa: E402
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
        help="Override train.jepa_lambda. 0 = heatmap-only (Experiment A).",
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
    ).to(device)

    # Load + freeze the teacher decoder from Stage A.
    teacher_ckpt = Path(cfg.get("paths.ckpt_dir")) / "teacher.pt"
    dec_state = None
    if teacher_ckpt.exists():
        tstate = torch.load(teacher_ckpt, map_location=device)["model"]
        dec_state = {k[len("decoder.") :]: v for k, v in tstate.items() if k.startswith("decoder.")}
        model.load_frozen_decoder(dec_state)
        print(f"[studentC] loaded + froze teacher decoder from {teacher_ckpt}")
    else:
        print("[studentC] WARNING: teacher.pt not found; training decoder jointly (Stage-0-like).")

    print(f"[studentC] total params={count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.get("train.lr", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    focal_w = float(cfg.get("train.focal_weight", 1.0))
    gamma = float(cfg.get("train.focal_gamma", 2.0))
    lam = float(cfg.get("train.jepa_lambda", 0.0) if args.jepa_lambda is None else args.jepa_lambda)
    print(f"[studentC] jepa_lambda={lam}  (0 = Experiment A, heatmap-only)")

    def predict_fn(batch):
        return model.predict_future_heatmap(batch["frames"])

    epochs = args.epochs or int(cfg.get("train.student_epochs", 60))
    ckpt_path = Path(cfg.get("paths.ckpt_dir")) / "student.pt"
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, total_h, total_j, n = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            out = model(batch["frames"])
            l_h = weighted_focal_mse(out["h_plus_hat"], batch["h_future"], focal_w, gamma)
            if lam > 0:
                l_jepa = jepa_distance(out["z_plus_hat"], batch["z_plus"])
                loss = l_h + lam * l_jepa
            else:
                l_jepa = l_h.new_zeros(())
                loss = l_h
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = batch["h_future"].shape[0]
            total += float(loss.detach()) * bs
            total_h += float(l_h.detach()) * bs
            total_j += float(l_jepa.detach()) * bs
            n += bs
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
        print(
            f"[studentC] epoch {epoch:03d}  loss={total / max(n, 1):.4f}  "
            f"L_H={total_h / max(n, 1):.4f}  L_JEPA={total_j / max(n, 1):.4f}  "
            + format_summary("val", res["overall"])
            + tag
        )

    print(f"[studentC] best selection metric={best:.4f}  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
