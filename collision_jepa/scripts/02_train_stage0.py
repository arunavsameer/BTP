"""Stage 0: train the tiny CNN to predict the future 5x5 heatmap directly.

No V-JEPA, no Z. This must beat the copy baseline (especially high-threat recall)
before we invest in the teacher. Runs on a 4 GB GPU.

Run:  python scripts/02_train_stage0.py
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
from collision_jepa.data.dataset import CollisionDataset  # noqa: E402
from collision_jepa.data.splits import load_split  # noqa: E402
from collision_jepa.engine import (  # noqa: E402
    configure_runtime,
    evaluate_future_heatmap,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)
from collision_jepa.losses import weighted_focal_mse  # noqa: E402
from collision_jepa.metrics import format_summary  # noqa: E402
from collision_jepa.models.baseline import Stage0Model  # noqa: E402
from collision_jepa.models.tiny_cnn import count_params  # noqa: E402


def build_loaders(cfg: Config, split: dict):
    cache_dir = cfg.get("data.cache_dir")
    size = int(cfg.get("student.img_size"))
    offsets = cfg.get("student.frame_offsets")
    tau = int(cfg.get("horizon.tau_frames"))
    n_frames = int(cfg.get("data.n_frames", 150))
    bs = int(cfg.get("train.batch_size"))
    workers = int(cfg.get("train.num_workers", 0))

    aug = ClipAugmentor(seed=int(cfg.get("seed", 1337)))
    train_ds = CollisionDataset(
        cache_dir, split["train"], size, offsets, tau, n_frames, stride=2, train=True, augmentor=aug
    )
    val_ds = CollisionDataset(
        cache_dir, split["val"], size, offsets, tau, n_frames, stride=3, train=False
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
    parser.add_argument("--epochs", type=int, default=0, help="Override config epochs.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 1337)))
    configure_runtime()
    device = resolve_device(cfg.get("train.device", "auto"))
    print(f"[stage0] device={device}")

    split = load_split(cfg.get("data.split_file"))
    has_rare = split.get("has_rare", {})
    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg, split)
    print(f"[stage0] train samples={len(train_ds)}  val samples={len(val_ds)}")

    model = Stage0Model(
        width=int(cfg.get("student.cnn_width", 32)),
        feature_grid=int(cfg.get("student.feature_grid", 5)),
    ).to(device)
    print(f"[stage0] params={count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("train.lr", 1e-3)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
    )
    focal_w = float(cfg.get("train.focal_weight", 1.0))
    gamma = float(cfg.get("train.focal_gamma", 2.0))

    def loss_step(batch):
        pred = model(batch["frames"])
        return weighted_focal_mse(pred, batch["h_future"], focal_w, gamma)

    def predict_fn(batch):
        return model(batch["frames"])

    epochs = args.epochs or int(cfg.get("train.stage0_epochs", 40))
    ckpt_path = Path(cfg.get("paths.ckpt_dir")) / "stage0.pt"
    best_key = -1.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, loss_step)
        model.eval()
        res = evaluate_future_heatmap(predict_fn, val_loader, cfg, device, val_ds.episodes, has_rare)
        # Selection metric: prioritize catching real threats.
        sel = res["overall"]["high_threat_recall"] + res["overall"]["stop_f1"]
        tag = ""
        if sel > best_key:
            best_key = sel
            save_checkpoint(
                {"model": model.state_dict(), "cfg": cfg.raw, "epoch": epoch, "val": res},
                ckpt_path,
            )
            tag = "  *saved"
        print(f"[stage0] epoch {epoch:03d}  loss={loss:.4f}  " + format_summary("val", res["overall"]) + tag)

    print(f"[stage0] best selection metric={best_key:.4f}  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
