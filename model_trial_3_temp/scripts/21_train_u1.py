#!/usr/bin/env python3
"""Train a frozen-G U1 variance head. Does not overwrite G checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.training.checkpoint import extract_teacher_state, load_checkpoint, save_checkpoint  # noqa: E402
from collision_jepa3.u1_runtime import (  # noqa: E402
    cache_fine_features,
    load_dump,
    load_frozen_g,
    score_u1,
    weighted_var_loss,
)
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402


def _dataset(cfg, split: str, records, splits) -> TeacherDataset:
    return TeacherDataset(cfg, records, split, splits, dummy_backbone=False, use_token_cache=True)


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out-root", default="reports/direct_h1s_u1")
    parser.add_argument("--checkpoint-root", default="checkpoints/direct_h1s_u1")
    parser.add_argument("--logs-root", default="logs/direct_h1s_u1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=9.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", True)
    cfg.set("evaluation.val_subset_fraction", 1.0)
    cfg.set("training.micro_batch_size", int(args.batch_size))
    cfg.set("training.num_workers", int(args.num_workers))
    device = resolve_device(args.device)
    out_root = Path(args.out_root)
    ckpt_root = Path(args.checkpoint_root)
    log_root = Path(args.logs_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(describe_hardware(device), indent=2), flush=True)

    oracle_path = out_root / "oracles.json"
    val_dump_path = out_root / "dump_val.npz"
    if not oracle_path.exists() or not val_dump_path.exists():
        raise SystemExit("run scripts/20_u1_oracles.py first (needs dump_val.npz and oracles.json)")
    oracles = json.loads(oracle_path.read_text(encoding="utf-8"))
    val_dump = load_dump(val_dump_path)

    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    train_ds = _dataset(cfg, "train", records, splits)
    val_ds = _dataset(cfg, "val", records, splits)

    model = load_frozen_g(cfg, Path(args.teacher_ckpt), device, variance_head=True)
    n_train = model.freeze_except_variance()
    print(f"trainable variance params={n_train}", flush=True)
    if n_train > 200:
        raise SystemExit(f"U1 head too large: {n_train} params")
    print("caching frozen 16x16 features (one adapter pass)", flush=True)
    train_cache = cache_fine_features(model, train_ds, cfg, device, canonical_only=True)
    val_cache = cache_fine_features(model, val_ds, cfg, device, canonical_only=False)
    print(
        json.dumps(
            {
                "train_canonical": int(train_cache["feat"].shape[0]),
                "val_windows": int(val_cache["feat"].shape[0]),
                "feat_shape": list(train_cache["feat"].shape[1:]),
            }
        ),
        flush=True,
    )
    opt = torch.optim.AdamW(model.variance_head.parameters(), lr=float(args.lr), weight_decay=1e-4)
    bs = int(args.batch_size)
    n_tr = int(train_cache["feat"].shape[0])

    history = []
    best_mae = float("inf")
    best_epoch = -1
    bad = 0
    t0 = time.time()
    for epoch in range(int(args.epochs)):
        model.eval()
        model.variance_head.train()
        running = 0.0
        n_batches = 0
        perm = torch.randperm(n_tr)
        for start in range(0, n_tr, bs):
            idx = perm[start : start + bs]
            feat = train_cache["feat"][idx].to(device=device, dtype=torch.float32)
            v_true = train_cache["v_true"][idx].to(device=device, dtype=torch.float32)
            mask = torch.ones(len(idx), dtype=torch.bool, device=device)
            v_hat = model.variance_head(feat)
            loss = weighted_var_loss(v_hat, v_true, mask, pos_weight=float(args.pos_weight))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            n_batches += 1
        model.variance_head.eval()
        with torch.no_grad():
            v_val = []
            for start in range(0, int(val_cache["feat"].shape[0]), 64):
                feat = val_cache["feat"][start : start + 64].to(device=device, dtype=torch.float32)
                v_val.append(model.variance_head(feat).cpu())
            v_val_t = torch.cat(v_val, dim=0)
            sel = val_cache["mask"].view(-1, 1, 1)
            n_cells = float(val_cache["mask"].sum().item() * 25)
            pair_mae = float(((v_val_t - val_cache["v_true"]).abs() * sel).sum() / max(n_cells, 1.0))
        val = {"pair_mae": pair_mae, "n_pair_cells": n_cells}
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "train_pair_windows": n_tr,
            **val,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        mae = float(val["pair_mae"])
        if mae + 1e-8 < best_mae:
            best_mae = mae
            best_epoch = epoch
            bad = 0
            blob = extract_teacher_state(model)
            blob["epoch"] = epoch
            blob["u1_pair_mae"] = mae
            blob["teacher_ckpt"] = str(args.teacher_ckpt)
            save_checkpoint(blob, ckpt_root / "u1_best.pt")
        else:
            bad += 1
            if bad >= int(args.patience) and epoch >= 4:
                print(f"early stop at epoch {epoch} best={best_epoch}", flush=True)
                break

    save_checkpoint(
        {**extract_teacher_state(model), "epoch": history[-1]["epoch"] if history else 0},
        ckpt_root / "u1_last.pt",
    )
    best_path = ckpt_root / "u1_best.pt"
    blob = load_checkpoint(best_path, map_location="cpu")
    model.variance_head.load_state_dict(blob["variance_head"])
    model.eval()
    model.variance_head.eval()
    with torch.no_grad():
        v_hat_chunks = []
        for start in range(0, int(val_cache["feat"].shape[0]), 64):
            feat = val_cache["feat"][start : start + 64].to(device=device, dtype=torch.float32)
            v_hat_chunks.append(model.variance_head(feat).cpu().numpy())
    v_hat = np.concatenate(v_hat_chunks, axis=0).astype(np.float32)
    mu_dump = val_dump["arrays"]["pred_mu"]
    mu_now = val_cache["mu"].float().numpy()
    mu_vs_dump = float(np.max(np.abs(mu_now.astype(np.float32) - mu_dump)))
    comparison = score_u1(val_dump, v_hat, oracles, n_boot=int(args.n_boot))
    comparison["mu_lock"] = {
        "max_abs_vs_g_dump": mu_vs_dump,
        "ok": bool(mu_vs_dump < 1e-5),
    }
    comparison["train"] = {
        "best_epoch": best_epoch,
        "best_pair_mae": best_mae,
        "trainable_params": n_train,
        "history": history,
        "teacher_ckpt": str(args.teacher_ckpt),
        "checkpoint": str(best_path),
    }
    np.savez_compressed(out_root / "val_vhat.npz", v_hat=v_hat)
    out = out_root / "u1_comparison.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (log_root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in history) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "selection": comparison["selection"], "mu_lock": comparison["mu_lock"], "var": comparison["variance_vs_true"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
