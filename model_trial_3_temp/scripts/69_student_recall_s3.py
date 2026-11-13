#!/usr/bin/env python3
"""Collision-JEPA S3 — Train Student From Scratch With Direct STOP/SAFE Supervision.

This script executes the complete S3 training from scratch:
1. Untrained backbone initialization from shared_backbone_init.pt (seed 1337).
2. Fresh spatial projection initialization.
3. Frozen Pair-CF parent decoder (hash 77f78c8714667617).
4. Direct two-threshold ordinal objective (0.25 and 0.60) with scenario-group balancing.
5. Gradient calibration for eta_F and lambda_ordinal.
6. 20 epochs training with AdamW and cosine decay.
7. Development evaluation at init, 5, 10, 15, 20.
8. Full 5-model comparison (S3 vs S2 vs S1 vs S0 vs Parent).
9. Paired scenario-group bootstrap (2,000 resamples).
10. Comprehensive report and artifact generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_dataset import (  # noqa: E402
    FDecPilotDataset,
    build_frame_store,
    make_fdec_loader,
)
from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report  # noqa: E402
from collision_jepa3.student_research.fdec_student import (  # noqa: E402
    FDEC_FRAME_OFFSETS,
    FDecNormStats,
    StudentS0,
    StudentS1,
    StudentS2,
    build_parent_decoder,
    decoder_state_hash,
    extract_backbone_state,
    init_shared_backbone,
    load_backbone_state,
)
from collision_jepa3.student_research.fdec_pilot_train import _backbone_hash, _huber_kd, apply_backbone  # noqa: E402
from collision_jepa3.student_research.s3_recall import (  # noqa: E402
    StudentS3,
    build_data_audit,
    calibrate_gradients_s3,
    classify_peak_tier,
    compute_balanced_weights,
    compute_module_hash,
    compute_ordinal_logits,
    compute_ordinal_targets,
    evaluate_predictions_comprehensive,
    paired_scenario_bootstrap,
    smooth_peak,
)
from collision_jepa3.utils.device import resolve_device  # noqa: E402

EXP_DIR = ROOT / "experiments/student_recall_s3"
REPORT_DIR = ROOT / "reports/student_recall_experiments"
TESTS_REPORT_DIR = ROOT / "reports/student_architecture_tests"
TEACHER_CACHE = TESTS_REPORT_DIR / "teacher_cache_train_pilot"
BASELINE_DIR = ROOT / "experiments/student_fdec_unoptimized_baseline"
EXP_S0 = ROOT / "experiments/student_fdec_s0"
EXP_S1 = ROOT / "experiments/student_fdec_s1"
EXP_S2 = ROOT / "experiments/student_fdec_s2_frozen_decoder"


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("s3_recall")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Collision-JEPA S3 Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    parser.set_defaults(seed=1337)
    args = parser.parse_args()

    device = str(resolve_device(args.device))
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    warning = cfg.get("warning") or {"caution_threshold": 0.25, "stop_threshold": 0.60}

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(EXP_DIR / "run.log")
    logger.info("=================================================================")
    logger.info("Collision-JEPA S3 — Train Student From Scratch With Direct Supervision")
    logger.info(f"Device: {device}, Seed: {args.seed}, Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}")
    logger.info("=================================================================")

    if (EXP_DIR / ".success").exists() and not args.force:
        logger.info(f".success already exists in {EXP_DIR}. Use --force to overwrite.")
        return

    # Set deterministic seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------------------
    # PART 1 — Initialization from scratch
    # -------------------------------------------------------------
    logger.info(">>> PART 1: Verifying initialization from scratch...")

    # 1.1 Verify shared_backbone_init.pt
    baseline_bb_path = BASELINE_DIR / "shared_backbone_init.pt"
    if not baseline_bb_path.exists():
        baseline_bb_path = TESTS_REPORT_DIR / "shared_backbone_init.pt"
    if not baseline_bb_path.exists():
        raise FileNotFoundError(f"Missing shared_backbone_init.pt in {BASELINE_DIR} or {TESTS_REPORT_DIR}")

    bb_state = torch.load(baseline_bb_path, map_location="cpu", weights_only=True)
    hash_bb_file = _backbone_hash(bb_state)
    logger.info(f"Loaded backbone initialization from {baseline_bb_path}, hash: {hash_bb_file}")

    fresh_bb_state = init_shared_backbone(seed=args.seed, n_frames=len(FDEC_FRAME_OFFSETS))
    hash_bb_fresh = _backbone_hash(fresh_bb_state)
    logger.info(f"Fresh untrained backbone initialization (seed {args.seed}) hash: {hash_bb_fresh}")

    if hash_bb_file != hash_bb_fresh or hash_bb_file != "057b56dde465bfec":
        raise ValueError(
            f"Backbone hash mismatch: file={hash_bb_file}, fresh={hash_bb_fresh}, expected=057b56dde465bfec. "
            "Refusing to train from non-verified initialization."
        )
    logger.info("Backbone verified 100% untrained (seed 1337).")

    # 1.2 Fresh spatial projection
    torch.manual_seed(args.seed)
    parent_dec = build_parent_decoder(cfg, device="cpu")
    dec_hash = decoder_state_hash(parent_dec)
    logger.info(f"Loaded Pair-CF parent decoder, hash: {dec_hash}")
    if dec_hash != "77f78c8714667617":
        raise ValueError(f"Parent decoder hash {dec_hash} != 77f78c8714667617. Refusing to continue.")

    # Instantiate StudentS3
    model = StudentS3(n_frames=len(FDEC_FRAME_OFFSETS), parent_decoder=parent_dec, proj_out_grid=16)
    load_backbone_state(model.backbone, bb_state)

    # Compute hashes and parameter audit
    bb_current_hash = _backbone_hash(extract_backbone_state(model.backbone))
    proj_hash = compute_module_hash(model.proj)
    param_audit = model.param_audit()

    logger.info(f"Model instantiated: Backbone hash: {bb_current_hash}, Proj hash: {proj_hash}, Decoder hash: {dec_hash}")
    logger.info(f"Parameter budget: {json.dumps(param_audit, indent=2)}")

    if not param_audit["under_700k"]:
        raise ValueError(f"Total parameters {param_audit['total_params']} exceeds 700,000 budget!")

    # Verify no trained student weights
    if EXP_S1.exists() and (EXP_S1 / "checkpoint_epoch_15.pt").exists():
        s1_trained = torch.load(EXP_S1 / "checkpoint_epoch_15.pt", map_location="cpu", weights_only=True)
        s1_bb_hash = _backbone_hash({k.replace("backbone.", ""): v for k, v in s1_trained.items() if k.startswith("backbone.") and k.replace("backbone.", "").split(".")[0] in ("stem", "encoder", "head")})
        if bb_current_hash == s1_bb_hash:
            raise RuntimeError("CRITICAL: Student backbone matches trained S1 weights!")

    # Save initial checkpoint
    init_ckpt = {
        "model_state_dict": model.state_dict(),
        "backbone_initialization_hash": bb_current_hash,
        "projection_initialization_hash": proj_hash,
        "parent_decoder_hash": dec_hash,
        "param_counts": param_audit,
        "seed": args.seed,
        "epoch": 0,
        "untrained_verification": {
            "backbone_is_untrained": True,
            "projection_is_fresh": True,
            "decoder_is_frozen_parent": True,
            "no_trained_student_weights": True,
        },
    }
    torch.save(init_ckpt, EXP_DIR / "checkpoint_initial.pt")
    logger.info(f"Saved initial checkpoint to {EXP_DIR / 'checkpoint_initial.pt'}")

    # -------------------------------------------------------------
    # PART 2 — Data and class audit
    # -------------------------------------------------------------
    logger.info(">>> PART 2: Data and class audit...")
    train_ids = json.loads((TESTS_REPORT_DIR / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((TESTS_REPORT_DIR / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    rows = [json.loads(line) for line in (TEACHER_CACHE / "index.jsonl").open()]
    train_rows = [rows[rid] for rid in train_ids]
    dev_rows = [rows[rid] for rid in dev_ids]

    gt_npz = np.load(TEACHER_CACHE / "gt_heatmaps.npz")
    train_gt = gt_npz["gt_1s"][train_ids]
    dev_gt = gt_npz["gt_1s"][dev_ids]

    integrity, fdec_norm_dict = build_integrity_report(cfg, TEACHER_CACHE, train_ids, dev_ids, device="cpu")
    verified_pairs_train = len(integrity.get("verified_pair_keys_train", integrity.get("verified_pair_keys", [])))
    verified_pairs_dev = [tuple(x) for x in integrity.get("verified_pair_keys_dev", [])]
    verified_dev_set = {tuple(x) for x in verified_pairs_dev}

    data_audit = build_data_audit(train_rows, train_gt, warning, verified_pairs_train)
    (EXP_DIR / "data_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")
    logger.info(f"Saved data audit to {EXP_DIR / 'data_audit.json'}")
    logger.info(f"Train counts: {data_audit['class_counts']}")
    logger.info(f"Train groups per class: {data_audit['independent_scenario_groups_by_class']}")

    # Precompute balanced weights
    train_peaks = train_gt.max(axis=(1, 2))
    w25_arr, w60_arr, weight_audit = compute_balanced_weights(train_rows, train_peaks)
    logger.info(f"Balanced weights audit: {json.dumps(weight_audit, indent=2)}")

    # Map row_id to sample weight
    max_row_id = max(max(train_ids), max(dev_ids)) + 1
    w25_table = torch.ones(max_row_id, dtype=torch.float32)
    w60_table = torch.ones(max_row_id, dtype=torch.float32)
    for rid, w25, w60 in zip(train_ids, w25_arr, w60_arr):
        w25_table[rid] = float(w25)
        w60_table[rid] = float(w60)

    # Prepare datasets & dataloaders
    logger.info("Preparing in-memory frame store (optimized_v2)...")
    store, _ = build_frame_store(cfg, TEACHER_CACHE, sorted(set(train_ids) | set(dev_ids)), FDEC_FRAME_OFFSETS)

    train_ds = FDecPilotDataset(TEACHER_CACHE, train_ids, FDEC_FRAME_OFFSETS, load_f_dec=True)
    dev_ds = FDecPilotDataset(TEACHER_CACHE, dev_ids, FDEC_FRAME_OFFSETS, load_f_dec=True)
    train_ds.attach_frame_store(store, {})
    dev_ds.attach_frame_store(store, {})

    train_loader = make_fdec_loader(train_ds, batch=args.batch, shuffle=True, num_workers=args.num_workers)
    dev_loader = make_fdec_loader(dev_ds, batch=args.batch, shuffle=False, num_workers=args.num_workers)

    # Normalization stats for F_dec (train targets only)
    fdec_norm_npz = TESTS_REPORT_DIR / "fdec_norm_train_only.npz"
    if fdec_norm_npz.exists():
        fn = np.load(fdec_norm_npz)
        norm = FDecNormStats(torch.from_numpy(fn["mu"]).view(-1, 1, 1), torch.from_numpy(fn["sigma"]).view(-1, 1, 1))
    else:
        norm = FDecNormStats(torch.from_numpy(fdec_norm_dict["mu"]).view(-1, 1, 1), torch.from_numpy(fdec_norm_dict["sigma"]).view(-1, 1, 1))
    norm_device = norm.to(device)

    # -------------------------------------------------------------
    # PART 4 & PART 7 — Gradient Calibration
    # -------------------------------------------------------------
    logger.info(">>> PART 4 & 7: Calibrating F_dec and ordinal loss coefficients...")
    model.to(device)

    # Grab 4 training batches
    calib_batches = []
    for b in train_loader:
        calib_batches.append(b)
        if len(calib_batches) == 4:
            break

    calib_results = calibrate_gradients_s3(
        model,
        calib_batches,
        norm_device,
        w25_table,
        w60_table,
        device=device,
        alpha_kd=0.5,
    )
    eta_f = float(calib_results["eta_f_selected"])
    lambda_ordinal = float(calib_results["lambda_ordinal_selected"])
    (EXP_DIR / "gradient_calibration.json").write_text(json.dumps(calib_results, indent=2), encoding="utf-8")
    logger.info(f"Gradient calibration results: {json.dumps(calib_results, indent=2)}")

    # -------------------------------------------------------------
    # PART 9 — Training schedule
    # -------------------------------------------------------------
    logger.info(">>> PART 9: Training schedule (20 epochs)...")
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Initial dev evaluation (epoch 0)
    logger.info("Running dev evaluation at initialization (epoch 0)...")
    dev_preds_init = []
    model.eval()
    with torch.no_grad():
        for b in dev_loader:
            b_clip = b["clip"].to(device)
            out = model(b_clip)
            dev_preds_init.append(out["Hhat_1s"].float().cpu().numpy())
    dev_preds_init = np.concatenate(dev_preds_init, axis=0)
    parent_pred_npz = np.load(TEACHER_CACHE / "teacher_pred.npz")
    parent_dev_hms = np.asarray(parent_pred_npz["H"][dev_ids], dtype=np.float32)

    dev_metrics_0 = evaluate_predictions_comprehensive(
        dev_preds_init, dev_gt, parent_dev_hms, dev_rows, warning, verified_dev_set
    )
    (EXP_DIR / "metrics_dev_epoch_00.json").write_text(json.dumps(dev_metrics_0, indent=2), encoding="utf-8")
    logger.info(f"Epoch 0 Dev: STOP recall={dev_metrics_0['stop_recall']:.4f}, SAFE recall={dev_metrics_0['safe_recall']:.4f}, Bal={dev_metrics_0['balanced_stop_safe_recall']:.4f}")

    decoder_hashes = [("epoch_00", dec_hash)]
    train_history = []
    dev_history = [{"epoch": 0, **dev_metrics_0}]

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_accum = defaultdict(float)
        exposed_counts = {"SAFE": 0, "THREAT": 0, "STOP": 0}
        n_batches = 0

        for batch in train_loader:
            b_clip = batch["clip"].to(device)
            gt_1s = batch["gt_1s"].to(device)
            h_teacher = batch["H_teacher"].to(device)
            f_teacher = batch["F_teacher"].to(device)
            rids = batch["row_id"].cpu()

            w25_b = w25_table[rids].to(device)
            w60_b = w60_table[rids].to(device)

            # Track exposed counts
            p_gt_b = gt_1s.view(gt_1s.shape[0], -1).amax(dim=-1).cpu().numpy()
            for p in p_gt_b:
                tier = classify_peak_tier(p, warning)
                exposed_counts[tier] += 1

            optimizer.zero_grad(set_to_none=True)

            out = model(b_clip)
            h_s = out["Hhat_1s"]
            f_s = out["F_student"]

            # 1. L_dense = L_gt + 0.5 * L_kd
            l_gt = F.mse_loss(h_s, gt_1s)
            l_kd = _huber_kd(h_s, h_teacher)
            l_dense = l_gt + 0.5 * l_kd

            # 2. L_fdec
            f_sn = (f_s - norm_device.mu) / (norm_device.sigma + 1e-6)
            f_tn = (f_teacher - norm_device.mu) / (norm_device.sigma + 1e-6)
            l_fdec = F.smooth_l1_loss(f_sn, f_tn)

            # 3. L_ordinal
            p_s = smooth_peak(h_s, tau=0.02)
            p_gt_t = gt_1s.view(gt_1s.shape[0], -1).amax(dim=-1)
            y25, y60 = compute_ordinal_targets(p_gt_t)
            z25, z60 = compute_ordinal_logits(p_s)
            l25 = F.binary_cross_entropy_with_logits(z25, y25, weight=w25_b)
            l60 = F.binary_cross_entropy_with_logits(z60, y60, weight=w60_b)
            l_ord = 0.5 * (l25 + l60)

            total_loss = l_dense + eta_f * l_fdec + lambda_ordinal * l_ord

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {ep}: {total_loss.item()}")

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
            optimizer.step()

            train_loss_accum["loss_total"] += float(total_loss.item())
            train_loss_accum["loss_dense"] += float(l_dense.item())
            train_loss_accum["loss_gt"] += float(l_gt.item())
            train_loss_accum["loss_kd"] += float(l_kd.item())
            train_loss_accum["loss_fdec"] += float(l_fdec.item())
            train_loss_accum["loss_ordinal"] += float(l_ord.item())
            train_loss_accum["loss_25"] += float(l25.item())
            train_loss_accum["loss_60"] += float(l60.item())
            n_batches += 1

        scheduler.step()
        ep_time = time.time() - t0

        cur_dec_hash = decoder_state_hash(model.decoder)
        if cur_dec_hash != dec_hash:
            raise RuntimeError(f"Decoder hash altered in epoch {ep}: {cur_dec_hash} != {dec_hash}")
        decoder_hashes.append((f"epoch_{ep:02d}", cur_dec_hash))

        ep_train_metrics = {
            "epoch": ep,
            "epoch_time_seconds": ep_time,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "losses": {k: v / max(n_batches, 1) for k, v in train_loss_accum.items()},
            "exposed_sample_counts": exposed_counts,
            "decoder_hash": cur_dec_hash,
            "decoder_hash_unchanged": bool(cur_dec_hash == dec_hash),
        }
        train_history.append(ep_train_metrics)
        (EXP_DIR / f"metrics_train_epoch_{ep:02d}.json").write_text(json.dumps(ep_train_metrics, indent=2), encoding="utf-8")

        # Save epoch checkpoint
        ckpt_payload = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "decoder_hash": cur_dec_hash,
            "param_counts": param_audit,
            "train_metrics": ep_train_metrics,
        }
        torch.save(ckpt_payload, EXP_DIR / f"checkpoint_epoch_{ep:02d}.pt")

        logger.info(
            f"Epoch {ep:02d}/{args.epochs} ({ep_time:.2f}s) — "
            f"Loss: {ep_train_metrics['losses']['loss_total']:.4f} "
            f"(dense: {ep_train_metrics['losses']['loss_dense']:.4f}, fdec: {ep_train_metrics['losses']['loss_fdec']:.4f}, ord: {ep_train_metrics['losses']['loss_ordinal']:.4f}) | "
            f"Exposed: STOP={exposed_counts['STOP']}, THREAT={exposed_counts['THREAT']}, SAFE={exposed_counts['SAFE']}"
        )

        # Dev evaluation at epochs 5, 10, 15, 20
        if ep in (5, 10, 15, 20):
            logger.info(f"Evaluating on development split at epoch {ep}...")
            model.eval()
            dev_preds = []
            with torch.no_grad():
                for b in dev_loader:
                    b_clip = b["clip"].to(device)
                    out = model(b_clip)
                    dev_preds.append(out["Hhat_1s"].float().cpu().numpy())
            dev_preds = np.concatenate(dev_preds, axis=0)
            dev_metrics = evaluate_predictions_comprehensive(
                dev_preds, dev_gt, parent_dev_hms, dev_rows, warning, verified_dev_set
            )
            dev_history.append({"epoch": ep, **dev_metrics})
            (EXP_DIR / f"metrics_dev_epoch_{ep:02d}.json").write_text(json.dumps(dev_metrics, indent=2), encoding="utf-8")
            logger.info(
                f"Epoch {ep:02d} Dev Metrics: "
                f"STOP Recall={dev_metrics['stop_recall']:.4f}, "
                f"SAFE Recall={dev_metrics['safe_recall']:.4f}, "
                f"Balanced={dev_metrics['balanced_stop_safe_recall']:.4f}, "
                f"AP_high={dev_metrics['ap_high']:.4f}, "
                f"Rec@5FA={dev_metrics['recall_at_5fa']:.4f}"
            )

    # Save final checkpoints
    shutil.copyfile(EXP_DIR / f"checkpoint_epoch_{args.epochs:02d}.pt", EXP_DIR / "checkpoint_epoch_20.pt")
    shutil.copyfile(EXP_DIR / f"checkpoint_epoch_{args.epochs:02d}.pt", EXP_DIR / "checkpoint_last.pt")

    decoder_integrity = {
        "verified_constant_hash": dec_hash,
        "all_epochs_checked": True,
        "n_epochs": args.epochs,
        "history": decoder_hashes,
    }
    (EXP_DIR / "decoder_integrity.json").write_text(json.dumps(decoder_integrity, indent=2), encoding="utf-8")

    # -------------------------------------------------------------
    # PART 10 — Full 5-Model Development Evaluation
    # -------------------------------------------------------------
    logger.info(">>> PART 10: Running full 5-model development evaluation on all 816 dev windows...")

    # 1. S3 predictions
    s3_preds = dev_preds  # from epoch 20

    # 2. S2 predictions
    logger.info("Loading S2 epoch 15 checkpoint...")
    s2_model = StudentS2(n_frames=5, parent_decoder=build_parent_decoder(cfg, "cpu"))
    s2_ckpt_path = EXP_S2 / "checkpoint_epoch_15.pt"
    s2_model.load_state_dict(torch.load(s2_ckpt_path, map_location="cpu", weights_only=True))
    s2_model.to(device).eval()
    s2_preds = []
    with torch.no_grad():
        for b in dev_loader:
            s2_preds.append(s2_model(b["clip"].to(device))["Hhat_1s"].float().cpu().numpy())
    s2_preds = np.concatenate(s2_preds, axis=0)

    # 3. S1 predictions
    logger.info("Loading S1 epoch 15 checkpoint...")
    s1_model = StudentS1(n_frames=5, parent_decoder=build_parent_decoder(cfg, "cpu"))
    s1_ckpt_path = EXP_S1 / "checkpoint_epoch_15.pt"
    s1_model.load_state_dict(torch.load(s1_ckpt_path, map_location="cpu", weights_only=True))
    s1_model.to(device).eval()
    s1_preds = []
    with torch.no_grad():
        for b in dev_loader:
            s1_preds.append(s1_model(b["clip"].to(device))["Hhat_1s"].float().cpu().numpy())
    s1_preds = np.concatenate(s1_preds, axis=0)

    # 4. S0 predictions
    logger.info("Loading S0 epoch 15 checkpoint...")
    s0_model = StudentS0(n_frames=5)
    s0_ckpt_path = EXP_S0 / "checkpoint_epoch_15.pt"
    s0_model.load_state_dict(torch.load(s0_ckpt_path, map_location="cpu", weights_only=True))
    s0_model.to(device).eval()
    s0_preds = []
    with torch.no_grad():
        for b in dev_loader:
            s0_preds.append(s0_model(b["clip"].to(device))["Hhat_1s"].float().cpu().numpy())
    s0_preds = np.concatenate(s0_preds, axis=0)

    # 5. Parent predictions
    parent_preds = parent_dev_hms

    # Compute comprehensive metrics for all 5 models
    all_models_eval = {
        "Parent": evaluate_predictions_comprehensive(parent_preds, dev_gt, parent_dev_hms, dev_rows, warning, verified_dev_set),
        "S0": evaluate_predictions_comprehensive(s0_preds, dev_gt, parent_dev_hms, dev_rows, warning, verified_dev_set),
        "S1": evaluate_predictions_comprehensive(s1_preds, dev_gt, parent_dev_hms, dev_rows, warning, verified_dev_set),
        "S2": evaluate_predictions_comprehensive(s2_preds, dev_gt, parent_dev_hms, dev_rows, warning, verified_dev_set),
        "S3_epoch_20": evaluate_predictions_comprehensive(s3_preds, dev_gt, parent_dev_hms, dev_rows, warning, verified_dev_set),
    }

    # -------------------------------------------------------------
    # PART 11 — Statistical comparison and final decision
    # -------------------------------------------------------------
    logger.info(">>> PART 11: Paired scenario-group bootstrap (2,000 resamples)...")
    bootstrap_results = paired_scenario_bootstrap(
        s3_preds, s2_preds, parent_preds, dev_gt, dev_rows, warning, n_resamples=2000, seed=args.seed
    )

    # Decision rule logic
    # 1. Preserve SAFE recall to within 0.02 of parent's measured SAFE recall?
    parent_safe = all_models_eval["Parent"]["safe_recall"]
    s3_safe = all_models_eval["S3_epoch_20"]["safe_recall"]
    s3_stop = all_models_eval["S3_epoch_20"]["stop_recall"]
    s2_stop = all_models_eval["S2"]["stop_recall"]
    safe_preserved = bool(s3_safe >= parent_safe - 0.02)

    ci_bal = bootstrap_results["diff_s3_minus_s2"]["balanced_recall"]
    bal_improved = bool(ci_bal["ci_95_lower"] > 0.0)
    bal_uncertain = bool(ci_bal["ci_95_lower"] <= 0.0 <= ci_bal["ci_95_upper"])

    if s3_stop == 0.0:
        experiment_status = "FAILED (STOP recall remained zero; collapsed to SAFE)"
    elif not safe_preserved:
        experiment_status = "FAILED (SAFE recall violated condition: fell >0.02 below parent)"
    elif bal_uncertain:
        experiment_status = "UNCERTAIN (Balanced recall improvement 95% CI overlaps zero)"
    elif bal_improved:
        experiment_status = "SUCCESS (Statistically significant improvement in balanced recall while preserving SAFE recall)"
    else:
        experiment_status = "INCONCLUSIVE"

    logger.info(f"Decision Rule Outcome: {experiment_status}")

    # -------------------------------------------------------------
    # PART 12 — Required Artifacts Generation
    # -------------------------------------------------------------
    logger.info(">>> PART 12: Writing required artifacts...")

    # config.json
    config_dict = {
        "model_name": "StudentS3",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "num_train_samples": len(train_ids),
        "num_dev_samples": len(dev_ids),
        "frame_offsets": FDEC_FRAME_OFFSETS,
        "resolution": 160,
        "eta_f": eta_f,
        "lambda_ordinal": lambda_ordinal,
        "backbone_hash": bb_current_hash,
        "projection_hash": proj_hash,
        "decoder_hash": dec_hash,
        "decision_rule_status": experiment_status,
    }
    (EXP_DIR / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # Metrics JSON
    metrics_summary = {
        "models": {
            name: {
                "stop_recall": m["stop_recall"],
                "safe_recall": m["safe_recall"],
                "threat_recall": m["threat_recall"],
                "balanced_stop_safe_recall": m["balanced_stop_safe_recall"],
                "stop_precision": m["stop_precision"],
                "safe_false_alarm_rate": m["safe_false_alarm_rate"],
                "fraction_gt_stop_as_safe": m["fraction_gt_stop_as_safe"],
                "fraction_gt_threat_as_safe": m["fraction_gt_threat_as_safe"],
                "recall_at_5fa": m["recall_at_5fa"],
                "ap_high": m["ap_high"],
                "auc_stop_vs_safe": m["auc_stop_vs_safe"],
                "auc_threat_vs_safe": m["auc_threat_vs_safe"],
                "gt_heatmap_mse": m["gt_heatmap_mse"],
                "parent_heatmap_mae": m["parent_heatmap_mae"],
                "pre_fork_stop_recall": m["fork_evaluation"]["pre_fork"]["stop_recall"],
                "post_fork_stop_recall": m["fork_evaluation"]["post_fork"]["stop_recall"],
                "pre_fork_safe_recall": m["fork_evaluation"]["pre_fork"]["safe_recall"],
                "post_fork_safe_recall": m["fork_evaluation"]["post_fork"]["safe_recall"],
            }
            for name, m in all_models_eval.items()
        },
        "decision_rule_status": experiment_status,
    }
    (REPORT_DIR / "s3_from_scratch_metrics.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    # Confusion matrices JSON
    confusion_matrices = {name: m["confusion_matrix"] for name, m in all_models_eval.items()}
    (REPORT_DIR / "s3_confusion_matrices.json").write_text(json.dumps(confusion_matrices, indent=2), encoding="utf-8")

    # Peak distributions CSV
    with (REPORT_DIR / "s3_peak_distributions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "gt_class", "count", "min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean", "std", "frac_ge_0_25", "frac_ge_0_60"])
        for name, m in all_models_eval.items():
            for c, dist in m["peak_distributions_by_class"].items():
                w.writerow([name, c, dist["count"], dist["min"], dist["p10"], dist["p25"], dist["p50"], dist["p75"], dist["p90"], dist["p95"], dist["p99"], dist["max"], dist["mean"], dist["std"], dist["frac_ge_0_25"], dist["frac_ge_0_60"]])

    # Scenario bootstrap JSON
    (REPORT_DIR / "s3_scenario_bootstrap.json").write_text(json.dumps(bootstrap_results, indent=2), encoding="utf-8")

    # Pair metrics JSON
    pair_metrics_all = {name: m["pair_evaluation"] for name, m in all_models_eval.items()}
    (REPORT_DIR / "s3_pair_metrics.json").write_text(json.dumps(pair_metrics_all, indent=2), encoding="utf-8")

    # Parameter budget JSON
    (REPORT_DIR / "s3_parameter_budget.json").write_text(json.dumps(param_audit, indent=2), encoding="utf-8")

    # Detailed Markdown Report
    m3 = all_models_eval["S3_epoch_20"]
    m2 = all_models_eval["S2"]
    m1 = all_models_eval["S1"]
    m0 = all_models_eval["S0"]
    mp = all_models_eval["Parent"]

    report_md = f"""# Collision-JEPA S3 — Train Student From Scratch Final Report

## Executive Summary

- **Experiment Status:** `{experiment_status}`
- **Model:** S3 fresh Base RGB student + fresh spatial projection + frozen Pair-CF parent decoder (`77f78c8714667617`).
- **Total Parameters:** `{param_audit['total_params']:,}` (Trainable: `{param_audit['trainable_params']:,}`, Frozen: `{param_audit['frozen_params']:,}`). Under 700K: `{param_audit['under_700k']}`.
- **Untrained Verification:** Backbone hash `{bb_current_hash}` matches seed 1337 initialization exactly. No trained student weights loaded.

---

## 1. 5-Model Development Scorecard (816 Windows)

| Metric | Parent (Fixed) | S0 (Owned Dec) | S1 (Adapt Dec) | S2 (Frozen Dec) | S3 (Direct Ordinal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **STOP Recall** | {mp['stop_recall']:.4f} | {m0['stop_recall']:.4f} | {m1['stop_recall']:.4f} | {m2['stop_recall']:.4f} | **{m3['stop_recall']:.4f}** |
| **SAFE Recall** | {mp['safe_recall']:.4f} | {m0['safe_recall']:.4f} | {m1['safe_recall']:.4f} | {m2['safe_recall']:.4f} | **{m3['safe_recall']:.4f}** |
| **THREAT Recall** | {mp['threat_recall']:.4f} | {m0['threat_recall']:.4f} | {m1['threat_recall']:.4f} | {m2['threat_recall']:.4f} | **{m3['threat_recall']:.4f}** |
| **Balanced Recall** | {mp['balanced_stop_safe_recall']:.4f} | {m0['balanced_stop_safe_recall']:.4f} | {m1['balanced_stop_safe_recall']:.4f} | {m2['balanced_stop_safe_recall']:.4f} | **{m3['balanced_stop_safe_recall']:.4f}** |
| **STOP Precision** | {mp['stop_precision']:.4f} | {m0['stop_precision']:.4f} | {m1['stop_precision']:.4f} | {m2['stop_precision']:.4f} | **{m3['stop_precision']:.4f}** |
| **SAFE False-Alarm Rate** | {mp['safe_false_alarm_rate']:.4f} | {m0['safe_false_alarm_rate']:.4f} | {m1['safe_false_alarm_rate']:.4f} | {m2['safe_false_alarm_rate']:.4f} | **{m3['safe_false_alarm_rate']:.4f}** |
| **GT STOP miss as SAFE** | {mp['fraction_gt_stop_as_safe']:.4f} | {m0['fraction_gt_stop_as_safe']:.4f} | {m1['fraction_gt_stop_as_safe']:.4f} | {m2['fraction_gt_stop_as_safe']:.4f} | **{m3['fraction_gt_stop_as_safe']:.4f}** |
| **GT THREAT miss as SAFE** | {mp['fraction_gt_threat_as_safe']:.4f} | {m0['fraction_gt_threat_as_safe']:.4f} | {m1['fraction_gt_threat_as_safe']:.4f} | {m2['fraction_gt_threat_as_safe']:.4f} | **{m3['fraction_gt_threat_as_safe']:.4f}** |
| **Recall @ 5% FA** | {mp['recall_at_5fa']:.4f} | {m0['recall_at_5fa']:.4f} | {m1['recall_at_5fa']:.4f} | {m2['recall_at_5fa']:.4f} | **{m3['recall_at_5fa']:.4f}** |
| **AP_high** | {mp['ap_high']:.4f} | {m0['ap_high']:.4f} | {m1['ap_high']:.4f} | {m2['ap_high']:.4f} | **{m3['ap_high']:.4f}** |
| **AUC (STOP vs SAFE)** | {mp['auc_stop_vs_safe']:.4f} | {m0['auc_stop_vs_safe']:.4f} | {m1['auc_stop_vs_safe']:.4f} | {m2['auc_stop_vs_safe']:.4f} | **{m3['auc_stop_vs_safe']:.4f}** |
| **AUC (THREAT vs SAFE)** | {mp['auc_threat_vs_safe']:.4f} | {m0['auc_threat_vs_safe']:.4f} | {m1['auc_threat_vs_safe']:.4f} | {m2['auc_threat_vs_safe']:.4f} | **{m3['auc_threat_vs_safe']:.4f}** |
| **GT Heatmap MSE** | {mp['gt_heatmap_mse']:.6f} | {m0['gt_heatmap_mse']:.6f} | {m1['gt_heatmap_mse']:.6f} | {m2['gt_heatmap_mse']:.6f} | **{m3['gt_heatmap_mse']:.6f}** |
| **Parent Heatmap MAE** | {mp['parent_heatmap_mae']:.6f} | {m0['parent_heatmap_mae']:.6f} | {m1['parent_heatmap_mae']:.6f} | {m2['parent_heatmap_mae']:.6f} | **{m3['parent_heatmap_mae']:.6f}** |

---

## 2. Fork Breakdown (Pre-fork vs Post-fork)

| Metric | Parent Pre / Post | S2 Pre / Post | S3 Pre / Post |
| :--- | :---: | :---: | :---: |
| **STOP Recall** | {mp['fork_evaluation']['pre_fork']['stop_recall']:.4f} / {mp['fork_evaluation']['post_fork']['stop_recall']:.4f} | {m2['fork_evaluation']['pre_fork']['stop_recall']:.4f} / {m2['fork_evaluation']['post_fork']['stop_recall']:.4f} | **{m3['fork_evaluation']['pre_fork']['stop_recall']:.4f}** / **{m3['fork_evaluation']['post_fork']['stop_recall']:.4f}** |
| **SAFE Recall** | {mp['fork_evaluation']['pre_fork']['safe_recall']:.4f} / {mp['fork_evaluation']['post_fork']['safe_recall']:.4f} | {m2['fork_evaluation']['pre_fork']['safe_recall']:.4f} / {m2['fork_evaluation']['post_fork']['safe_recall']:.4f} | **{m3['fork_evaluation']['pre_fork']['safe_recall']:.4f}** / **{m3['fork_evaluation']['post_fork']['safe_recall']:.4f}** |
| **False-Alarm Rate** | {mp['fork_evaluation']['pre_fork']['false_alarm_rate']:.4f} / {mp['fork_evaluation']['post_fork']['false_alarm_rate']:.4f} | {m2['fork_evaluation']['pre_fork']['false_alarm_rate']:.4f} / {m2['fork_evaluation']['post_fork']['false_alarm_rate']:.4f} | **{m3['fork_evaluation']['pre_fork']['false_alarm_rate']:.4f}** / **{m3['fork_evaluation']['post_fork']['false_alarm_rate']:.4f}** |

---

## 3. Background Pair Evaluation (398 Verified Pairs)

| Metric | Parent | S2 | S3 |
| :--- | :---: | :---: | :---: |
| **Median Peak Gap** | {mp['pair_evaluation']['median_peak_gap']:.4f} | {m2['pair_evaluation']['median_peak_gap']:.4f} | **{m3['pair_evaluation']['median_peak_gap']:.4f}** |
| **P90 Peak Gap** | {mp['pair_evaluation']['p90_peak_gap']:.4f} | {m2['pair_evaluation']['p90_peak_gap']:.4f} | **{m3['pair_evaluation']['p90_peak_gap']:.4f}** |
| **Warning Tier Flips** | {mp['pair_evaluation']['warning_tier_flips']} ({mp['pair_evaluation']['warning_tier_flip_rate']:.4f}) | {m2['pair_evaluation']['warning_tier_flips']} ({m2['pair_evaluation']['warning_tier_flip_rate']:.4f}) | **{m3['pair_evaluation']['warning_tier_flips']}** (**{m3['pair_evaluation']['warning_tier_flip_rate']:.4f}**) |
| **Both-Correct STOP Rate** | {mp['pair_evaluation']['both_background_correct_stop_rate']:.4f} | {m2['pair_evaluation']['both_background_correct_stop_rate']:.4f} | **{m3['pair_evaluation']['both_background_correct_stop_rate']:.4f}** |
| **Both-Correct SAFE Rate** | {mp['pair_evaluation']['both_background_correct_safe_rate']:.4f} | {m2['pair_evaluation']['both_background_correct_safe_rate']:.4f} | **{m3['pair_evaluation']['both_background_correct_safe_rate']:.4f}** |

---

## 4. Paired Scenario-Group Bootstrap (2,000 Resamples)

### S3 vs S2 Differences
- **STOP Recall Diff:** Mean = `{bootstrap_results['diff_s3_minus_s2']['stop_recall']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_s2']['stop_recall']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_s2']['stop_recall']['ci_95_upper']:.4f}]`
- **SAFE Recall Diff:** Mean = `{bootstrap_results['diff_s3_minus_s2']['safe_recall']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_s2']['safe_recall']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_s2']['safe_recall']['ci_95_upper']:.4f}]`
- **Balanced Recall Diff:** Mean = `{bootstrap_results['diff_s3_minus_s2']['balanced_recall']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_s2']['balanced_recall']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_s2']['balanced_recall']['ci_95_upper']:.4f}]`
- **Post-fork STOP Recall Diff:** Mean = `{bootstrap_results['diff_s3_minus_s2']['post_fork_stop_recall']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_s2']['post_fork_stop_recall']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_s2']['post_fork_stop_recall']['ci_95_upper']:.4f}]`
- **SAFE False-Alarm Rate Diff:** Mean = `{bootstrap_results['diff_s3_minus_s2']['safe_false_alarm_rate']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_s2']['safe_false_alarm_rate']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_s2']['safe_false_alarm_rate']['ci_95_upper']:.4f}]`

### S3 vs Parent Differences
- **STOP Recall Diff:** Mean = `{bootstrap_results['diff_s3_minus_parent']['stop_recall']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_parent']['stop_recall']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_parent']['stop_recall']['ci_95_upper']:.4f}]`
- **SAFE Recall Diff:** Mean = `{bootstrap_results['diff_s3_minus_parent']['safe_recall']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_parent']['safe_recall']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_parent']['safe_recall']['ci_95_upper']:.4f}]`
- **Balanced Recall Diff:** Mean = `{bootstrap_results['diff_s3_minus_parent']['balanced_recall']['mean_diff']:.4f}`, 95% CI = `[{bootstrap_results['diff_s3_minus_parent']['balanced_recall']['ci_95_lower']:.4f}, {bootstrap_results['diff_s3_minus_parent']['balanced_recall']['ci_95_upper']:.4f}]`

---

## 5. Confusion Matrices (Rows: GT, Columns: Pred)

### S3 Epoch 20:
```
{json.dumps(m3['confusion_matrix'], indent=2)}
```

### S2 Epoch 15:
```
{json.dumps(m2['confusion_matrix'], indent=2)}
```

### Fixed Parent:
```
{json.dumps(mp['confusion_matrix'], indent=2)}
```
"""
    (REPORT_DIR / "s3_from_scratch_report.md").write_text(report_md, encoding="utf-8")
    (EXP_DIR / ".success").write_text(f"success epoch {args.epochs}\n", encoding="utf-8")
    logger.info(">>> S3 experiment completed successfully! All artifacts written.")


if __name__ == "__main__":
    main()
