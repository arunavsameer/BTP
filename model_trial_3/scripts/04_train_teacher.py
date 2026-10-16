#!/usr/bin/env python3
"""Train the Collision-JEPA teacher (frozen and/or LoRA phases)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.config import dump_resolved  # noqa: E402
from collision_jepa3.data.cache import prepare_token_cache  # noqa: E402
from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.preprocessing import signature_from_cfg  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.reproducibility import configure_runtime, set_seed  # noqa: E402
from collision_jepa3.training.trainer import TeacherTrainer  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--phase", choices=["frozen", "lora", "both"], default="frozen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--dummy", action="store_true", help="Train a dummy encoder (debug only)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frozen-epochs", type=int, default=None)
    parser.add_argument("--lora-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop each epoch after N optimizer steps (0 = full epoch)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--bg-weight", type=float, default=None, help="Override loss.background_consistency_weight")
    parser.add_argument("--heads-mode", choices=["residual", "direct_h1s"], default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early-stop-metric", choices=["selection", "val_mse", "recall_at_5fa"], default=None)
    parser.add_argument("--recall-beta", type=float, default=None, help="L_1s extra weight on under-predicted high-threat cells")
    parser.add_argument("--val-subset-fraction", type=float, default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--logs-root", default=None)
    parser.add_argument("--reports-root", default=None)
    args = parser.parse_args()
    cfg = load_cfg(args)
    if args.frozen_epochs is not None:
        cfg.set("training.frozen_epochs", int(args.frozen_epochs))
    if args.lora_epochs is not None:
        cfg.set("training.lora_epochs", int(args.lora_epochs))
    if args.max_steps is not None:
        cfg.set("training.max_steps", int(args.max_steps))
    if args.num_workers is not None:
        cfg.set("training.num_workers", int(args.num_workers))
    if args.bg_weight is not None:
        cfg.set("loss.background_consistency_weight", float(args.bg_weight))
    if args.heads_mode is not None:
        cfg.set("heads.mode", args.heads_mode)
    if args.patience is not None:
        cfg.set("training.patience", int(args.patience))
    if args.early_stop_metric is not None:
        cfg.set("training.early_stop_metric", args.early_stop_metric)
    if args.recall_beta is not None:
        cfg.set("loss.recall_weight_beta", float(args.recall_beta))
    if args.val_subset_fraction is not None:
        cfg.set("evaluation.val_subset_fraction", float(args.val_subset_fraction))
    if args.checkpoint_root:
        cfg.set("paths.checkpoint_root", args.checkpoint_root)
    if args.logs_root:
        cfg.set("paths.logs_root", args.logs_root)
    if args.reports_root:
        cfg.set("paths.reports_root", args.reports_root)
    set_seed(int(cfg.get("project.seed", 1337)))
    configure_runtime(int(cfg.get("training.num_workers", 2)))
    log = setup_logger("train", cfg.path("paths.logs_root"))
    log.info("resolved config: %s", json.dumps(dump_resolved(cfg)["_resolved"], indent=2))
    device = resolve_device(args.device)
    log.info("hardware %s", describe_hardware(device, cfg.get("backbone.model_id")))

    man = cfg.path("paths.manifest")
    split_file = cfg.path("paths.split_file")
    if not man.exists() or not split_file.exists():
        raise SystemExit("Run 01_build_manifest.py before training.")
    records = load_manifest(man)
    splits = load_splits(split_file)
    dummy = bool(args.dummy)
    train_ds = TeacherDataset(cfg, records, "train", splits, dummy_backbone=dummy)
    val_ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=dummy)
    frac = float(cfg.get("evaluation.val_subset_fraction", 1.0))
    if 0.0 < frac < 1.0 and len(val_ds):
        n_clips = len(val_ds.clip_ids())
        n_before = len(val_ds)
        val_ds.subsample_clips(frac, int(cfg.get("project.seed", 1337)))
        log.info(
            "val subset fraction=%.2f clips %d→%d anchors %d→%d",
            frac,
            n_clips,
            len(val_ds.clip_ids()),
            n_before,
            len(val_ds),
        )
    log.info(
        "train clips=%d anchors=%d  val clips=%d anchors=%d  amb=%s",
        len(train_ds.records),
        len(train_ds),
        len(val_ds.clip_ids()),
        len(val_ds),
        train_ds.ambiguity_counts(),
    )
    model = CollisionTeacher3(cfg, dummy_backbone=dummy)
    model.to(device)
    log.info("audit %s", model.parameter_audit())
    if args.dry_run:
        print("dry-run OK", model.parameter_audit())
        return
    if bool(cfg.get("cache.cache_encoder_tokens", False)) and args.phase in {"frozen", "both"}:
        tok = prepare_token_cache(records, cfg, model, device)
        log.info("token cache %s", tok)
    train_loader = build_dataloader(train_ds, cfg, shuffle=True, drop_last=False)
    val_loader = build_dataloader(val_ds, cfg, shuffle=False) if len(val_ds) else None
    resume_path = None
    if args.resume or args.resume_from:
        resume_path = Path(args.resume_from) if args.resume_from else cfg.path("paths.checkpoint_root") / (
            "teacher_phase_b_last.pt" if args.phase == "lora" else "teacher_phase_a_last.pt"
        )
        if not resume_path.exists():
            log.warning("resume path %s missing; starting fresh", resume_path)
            resume_path = None
    trainer = TeacherTrainer(
        cfg,
        model,
        train_loader,
        val_loader,
        device,
        split_fingerprint=split_file.name,
        manifest_fingerprint=man.name,
        preprocess_signature=signature_from_cfg(cfg).as_dict(),
    )
    result = trainer.run(phase=args.phase, resume_path=resume_path)
    out = cfg.path("paths.reports_root") / "train_report.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
