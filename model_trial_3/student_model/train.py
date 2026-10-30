#!/usr/bin/env python3
"""Distill Collision-JEPA teacher G into a <1M RGB student. No V-JEPA at inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.config import dump_resolved  # noqa: E402
from collision_jepa3.data.extra_same_dir import extra_same_dir_groups  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.preprocessing import signature_from_cfg  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone  # noqa: E402
from collision_jepa3.reproducibility import configure_runtime, set_seed  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402

from student_model.dataset import StudentDataset, build_student_loader, upsample_small_appear  # noqa: E402
from student_model.evaluate import evaluate_student  # noqa: E402
from student_model.student import CollisionStudent, MAX_PARAMS, load_frozen_teacher_heads  # noqa: E402
from student_model.trainer import StudentTrainer  # noqa: E402


def _apply_g_recipe(cfg, args) -> None:
    cfg.set("heads.mode", "direct_h1s")
    if args.close_weight is not None:
        cfg.set("loss.close_weight", float(args.close_weight))
    else:
        cfg.set("loss.close_weight", 1.0)
    if args.stop_hinge_weight is not None:
        cfg.set("loss.stop_hinge_weight", float(args.stop_hinge_weight))
    else:
        cfg.set("loss.stop_hinge_weight", 1.0)
    if args.safe_hinge_weight is not None:
        cfg.set("loss.safe_hinge_weight", float(args.safe_hinge_weight))
    else:
        cfg.set("loss.safe_hinge_weight", 0.5)
    if args.small_blob_weight is not None:
        cfg.set("loss.small_blob_weight", float(args.small_blob_weight))
    else:
        cfg.set("loss.small_blob_weight", 2.0)
    if args.extra_same_dir is not None:
        cfg.set("data.extra_same_dir.enabled", bool(args.extra_same_dir))
    if args.extra_same_dir_min is not None:
        cfg.set("data.extra_same_dir.sample_min", int(args.extra_same_dir_min))
    if args.extra_same_dir_max is not None:
        cfg.set("data.extra_same_dir.sample_max", int(args.extra_same_dir_max))
    if args.frozen_epochs is not None:
        cfg.set("training.frozen_epochs", int(args.frozen_epochs))
    if args.patience is not None:
        cfg.set("training.patience", int(args.patience))
    if args.patience_start_epoch is not None:
        cfg.set("training.patience_start_epoch", int(args.patience_start_epoch))
    if args.post_fork_stop_floor is not None:
        cfg.set("training.post_fork_stop_floor", float(args.post_fork_stop_floor))
    if args.stop_recall_floor is not None:
        cfg.set("training.stop_recall_floor", float(args.stop_recall_floor))
    if args.safe_recall_floor is not None:
        cfg.set("training.safe_recall_floor", float(args.safe_recall_floor))
    if args.max_safe_stop_rate is not None:
        cfg.set("training.max_safe_stop_rate", float(args.max_safe_stop_rate))
    if args.adapter_lr is not None:
        cfg.set("optimizer.adapter_lr", float(args.adapter_lr))
    if args.warmup_epochs is not None:
        cfg.set("optimizer.warmup_epochs", int(args.warmup_epochs))
    if args.early_stop_metric is not None:
        cfg.set("training.early_stop_metric", args.early_stop_metric)
    if args.val_subset_fraction is not None:
        cfg.set("evaluation.val_subset_fraction", float(args.val_subset_fraction))
    if args.num_workers is not None:
        cfg.set("training.num_workers", int(args.num_workers))
    if args.checkpoint_root:
        cfg.set("paths.checkpoint_root", args.checkpoint_root)
    if args.logs_root:
        cfg.set("paths.logs_root", args.logs_root)
    if args.reports_root:
        cfg.set("paths.reports_root", args.reports_root)
    cfg.set("student.feat_weight", float(args.feat_weight))
    cfg.set("student.z_weight", float(args.z_weight))
    if getattr(args, "cos_weight", None) is not None:
        cfg.set("student.cos_weight", float(args.cos_weight))
    if getattr(args, "mgd_weight", None) is not None:
        cfg.set("student.mgd_weight", float(args.mgd_weight))
    if getattr(args, "rel_weight", None) is not None:
        cfg.set("student.rel_weight", float(args.rel_weight))
    cfg.set("student.kd_weight", float(args.kd_weight))
    cfg.set("student.gt_weight", float(args.gt_weight))
    cfg.set("student.at_weight", float(args.at_weight))
    cfg.set("student.distill_warmup_epochs", int(args.distill_warmup_epochs))
    if getattr(args, "predictor_lr", None) is not None:
        cfg.set("student.predictor_lr", float(args.predictor_lr))
    if getattr(args, "hint_lr", None) is not None:
        cfg.set("student.hint_lr", float(args.hint_lr))
    if getattr(args, "decoder_lr", None) is not None:
        cfg.set("student.decoder_lr", float(args.decoder_lr))
    if getattr(args, "energy_alpha", None) is not None:
        cfg.set("student.energy_alpha", float(args.energy_alpha))
    if getattr(args, "appear_boost", None) is not None:
        cfg.set("student.appear_boost", float(args.appear_boost))
    if getattr(args, "stop_boost", None) is not None:
        cfg.set("student.stop_boost", float(args.stop_boost))
    if getattr(args, "upsample_repeats", None) is not None:
        cfg.set("student.upsample_repeats", int(args.upsample_repeats))
    if getattr(args, "h_weight", None) is not None:
        cfg.set("student.h_weight", float(args.h_weight))
    if getattr(args, "peak_stop_weight", None) is not None:
        cfg.set("student.peak_stop_weight", float(args.peak_stop_weight))
    if getattr(args, "peak_safe_weight", None) is not None:
        cfg.set("student.peak_safe_weight", float(args.peak_safe_weight))
    if getattr(args, "logit_weight", None) is not None:
        cfg.set("student.logit_weight", float(args.logit_weight))
    if getattr(args, "encoder_variant", None) is not None:
        cfg.set("student.encoder_variant", str(args.encoder_variant))


def _to_device(module, device: str, *, tries: int = 18, delay: float = 10.0):
    """Move to GPU with retries for transient CUDA busy/unavailable on shared nodes."""
    import time

    last: BaseException | None = None
    for attempt in range(int(tries)):
        try:
            return module.to(device)
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            if "busy" not in msg and "unavailable" not in msg:
                raise
            print(f"cuda busy, retry {attempt + 1}/{tries}: {exc}", flush=True)
            time.sleep(float(delay) * (1 if attempt < 3 else 2))
    assert last is not None
    raise last


def build_frozen_teacher(cfg, ckpt: Path, device: str):
    """Adapter + decoder from G. Dummy 1408-D backbone so Giant is never loaded."""
    backbone = DummyVJEPA2Backbone(
        hidden_size=1408,
        crop_size=256,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        num_layers=1,
    )
    teacher = CollisionTeacher3(cfg, backbone=backbone)
    blob = load_checkpoint(ckpt, map_location="cpu")
    load_teacher_state(teacher, blob, strict=False)
    _to_device(teacher, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, blob


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frozen-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--patience-start-epoch", type=int, default=None)
    parser.add_argument("--post-fork-stop-floor", type=float, default=None)
    parser.add_argument("--stop-recall-floor", type=float, default=None)
    parser.add_argument("--safe-recall-floor", type=float, default=None)
    parser.add_argument("--max-safe-stop-rate", type=float, default=None)
    parser.add_argument("--stop-hinge-weight", type=float, default=None)
    parser.add_argument("--safe-hinge-weight", type=float, default=None)
    parser.add_argument("--small-blob-weight", type=float, default=None)
    parser.add_argument("--close-weight", type=float, default=None)
    parser.add_argument("--adapter-lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument(
        "--early-stop-metric",
        choices=[
            "selection",
            "val_mse",
            "recall_at_5fa",
            "prefork_stop",
            "balanced_warning",
            "latent_l1",
            "blob_l1",
            "small_stop_recall",
        ],
        default=None,
    )
    parser.add_argument("--val-subset-fraction", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--logs-root", default=None)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--extra-same-dir", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--extra-same-dir-min", type=int, default=None)
    parser.add_argument("--extra-same-dir-max", type=int, default=None)
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--z-weight", type=float, default=0.3)
    parser.add_argument("--cos-weight", type=float, default=0.2)
    parser.add_argument("--mgd-weight", type=float, default=0.5)
    parser.add_argument("--rel-weight", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--gt-weight", type=float, default=0.0)
    parser.add_argument("--at-weight", type=float, default=0.0)
    parser.add_argument("--distill-warmup-epochs", type=int, default=0)
    parser.add_argument("--predictor-lr", type=float, default=0.0003)
    parser.add_argument("--hint-lr", type=float, default=0.0003)
    parser.add_argument("--decoder-lr", type=float, default=0.0001)
    parser.add_argument(
        "--cache-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read letterboxed uint8 frames from cache_root/frames (default on).",
    )
    parser.add_argument(
        "--prepare-frame-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build missing frame-cache .npy files before train (default on).",
    )
    parser.add_argument("--frame-cache-workers", type=int, default=8)
    parser.add_argument("--frame-cache-max-gb", type=float, default=80.0)
    parser.add_argument("--energy-alpha", type=float, default=None)
    parser.add_argument("--appear-boost", type=float, default=None)
    parser.add_argument("--stop-boost", type=float, default=None)
    parser.add_argument("--upsample-repeats", type=int, default=None)
    parser.add_argument("--h-weight", type=float, default=None)
    parser.add_argument("--peak-stop-weight", type=float, default=None)
    parser.add_argument("--peak-safe-weight", type=float, default=None)
    parser.add_argument("--logit-weight", type=float, default=None)
    parser.add_argument("--encoder-variant", choices=["lat", "fast"], default=None)
    args = parser.parse_args()
    cfg = load_cfg(args)
    _apply_g_recipe(cfg, args)
    cfg.set("cache.cache_frames", bool(args.cache_frames))
    cfg.set("cache.lru_clips_in_ram", max(8, int(cfg.get("cache.lru_clips_in_ram", 4))))
    set_seed(int(cfg.get("project.seed", 1337)))
    configure_runtime(int(cfg.get("training.num_workers", 2)))
    log = setup_logger("student", cfg.path("paths.logs_root"))
    setup_logger("student_model.trainer", cfg.path("paths.logs_root"))
    setup_logger("student_model.prepare_frames", cfg.path("paths.logs_root"))
    log.info("resolved config: %s", json.dumps(dump_resolved(cfg)["_resolved"], indent=2))

    teacher_ckpt = Path(args.teacher_ckpt)
    if not teacher_ckpt.is_absolute():
        teacher_ckpt = ROOT / teacher_ckpt
    if not teacher_ckpt.exists():
        raise SystemExit(f"teacher checkpoint missing: {teacher_ckpt}")

    man = cfg.path("paths.manifest")
    split_file = cfg.path("paths.split_file")
    if not man.exists() or not split_file.exists():
        raise SystemExit("Run 01_build_manifest.py before training.")
    records = load_manifest(man)
    splits = load_splits(split_file)
    if bool(args.prepare_frame_cache) and bool(args.cache_frames):
        from student_model.prepare_frames import prepare_frame_cache_parallel

        cache_stats = prepare_frame_cache_parallel(
            records,
            cfg,
            workers=int(args.frame_cache_workers),
            max_gb=float(args.frame_cache_max_gb) if args.frame_cache_max_gb else None,
        )
        log.info("frame cache %s", cache_stats)
        print("frame cache", cache_stats, flush=True)
    train_ds = StudentDataset(cfg, records, "train", splits, dummy_backbone=False)
    val_ds = StudentDataset(cfg, records, "val", splits, dummy_backbone=False)
    up_repeats = int(cfg.get("student.upsample_repeats", 0) or 0)
    if up_repeats > 0:
        up_stats = upsample_small_appear(train_ds, repeats=up_repeats)
        log.info("upsample small-appear %s", up_stats)
        print("upsample small-appear", up_stats, flush=True)
    frac = float(cfg.get("evaluation.val_subset_fraction", 1.0))
    if 0.0 < frac < 1.0 and len(val_ds):
        val_ds.subsample_clips(frac, int(cfg.get("project.seed", 1337)))
    extra_on = bool(cfg.get("data.extra_same_dir.enabled", True))
    extra_groups = extra_same_dir_groups(records, cfg)
    extra_n = sum(1 for r in train_ds.records if r.scenario_group_id in extra_groups)
    log.info(
        "train clips=%d anchors=%d  val clips=%d anchors=%d  extra_same_dir=%s extra_clips=%d",
        len(train_ds.records),
        len(train_ds),
        len(val_ds.clip_ids()),
        len(val_ds),
        extra_on,
        extra_n,
    )

    # CUDA must not be initialized before the frame-cache process pool, or
    # fork-after-CUDA leaves the GPU busy/unavailable on shared DGX nodes.
    device = resolve_device(args.device)
    log.info("hardware %s", describe_hardware(device, "student_g"))
    teacher, blob = build_frozen_teacher(cfg, teacher_ckpt, device)
    student = CollisionStudent(cfg)
    load_frozen_teacher_heads(student, blob)
    _to_device(student, device)
    audit = student.parameter_audit()
    log.info("student audit %s", audit)
    if audit["total"] >= MAX_PARAMS:
        raise SystemExit(f"student has {audit['total']} params, cap is {MAX_PARAMS}")
    if args.dry_run:
        print("dry-run OK", audit)
        return

    train_loader = build_student_loader(train_ds, cfg, shuffle=True, drop_last=False)
    val_loader = build_student_loader(val_ds, cfg, shuffle=False) if len(val_ds) else None
    trainer = StudentTrainer(
        cfg,
        student,
        teacher,
        train_loader,
        val_loader,
        device,
        split_fingerprint=split_file.name,
        manifest_fingerprint=man.name,
        preprocess_signature=signature_from_cfg(cfg).as_dict(),
        teacher_ckpt=str(teacher_ckpt),
    )
    result = trainer.run(int(cfg.get("training.frozen_epochs", 100)))
    out = cfg.path("paths.reports_root") / "train_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")

    best = cfg.path("paths.checkpoint_root") / "student_best.pt"
    if best.exists() and val_loader is not None:
        from collision_jepa3.training.checkpoint import load_checkpoint as _load

        ck = _load(best, map_location=device)
        student.encoder.load_state_dict(ck["encoder"])
        if ck.get("predictor"):
            student.predictor.load_state_dict(ck["predictor"])
        elif ck.get("hint") and hasattr(student, "predictor"):
            try:
                student.predictor.load_state_dict(ck["hint"])
            except Exception:
                pass
        if ck.get("bottleneck"):
            student.bottleneck.load_state_dict(ck["bottleneck"])
        if ck.get("norm"):
            student.norm.load_state_dict(ck["norm"])
        if ck.get("decoder"):
            student.decoder.load_state_dict(ck["decoder"])
        if ck.get("close_head"):
            student.close_head.load_state_dict(ck["close_head"])
        student.to(device).eval()
        val = evaluate_student(student, val_loader, cfg, device, teacher=teacher)
        score_path = cfg.path("paths.reports_root") / "full_val_operating.json"
        score_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(best),
                    "epoch": ck.get("epoch"),
                    "operating": val.get("operating"),
                    "selection": val.get("selection"),
                    "fork": (val.get("horizons") or {}).get("1s"),
                    "latent": val.get("latent"),
                    "parameter_audit": student.parameter_audit(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print("wrote", score_path)
        print(json.dumps(val.get("operating") or {}, indent=2))


if __name__ == "__main__":
    main()
