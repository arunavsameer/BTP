#!/usr/bin/env python3
"""Train the Collision-JEPA teacher (frozen and/or LoRA phases)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.config import dump_resolved  # noqa: E402
from collision_jepa3.data.cache import prepare_token_cache  # noqa: E402
from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.extra_same_dir import extra_same_dir_groups  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.preprocessing import signature_from_cfg  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.reproducibility import configure_runtime, set_seed  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.training.trainer import TeacherTrainer  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--phase", choices=["frozen", "lora", "both"], default="frozen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help="Load adapter/decoder weights only (fresh optimizer, new run dir). Does not resume epoch.",
    )
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
    parser.add_argument("--patience-start-epoch", type=int, default=None)
    parser.add_argument("--post-fork-stop-floor", type=float, default=None)
    parser.add_argument("--stop-recall-floor", type=float, default=None)
    parser.add_argument("--safe-recall-floor", type=float, default=None)
    parser.add_argument("--stop-hinge-weight", type=float, default=None)
    parser.add_argument("--stop-hinge-target", type=float, default=None)
    parser.add_argument("--safe-hinge-weight", type=float, default=None)
    parser.add_argument("--small-blob-weight", type=float, default=None)
    parser.add_argument("--disappear-weight", type=float, default=None)
    parser.add_argument("--close-weight", type=float, default=None)
    parser.add_argument("--variance-head", action="store_true", help="Add 16×16 counterfactual variance head")
    parser.add_argument("--var-weight", type=float, default=None)
    parser.add_argument("--var-pos-weight", type=float, default=None)
    parser.add_argument("--beta-nll-weight", type=float, default=None)
    parser.add_argument("--evidential-weight", type=float, default=None)
    parser.add_argument("--prob-stop-weight", type=float, default=None)
    parser.add_argument("--prob-safe-weight", type=float, default=None)
    parser.add_argument("--adapter-lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--max-safe-stop-rate", type=float, default=None)
    parser.add_argument(
        "--early-stop-metric",
        choices=["selection", "val_mse", "recall_at_5fa", "prefork_stop", "balanced_warning"],
        default=None,
    )
    parser.add_argument("--recall-beta", type=float, default=None, help="L_1s extra weight on under-predicted high-threat cells")
    parser.add_argument("--val-subset-fraction", type=float, default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--logs-root", default=None)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--train-anchor-stride", type=int, default=None)
    parser.add_argument("--val-anchor-stride", type=int, default=None)
    parser.add_argument(
        "--extra-same-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use extra_same_dir sample_min..sample_max in train (default: yaml, on). --no-extra-same-dir turns them off.",
    )
    parser.add_argument("--extra-same-dir-min", type=int, default=None)
    parser.add_argument("--extra-same-dir-max", type=int, default=None)
    parser.add_argument(
        "--task-whiten",
        action="store_true",
        help="Fit task-whitened pair-covariance drop on train F, then train G through T(F).",
    )
    parser.add_argument("--task-whiten-drop-k", type=int, default=16)
    parser.add_argument(
        "--pair-dir-adv",
        action="store_true",
        help="Train on the hardest interpolant along matched-background ΔF.",
    )
    parser.add_argument("--pair-dir-adv-weight", type=float, default=0.25)
    parser.add_argument(
        "--pair-pcgrad",
        action="store_true",
        help="PCGrad on matched-background adapter/decoder gradients.",
    )
    parser.add_argument(
        "--padr",
        action="store_true",
        help="Pair-anchored directional robustness: actual-pair Huber + local tangent.",
    )
    parser.add_argument("--padr-warmup-epochs", type=int, default=18)
    parser.add_argument("--padr-ramp-epochs", type=int, default=12)
    parser.add_argument("--padr-rho-lo", type=float, default=0.05)
    parser.add_argument("--padr-rho-hi", type=float, default=0.10)
    parser.add_argument("--padr-rms-frac", type=float, default=0.05)
    parser.add_argument("--padr-grad-frac", type=float, default=0.08)
    parser.add_argument(
        "--no-padr-gt-filter",
        action="store_true",
        help="Do not drop PADR pairs whose mate h_1s differs from anchor (default: filter on).",
    )
    parser.add_argument(
        "--padr-gt-tol",
        type=float,
        default=None,
        help="Max cell |Δ| for mate h_1s agreement (default: training.padr_gt_tol).",
    )
    parser.add_argument(
        "--logit-hinges",
        action="store_true",
        help="STOP/SAFE hinges on max heatmap logits vs logit(0.60)/logit(0.25).",
    )
    parser.add_argument("--logit-hinge-margin", type=float, default=0.1)
    parser.add_argument(
        "--heatmap-balanced",
        action="store_true",
        help="L_1s = ½(occupied BCE + empty BCE). Empty maps use the empty term only.",
    )
    parser.add_argument(
        "--bg-reg",
        action="store_true",
        help="Enable matched-background pair batching + optional consistency/dropout (training.bg_reg).",
    )
    parser.add_argument(
        "--bg-reg-pair-consistency",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Output pair consistency loss L_bg_pair (default: false).",
    )
    parser.add_argument(
        "--bg-reg-targeted-dropout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Channel dropout on P4 low-ICC set (default: false).",
    )
    parser.add_argument(
        "--bg-reg-random-dropout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Random dropout on same count/eligible pool as targeted (R3).",
    )
    parser.add_argument(
        "--bg-reg-no-early-stop",
        action="store_true",
        help="Disable early stopping for fixed budget (training.bg_reg.disable_early_stopping).",
    )
    parser.add_argument("--bg-reg-p4", default=None, help="Path to p4_report.json")
    parser.add_argument(
        "--bg-reg-clean-anchor",
        action="store_true",
        help="Clean-anchored ICC channel dropout auxiliary loss (training.bg_reg.clean_anchored_dropout).",
    )
    parser.add_argument(
        "--bg-reg-dropout-prob",
        type=float,
        default=None,
        help="Override training.bg_reg.dropout_prob (e.g. 0.25 for clean-anchor).",
    )
    parser.add_argument(
        "--bg-reg-clean-anchor-pair-frac",
        type=float,
        default=None,
        help="Fraction of pair batches that run the auxiliary dropout path (default 0.5).",
    )
    parser.add_argument(
        "--bg-reg-clean-anchor-grad-scale",
        type=float,
        default=None,
        help="β = scale * median(||g_clean||/||g_drop||) scale factor (default 0.10).",
    )
    parser.add_argument(
        "--bg-reg-pair-cf",
        action="store_true",
        help="Pair-conditioned pooled PCA counterfactual auxiliary (training.bg_reg.pair_counterfactual_shrink).",
    )
    parser.add_argument("--bg-reg-pair-cf-subspace", default=None, help="Path to learned_k*.npz with U [k,C].")
    parser.add_argument("--bg-reg-pair-cf-frac", type=float, default=None, help="Pair-batch aux probability (default 0.5).")
    parser.add_argument("--bg-reg-pair-cf-grad-scale", type=float, default=None, help="Target cf/clean grad ratio scale (default 0.10).")
    args = parser.parse_args()
    if args.pair_dir_adv and args.pair_pcgrad:
        raise SystemExit("run --pair-dir-adv and --pair-pcgrad separately")
    if args.padr and (args.pair_dir_adv or args.pair_pcgrad):
        raise SystemExit("run --padr separately from --pair-dir-adv / --pair-pcgrad")
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
    if args.patience_start_epoch is not None:
        cfg.set("training.patience_start_epoch", int(args.patience_start_epoch))
    if args.post_fork_stop_floor is not None:
        cfg.set("training.post_fork_stop_floor", float(args.post_fork_stop_floor))
    if args.stop_recall_floor is not None:
        cfg.set("training.stop_recall_floor", float(args.stop_recall_floor))
    if args.safe_recall_floor is not None:
        cfg.set("training.safe_recall_floor", float(args.safe_recall_floor))
    if args.stop_hinge_weight is not None:
        cfg.set("loss.stop_hinge_weight", float(args.stop_hinge_weight))
    if args.stop_hinge_target is not None:
        cfg.set("loss.stop_hinge_target", float(args.stop_hinge_target))
    if args.safe_hinge_weight is not None:
        cfg.set("loss.safe_hinge_weight", float(args.safe_hinge_weight))
    if args.small_blob_weight is not None:
        cfg.set("loss.small_blob_weight", float(args.small_blob_weight))
    if args.disappear_weight is not None:
        cfg.set("loss.bce_disappear_weight", float(args.disappear_weight))
    if args.close_weight is not None:
        cfg.set("loss.close_weight", float(args.close_weight))
    if args.variance_head:
        cfg.set("adapter.variance_head", True)
        if args.var_weight is None and cfg.get("loss.var_weight", 0.0) in {0, 0.0, None}:
            cfg.set("loss.var_weight", 1.0)
    if args.var_weight is not None:
        cfg.set("loss.var_weight", float(args.var_weight))
        if float(args.var_weight) > 0:
            cfg.set("adapter.variance_head", True)
    if args.var_pos_weight is not None:
        cfg.set("loss.var_pos_weight", float(args.var_pos_weight))
    if args.beta_nll_weight is not None:
        cfg.set("loss.beta_nll_weight", float(args.beta_nll_weight))
    if args.evidential_weight is not None:
        cfg.set("loss.evidential_weight", float(args.evidential_weight))
    if args.prob_stop_weight is not None:
        cfg.set("loss.prob_stop_weight", float(args.prob_stop_weight))
    if args.prob_safe_weight is not None:
        cfg.set("loss.prob_safe_weight", float(args.prob_safe_weight))
    if args.max_safe_stop_rate is not None:
        cfg.set("training.max_safe_stop_rate", float(args.max_safe_stop_rate))
    if args.adapter_lr is not None:
        cfg.set("optimizer.adapter_lr", float(args.adapter_lr))
    if args.warmup_epochs is not None:
        cfg.set("optimizer.warmup_epochs", int(args.warmup_epochs))
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
    if args.train_anchor_stride is not None:
        cfg.set("sampling.train_anchor_stride", int(args.train_anchor_stride))
    if args.val_anchor_stride is not None:
        cfg.set("sampling.val_anchor_stride", int(args.val_anchor_stride))
    if args.extra_same_dir is not None:
        cfg.set("data.extra_same_dir.enabled", bool(args.extra_same_dir))
    if args.extra_same_dir_min is not None:
        cfg.set("data.extra_same_dir.sample_min", int(args.extra_same_dir_min))
    if args.extra_same_dir_max is not None:
        cfg.set("data.extra_same_dir.sample_max", int(args.extra_same_dir_max))
    if args.task_whiten:
        cfg.set("adapter.task_whiten", True)
        cfg.set("adapter.task_whiten_drop_k", int(args.task_whiten_drop_k))
    if args.pair_dir_adv:
        cfg.set("loss.pair_dir_adv_weight", float(args.pair_dir_adv_weight))
    if args.pair_pcgrad:
        cfg.set("training.pair_pcgrad", True)
    if args.padr:
        cfg.set("training.padr", True)
        cfg.set("training.padr_warmup_epochs", int(args.padr_warmup_epochs))
        cfg.set("training.padr_ramp_epochs", int(args.padr_ramp_epochs))
        cfg.set("training.padr_rho_lo", float(args.padr_rho_lo))
        cfg.set("training.padr_rho_hi", float(args.padr_rho_hi))
        cfg.set("training.padr_rms_frac", float(args.padr_rms_frac))
        cfg.set("training.padr_grad_frac", float(args.padr_grad_frac))
        cfg.set("training.padr_gt_filter", not bool(args.no_padr_gt_filter))
        if args.padr_gt_tol is not None:
            cfg.set("training.padr_gt_tol", float(args.padr_gt_tol))
    if args.logit_hinges:
        cfg.set("loss.logit_hinges", True)
        cfg.set("loss.logit_hinge_margin", float(args.logit_hinge_margin))
    if args.heatmap_balanced:
        cfg.set("loss.heatmap_balanced", True)
    if args.bg_reg:
        cfg.set("training.bg_reg.enabled", True)
    if args.bg_reg_pair_consistency is not None:
        cfg.set("training.bg_reg.pair_consistency", bool(args.bg_reg_pair_consistency))
    if args.bg_reg_targeted_dropout is not None:
        cfg.set("training.bg_reg.targeted_dropout", bool(args.bg_reg_targeted_dropout))
    if args.bg_reg_random_dropout is not None:
        cfg.set("training.bg_reg.random_dropout", bool(args.bg_reg_random_dropout))
    if args.bg_reg_no_early_stop:
        cfg.set("training.bg_reg.disable_early_stopping", True)
    if args.bg_reg_p4:
        cfg.set("training.bg_reg.p4_report", args.bg_reg_p4)
    if args.bg_reg_dropout_prob is not None:
        cfg.set("training.bg_reg.dropout_prob", float(args.bg_reg_dropout_prob))
    if args.bg_reg_clean_anchor:
        cfg.set("training.bg_reg.enabled", True)
        cfg.set("training.bg_reg.clean_anchored_dropout", True)
        cfg.set("training.bg_reg.pair_consistency", False)
        cfg.set("training.bg_reg.targeted_dropout", False)
        cfg.set("training.bg_reg.random_dropout", False)
    if args.bg_reg_clean_anchor_pair_frac is not None:
        cfg.set("training.bg_reg.clean_anchor_pair_frac", float(args.bg_reg_clean_anchor_pair_frac))
    if args.bg_reg_clean_anchor_grad_scale is not None:
        cfg.set("training.bg_reg.clean_anchor_grad_scale", float(args.bg_reg_clean_anchor_grad_scale))
    if getattr(args, "bg_reg_pair_cf", False):
        cfg.set("training.bg_reg.enabled", True)
        cfg.set("training.bg_reg.pair_counterfactual_shrink", True)
        cfg.set("training.bg_reg.pair_consistency", False)
        cfg.set("training.bg_reg.targeted_dropout", False)
        cfg.set("training.bg_reg.random_dropout", False)
        cfg.set("training.bg_reg.clean_anchored_dropout", False)
    if getattr(args, "bg_reg_pair_cf_subspace", None):
        cfg.set("training.bg_reg.pair_cf_subspace_npz", str(args.bg_reg_pair_cf_subspace))
    if getattr(args, "bg_reg_pair_cf_frac", None) is not None:
        cfg.set("training.bg_reg.pair_cf_pair_frac", float(args.bg_reg_pair_cf_frac))
    if getattr(args, "bg_reg_pair_cf_grad_scale", None) is not None:
        cfg.set("training.bg_reg.pair_cf_grad_scale", float(args.bg_reg_pair_cf_grad_scale))
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
    extra_on = bool(cfg.get("data.extra_same_dir.enabled", True))
    extra_groups = extra_same_dir_groups(records, cfg)
    extra_n = sum(1 for r in train_ds.records if r.scenario_group_id in extra_groups)
    log.info(
        "train clips=%d anchors=%d  val clips=%d anchors=%d  amb=%s  extra_same_dir=%s extra_clips=%d",
        len(train_ds.records),
        len(train_ds),
        len(val_ds.clip_ids()),
        len(val_ds),
        train_ds.ambiguity_counts(),
        extra_on,
        extra_n,
    )
    model = CollisionTeacher3(cfg, dummy_backbone=dummy)
    model.to(device)
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise SystemExit(f"--init-from missing: {init_path}")
        load_teacher_state(model, load_checkpoint(init_path, map_location=device))
        log.info("initialized adapter/decoder from %s (fresh optimizer)", init_path)
    if bool(cfg.get("adapter.task_whiten", False)) and (args.resume_from is None and not args.resume):
        from collision_jepa3.models.task_whiten import fit_and_install  # noqa: PLC0415

        drop_k = int(cfg.get("adapter.task_whiten_drop_k", 16))
        log.info("fitting task-whitened drop-%d on TRAIN features (T disabled)", drop_k)
        tw_stats = fit_and_install(model, train_ds, cfg, device, drop_k=drop_k)
        log.info("task-whiten %s", tw_stats)
        tw_path = cfg.path("paths.checkpoint_root") / "task_whiten_fit.json"
        tw_path.parent.mkdir(parents=True, exist_ok=True)
        tw_path.write_text(json.dumps(tw_stats, indent=2), encoding="utf-8")
        torch.save({"W": model.task_whiten.W.detach().cpu(), **tw_stats}, cfg.path("paths.checkpoint_root") / "task_whiten_W.pt")
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
