#!/usr/bin/env python3
"""Evaluate a teacher checkpoint on val/test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg, resolve_checkpoint  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.evaluator import evaluate_loader  # noqa: E402
from collision_jepa3.evaluation.reports import write_json, write_metrics_csv  # noqa: E402
from collision_jepa3.losses.composite import TeacherLoss  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_cfg(args)
    log = setup_logger("eval", cfg.path("paths.logs_root"))
    device = resolve_device(args.device)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds = TeacherDataset(cfg, records, args.split, splits)
    loader = build_dataloader(ds, cfg, shuffle=False)
    model = CollisionTeacher3(cfg, dummy_backbone=bool(args.dummy))
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    blob = load_checkpoint(ckpt, map_location=device)
    if blob.get("lora_enabled") and not getattr(model.backbone, "lora_enabled", False):
        model.enable_lora()
    load_teacher_state(model, blob)
    model.to(device)
    metrics = evaluate_loader(model, loader, cfg, device, loss_fn=TeacherLoss(cfg))
    out_json = cfg.path("paths.reports_root") / f"eval_{args.split}.json"
    out_csv = cfg.path("paths.reports_root") / f"eval_{args.split}.csv"
    write_json(out_json, metrics)
    write_metrics_csv(out_csv, metrics["horizons"])
    log.info("wrote %s", out_json)
    print(json.dumps({"selection": metrics["selection"], "horizons": metrics["horizons"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
