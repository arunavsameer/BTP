#!/usr/bin/env python3
"""Zero-train U1 controls: dump G μ, then G-thr / μ-sigma / envelope / oracle-σ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.u1_runtime import collect_split, load_dump, load_frozen_g, run_oracles, save_dump  # noqa: E402
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-dump", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("evaluation.val_subset_fraction", 1.0)
    cfg.set("adapter.variance_head", False)
    device = resolve_device(args.device)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(describe_hardware(device), indent=2), flush=True)

    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    train_path = out_root / "dump_train.npz"
    val_path = out_root / "dump_val.npz"

    if args.reuse_dump and train_path.exists() and val_path.exists():
        print("reusing dumps", train_path, val_path, flush=True)
        train_dump = load_dump(train_path)
        val_dump = load_dump(val_path)
    else:
        model = load_frozen_g(cfg, Path(args.teacher_ckpt), device, variance_head=False)
        print("dumping train", flush=True)
        train_ds = _dataset(cfg, "train", records, splits)
        train_dump = collect_split(model, train_ds, cfg, device)
        save_dump(train_path, train_dump)
        print("dumping val", flush=True)
        val_ds = _dataset(cfg, "val", records, splits)
        val_dump = collect_split(model, val_ds, cfg, device)
        save_dump(val_path, val_dump)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    report = run_oracles(train_dump, val_dump)
    out = out_root / "oracles.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    v = report["verdict"]
    print(json.dumps({"wrote": str(out), "verdict": v, "g_clip": report["g_clip_own"]["headline"], "g_mean": report["g_mean_target"]["headline"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
