#!/usr/bin/env python3
"""Score direct H_1s checkpoints on the full val set (recall@5% FA, AP, peak mean, MSE)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.evaluator import evaluate_loader  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("evaluation.val_subset_fraction", 1.0)
    device = resolve_device(args.device)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=bool(args.dummy), use_token_cache=True)
    loader = build_dataloader(ds, cfg, shuffle=False)
    report = {
        "split": "val",
        "n_clips": len(ds.clip_ids()),
        "n_windows": len(ds),
        "runs": {},
    }
    for item in args.checkpoints:
        name, path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        print(f"scoring {name} {path}", flush=True)
        model = CollisionTeacher3(cfg, dummy_backbone=bool(args.dummy))
        blob = load_checkpoint(path, map_location=device)
        load_teacher_state(model, blob)
        model.to(device).eval()
        val = evaluate_loader(model, loader, cfg, device)
        op = val.get("operating") or {}
        report["runs"][name] = {
            "checkpoint": str(path),
            "epoch": blob.get("epoch"),
            "operating": op,
            "selection": val.get("selection"),
            "mse_horizon": (val.get("horizons") or {}).get("1s", {}).get("mse"),
        }
        print(json.dumps(report["runs"][name]["operating"], indent=2), flush=True)
        del model
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
