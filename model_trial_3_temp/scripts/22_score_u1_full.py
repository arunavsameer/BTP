#!/usr/bin/env python3
"""Score a jointly trained U1-full checkpoint: κ grid vs G oracles. Does not train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.u1_runtime import collect_split, feasible_vs_g, load_dump, load_frozen_g, score_pred, score_u1  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oracle-json", default="reports/direct_h1s_u1/oracles.json")
    parser.add_argument("--val-dump", default="reports/direct_h1s_u1/dump_val.npz")
    parser.add_argument("--out-root", default="reports/direct_h1s_u1_full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", True)
    cfg.set("evaluation.val_subset_fraction", 1.0)
    device = resolve_device(args.device)
    print(json.dumps(describe_hardware(device), indent=2), flush=True)
    oracles = json.loads(Path(args.oracle_json).read_text(encoding="utf-8"))
    val_dump = load_dump(Path(args.val_dump))
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    val_ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
    model = load_frozen_g(
        cfg, Path(args.checkpoint), device, variance_head=True, freeze_variance_only=False
    )
    model.eval()
    pack = collect_split(model, val_ds, cfg, device)
    pred_mu = pack["arrays"]["pred_mu"]
    if "v_hat" not in pack["arrays"]:
        raise SystemExit("checkpoint has no Hhat_var")
    v_hat = pack["arrays"]["v_hat"]
    comparison = score_u1(val_dump, v_hat, oracles, n_boot=int(args.n_boot), pred_mu=pred_mu)
    mean_only = score_pred(pred_mu, val_dump, gt_key="gt_clip")
    comparison["mean_only"] = {
        "headline": mean_only["headline"],
        "vs_g_clip": feasible_vs_g(mean_only, oracles["g_clip_own"]),
    }
    comparison["checkpoint"] = str(args.checkpoint)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_root / "val_pred.npz", pred_mu=pred_mu, v_hat=v_hat)
    out = out_root / "u1_full_comparison.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(out),
                "selection": comparison["selection"],
                "mean_only": comparison["mean_only"]["headline"],
                "var": comparison["variance_vs_true"],
                "beats_mu_lookup_mae": comparison["beats_mu_lookup_mae"],
            },
            indent=2,
        ),
        flush=True,
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
