#!/usr/bin/env python3
"""Read-only S0 dev diagnostics; writes to reports/ only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.fdec_pilot_dataset import (  # noqa: E402
    FDecPilotDataset,
    build_frame_store,
    make_fdec_loader,
)
from collision_jepa3.student_research.fdec_pilot_diagnostics import build_s0_dev_diagnostics_report  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_train import apply_backbone, prepare_shared_backbone  # noqa: E402
from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS, StudentS0  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402

REPORT = ROOT / "reports/student_architecture_tests"
CACHE = REPORT / "teacher_cache_train_pilot"
EXP_S0 = ROOT / "experiments/student_fdec_s0"
BASELINE = ROOT / "experiments/student_fdec_unoptimized_baseline/shared_backbone_init.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--checkpoint", type=Path, default=None)
    add_config_args(parser)
    parser.set_defaults(seed=1337)
    args = parser.parse_args()
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    device = str(resolve_device(args.device))

    ckpt = args.checkpoint or (EXP_S0 / f"checkpoint_epoch_{args.epoch:02d}.pt")
    if not ckpt.exists():
        ckpt = EXP_S0 / "checkpoint_last.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint under {EXP_S0}")

    dev_ids = json.loads((REPORT / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    train_ids = json.loads((REPORT / "part1_train_manifest.json").read_text())["train_row_ids"]
    integrity, _ = build_integrity_report(cfg, CACHE, train_ids, dev_ids, device="cpu")
    (REPORT / "student_fdec_pilot_data_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    verified_dev = [tuple(x) for x in integrity["verified_pair_keys_dev"]]
    offsets = FDEC_FRAME_OFFSETS
    store, stats = build_frame_store(cfg, CACHE, sorted(set(train_ids) | set(dev_ids)), offsets)
    ds = FDecPilotDataset(CACHE, dev_ids, offsets, load_f_dec=False)
    ds.attach_frame_store(store, stats)
    bench = json.loads((REPORT / "student_fdec_dataloader_benchmark.json").read_text())
    loader = make_fdec_loader(ds, batch=8, shuffle=False, num_workers=int(bench["selected_num_workers"]))

    shared = prepare_shared_backbone(REPORT, seed=args.seed, baseline_init=BASELINE)
    model = StudentS0(n_frames=len(offsets))
    apply_backbone(model, shared)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)

    report = build_s0_dev_diagnostics_report(
        cfg,
        CACHE,
        dev_ids,
        model,
        loader,
        device,
        verified_pair_keys_dev=verified_dev,
        checkpoint_path=ckpt,
        epoch=args.epoch,
    )
    arrays_path = REPORT / f"student_fdec_s0_dev_preds_epoch{args.epoch:02d}.npz"
    from collision_jepa3.student_research.fdec_pilot_diagnostics import collect_dev_predictions  # noqa: E402

    warning = cfg.get("warning") or {}
    full = collect_dev_predictions(model, loader, device, warning)
    np.savez_compressed(
        arrays_path,
        row_ids=np.asarray(full["row_ids"], dtype=np.int32),
        student_heatmaps=full["student_heatmaps"],
        parent_heatmaps=full["parent_heatmaps"],
        gt_heatmaps=full["gt_heatmaps"],
        student_peaks=full["student_peaks"],
        parent_peaks=full["parent_peaks"],
        pre_fork=full["pre_fork"],
    )
    report["saved_arrays"] = str(arrays_path)

    out_json = REPORT / "student_fdec_s0_dev_diagnostics.json"
    out_md = REPORT / "student_fdec_s0_dev_diagnostics.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ps = report["predictions_summary"]
    lines = [
        "# S0 development diagnostics (read-only)",
        "",
        f"Checkpoint: `{ckpt}` (not modified)",
        "",
        "## Pair metrics bug (legacy epoch JSON)",
        report["legacy_pair_metrics_bug"]["issue"],
        f"- Dev verified pairs: **{report['legacy_pair_metrics_bug']['dev_verified_pairs_available']}**",
        "",
        "## Corrected dev pair metrics",
        json.dumps(report["dev_eval_corrected_pair_metrics"], indent=2),
        "",
        "## Student peak percentiles",
        json.dumps(ps["student_peak_percentiles"], indent=2),
        "",
        "## Parent peak percentiles",
        json.dumps(ps["parent_peak_percentiles"], indent=2),
        "",
        "## Crossings",
        f"- Parent ≥0.25: {ps['parent_pct_crossing_0_25']:.4%}" if ps.get("parent_pct_crossing_0_25") is not None else "",
        f"- Parent ≥0.60: {ps['parent_pct_crossing_0_60']:.4%}" if ps.get("parent_pct_crossing_0_60") is not None else "",
        f"- Student ≥0.25: {ps['student_pct_crossing_0_25']:.4%}" if ps["student_pct_crossing_0_25"] is not None else "- Student ≥0.25: N/A",
        f"- Student ≥0.60: {ps['student_pct_crossing_0_60']:.4%}" if ps["student_pct_crossing_0_60"] is not None else "- Student ≥0.60: N/A",
        "",
        "## GT severity counts",
        json.dumps(ps["gt_severity_counts"], indent=2),
        "",
        "## STOP confusion (student vs GT)",
        json.dumps(ps["stop_confusion"], indent=2),
        "",
        "## STOP recall pre/post fork",
        f"- pre_fork: {ps['stop_recall_pre_fork']} (n_stop={ps['pre_fork_stop_n']})",
        f"- post_fork: {ps['stop_recall_post_fork']} (n_stop={ps['post_fork_stop_n']})",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(out_json), "npz": str(arrays_path)}, indent=2))


if __name__ == "__main__":
    main()
