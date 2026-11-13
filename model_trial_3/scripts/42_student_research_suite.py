#!/usr/bin/env python3
"""Orchestrate student architecture research (Parts 1–11). Does not pick a winner."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.student_research.cnn import preset_architectures, StudentRGBNet  # noqa: E402
from collision_jepa3.student_research.dataset_pilot import (  # noqa: E402
    PilotAnchorDataset,
    filter_rows_by_groups,
    make_loader,
)
from collision_jepa3.student_research.export_quant import sweep_architectures  # noqa: E402
from collision_jepa3.student_research.frames import FRAME_CANDIDATES  # noqa: E402
from collision_jepa3.student_research.pair_gap import part8_pair_robustness  # noqa: E402
from collision_jepa3.student_research.pca_latent import run_part6  # noqa: E402
from collision_jepa3.student_research.pilot_train import run_pilot  # noqa: E402
from collision_jepa3.student_research.pilot_data import pilot_cache_split  # noqa: E402
from collision_jepa3.student_research.splits_dev import write_split_manifest  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stage_b_frames(cfg, out: Path, cache_dir: Path, device: str, epochs: int, max_rows: int, seed: int) -> dict:
    cache_split, tr_g, dev_g, _ = pilot_cache_split(cfg, cache_dir)
    write_split_manifest(
        out / "pilot_split.json",
        tr_g,
        dev_g,
        seed,
        extra={"cache_split": cache_split, "note": "pilots use rows present in teacher cache"},
    )
    all_train = filter_rows_by_groups(cache_dir, set(tr_g), split=cache_split)
    all_dev = filter_rows_by_groups(cache_dir, set(dev_g), split=cache_split)
    if max_rows > 0:
        rng = random.Random(seed)
        all_train = rng.sample(all_train, min(max_rows, len(all_train)))
        all_dev = rng.sample(all_dev, min(max_rows // 4, len(all_dev)))
    arch = StudentRGBNet(preset_architectures()[0], n_frames=3, input_size=160)
    results = {}
    for name, offsets in FRAME_CANDIDATES.items():
        if name.startswith("repeat") or name == "single_current":
            continue
        tr_ds = PilotAnchorDataset(cache_dir, row_ids=all_train, frame_offsets=offsets, input_size=160)
        va_ds = PilotAnchorDataset(cache_dir, row_ids=all_dev, frame_offsets=offsets, input_size=160)
        tr_ds.set_cache_root(cfg)
        va_ds.set_cache_root(cfg)
        m = StudentRGBNet(arch.cfg, n_frames=len(offsets), input_size=160)
        rep = run_pilot(
            m,
            make_loader(tr_ds, 8, True),
            make_loader(va_ds, 8, False),
            device=device,
            epochs=epochs,
            lr=3e-4,
            alpha_kd=0.5,
            cfg_eval=cfg,
        )
        rep["frame_offsets"] = offsets
        rep["n_train"] = len(all_train)
        results[name] = rep
        path = out / "part3_frame_selection.json"
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"part3 checkpoint {name} n_train={len(all_train)}", flush=True)
    return results


def stage_b_arch(out: Path, input_size: int) -> dict:
    rows = sweep_architectures(preset_architectures(), input_size=input_size)
    for res in (128, 160):
        if res != input_size:
            rows += sweep_architectures(preset_architectures()[:2], input_size=res)
    report = {"models": rows}
    (out / "part4_architecture_feasibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def stage_c_pca(out: Path, cache_dir: Path) -> dict:
    return run_part6(cache_dir, out)


def stage_c_distill(cfg, out: Path, cache_dir: Path, device: str, epochs: int, max_rows: int, seed: int) -> dict:
    cache_split, tr_g, dev_g, _ = pilot_cache_split(cfg, cache_dir)
    rows = filter_rows_by_groups(cache_dir, set(tr_g), split=cache_split)
    dev_rows = filter_rows_by_groups(cache_dir, set(dev_g), split=cache_split)
    if max_rows > 0:
        dev_rows = dev_rows[: max_rows // 4]
    if max_rows > 0:
        rows = random.Random(seed).sample(rows, min(max_rows, len(rows)))
    offsets = FRAME_CANDIDATES["F2"]
    arch = preset_architectures()[0]
    strategies = {}
    for tag, alpha, eta in [
        ("A_heatmap_only", 0.5, 0.0),
        ("B_latent_heatmap", 0.5, 0.5),
    ]:
        tr_ds = PilotAnchorDataset(cache_dir, row_ids=rows, frame_offsets=offsets, input_size=160)
        va_ds = PilotAnchorDataset(cache_dir, row_ids=dev_rows, frame_offsets=offsets, input_size=160)
        tr_ds.set_cache_root(cfg)
        va_ds.set_cache_root(cfg)
        m = StudentRGBNet(arch, n_frames=3, input_size=160)
        strategies[tag] = run_pilot(
            m,
            make_loader(tr_ds, 8, True),
            make_loader(va_ds, 8, False),
            device=device,
            epochs=epochs,
            lr=3e-4,
            alpha_kd=alpha,
            eta_latent=eta,
            cfg_eval=cfg,
        )
    (out / "part5_latent_distillation.json").write_text(json.dumps(strategies, indent=2), encoding="utf-8")
    return strategies


def stage_d_quant(out: Path) -> dict:
    rows = sweep_architectures(preset_architectures())
    report = {"exports": rows}
    (out / "part11_quantization_deployment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_index(out: Path) -> None:
    idx = {
        "report_root": str(out),
        "parent_ckpt": "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt",
        "parts": [
            "part1_parent_latent_shapes.json",
            "part2_rgb_availability.json",
            "part3_frame_selection.json",
            "part4_architecture_feasibility.json",
            "part5_latent_distillation.json",
            "part6_latent_pca_analysis.json",
            "part11_quantization_deployment.json",
        ],
        "note": "Evidence-only bundle; final architecture not selected by this suite.",
    }
    (out / "experiment_index.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")
    csv_path = out / "all_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part", "key", "stop_recall", "safe_recall", "mse_gt"])
        p3 = out / "part3_frame_selection.json"
        if p3.exists():
            data = json.loads(p3.read_text())
            for k, v in data.items():
                best = v.get("best") or {}
                w.writerow(["part3", k, best.get("stop_recall"), best.get("safe_recall"), best.get("mse_gt")])


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--stage", choices=["B", "C", "D", "index", "all"], default="index")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.out)
    cache_dir = out / "teacher_cache"
    cfg = load_cfg(args)
    seed = int(cfg.get("project.seed", 1337))
    _seed_all(seed)
    device = resolve_device(args.device)
    if args.stage in ("B", "all"):
        if not (cache_dir / "teacher_pred.npz").exists():
            raise SystemExit("teacher cache missing; run scripts/30_student_part1_parent.py first")
        stage_b_frames(cfg, out, cache_dir, device, args.epochs, args.max_rows, seed)
        stage_b_arch(out, 160)
    if args.stage in ("C", "all"):
        stage_c_pca(out, cache_dir)
        stage_c_distill(cfg, out, cache_dir, device, args.epochs, args.max_rows, seed)
        part8_pair_robustness(cache_dir, out)
        stub = {
            "status": "partial",
            "note": "Augmentation sweeps and S0–S8 matrix scheduled after frame/distill pilots.",
        }
        (out / "part7_pca_distillation_comparison.json").write_text(json.dumps(stub, indent=2), encoding="utf-8")
        (out / "part9_augmentation_and_temporal_robustness.json").write_text(json.dumps(stub, indent=2), encoding="utf-8")
    if args.stage in ("D", "all"):
        stage_d_quant(out)
    write_index(out)
    md = out / "final_experiment_report.md"
    md.write_text(
        "# Student architecture research (in progress)\n\n"
        "This report aggregates measured pilots under `reports/student_architecture_tests/`. "
        "No final student architecture is selected here.\n",
        encoding="utf-8",
    )
    print(f"suite done → {out}", flush=True)


if __name__ == "__main__":
    main()
