#!/usr/bin/env python3
"""Controlled S0 vs S1 F_dec pilot."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.experiment_runner import gpu_mem_mb  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_compare import write_comparison  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_dataset import (  # noqa: E402
    FDecPilotDataset,
    build_frame_store,
    make_fdec_loader,
)
from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_train import (  # noqa: E402
    apply_backbone,
    calibrate_eta_s1,
    prepare_shared_backbone,
    run_fdec_pilot,
)
from collision_jepa3.student_research.fdec_student import (  # noqa: E402
    FDecNormStats,
    FDEC_FRAME_OFFSETS,
    StudentS0,
    StudentS1,
    build_parent_decoder,
)
from collision_jepa3.utils.device import resolve_device  # noqa: E402

REPORT = ROOT / "reports/student_architecture_tests"
CACHE = REPORT / "teacher_cache_train_pilot"
EXP_S0 = ROOT / "experiments/student_fdec_s0"
EXP_S1 = ROOT / "experiments/student_fdec_s1"
BASELINE_DIR = ROOT / "experiments/student_fdec_unoptimized_baseline"
BENCH_PATH = REPORT / "student_fdec_dataloader_benchmark.json"


def _epoch0(exp_dir: Path) -> int:
    start = 0
    for pat in ("metrics_train_epoch_*.json", "metrics_epoch_*.json"):
        for p in sorted(exp_dir.glob(pat)):
            start = max(start, int(p.stem.split("_")[-1]))
    return start


def _run_variant(
    variant: str,
    cfg,
    device: str,
    train_ids: list[int],
    dev_ids: list[int],
    integrity: dict,
    fdec_norm: dict,
    verified_keys: list[tuple],
    shared_backbone: dict,
    exp_dir: Path,
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    force: bool,
    redo_s0: bool = False,
) -> dict:
    exp_dir.mkdir(parents=True, exist_ok=True)
    if (exp_dir / ".success").exists():
        if not force or (variant == "S0" and not redo_s0):
            return json.loads((exp_dir / "result.json").read_text(encoding="utf-8"))

    offsets = FDEC_FRAME_OFFSETS
    store, rgb_stats = build_frame_store(cfg, CACHE, sorted(set(train_ids) | set(dev_ids)), offsets)
    load_f = variant == "S1"
    tr_ds = FDecPilotDataset(CACHE, train_ids, offsets, load_f_dec=load_f)
    va_ds = FDecPilotDataset(CACHE, dev_ids, offsets, load_f_dec=load_f)
    tr_ds.attach_frame_store(store, rgb_stats)
    va_ds.attach_frame_store(store, rgb_stats)
    num_workers = 0
    if BENCH_PATH.exists():
        num_workers = int(json.loads(BENCH_PATH.read_text())["selected_num_workers"])
    tr_loader = make_fdec_loader(tr_ds, batch, shuffle=True, num_workers=num_workers)
    va_loader = make_fdec_loader(va_ds, batch, shuffle=False, num_workers=num_workers)

    mu = torch.from_numpy(fdec_norm["mu"]).view(-1, 1, 1)
    sigma = torch.from_numpy(fdec_norm["sigma"]).view(-1, 1, 1)
    norm = FDecNormStats(mu, sigma)
    eta = 0.0
    grad_cal = None

    if variant == "S0":
        model = StudentS0(n_frames=len(offsets))
        apply_backbone(model, shared_backbone)
    else:
        pdec = build_parent_decoder(cfg, device="cpu")
        model = StudentS1(n_frames=len(offsets), parent_decoder=pdec)
        apply_backbone(model, shared_backbone)
        model.to(device)
        model.set_decoder_trainable(False)
        batch0 = next(iter(tr_loader))
        grad_cal = calibrate_eta_s1(model, batch0, norm.to(device), device, alpha=0.5, target_frac=0.25)
        eta = float(grad_cal["eta_selected"])

    audit = model.param_audit()
    if audit["total_params"] > 700_000:
        raise RuntimeError(f"{variant} exceeds 700K params: {audit}")

    start_ep = _epoch0(exp_dir)
    if start_ep > 0 and (exp_dir / "checkpoint_last.pt").exists():
        model.load_state_dict(torch.load(exp_dir / "checkpoint_last.pt", map_location="cpu", weights_only=True))

    config = {
        "variant": variant,
        "frame_offsets": offsets,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "alpha_kd": 0.5,
        "eta_fdec": eta,
        "seed": seed,
        "train_rows": len(train_ids),
        "dev_rows": len(dev_ids),
        **audit,
        "rgb_prepare": rgb_stats,
        "num_workers": num_workers,
        "load_f_dec": load_f,
        "pipeline_version": "optimized_v2",
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if start_ep == 0:
        torch.save(model.state_dict(), exp_dir / "checkpoint_initial.pt")

    if grad_cal:
        (REPORT / "student_fdec_gradient_calibration.json").write_text(json.dumps(grad_cal, indent=2), encoding="utf-8")

    t0 = time.time()
    try:
        hist = run_fdec_pilot(
            variant,
            model,
            tr_loader,
            va_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            alpha_kd=0.5,
            eta_fdec=eta,
            norm=norm if variant == "S1" else None,
            cfg_eval=cfg,
            exp_dir=exp_dir,
            report_root=REPORT,
            verified_pair_keys=verified_keys,
            start_epoch=start_ep,
            decoder_freeze_epochs=3,
            eval_epochs=(1, 3, 6, 9, 12, 15),
        )
        summary = {
            "variant": variant,
            "elapsed_sec": time.time() - t0,
            "params": audit,
            "last_epoch": hist["history"][-1]["epoch"] if hist["history"] else 0,
            "gpu_mem": gpu_mem_mb(),
        }
        (exp_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (exp_dir / ".success").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        return summary
    except Exception:
        (exp_dir / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("S0", "S1", "both", "compare", "integrity"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--redo-s0", action="store_true", help="allow overwriting a completed S0 run")
    add_config_args(parser)
    parser.set_defaults(seed=1337)
    args = parser.parse_args()
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    device = str(resolve_device(args.device))

    train_ids = json.loads((REPORT / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((REPORT / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    integrity, fdec_norm = build_integrity_report(cfg, CACHE, train_ids, dev_ids, device="cpu")
    np.savez_compressed(REPORT / "fdec_norm_train_only.npz", **fdec_norm)
    (REPORT / "student_fdec_pilot_data_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    if not integrity["ok_for_s1"]:
        raise SystemExit("F_dec missing — stop S1")
    if args.variant == "integrity":
        print(json.dumps({"ok": True, "integrity": integrity}, indent=2))
        return
    verified_keys_dev = [tuple(x) for x in integrity.get("verified_pair_keys_dev", integrity["verified_pair_keys"])]

    baseline_init = BASELINE_DIR / "shared_backbone_init.pt"
    shared = prepare_shared_backbone(REPORT, seed=args.seed, baseline_init=baseline_init)
    budget = {}

    if args.variant in ("S0", "both"):
        s0 = StudentS0(n_frames=5)
        apply_backbone(s0, shared)
        budget["S0"] = s0.param_audit()
    if args.variant in ("S1", "both"):
        m = StudentS1(n_frames=5, parent_decoder=build_parent_decoder(cfg, "cpu"))
        apply_backbone(m, shared)
        budget["S1"] = m.param_audit()
    (REPORT / "student_fdec_parameter_budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

    grad_cal = None
    if args.variant == "compare":
        write_comparison(REPORT, EXP_S0, EXP_S1, integrity=integrity, budget=budget, grad_cal=None)
        print(json.dumps({"ok": True, "compare": True}, indent=2))
        return

    summaries = {}
    if args.variant in ("S0", "both"):
        summaries["S0"] = _run_variant(
            "S0", cfg, device, train_ids, dev_ids, integrity, fdec_norm, verified_keys_dev, shared,
            EXP_S0,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
            force=args.force,
            redo_s0=args.redo_s0,
        )
    if args.variant in ("S1", "both"):
        gc_path = REPORT / "student_fdec_gradient_calibration.json"
        grad_cal = json.loads(gc_path.read_text()) if gc_path.exists() else None
        summaries["S1"] = _run_variant(
            "S1", cfg, device, train_ids, dev_ids, integrity, fdec_norm, verified_keys_dev, shared,
            EXP_S1,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
            force=args.force,
            redo_s0=args.redo_s0,
        )
        if gc_path.exists():
            grad_cal = json.loads(gc_path.read_text())

    if args.variant == "both":
        write_comparison(REPORT, EXP_S0, EXP_S1, integrity=integrity, budget=budget, grad_cal=grad_cal)

    print(json.dumps({"ok": True, "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
