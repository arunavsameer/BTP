#!/usr/bin/env python3
"""Run one resumable student experiment (Part 3 frame pilot, Part 5 distill, etc.)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.pilot_data import load_pilot_split_train, pilot_cache_split, pilot_row_ids  # noqa: E402

from collision_jepa3.student_research.cnn import (  # noqa: E402
    MAX_TOTAL_PARAMS,
    assert_param_budget,
    preset_architectures,
    StudentRGBNet,
)
from collision_jepa3.student_research.dataset_pilot import (  # noqa: E402
    PilotAnchorDataset,
    filter_rows_by_groups,
    make_loader,
)
from collision_jepa3.student_research.experiment_runner import (  # noqa: E402
    gpu_mem_mb,
    run_wrapped,
    write_epoch_metrics,
)
from collision_jepa3.student_research.frames import FRAME_CANDIDATES  # noqa: E402
from collision_jepa3.student_research.pilot_train import run_pilot  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def _arch_by_name(name: str):
    for a in preset_architectures():
        if a.name == name or a.budget == name:
            return a
    if name in {"micro", "micro_concat"}:
        return preset_architectures()[0]
    raise ValueError(f"unknown arch {name}")


def _pilot_data(cfg, cache_dir: Path, report_root: Path, max_rows: int, seed: int, offsets: list[int]):
    split = load_pilot_split_train(report_root)
    cache_split, tr_g, dev_g, _ = pilot_cache_split(cfg, cache_dir, report_root)
    all_train, all_dev, _, _ = pilot_row_ids(report_root)
    if max_rows > 0 and not split.get("cache_built"):
        all_train = filter_rows_by_groups(cache_dir, set(tr_g), split=cache_split)
        all_dev = filter_rows_by_groups(cache_dir, set(dev_g), split=cache_split)
        rng = random.Random(seed)
        all_train = rng.sample(all_train, min(max_rows, len(all_train)))
        all_dev = rng.sample(all_dev, min(max_rows // 4, len(all_dev))) if all_dev else []
    tr_ds = PilotAnchorDataset(cache_dir, row_ids=all_train, frame_offsets=offsets, input_size=160)
    va_ds = PilotAnchorDataset(cache_dir, row_ids=all_dev, frame_offsets=offsets, input_size=160)
    tr_ds.set_cache_root(cfg)
    va_ds.set_cache_root(cfg)
    tr_ds.preload_clips()
    va_ds.preload_clips()
    return tr_ds, va_ds, len(all_train), len(all_dev), cache_split


def run_part3_frame(exp_d: Path, config: dict) -> dict:
    ns = argparse.Namespace(
        config=config.get("config_path", str(ROOT / "configs/default.yaml")),
        dataset_root=None,
        seed=config.get("seed"),
    )
    cfg = load_cfg(ns)
    device = config["device"]
    report_root = Path(config["report_root"])
    cache_dir = Path(config.get("cache_dir", report_root / "teacher_cache_train_pilot"))
    offsets = FRAME_CANDIDATES[config["frame_set"]]
    arch = _arch_by_name(config.get("arch", "micro_concat"))
    tr_ds, va_ds, n_tr, n_va, cache_split = _pilot_data(
        cfg, cache_dir, report_root, int(config["max_rows"]), int(config["seed"]), offsets
    )
    model = StudentRGBNet(arch, n_frames=len(offsets), input_size=160)
    assert_param_budget(model, MAX_TOTAL_PARAMS)
    start_ep = 0
    for p in sorted(exp_d.glob("metrics_epoch_*.json")):
        start_ep = max(start_ep, int(p.stem.split("_")[-1]))
    if start_ep > 0 and (exp_d / "checkpoint_last.pt").exists():
        model.load_state_dict(torch.load(exp_d / "checkpoint_last.pt", map_location="cpu", weights_only=True))

    frame_set = config["frame_set"]
    best_val: dict | None = None

    def on_epoch(ep: int, rec: dict) -> None:
        nonlocal best_val
        payload = {**rec, "gpu_mem": gpu_mem_mb()}
        write_epoch_metrics(exp_d, ep, payload)
        torch.save(model.state_dict(), exp_d / "checkpoint_last.pt")
        val = rec.get("val") or {}
        score = val.get("stop_recall")
        if best_val is None or (score == score and score > best_val.get("stop_recall", -1)):
            best_val = {"epoch": ep, **val}
        _merge_part3(
            report_root,
            frame_set,
            {
                "frame_set": frame_set,
                "frame_offsets": offsets,
                "arch": arch.name,
                "params": sum(p.numel() for p in model.parameters()),
                "n_train": n_tr,
                "n_dev": n_va,
                "cache_split": cache_split,
                "cache_dir": str(cache_dir),
                "last_epoch": ep,
                "best": best_val,
                "partial": ep < int(config["epochs"]),
                "history_tail": payload,
            },
        )

    rep = run_pilot(
        model,
        make_loader(tr_ds, int(config["batch"]), True),
        make_loader(va_ds, int(config["batch"]), False),
        device=device,
        epochs=int(config["epochs"]),
        lr=float(config["lr"]),
        alpha_kd=float(config["alpha_kd"]),
        eta_latent=float(config.get("eta_latent", 0)),
        cfg_eval=cfg,
        start_epoch=start_ep,
        on_epoch_end=on_epoch,
    )
    train_seconds = sum(h.get("seconds", 0.0) for h in rep.get("history", []))
    rep.update(
        {
            "frame_set": config["frame_set"],
            "frame_offsets": offsets,
            "arch": arch.name,
            "params": sum(p.numel() for p in model.parameters()),
            "n_train": n_tr,
            "n_dev": n_va,
            "cache_split": cache_split,
            "cache_dir": str(cache_dir),
            "train_seconds_total": train_seconds,
            "partial": False,
        }
    )
    _merge_part3(report_root, frame_set, rep)
    return rep


def _merge_part3(report_root: Path, frame_set: str, rep: dict) -> None:
    path = report_root / "part3_frame_selection.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[frame_set] = rep
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--kind", choices=["part3_frame", "part5_heatmap", "part5_latent"], default="part3_frame")
    parser.add_argument("--frame-set", default="F2")
    parser.add_argument("--arch", default="micro_concat")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-kd", type=float, default=0.5)
    parser.add_argument("--eta-latent", type=float, default=0.0)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument(
        "--cache-dir",
        default="",
        help="train-pilot teacher cache (default: <out>/teacher_cache_train_pilot)",
    )
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    args = parser.parse_args()
    seed = int(args.seed if args.seed is not None else 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    report_root = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else report_root / "teacher_cache_train_pilot"
    config = {
        "kind": args.kind,
        "frame_set": args.frame_set,
        "arch": args.arch,
        "epochs": args.epochs,
        "max_rows": args.max_rows,
        "batch": args.batch,
        "device": resolve_device(args.device),
        "lr": args.lr,
        "alpha_kd": args.alpha_kd,
        "eta_latent": args.eta_latent,
        "seed": seed,
        "report_root": str(report_root),
        "cache_dir": str(cache_dir),
        "cfg_args": {"config": args.config, "dataset_root": args.dataset_root, "seed": seed},
        "config_path": args.config,
        "param_budget_max": MAX_TOTAL_PARAMS,
    }
    if args.kind == "part3_frame":
        fn = run_part3_frame
    else:
        raise SystemExit(f"kind {args.kind} not implemented in this script yet")
    summary = run_wrapped(report_root, args.exp_id, config, fn, force=args.force)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
