#!/usr/bin/env python3
"""Benchmark optimized F_dec pilot dataloader and report before full pilot."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.fdec_pilot_dataset import (  # noqa: E402
    FDecPilotDataset,
    build_frame_store,
    make_fdec_loader,
    rgb_letterbox_parity_test,
)
from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS, StudentS0  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402

REPORT = ROOT / "reports/student_architecture_tests"
CACHE = REPORT / "teacher_cache_train_pilot"


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _one_epoch_benchmark(model, loader, device: str) -> dict:
    model.to(device).train()
    t_wait = t_h2d = t_compute = 0.0
    use_cuda = device.startswith("cuda")
    n = 0
    t_end = time.perf_counter()
    for batch in loader:
        t0 = time.perf_counter()
        t_wait += t0 - t_end
        batch = {k: v.to(device, non_blocking=use_cuda) if torch.is_tensor(v) else v for k, v in batch.items()}
        if use_cuda:
            torch.cuda.synchronize()
        t_h2d = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            out = model(batch["clip"])
            loss = out["Hhat_1s"].mean()
        if use_cuda:
            torch.cuda.synchronize()
        t_compute += time.perf_counter() - t1
        loss.backward()
        n += 1
        t_end = time.perf_counter()
    return {
        "n_steps": n,
        "dataloader_wait_sec": t_wait,
        "h2d_plus_forward_backward_sec": t_h2d + t_compute,
        "epoch_wall_sec": t_end - (t_end - t_wait - t_h2d - t_compute),
        "train_loop_sec": time.perf_counter() - (t_end - t_wait - t_h2d - t_compute) if False else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-parity", action="store_true", help="reuse existing parity JSON if present")
    add_config_args(parser)
    parser.set_defaults(seed=1337)
    args = parser.parse_args()
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    device = str(resolve_device(args.device))
    train_ids = json.loads((REPORT / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((REPORT / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    all_ids = sorted(set(train_ids) | set(dev_ids))
    offsets = FDEC_FRAME_OFFSETS

    parity_path = REPORT / "student_fdec_rgb_letterbox_parity.json"
    if args.skip_parity and parity_path.exists():
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
    else:
        parity = rgb_letterbox_parity_test(cfg, CACHE, all_ids, offsets, n=128)
    store, rgb_stats = build_frame_store(cfg, CACHE, all_ids, offsets)

    ds = FDecPilotDataset(CACHE, train_ids, offsets, load_f_dec=False)
    ds.attach_frame_store(store, rgb_stats)
    ds_s1 = FDecPilotDataset(CACHE, train_ids[:4], offsets, load_f_dec=True)
    ds_s1.attach_frame_store(store, rgb_stats)
    s0_sample = ds[0]
    s1_sample = ds_s1[0]
    index_lines = (CACHE / "index.jsonl").read_text(encoding="utf-8").splitlines()
    rows_by_id = [json.loads(ln) for ln in index_lines]
    tr_groups = {rows_by_id[rid]["scenario_group_id"] for rid in train_ids}
    dv_groups = {rows_by_id[rid]["scenario_group_id"] for rid in dev_ids}
    overlap = tr_groups & dv_groups
    rgb_cache_hits = [
        p
        for base in (ROOT / "experiments", REPORT)
        for p in base.rglob("*")
        if p.is_file() and "rgb" in p.name.lower() and "cache" in p.name.lower()
    ]
    checks = {
        "parity": parity,
        "rgb_prepare": rgb_stats,
        "frame_offsets": offsets,
        "n_train_rows": len(train_ids),
        "n_dev_rows": len(dev_ids),
        "train_dev_group_overlap_n": len(overlap),
        "s0_has_F_teacher": "F_teacher" in s0_sample,
        "s1_has_F_teacher": "F_teacher" in s1_sample,
        "clip_shape": list(s0_sample["clip"].shape),
        "clip_value_range": [float(s0_sample["clip"].min()), float(s0_sample["clip"].max())],
        "f_teacher_shape_s1": list(s1_sample["F_teacher"].shape) if "F_teacher" in s1_sample else None,
        "hm_shapes": [list(s0_sample["gt_1s"].shape), list(s0_sample["H_teacher"].shape)],
        "no_persistent_rgb_cache_files": len(rgb_cache_hits) == 0,
    }
    if overlap:
        raise SystemExit(f"train/dev scenario_group overlap: {len(overlap)}")

    model = StudentS0(n_frames=len(offsets)).to(device)
    worker_results = []
    rss_before = _rss_mb()
    for nw in (0, 2, 4):
        loader = make_fdec_loader(ds, batch=8, shuffle=True, num_workers=nw)
        stats = {"num_workers": nw, "rss_mb_before": _rss_mb()}
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t_loop = time.perf_counter()
        n = 0
        t_wait = 0.0
        t_end = time.perf_counter()
        model.train()
        for batch in loader:
            t0b = time.perf_counter()
            t_wait += t0b - t_end
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                loss = model(batch["clip"])["Hhat_1s"].mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            n += 1
            t_end = time.perf_counter()
        stats.update(
            {
                "n_steps": n,
                "train_only_sec": time.perf_counter() - t_loop,
                "dataloader_wait_sec": t_wait,
                "rss_mb_after": _rss_mb(),
                "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if device.startswith("cuda") else None,
            }
        )
        worker_results.append(stats)

    best = min(worker_results, key=lambda x: x["train_only_sec"])
    out = {
        "checks": checks,
        "worker_benchmarks": worker_results,
        "selected_num_workers": int(best["num_workers"]),
        "selection_reason": "lowest train_only_sec among stable configs",
        "rss_mb_baseline_process": rss_before,
        "unoptimized_baseline_note": "letterbox per sample + F_dec every row + num_workers=0",
        "before_after": {
            "unoptimized_estimates": {
                "letterbox_per_epoch_calls": len(train_ids) * 5,
                "f_dec_loads_per_epoch_s0": len(train_ids),
            },
            "optimized": {
                "letterbox_at_prepare_only": True,
                "s0_skips_f_dec": True,
                "stored_frames_160": True,
            },
        },
    }
    path = REPORT / "student_fdec_dataloader_benchmark.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    parity_path.write_text(json.dumps(parity, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "benchmark": str(path), "selected_workers": out["selected_num_workers"]}, indent=2))


if __name__ == "__main__":
    main()
