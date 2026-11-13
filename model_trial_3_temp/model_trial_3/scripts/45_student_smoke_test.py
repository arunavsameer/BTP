#!/usr/bin/env python3
"""Part 14: one-batch student smoke test before pilots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.cnn import assert_param_budget, preset_architectures, StudentRGBNet  # noqa: E402
from collision_jepa3.student_research.dataset_pilot import PilotAnchorDataset, collate_pilot, filter_rows_by_groups  # noqa: E402
from collision_jepa3.student_research.experiment_runner import gpu_mem_mb  # noqa: E402
from collision_jepa3.student_research.frames import FRAME_CANDIDATES  # noqa: E402
from collision_jepa3.student_research.splits_dev import pilot_train_dev_groups  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args)
    device = resolve_device(args.device)
    cache = out / "teacher_cache"
    if not (cache / "teacher_pred.npz").exists():
        raise SystemExit("teacher cache missing")
    groups = load_splits(cfg.path("paths.split_file")).get("val", [])
    tr_g, _ = pilot_train_dev_groups(groups, seed=int(cfg.get("project.seed", 1337)))
    row = filter_rows_by_groups(cache, set(tr_g[:1]), split="val")[:16]
    if not row:
        raise SystemExit("no rows for smoke")
    offsets = FRAME_CANDIDATES["F2"]
    ds = PilotAnchorDataset(cache, row_ids=row, frame_offsets=offsets, input_size=160)
    ds.set_cache_root(cfg)
    ds.preload_clips()
    batch = collate_pilot([ds[0]])
    arch = preset_architectures()[0]
    model = StudentRGBNet(arch, n_frames=len(offsets), input_size=160).to(device)
    n = assert_param_budget(model)
    clip = batch["clip"].to(device)
    model.train()
    out_t = model(clip)
    loss = ((out_t["Hhat_1s"] - batch["gt_1s"].to(device)) ** 2).mean()
    loss.backward()
    rep = {
        "ok": True,
        "params": n,
        "arch": arch.name,
        "batch": int(clip.shape[0]),
        "device": str(device),
        "gpu_mem": gpu_mem_mb(),
    }
    (out / "smoke_test.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
