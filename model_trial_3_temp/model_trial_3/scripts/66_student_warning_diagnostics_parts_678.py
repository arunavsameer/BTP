#!/usr/bin/env python3
"""Complete warning diagnostics parts 6–8 (read-only)."""

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
from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_train import apply_backbone  # noqa: E402
from collision_jepa3.student_research.fdec_student import (  # noqa: E402
    FDEC_FRAME_OFFSETS,
    FDecNormStats,
    StudentS1,
    build_parent_decoder,
)
from collision_jepa3.student_research.warning_diagnostics_parts_678 import (  # noqa: E402
    _infer_f_student,
    run_part6_fdec_alignment,
    run_part7_gradients,
    run_part8,
    write_final_report,
)
from collision_jepa3.utils.device import resolve_device  # noqa: E402

REPORT = ROOT / "reports/student_architecture_tests"
OUT = ROOT / "reports/student_warning_diagnostics"
CACHE = REPORT / "teacher_cache_train_pilot"
BASELINE = ROOT / "experiments/student_fdec_unoptimized_baseline/shared_backbone_init.pt"
EXP_S0 = ROOT / "experiments/student_fdec_s0"
EXP_S1 = ROOT / "experiments/student_fdec_s1"
NPZ = OUT / "development_predictions.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=1337))
    device = str(resolve_device(args.device))
    warning = cfg.get("warning") or {}

    train_ids = json.loads((REPORT / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((REPORT / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    integrity, fdec_norm = build_integrity_report(cfg, CACHE, train_ids, dev_ids, device="cpu")
    rows = [json.loads(l) for l in (CACHE / "index.jsonl").open()]
    pred_npz = np.load(CACHE / "teacher_pred.npz")
    f_teacher = np.asarray(pred_npz["F_dec"][dev_ids], dtype=np.float32)

    if not NPZ.exists():
        raise SystemExit(f"missing {NPZ}; run round 2 first")
    z = np.load(NPZ)
    gts = z["gt"]
    s0_h, s1_h, parent_h = z["s0"], z["s1"], z["parent_cached"]
    pre_fork = np.array([bool(rows[rid].get("pre_fork")) for rid in dev_ids])

    bench = json.loads((REPORT / "student_fdec_dataloader_benchmark.json").read_text())
    nw = int(bench["selected_num_workers"])
    store, _ = build_frame_store(cfg, CACHE, sorted(set(train_ids) | set(dev_ids)), FDEC_FRAME_OFFSETS)
    ds = FDecPilotDataset(CACHE, dev_ids, FDEC_FRAME_OFFSETS, load_f_dec=True)
    ds.attach_frame_store(store, {})
    loader = make_fdec_loader(ds, 8, False, num_workers=nw)

    parent_dec = build_parent_decoder(cfg, device)
    s1 = StudentS1(n_frames=5, parent_decoder=build_parent_decoder(cfg, "cpu"))
    s1.load_state_dict(torch.load(EXP_S1 / "checkpoint_epoch_15.pt", map_location="cpu", weights_only=True))
    f_student = _infer_f_student(s1, loader, device)

    part6 = run_part6_fdec_alignment(dev_ids, f_student, f_teacher, gts, warning, parent_dec, device)
    (OUT / "part6_fdec_alignment.json").write_text(json.dumps(part6, indent=2), encoding="utf-8")

    mu = torch.from_numpy(fdec_norm["mu"]).view(-1, 1, 1)
    sigma = torch.from_numpy(fdec_norm["sigma"]).view(-1, 1, 1)
    norm = FDecNormStats(mu, sigma)
    fdec_norm_npz = REPORT / "fdec_norm_train_only.npz"
    if fdec_norm_npz.exists():
        fn = np.load(fdec_norm_npz)
        norm = FDecNormStats(torch.from_numpy(fn["mu"]).view(-1, 1, 1), torch.from_numpy(fn["sigma"]).view(-1, 1, 1))

    part7 = run_part7_gradients(
        cfg,
        CACHE,
        train_ids,
        REPORT,
        BASELINE,
        device,
        norm,
        s0_ckpt=EXP_S0 / "checkpoint_epoch_15.pt",
        s1_ckpt=EXP_S1 / "checkpoint_epoch_15.pt",
        num_workers=nw,
    )
    (OUT / "part7_late_gradient_diagnostics.json").write_text(json.dumps(part7, indent=2), encoding="utf-8")

    part8 = run_part8(
        rows,
        dev_ids,
        integrity["verified_pair_keys_dev"],
        warning,
        NPZ,
        s0_h=s0_h,
        s1_h=s1_h,
        parent_h=parent_h,
        gts=gts,
        pre_fork=pre_fork,
    )
    (OUT / "part8_group_and_pair_diagnostics.json").write_text(json.dumps(part8, indent=2), encoding="utf-8")

    write_final_report(OUT, {})
    print(json.dumps({"ok": True, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
