#!/usr/bin/env python3
"""S2: permanently frozen parent decoder pilot."""

from __future__ import annotations

import argparse
import hashlib
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

from collision_jepa3.student_research.fdec_pilot_dataset import (  # noqa: E402
    FDecPilotDataset,
    build_frame_store,
    make_fdec_loader,
)
from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_train import run_fdec_pilot  # noqa: E402
from collision_jepa3.student_research.fdec_student import (  # noqa: E402
    FDEC_FRAME_OFFSETS,
    FDecNormStats,
    StudentS2,
    build_parent_decoder,
    decoder_state_hash,
)
from collision_jepa3.utils.device import resolve_device  # noqa: E402

REPORT = ROOT / "reports/student_architecture_tests"
CACHE = REPORT / "teacher_cache_train_pilot"
EXP_S1 = ROOT / "experiments/student_fdec_s1"
EXP = ROOT / "experiments/student_fdec_s2_frozen_decoder"
BENCH = REPORT / "student_fdec_dataloader_benchmark.json"
ETA = 3.7169490058702985


def _state_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def gradient_flow_check(model: StudentS2, device: str) -> dict:
    model.train()
    f = torch.randn(2, 128, 16, 16, device=device, requires_grad=True)
    h, _ = model.decoder.forward_with_logits(f)
    loss = h.mean()
    loss.backward()
    dec_grad = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)
    return {"f_dec_has_grad": f.grad is not None and float(f.grad.abs().sum()) > 0, "decoder_param_grad_sum": dec_grad}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    add_config_args(parser)
    parser.set_defaults(seed=1337)
    args = parser.parse_args()
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    device = str(resolve_device(args.device))
    EXP.mkdir(parents=True, exist_ok=True)

    if (EXP / ".success").exists() and not args.force:
        print(json.dumps({"ok": True, "skipped": True}, indent=2))
        return

    s1_init = EXP_S1 / "checkpoint_initial.pt"
    if not s1_init.exists():
        raise SystemExit("missing S1 checkpoint_initial.pt")

    init_meta = {
        "source": str(s1_init),
        "sha256_prefix": _state_hash(s1_init),
        "note": "S2 starts from S1 initial weights (not epoch 15)",
    }
    (EXP / "s2_init_verification.json").write_text(json.dumps(init_meta, indent=2), encoding="utf-8")

    train_ids = json.loads((REPORT / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((REPORT / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    integrity, fdec_norm = build_integrity_report(cfg, CACHE, train_ids, dev_ids, device="cpu")
    verified_dev = [tuple(x) for x in integrity["verified_pair_keys_dev"]]

    store, rgb_stats = build_frame_store(cfg, CACHE, sorted(set(train_ids) | set(dev_ids)), FDEC_FRAME_OFFSETS)
    tr_ds = FDecPilotDataset(CACHE, train_ids, FDEC_FRAME_OFFSETS, load_f_dec=True)
    va_ds = FDecPilotDataset(CACHE, dev_ids, FDEC_FRAME_OFFSETS, load_f_dec=True)
    tr_ds.attach_frame_store(store, rgb_stats)
    va_ds.attach_frame_store(store, rgb_stats)
    nw = int(json.loads(BENCH.read_text())["selected_num_workers"])
    tr_loader = make_fdec_loader(tr_ds, 8, True, num_workers=nw)
    va_loader = make_fdec_loader(va_ds, 8, False, num_workers=nw)

    parent_dec = build_parent_decoder(cfg, device="cpu")
    model = StudentS2(n_frames=5, parent_decoder=parent_dec)
    model.load_state_dict(torch.load(s1_init, map_location="cpu", weights_only=True))
    model.set_decoder_trainable(False)
    for p in model.decoder.parameters():
        p.requires_grad = False

    audit = model.param_audit()
    flow = gradient_flow_check(model.to(device), device)
    dec_h = decoder_state_hash(model.decoder)
    integrity_rec = {"decoder_hash_initial": dec_h, "gradient_flow": flow, **audit}
    (EXP / "student_s2_decoder_integrity.json").write_text(json.dumps(integrity_rec, indent=2), encoding="utf-8")
    diag = ROOT / "reports/student_warning_diagnostics/student_s2_decoder_integrity.json"
    diag.parent.mkdir(parents=True, exist_ok=True)
    diag.write_text(json.dumps(integrity_rec, indent=2), encoding="utf-8")

    mu = torch.from_numpy(fdec_norm["mu"]).view(-1, 1, 1)
    sigma = torch.from_numpy(fdec_norm["sigma"]).view(-1, 1, 1)
    norm = FDecNormStats(mu, sigma)

    config = {
        "variant": "S2",
        "parent_decoder_frozen_all_epochs": True,
        "init_from": str(s1_init),
        "eta_fdec": ETA,
        "alpha_kd": 0.5,
        "lr": 3e-4,
        "batch": 8,
        "seed": 1337,
        "epochs": 15,
        "frame_offsets": FDEC_FRAME_OFFSETS,
        "pipeline_version": "optimized_v2",
        "num_workers": nw,
        "s1_reference": {"decoder_unfreeze_epoch": 3, "eta_fdec": ETA},
        **audit,
        "rgb_prepare": rgb_stats,
    }
    (EXP / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), EXP / "checkpoint_initial.pt")
    (REPORT / "student_s2_parameter_budget.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    t0 = time.time()
    log_path = EXP / "run.log"
    try:
        hist = run_fdec_pilot(
            "S2",
            model,
            tr_loader,
            va_loader,
            device=device,
            epochs=15,
            lr=3e-4,
            alpha_kd=0.5,
            eta_fdec=ETA,
            norm=norm,
            cfg_eval=cfg,
            exp_dir=EXP,
            report_root=REPORT,
            verified_pair_keys=verified_dev,
            start_epoch=0,
            decoder_freeze_epochs=999,
            eval_epochs=(1, 3, 6, 9, 12, 15),
        )
        summary = {"elapsed_sec": time.time() - t0, "history_epochs": len(hist["history"])}
        (EXP / "student_s2_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (ROOT / "reports/student_warning_diagnostics/student_s2_training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (EXP / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (EXP / ".success").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        print(json.dumps({"ok": True, "summary": summary}, indent=2))
    except Exception:
        (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
