#!/usr/bin/env python3
"""Dequant int8 CollisionStudentV2 weights and compare dummy forward vs fp32 source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student_i8.constants import STOP_THRESHOLD  # noqa: E402
from student_i8.load import load_student  # noqa: E402

DEFAULT_I8 = ROOT / "ckpts" / "checkpoint_epoch_22_i8.pt"
DEFAULT_FP32 = Path("/storage/BTP/model_trial_4/ckpts/checkpoint_epoch_22.pt")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_I8)
    p.add_argument("--ref", type=Path, default=DEFAULT_FP32)
    p.add_argument("--compute", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dtype = torch.float16 if args.compute == "fp16" else torch.float32
    device = torch.device("cpu")
    q_model, meta = load_student(args.ckpt, compute_dtype=dtype)
    print(f"arch={meta.get('arch')} epoch={meta.get('epoch')} dtype={meta.get('dtype')}")
    print(f"quant={meta.get('quant_scheme')} weight_recon_max_abs={meta.get('weight_recon_max_abs')}")

    torch.manual_seed(args.seed)
    x = torch.rand(1, 16, 3, 256, 256)
    if dtype == torch.float16:
        x_in = x.half()
        try:
            with torch.no_grad():
                q_out = q_model(x_in)
        except Exception as exc:
            print(f"fp16 dequant-forward failed ({type(exc).__name__}: {exc}); using fp32")
            q_model.float()
            dtype = torch.float32
            with torch.no_grad():
                q_out = q_model(x)
    else:
        with torch.no_grad():
            q_out = q_model(x.to(device))

    h = q_out["Hhat_1s"]
    print(f"Hhat_1s shape={tuple(h.shape)} dtype={h.dtype} peak={float(h.float().amax()):.4f} STOP_thr={STOP_THRESHOLD}")

    if args.ref.is_file():
        ref, _ = load_student(args.ref, compute_dtype=torch.float32)
        with torch.no_grad():
            r = ref(x)["Hhat_1s"]
        err = (h.float() - r.float()).abs()
        print(f"vs fp32 source: max_abs={float(err.amax()):.6e} mean_abs={float(err.mean()):.6e}")
    print("SMOKE_OK" if tuple(h.shape) == (1, 5, 5) else "SMOKE_FAIL")


if __name__ == "__main__":
    main()
