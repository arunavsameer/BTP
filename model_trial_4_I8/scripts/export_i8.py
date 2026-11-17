#!/usr/bin/env python3
"""Dump CollisionStudentV2 weights as int8 + per-channel/per-tensor affine scales."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student_i8.constants import ARCH  # noqa: E402
from student_i8.load import load_student  # noqa: E402
from student_i8.quant import dequant_state, quantize_state  # noqa: E402

DEFAULT_SRC = Path("/storage/BTP/model_trial_4/ckpts/checkpoint_epoch_22.pt")
DEFAULT_DST = ROOT / "ckpts" / "checkpoint_epoch_22_i8.pt"

NOTE = (
    "Weight-only symmetric int8 (qmin=-128,qmax=127, zero_point=0). "
    "Conv/Linear weights: per-out-channel scale (axis 0). "
    "1D affine (GN/LN weight/bias, most biases): per-tensor scale. "
    "This is NOT fused int8 GEMM / QAT / calibrated PTQ. "
    "Python inference dequantizes to fp32 (or fp16) then runs the original modules. "
    "No local RGB cache was used for calibration."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = p.parse_args()
    model, meta = load_student(args.src)
    bb_q = quantize_state(model.backbone.state_dict())
    dec_q = quantize_state(model.decoder.state_dict())

    # reconstruction error on weights
    bb_hat = dequant_state(bb_q, dtype=torch.float32)
    max_err = 0.0
    for k, v in model.backbone.state_dict().items():
        if torch.is_floating_point(v):
            max_err = max(max_err, float((v.cpu().float() - bb_hat[k]).abs().amax()))

    payload = {
        "epoch": meta.get("epoch"),
        "arch": ARCH,
        "backbone_q": bb_q,
        "decoder_q": dec_q,
        "audit": meta.get("audit"),
        "lam_feat": meta.get("lam_feat"),
        "global_step": meta.get("global_step"),
        "config": {
            "arch": ARCH,
            "ffn_hidden": 155,
            "decoder": {"z_channels": 128, "hidden": 48, "dilated": True, "out_grid": 5, "pool": "max"},
            "input": {"T": 16, "C": 3, "H": 256, "W": 256, "layout": "B,T,C,H,W", "value_range": "[0,1]"},
        },
        "dtype": "int8_weight_only",
        "quant_scheme": "symmetric_affine_int8_per_channel_conv_linear",
        "dtype_note": NOTE,
        "weight_recon_max_abs": max_err,
        "source_ckpt": str(args.src),
    }
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.dst)
    print(f"wrote {args.dst} ({args.dst.stat().st_size} bytes)")
    print(f"backbone weight recon max_abs={max_err:.6e}")


if __name__ == "__main__":
    main()
