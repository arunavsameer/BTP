#!/usr/bin/env python3
"""Convert CollisionStudentV2 fp32 checkpoint tensors to float16 (inference-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student_f16.constants import ARCH  # noqa: E402
from student_f16.load import load_student  # noqa: E402

DEFAULT_SRC = Path("/storage/BTP/model_trial_4/ckpts/checkpoint_epoch_22.pt")
DEFAULT_DST = ROOT / "ckpts" / "checkpoint_epoch_22_f16.pt"

DTYPE_NOTE = (
    "All floating parameters and persistent floating buffers stored as torch.float16. "
    "Integer tensors (none in this student) would be left unchanged. "
    "This architecture uses GroupNorm + LayerNorm (no BatchNorm). "
    "GN/LN affine weights and biases are fp16. "
    "Non-persistent buffers (_mean, _std, sincos pos) are recreated at init and cast via model.half()."
)


def to_f16_state(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if torch.is_floating_point(v):
            out[k] = v.detach().cpu().contiguous().to(torch.float16)
        else:
            out[k] = v.detach().cpu().contiguous()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = p.parse_args()
    model, meta = load_student(args.src)
    model.half()
    payload = {
        "epoch": meta.get("epoch"),
        "arch": ARCH,
        "backbone": to_f16_state(model.backbone.state_dict()),
        "decoder": to_f16_state(model.decoder.state_dict()),
        "audit": meta.get("audit"),
        "lam_feat": meta.get("lam_feat"),
        "global_step": meta.get("global_step"),
        "config": {
            "arch": ARCH,
            "ffn_hidden": 155,
            "decoder": {"z_channels": 128, "hidden": 48, "dilated": True, "out_grid": 5, "pool": "max"},
            "input": {"T": 16, "C": 3, "H": 256, "W": 256, "layout": "B,T,C,H,W", "value_range": "[0,1]"},
        },
        "dtype": "float16",
        "dtype_note": DTYPE_NOTE,
        "source_ckpt": str(args.src),
    }
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.dst)
    print(f"wrote {args.dst} ({args.dst.stat().st_size} bytes)")
    n_fp16 = sum(int(t.is_floating_point() and t.dtype == torch.float16) for t in payload["backbone"].values())
    n_fp16 += sum(int(t.is_floating_point() and t.dtype == torch.float16) for t in payload["decoder"].values())
    print(f"fp16 tensors: {n_fp16}")


if __name__ == "__main__":
    main()
