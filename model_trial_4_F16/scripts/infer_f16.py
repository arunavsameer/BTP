#!/usr/bin/env python3
"""Load fp16 CollisionStudentV2 weights and run a dummy window forward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student_f16.constants import STOP_THRESHOLD  # noqa: E402
from student_f16.load import load_student  # noqa: E402

DEFAULT_CKPT = ROOT / "ckpts" / "checkpoint_epoch_22_f16.pt"


def pick_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def run_forward(model, x: torch.Tensor) -> dict:
    with torch.no_grad():
        return model(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--compute",
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="fp16: model.half()+inputs half. fp32: dequant weights to fp32 (CPU fallback). auto: fp16 on CUDA, try fp16 then fp32 on CPU.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(args.device)
    model, meta = load_student(args.ckpt, map_location="cpu")
    print(f"arch={meta.get('arch')} epoch={meta.get('epoch')} dtype={meta.get('dtype')}")
    print(f"device={device} cuda={torch.cuda.is_available()}")

    torch.manual_seed(args.seed)
    x_fp32 = torch.rand(1, 16, 3, 256, 256)

    compute = args.compute
    if compute == "auto":
        compute = "fp16" if device.type == "cuda" else "fp16"

    def try_fp16():
        m = model.to(device)
        m.half()
        x = x_fp32.to(device=device, dtype=torch.float16)
        return run_forward(m, x)

    used = compute
    fallback = False
    if compute == "fp16":
        try:
            out = try_fp16()
        except Exception as exc:  # CPU conv3d often lacks fp16 kernels
            print(f"fp16 forward failed ({type(exc).__name__}: {exc})")
            print("falling back to fp32 compute with fp16-stored weights")
            used = "fp32"
            fallback = True
            model.float()
            model.to(device)
            out = run_forward(model, x_fp32.to(device))
    else:
        model.float()
        model.to(device)
        out = run_forward(model, x_fp32.to(device))
        used = "fp32"

    h = out["Hhat_1s"]
    peak = float(h.float().amax().cpu())
    print(f"compute={used} fallback={fallback}")
    print(f"Hhat_1s shape={tuple(h.shape)} dtype={h.dtype} peak={peak:.4f} STOP_thr={STOP_THRESHOLD}")
    print("SMOKE_OK" if tuple(h.shape) == (1, 5, 5) else "SMOKE_FAIL")


if __name__ == "__main__":
    main()
