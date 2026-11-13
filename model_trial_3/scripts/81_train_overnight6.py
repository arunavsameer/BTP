#!/usr/bin/env python3
"""Train one overnight Pair-CF student (models 1–6). Selection only. No test."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.gpu_bind import bind_cuda  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=int, required=True, choices=list(range(1, 7)))
_parser.add_argument("--epochs", type=int, default=30)
_parser.add_argument("--batch", type=int, default=8)
_parser.add_argument("--lr", type=float, default=3e-4)
_parser.add_argument("--device", default="auto")
_parser.add_argument("--num-workers", type=int, default=4)
_args = _parser.parse_args()
bind_cuda(str(_args.device))

from collision_jepa3.student_research.overnight6.train import NAMES, exp_dir, pick_cuda, run_train  # noqa: E402


def main() -> None:
    device = pick_cuda(str(_args.device))
    print(f"training {NAMES[_args.model]} on {device} vis={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    try:
        run_train(
            _args.model,
            epochs=_args.epochs,
            batch=_args.batch,
            lr=_args.lr,
            device=device,
            num_workers=_args.num_workers,
            smoke=False,
        )
    except Exception:
        dest = exp_dir(_args.model)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
