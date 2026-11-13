#!/usr/bin/env python3
"""Build Z_teacher_16 sidecar from cached F_dec + frozen Pair-CF decoder.

Writes /raid/.../z16_sidecar_v1/Z_16.npy (float16 [N,1,16,16]). Does not modify
the Trial-3 parent_target_cache_v2 files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student.parent import load_parent_decoder  # noqa: E402
from student.paths import parent_cache, parent_ckpt, z16_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache", default=None)
    parser.add_argument("--parent-ckpt", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    par = Path(args.parent_cache) if args.parent_cache else parent_cache()
    out = Path(args.out) if args.out else z16_cache()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((par / "metadata.json").read_text())
    n = int(meta["num_training_anchors"])
    f = np.memmap(par / "F_dec.npy", dtype=np.float16, mode="r", shape=(n, 128, 16, 16))
    zpath = out / "Z_16.npy"
    z = np.memmap(zpath, dtype=np.float16, mode="w+", shape=(n, 1, 16, 16))
    decoder = load_parent_decoder(Path(args.parent_ckpt) if args.parent_ckpt else parent_ckpt())
    decoder.to(args.device).eval()
    bs = int(args.batch)
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            x = torch.from_numpy(np.asarray(f[sl], dtype=np.float32)).to(args.device)
            logits = decoder.forward_fine_logits(x).cpu().numpy().astype(np.float16)
            z[sl] = logits
            if i % (bs * 20) == 0:
                print(f"z16 {i}/{n}", flush=True)
    z.flush()
    (out / "COMPLETE").write_text(f"n={n}\nsource={par}\n", encoding="utf-8")
    print(f"wrote {zpath} n={n}")


if __name__ == "__main__":
    main()
