#!/usr/bin/env python3
"""Pair tier flips: cutoff-near vs large peak-gap (dense_dump)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.training.pair_signals import filter_gt_matched_pairs
CAUTION = 0.25
STOP = 0.60
NEAR_EPS = 0.10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default="reports/dense_dump")
    parser.add_argument("names", nargs="*", default=["gr40g", "padr"])
    parser.add_argument("--near", type=float, default=NEAR_EPS)
    parser.add_argument("--big-diff", type=float, default=0.3)
    parser.add_argument("--gt-matched-only", action="store_true")
    parser.add_argument("--gt-tol", type=float, default=1e-3)
    args = parser.parse_args()

    dump = ROOT / args.dump if not Path(args.dump).is_absolute() else Path(args.dump)
    idx = [json.loads(l) for l in (dump / "index.jsonl").open()]
    g: dict[tuple, list[int]] = {}
    for i, r in enumerate(idx):
        g.setdefault((r["sample_id"], r["outcome"], int(r["frame"])), []).append(i)
    a = np.array([v[0] for v in g.values()], dtype=np.int64)
    b = np.array([v[1] for v in g.values()], dtype=np.int64)
    if args.gt_matched_only:
        G = np.load(dump / "gt_heatmaps.npz")["gt_1s"]
        n_before = len(a)
        a, b, _ = filter_gt_matched_pairs(G, a, b, tol=args.gt_tol)
        print(f"gt-matched pairs: {len(a)} / {n_before} (tol={args.gt_tol})")
    post = np.array([not idx[i]["pre_fork"] for i in a], dtype=bool)

    mx = lambda h: h.reshape(len(h), -1).max(axis=1)
    tier = lambda s: np.where(s < CAUTION, 0, np.where(s < STOP, 1, 2))
    cuts = (CAUTION, STOP)

    for nm in args.names:
        path = dump / f"pred_{nm}.npz"
        if not path.exists():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        H = np.load(path)["H"].astype(np.float64)
        pa, pb = mx(H[a]), mx(H[b])
        fl = tier(pa) != tier(pb)
        near = np.minimum.reduce([abs(pa - c) for c in cuts] + [abs(pb - c) for c in cuts]) < float(args.near)
        print(f"\n== {nm} ==")
        for lab, m in [("post-fork", post), ("pre-fork", ~post)]:
            mm = m & fl
            n = int(mm.sum())
            if n == 0:
                print(nm, lab, "flips: 0")
                continue
            diff = np.abs(pa - pb)[mm]
            print(
                nm,
                lab,
                "flips:",
                n,
                f" both peaks within {args.near} of a cutoff:",
                round(float(near[mm].mean()), 3),
                " median |peak diff|:",
                round(float(np.median(diff)), 3),
                f" flips with |diff|>{args.big_diff}:",
                round(float((diff > args.big_diff).mean()), 3),
            )


if __name__ == "__main__":
    main()
