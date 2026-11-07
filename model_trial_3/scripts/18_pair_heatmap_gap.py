#!/usr/bin/env python3
"""Matched-background pair analysis on dense_dump H predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.training.pair_signals import filter_gt_matched_pairs, pair_indices_from_index


def load_H(dump_dir: Path, pred: str) -> np.ndarray:
    p = Path(pred)
    if p.suffix == ".npz":
        path = p
    else:
        path = dump_dir / f"pred_{pred}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    H = np.load(path)["H"].astype(np.float64)
    return H


def pair_indices(idx: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    """Pair rows with same (sample_id, outcome, frame) and two backgrounds."""
    a, b = pair_indices_from_index(idx)
    return a, b, int(a.shape[0])


def analyze(
    H: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int = 0,
    flip_thresholds: tuple[float, ...] = (0.25, 0.3, 0.5, 0.6, 0.7),
    caution: float = 0.25,
    stop: float = 0.60,
) -> dict:
    if len(a) == 0:
        return {"n_pairs": 0}
    valid = ~(np.isnan(H[a]).any(axis=(1, 2)) | np.isnan(H[b]).any(axis=(1, 2)))
    a, b = a[valid], b[valid]
    if len(a) == 0:
        return {"n_pairs": 0, "note": "all pair rows NaN in H"}

    pair_gap = np.abs(H[a] - H[b]).mean(axis=(1, 2))
    rng = np.random.default_rng(seed)
    other = rng.permutation(len(a))
    unrel = np.abs(H[a] - H[b[other]]).mean(axis=(1, 2))

    mx = lambda h: H[h].reshape(len(h), -1).max(axis=1)
    pa, pb = mx(a), mx(b)

    flips = {}
    for thr in flip_thresholds:
        flips[f"flip_max_ge_{thr:g}"] = float(((pa >= thr) != (pb >= thr)).mean())

    # Project warning bins (peak vs caution / stop cuts)
    def warn_bin(p):
        out = np.zeros_like(p, dtype=np.int8)
        out[p >= stop] = 2  # STOP
        out[(p >= caution) & (p < stop)] = 1  # THREAT
        return out

    flips["warning_label_flip"] = float((warn_bin(pa) != warn_bin(pb)).mean())
    flips["safe_vs_stop_flip"] = float(
        (((pa < caution) & (pb >= stop)) | ((pb < caution) & (pa >= stop))).mean()
    )

    return {
        "n_pairs": int(len(a)),
        "mean_abs_H_diff_matched": float(pair_gap.mean()),
        "mean_abs_H_diff_unrelated": float(unrel.mean()),
        "ratio_matched_over_unrelated": float(pair_gap.mean() / max(unrel.mean(), 1e-12)),
        **flips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default="reports/dense_dump", help="dense_dump directory")
    parser.add_argument(
        "--pred",
        nargs="*",
        default=["bce_fine"],
        help="pred stem (pred_<name>.npz) or path to .npz",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gt-matched-only",
        action="store_true",
        help="exclude pairs whose gt_1s differ (max cell |Δ| >= gt-tol)",
    )
    parser.add_argument("--gt-tol", type=float, default=1e-3)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    dump_dir = Path(args.dump)
    if not dump_dir.is_absolute():
        dump_dir = ROOT / dump_dir
    idx = [json.loads(l) for l in (dump_dir / "index.jsonl").open()]
    bgs = sorted({r["background_id"] for r in idx})
    print(f"dump={dump_dir} rows={len(idx)} backgrounds={bgs}")

    report: dict[str, dict] = {}
    for pred in args.pred:
        name = Path(pred).stem.replace("pred_", "") if pred.endswith(".npz") else pred
        H = load_H(dump_dir, pred)
        if H.shape[0] != len(idx):
            raise SystemExit(f"{pred}: H rows {H.shape[0]} != index {len(idx)}")
        a, b, n_raw = pair_indices(idx)
        if args.gt_matched_only:
            gt_path = dump_dir / "gt_heatmaps.npz"
            if not gt_path.exists():
                raise SystemExit(f"--gt-matched-only requires {gt_path}")
            G = np.load(gt_path)["gt_1s"]
            n_before = int(a.shape[0])
            a, b, _ = filter_gt_matched_pairs(G, a, b, tol=args.gt_tol)
            print(f"gt-matched pairs: {a.shape[0]} / {n_before} (tol={args.gt_tol})")
        stats = analyze(H, a, b, seed=args.seed)
        stats["pred"] = str(pred)
        stats["n_pairs_raw"] = n_raw
        stats["gt_matched_only"] = bool(args.gt_matched_only)
        report[name] = stats
        print(f"\n=== {name} ===")
        print("pairs:", stats.get("n_pairs", n_raw))
        if stats.get("n_pairs", 0):
            print(
                "mean |H diff|, same scene other background:",
                round(stats["mean_abs_H_diff_matched"], 4),
            )
            print(
                "mean |H diff|, unrelated scenes:           ",
                round(stats["mean_abs_H_diff_unrelated"], 4),
            )
            print("ratio matched/unrelated:", round(stats["ratio_matched_over_unrelated"], 4))
            for k, v in sorted(stats.items()):
                if k.startswith("flip_") or k.endswith("_flip"):
                    print(f"{k}:", round(v, 4))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
