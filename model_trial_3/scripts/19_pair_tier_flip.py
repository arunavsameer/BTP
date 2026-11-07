#!/usr/bin/env python3
"""Matched-background warning-tier flips on dense_dump (scene-bootstrap CIs)."""

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
FRAME_STEP = 6


def tier_from_peak(s: np.ndarray) -> np.ndarray:
    return np.where(s < CAUTION, 0, np.where(s < STOP, 1, 2)).astype(np.int8)


def build_pairs(idx: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(idx):
        groups.setdefault((r["sample_id"], r["outcome"], int(r["frame"])), []).append(i)
    bad = [k for k, v in groups.items() if len(v) != 2]
    if bad:
        raise SystemExit(f"expected 2 rows per pair key, got mismatches e.g. {bad[0]}")
    for v in groups.values():
        bgs = {idx[j]["background_id"] for j in v}
        if len(bgs) != 2:
            raise SystemExit(f"expected 2 backgrounds per pair, got {bgs}")
    a = np.array([v[0] for v in groups.values()], dtype=np.int64)
    b = np.array([v[1] for v in groups.values()], dtype=np.int64)
    sid = np.array([idx[i]["sample_id"] for i in a])
    post = np.array([not idx[i]["pre_fork"] for i in a], dtype=bool)
    scenes = {s: np.where(sid == s)[0] for s in set(sid)}
    return a, b, sid, post, scenes


def temporal_neighbors(idx: list[dict], step: int = FRAME_STEP) -> tuple[np.ndarray, np.ndarray]:
    pos = {(r["clip_id"], int(r["frame"])): i for i, r in enumerate(idx)}
    nxt = [(i, pos[(r["clip_id"], int(r["frame"]) + step)]) for i, r in enumerate(idx) if (r["clip_id"], int(r["frame"]) + step) in pos]
    if not nxt:
        return np.array([], dtype=int), np.array([], dtype=int)
    ni, nj = np.array(nxt, dtype=np.int64).T
    return ni, nj


def scene_bootstrap_ci(x: np.ndarray, mask: np.ndarray, scenes: dict[str, np.ndarray], rng: np.random.Generator, n: int = 500) -> np.ndarray:
    ks = [s for s in scenes if mask[scenes[s]].any()]
    if not ks:
        return np.array([np.nan, np.nan])
    vals = [x[scenes[s]][mask[scenes[s]]] for s in ks]
    means = []
    for _ in range(n):
        picks = rng.integers(0, len(ks), len(ks))
        means.append(np.concatenate([vals[k] for k in picks]).mean())
    return np.percentile(means, [2.5, 97.5])


def analyze_model(
    name: str,
    H: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    post: np.ndarray,
    gt_t: np.ndarray,
    ni: np.ndarray,
    nj: np.ndarray,
    scenes: dict,
    rng: np.random.Generator,
) -> dict:
    mx = lambda h: h.reshape(len(h), -1).max(axis=1)
    t = tier_from_peak(mx(H))
    ta, tb = t[a], t[b]
    flip = ta != tb
    scale_free = np.abs(H[a] - H[b]).sum(axis=(1, 2)) / (0.5 * (H[a] + H[b]).sum(axis=(1, 2)) + 1e-9)
    jit = float((t[ni] != t[nj]).mean()) if len(ni) else float("nan")

    out: dict = {
        "scale_free_gap_mean": float(scale_free.mean()),
        "mean_H": float(np.nanmean(H)),
        "tier_jitter_0p2s": jit,
    }
    for lab, m in [("all", np.ones(len(a), bool)), ("post_fork", post), ("pre_fork", ~post)]:
        ci = scene_bootstrap_ci(flip.astype(float), m, scenes, rng)
        out[lab] = {
            "flip_rate": float(flip[m].mean()) if m.any() else float("nan"),
            "flip_ci_2p5_97p5": [float(ci[0]), float(ci[1])],
            "stop_safe_cross": float((((ta == 2) & (tb == 0)) | ((ta == 0) & (tb == 2)))[m].mean())
            if m.any()
            else float("nan"),
        }
    for g, nmg in [(2, "GT_STOP"), (1, "GT_THREAT"), (0, "GT_SAFE")]:
        m = gt_t[a] == g
        if int(m.sum()) <= 30:
            continue
        out[nmg] = {
            "n": int(m.sum()),
            "flip_rate": float(flip[m].mean()),
            "one_side_stop": float(((ta == 2) != (tb == 2))[m].mean()),
            "one_side_safe": float(((ta == 0) != (tb == 0))[m].mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default="reports/dense_dump")
    parser.add_argument("names", nargs="*", default=["gr40g"])
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gt-matched-only", action="store_true")
    parser.add_argument("--gt-tol", type=float, default=1e-3)
    args = parser.parse_args()

    dump = Path(args.dump)
    if not dump.is_absolute():
        dump = ROOT / dump
    idx = [json.loads(l) for l in (dump / "index.jsonl").open()]
    G = np.load(dump / "gt_heatmaps.npz")["gt_1s"].astype(np.float64)
    mx = lambda h: h.reshape(len(h), -1).max(axis=1)
    gt_t = tier_from_peak(mx(G))
    a, b, sid, post, scenes = build_pairs(idx)
    if args.gt_matched_only:
        n_before = len(a)
        a, b, valid = filter_gt_matched_pairs(G, a, b, tol=args.gt_tol)
        sid = sid[valid]
        post = post[valid]
        scenes = {s: np.where(sid == s)[0] for s in set(sid)}
        print(f"gt-matched pairs: {len(a)} / {n_before} (tol={args.gt_tol})")
    ni, nj = temporal_neighbors(idx)
    rng = np.random.default_rng(args.seed)

    gt_jit = float((gt_t[ni] != gt_t[nj]).mean()) if len(ni) else float("nan")
    print("pairs", len(a), "scenes", len(scenes), " GT tier change in 0.2 s:", round(gt_jit, 3))

    names = args.names or ["gr40g"]
    report = {"gt_tier_jitter_0p2s": gt_jit, "n_pairs": len(a), "n_scenes": len(scenes), "models": {}}

    for nm in names:
        path = dump / f"pred_{nm}.npz"
        if not path.exists():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        H = np.load(path)["H"].astype(np.float64)
        if H.shape[0] != len(idx):
            raise SystemExit(f"{nm}: H rows {H.shape[0]} != index {len(idx)}")
        stats = analyze_model(nm, H, G, a, b, post, gt_t, ni, nj, scenes, rng)
        report["models"][nm] = stats
        print(f"\n== {nm} ==  scale-free gap {stats['scale_free_gap_mean']:.3f}  mean H {stats['mean_H']:.4f}")
        print(f" tier change in 0.2 s (same clip, includes real change): model {stats['tier_jitter_0p2s']:.3f}")
        for lab in ("all", "post_fork", "pre_fork"):
            block = stats[lab]
            ci = block["flip_ci_2p5_97p5"]
            print(
                f" [{lab:9s}] flip {block['flip_rate']:.3f} CI {np.round(ci, 3)}"
                f"  STOP<->SAFE {block['stop_safe_cross']:.4f}"
            )
        for key in ("GT_STOP", "GT_THREAT", "GT_SAFE"):
            if key not in stats:
                continue
            s = stats[key]
            print(
                f" {key:9s} n={s['n']:5d} flip {s['flip_rate']:.3f}"
                f"  one side STOP other not: {s['one_side_stop']:.3f}"
                f"  one side SAFE other not: {s['one_side_safe']:.3f}"
            )

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
