#!/usr/bin/env python3
"""A1 label audit, A2 heatmap-vs-event agreement, A3 pre-fork ambiguity ceiling."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collision_jepa3.evaluation.labels import (  # noqa: E402
    DEFAULT_WARNING,
    event_alert,
    heatmap_alert,
)


SIDES = ("LEFT", "CENTER", "RIGHT")
CUTOFFS = (0.42, 0.50, 0.70)


def load_dump(root: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = [json.loads(l) for l in (root / "index.jsonl").open()]
    gt = np.load(root / "gt_heatmaps.npz")
    return rows, {k: np.asarray(gt[k], dtype=np.float32) for k in gt.files}


def _hist(xs: list[float], bins: list[float]) -> dict[str, int]:
    arr = np.asarray(xs, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=bins)
    out = {}
    for i, c in enumerate(counts):
        out[f"[{edges[i]:.1f},{edges[i+1]:.1f})"] = int(c)
    return out


def a1(rows: list[dict]) -> dict:
    clips = {}
    for r in rows:
        clips.setdefault(r["clip_id"], r)
    clip_out = Counter(v["outcome"] for v in clips.values())
    clip_raw = Counter(v["outcome_raw"] for v in clips.values())
    sides = Counter(v["true_side"] for v in clips.values())
    tti = [float(r["tti_s"]) for r in rows if r["tti_s"] is not None]
    n = len(rows)
    within = {
        k: float(sum(bool(r[k]) for r in rows) / n)
        for k in ("collision_within_1s", "collision_within_2s", "collision_within_3s", "collision_within_4s")
    }
    pre = sum(bool(r["pre_fork"]) for r in rows)
    return {
        "n_clips": len(clips),
        "n_anchors": n,
        "clip_outcome": dict(clip_out),
        "clip_outcome_raw": dict(clip_raw),
        "safe_clips": int(clip_out.get("safe", 0)),
        "true_side_clips": dict(sides),
        "true_side_anchors": dict(Counter(r["true_side"] for r in rows)),
        "anchor_class": dict(Counter(r["anchor_class"] for r in rows)),
        "tti_s_collision_anchors": {
            "n": len(tti),
            "min": float(min(tti)) if tti else None,
            "p10": float(np.quantile(tti, 0.10)) if tti else None,
            "p50": float(np.median(tti)) if tti else None,
            "p90": float(np.quantile(tti, 0.90)) if tti else None,
            "max": float(max(tti)) if tti else None,
            "hist_s": _hist(tti, [-1, 0, 1, 2, 3, 4, 6, 8, 12, 20]) if tti else {},
        },
        "fraction_anchors_collision_within": within,
        "pre_fork_anchor_frac": float(pre / n) if n else None,
        "note": (
            "No dedicated safe clips exist (collision vs miss_left/miss_right only). "
            "anchor_class 'safe' means a collision clip with TTI > 4 s. "
            "hit_s is 14.35 s (frame 430) on every clip; heatmap peaks ~0.8 s earlier."
        ),
    }


def a2(rows: list[dict], gt: dict[str, np.ndarray], warning: dict) -> dict:
    n = len(rows)
    event = np.array([event_alert(r["tti_s"], 4.0) for r in rows], dtype=bool)
    out = {"n": n, "event_positive_frac": float(event.mean()), "by_cutoff": {}}
    for cut in CUTOFFS:
        pred = np.zeros(n, dtype=bool)
        hm_side = []
        for i, r in enumerate(rows):
            maps = {
                "now": gt["gt_now"][i],
                "1s": gt["gt_1s"][i],
                "1_5s": gt["gt_1_5s"][i],
                "2s": gt["gt_2s"][i],
            }
            amb = {"1s": r["amb_1s"], "1_5s": r["amb_1_5s"], "2s": r["amb_2s"]}
            fired, side, _peak = heatmap_alert(maps, cutoff=cut, warning=warning, amb=amb)
            pred[i] = fired
            hm_side.append(side)
        disagree = float(np.mean(pred != event))
        # confusion of heatmap side vs true side, on event-positive anchors
        cm = {s: {t: 0 for t in SIDES} for s in SIDES}
        n_evt = 0
        n_match = 0
        for i, r in enumerate(rows):
            if not event[i]:
                continue
            n_evt += 1
            a, b = hm_side[i], r["true_side"]
            if a in cm and b in cm[a]:
                cm[a][b] += 1
            n_match += int(a == b)
        out["by_cutoff"][str(cut)] = {
            "heatmap_positive_frac": float(pred.mean()),
            "disagreement_rate": disagree,
            "heatmap_alert_precision": float(np.mean(event[pred])) if pred.any() else None,
            "heatmap_alert_recall": float(np.mean(pred[event])) if event.any() else None,
            "direction_accuracy_on_event": float(n_match / n_evt) if n_evt else None,
            "direction_confusion_heatmap_row_true_col": cm,
        }
    return out


def a3(rows: list[dict]) -> dict:
    """Best alert probability given identical RGB history = collision share in the outcome group."""
    groups = defaultdict(list)
    gid_scen = {}
    for r in rows:
        gid_scen[r["outcome_group_id"]] = r["scenario_type"]
        if not r["pre_fork"]:
            continue
        key = (r["outcome_group_id"], int(r["frame"]))
        groups[key].append(r["outcome"] == "collision")
    by_scen = defaultdict(list)
    all_p = []
    n_groups = 0
    for (gid, _t), flags in groups.items():
        if len(flags) < 2:
            continue
        p = float(np.mean(flags))
        all_p.append(p)
        n_groups += 1
        by_scen[gid_scen[gid]].append(p)
    per = {s: {"n": len(v), "mean_p_collision": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))} for s, v in sorted(by_scen.items())}
    return {
        "n_pre_fork_identity_groups": n_groups,
        "p_collision_overall": float(np.mean(all_p)) if all_p else None,
        "p_collision_unique_values": sorted(set(round(x, 4) for x in all_p)),
        "per_scenario_type": per,
        "recommendation": (
            "Pre-fork RGB is shared by one collision and one miss clip in every group, "
            "so the Bayes-optimal alert probability is 0.5. Use soft targets pre-fork, not hard 0/1, "
            "and down-weight those anchors if the loss is cross-entropy on a hard collision label."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default="reports/dense_dump")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.dump)
    meta = json.loads((root / "meta.json").read_text())
    rows, gt = load_dump(root)
    report = {
        "dump_meta": {
            "n_rows": meta.get("n_rows"),
            "n_clips": meta.get("n_clips"),
            "fps_unique": meta.get("fps_unique"),
            "stride_s": meta.get("stride_s"),
            "scenario_types": meta.get("scenario_types"),
            "anchors_per_clip": meta.get("n_anchors_per_clip"),
        },
        "A1_label_audit": a1(rows),
        "A2_heatmap_vs_event": a2(rows, gt, DEFAULT_WARNING),
        "A3_ambiguity_ceiling": a3(rows),
    }
    out = Path(args.out) if args.out else root / "label_tests.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
