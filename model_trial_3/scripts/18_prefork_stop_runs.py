#!/usr/bin/env python3
"""Is pre-fork STOP a one-frame spike or a sustained run in the heatmap GT?"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.heatmap_annotations import targets_for_anchor  # noqa: E402
from collision_jepa3.data.pairing import apply_mean_future_targets  # noqa: E402
from collision_jepa3.data.sampling import available_horizons  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402

STOP = 0.60
CAUTION = 0.25
H1S = 30


def band(peak: float) -> str:
    if peak >= STOP:
        return "STOP"
    if peak >= CAUTION:
        return "THREAT"
    return "SAFE"


def runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Inclusive (start, end, length) of True runs."""
    out: list[tuple[int, int, int]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j, j - i + 1))
        i = j + 1
    return out


def run_at(rs: list[tuple[int, int, int]], f: int) -> tuple[int, int, int] | None:
    for a, b, L in rs:
        if a <= f <= b:
            return (a, b, L)
    return None


def hist_bins(lengths: list[int]) -> dict[str, int]:
    bins = {
        "1_frame": 0,
        "2_to_5": 0,
        "6_to_15 (~0.2-0.5s)": 0,
        "16_to_30 (~0.5-1s)": 0,
        "31_to_60 (~1-2s)": 0,
        "61_plus (>2s)": 0,
    }
    for L in lengths:
        if L <= 1:
            bins["1_frame"] += 1
        elif L <= 5:
            bins["2_to_5"] += 1
        elif L <= 15:
            bins["6_to_15 (~0.2-0.5s)"] += 1
        elif L <= 30:
            bins["16_to_30 (~0.5-1s)"] += 1
        elif L <= 60:
            bins["31_to_60 (~1-2s)"] += 1
        else:
            bins["61_plus (>2s)"] += 1
    return bins


def gt_h1s(ds: TeacherDataset, clip_id: str, t: int) -> tuple[np.ndarray, bool]:
    rec = ds.records_by_id[clip_id]
    tg = targets_for_anchor(ds._heatmaps[clip_id], t, ds.horizons, allow_missing=ds.allow_partial_horizons)
    amb = ds.ambiguity.get((clip_id, t), {n: False for n in ds.horizon_names})
    avail = available_horizons(t, ds.n_frames, ds.horizons)
    replaced: set[str] = set()
    if ds.ambiguous_policy == "mean_target":
        tg, replaced = apply_mean_future_targets(
            tg,
            rec=rec,
            anchor=t,
            amb=amb,
            clip_heatmaps=ds._heatmaps,
            outcome_ids=ds._outcome_ids,
            horizons=ds.horizons,
            horizon_names=ds.horizon_names,
            available=avail,
        )
    return np.asarray(tg["h_1s"], dtype=np.float32), bool("1s" in replaced)


def main() -> None:
    args = add_config_args(__import__("argparse").ArgumentParser(description=__doc__)).parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=False)
    fork_default = int(cfg.get("sampling.fork_frame", 365))

    sampled = []
    dense_run_lengths_pre = []
    dense_run_lengths_all = []
    clips_with_prefork_occ_stop = 0
    clips_prefork_occ_stop_isolated_only = 0

    for rec in ds.records:
        hm = ds._heatmaps[rec.clip_id]
        peak = hm.reshape(hm.shape[0], -1).max(axis=1)
        fork = int(rec.fork_frame or fork_default)
        occ_stop = peak >= STOP
        rs_all = runs(occ_stop)
        rs_pre = runs(occ_stop[:fork])
        dense_run_lengths_all.extend([L for _, _, L in rs_all])
        dense_run_lengths_pre.extend([L for _, _, L in rs_pre])
        if rs_pre:
            clips_with_prefork_occ_stop += 1
            if all(L <= 2 for _, _, L in rs_pre):
                clips_prefork_occ_stop_isolated_only += 1

        for t in ds.anchors_by_clip[rec.clip_id]:
            if int(t) >= fork:
                continue
            h1, replaced = gt_h1s(ds, rec.clip_id, int(t))
            p1 = float(h1.max())
            if p1 < STOP:
                continue
            fut = int(t) + H1S
            raw_peak_fut = float(peak[fut]) if 0 <= fut < peak.size else float("nan")
            raw_peak_now = float(peak[int(t)])
            r = run_at(rs_all, fut) if 0 <= fut < peak.size else None
            r_pre = run_at(rs_pre, fut) if fut < fork else None
            # Neighbor persistence on occupancy at the 1s target frame.
            def neigh(w: int) -> float:
                if fut < 0 or fut >= peak.size:
                    return float("nan")
                a = max(0, fut - w)
                b = min(peak.size, fut + w + 1)
                return float((peak[a:b] >= STOP).mean())

            sampled.append(
                {
                    "clip_id": rec.clip_id,
                    "outcome": rec.outcome_type,
                    "scenario": rec.scenario_type,
                    "anchor": int(t),
                    "future": fut,
                    "fork": fork,
                    "future_crossed_fork": bool(fut >= fork),
                    "seconds_before_fork": (fork - int(t)) / 30.0,
                    "gt_peak_1s": p1,
                    "raw_peak_now": raw_peak_now,
                    "raw_peak_1s": raw_peak_fut,
                    "now_band": band(raw_peak_now),
                    "mean_replaced": replaced,
                    "run_len_frames": None if r is None else int(r[2]),
                    "run_start": None if r is None else int(r[0]),
                    "run_end": None if r is None else int(r[1]),
                    "run_len_pre_fork_frames": None if r_pre is None else int(r_pre[2]),
                    "frac_stop_pm1": neigh(1),
                    "frac_stop_pm5": neigh(5),
                    "frac_stop_pm15": neigh(15),
                }
            )

    n = len(sampled)
    crossed = [s for s in sampled if s["future_crossed_fork"]]
    still = [s for s in sampled if not s["future_crossed_fork"]]
    isolated = [s for s in sampled if (s["run_len_frames"] or 0) <= 2]
    short = [s for s in sampled if 1 <= (s["run_len_frames"] or 0) <= 15]
    sustained = [s for s in sampled if (s["run_len_frames"] or 0) >= 16]
    long = [s for s in sampled if (s["run_len_frames"] or 0) >= 31]
    flicker_win = [s for s in sampled if (s["frac_stop_pm5"] or 0) < 0.5]
    now_safe = [s for s in sampled if s["now_band"] == "SAFE"]
    now_threat = [s for s in sampled if s["now_band"] == "THREAT"]
    now_stop = [s for s in sampled if s["now_band"] == "STOP"]
    replaced = [s for s in sampled if s["mean_replaced"]]
    by_outcome = Counter(s["outcome"] for s in sampled)

    def pct(xs):
        return None if n == 0 else round(100.0 * len(xs) / n, 1)

    report = {
        "definition": {
            "STOP": "max(H) >= 0.60",
            "H_1s_frame": "anchor + 30",
            "pre_fork_window": "anchor t < fork (365); the 1s label can still be post-fork",
            "run": "contiguous frames where occupancy peak >= 0.60, on the clip's own heatmap",
        },
        "n_val_clips": len(ds.records),
        "n_prefork_sampled_STOP_windows": n,
        "n_expected_from_cm": 504,
        "sampled_STOP": {
            "future_still_pre_fork": {"n": len(still), "pct": pct(still)},
            "future_crossed_fork": {"n": len(crossed), "pct": pct(crossed)},
            "mean_target_replaced": {"n": len(replaced), "pct": pct(replaced)},
            "run_len_1_2_frames": {"n": len(isolated), "pct": pct(isolated)},
            "run_len_le_0.5s": {"n": len(short), "pct": pct(short)},
            "run_len_ge_0.5s": {"n": len(sustained), "pct": pct(sustained)},
            "run_len_ge_1s": {"n": len(long), "pct": pct(long)},
            "not_mostly_STOP_in_pm5": {"n": len(flicker_win), "pct": pct(flicker_win)},
            "now_SAFE_future_STOP": {"n": len(now_safe), "pct": pct(now_safe)},
            "now_THREAT_future_STOP": {"n": len(now_threat), "pct": pct(now_threat)},
            "now_already_STOP": {"n": len(now_stop), "pct": pct(now_stop)},
            "by_outcome": dict(by_outcome),
            "run_len_histogram_frames": hist_bins([int(s["run_len_frames"] or 0) for s in sampled]),
            "median_run_len_frames": float(np.median([s["run_len_frames"] or 0 for s in sampled])) if sampled else None,
            "median_seconds_before_fork": float(np.median([s["seconds_before_fork"] for s in sampled])) if sampled else None,
            "seconds_before_fork": {
                "lt_1s": sum(s["seconds_before_fork"] < 1.0 for s in sampled),
                "1_to_2s": sum(1.0 <= s["seconds_before_fork"] < 2.0 for s in sampled),
                "2_to_4s": sum(2.0 <= s["seconds_before_fork"] < 4.0 for s in sampled),
                "ge_4s": sum(s["seconds_before_fork"] >= 4.0 for s in sampled),
            },
            "median_gt_peak": float(np.median([s["gt_peak_1s"] for s in sampled])) if sampled else None,
        },
        "still_pre_fork_future_only": {
            "n": len(still),
            "run_len_histogram_frames": hist_bins([int(s["run_len_frames"] or 0) for s in still]),
            "run_len_pre_fork_only_histogram": hist_bins(
                [int(s["run_len_pre_fork_frames"] or 0) for s in still]
            ),
            "median_run_len_frames": float(np.median([s["run_len_frames"] or 0 for s in still])) if still else None,
            "median_run_len_pre_fork_frames": (
                float(np.median([s["run_len_pre_fork_frames"] or 0 for s in still])) if still else None
            ),
            "now_SAFE": sum(s["now_band"] == "SAFE" for s in still),
            "now_THREAT": sum(s["now_band"] == "THREAT" for s in still),
            "now_STOP": sum(s["now_band"] == "STOP" for s in still),
            "seconds_before_fork": {
                "lt_1s": sum(s["seconds_before_fork"] < 1.0 for s in still),
                "1_to_2s": sum(1.0 <= s["seconds_before_fork"] < 2.0 for s in still),
                "2_to_4s": sum(2.0 <= s["seconds_before_fork"] < 4.0 for s in still),
                "ge_4s": sum(s["seconds_before_fork"] >= 4.0 for s in still),
            },
        },
        "crossed_fork_future_only": {
            "n": len(crossed),
            "run_len_histogram_frames": hist_bins([int(s["run_len_frames"] or 0) for s in crossed]),
            "median_run_len_frames": float(np.median([s["run_len_frames"] or 0 for s in crossed])) if crossed else None,
            "now_SAFE": sum(s["now_band"] == "SAFE" for s in crossed),
            "now_THREAT": sum(s["now_band"] == "THREAT" for s in crossed),
            "now_STOP": sum(s["now_band"] == "STOP" for s in crossed),
        },
        "dense_occupancy_STOP_runs": {
            "pre_fork_occupancy_runs": {
                "n_runs": len(dense_run_lengths_pre),
                "histogram": hist_bins(dense_run_lengths_pre),
                "median_len": float(np.median(dense_run_lengths_pre)) if dense_run_lengths_pre else None,
                "clips_with_any": clips_with_prefork_occ_stop,
                "clips_only_1_2_frame_runs": clips_prefork_occ_stop_isolated_only,
            },
            "all_time_occupancy_runs": {
                "n_runs": len(dense_run_lengths_all),
                "histogram": hist_bins(dense_run_lengths_all),
                "median_len": float(np.median(dense_run_lengths_all)) if dense_run_lengths_all else None,
            },
        },
    }
    out = Path("reports/direct_h1s_appear_blob_gr40b/prefork_stop_runs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
