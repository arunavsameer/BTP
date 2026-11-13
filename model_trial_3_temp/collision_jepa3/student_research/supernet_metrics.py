"""Metrics for temporal supernet (per-pattern eval + pair stats)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from collision_jepa3.evaluation.metrics import HorizonAccumulator, SAFE, STOP, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.student_research.temporal_supernet import TemporalSupernet, pack_pattern_clip


def _tier_metrics(acc: HorizonAccumulator) -> dict[str, float]:
    s = acc.summary()
    return {
        "stop_recall": s["stop_recall"],
        "safe_recall": s["safe_recall"],
        "threat_recall": s["high_threat_recall"],
        "mse": s["mse"],
        "pre_fork_stop_recall": s["pre_fork_stop_recall"],
        "post_fork_stop_recall": s["post_fork_stop_recall"],
    }


def eval_pattern(
    model: TemporalSupernet,
    loader,
    device: str,
    pattern: str,
    warning: dict,
) -> dict[str, Any]:
    model.eval()
    acc = HorizonAccumulator("1s", warning=warning)
    preds, gts, valids = [], [], []
    pair_peaks: dict[tuple, list[tuple[float, float, bool, bool]]] = defaultdict(list)
    stop_thr = float(warning.get("stop_threshold", 0.60))
    both_stop_ok = both_safe_ok = n_pairs = 0
    with torch.no_grad():
        for batch in loader:
            flat = batch["flat_frames"].to(device)
            gt = batch["gt_1s"].to(device)
            clip, mask, off = pack_pattern_clip(flat, pattern)
            out = model(clip, mask=mask, offsets_norm=off)
            h = out["Hhat_1s"].float().cpu().numpy()
            gt_np = gt.float().cpu().numpy()
            bsz = h.shape[0]
            pre = batch["pre_fork"]
            for i in range(bsz):
                acc.update(
                    h[i],
                    gt_np[i],
                    masked=False,
                    outcome=str(batch["outcome_type"][i]),
                    background="",
                    scenario="",
                    pre_fork=bool(pre[i]),
                )
                valid = np.ones_like(gt_np[i], dtype=bool)
                preds.append(h[i])
                gts.append(gt_np[i])
                valids.append(valid)
                pid = str(batch["background_pair_id"][i])
                t = int(batch["frame"][i])
                _, sev, _ = classify_heatmap(h[i], warning)
                _, gt_sev, _ = classify_heatmap(gt_np[i], warning)
                gt_stop = gt_sev >= STOP
                gt_safe = gt_sev == SAFE
                p_stop = sev >= STOP
                p_safe = sev < STOP
                if pid:
                    key = (pid, t)
                    pair_peaks[key].append((float(h[i].max()), gt_stop, p_stop, p_safe, gt_safe))
    if preds:
        op = operating_metrics(np.stack(preds), np.stack(gts), np.stack(valids), hi=0.5)
    else:
        op = {}
    gaps = []
    flips = 0
    for key, vals in pair_peaks.items():
        if len(vals) != 2:
            continue
        n_pairs += 1
        pa, pb = vals[0][0], vals[1][0]
        gaps.append(abs(pa - pb))
        if (pa >= stop_thr) != (pb >= stop_thr):
            flips += 1
        g0, g1 = vals[0][1], vals[1][1]
        if g0 and g1 and vals[0][2] and vals[1][2]:
            both_stop_ok += 1
        gs0, gs1 = vals[0][4], vals[1][4]
        if gs0 and gs1 and vals[0][3] and vals[1][3]:
            both_safe_ok += 1
    garr = np.asarray(gaps, dtype=np.float32)
    base = _tier_metrics(acc)
    base.update(
        {
            "recall_at_5fa": op.get("recall_at_5fa", float("nan")),
            "ap_high": op.get("ap_high", float("nan")),
            "pair_gap_mean": float(garr.mean()) if garr.size else float("nan"),
            "pair_gap_median": float(np.median(garr)) if garr.size else float("nan"),
            "pair_gap_p90": float(np.quantile(garr, 0.9)) if garr.size else float("nan"),
            "pair_warning_flip_rate": flips / max(n_pairs, 1),
            "both_background_correct_stop_rate": both_stop_ok / max(n_pairs, 1),
            "both_background_correct_safe_rate": both_safe_ok / max(n_pairs, 1),
            "n_pairs_evaluated": n_pairs,
            "n": int(acc.n),
            "pattern": pattern,
        }
    )
    return base
