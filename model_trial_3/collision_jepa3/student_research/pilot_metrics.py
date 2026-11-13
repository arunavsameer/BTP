"""Warning and operating metrics on pilot loaders."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from collision_jepa3.evaluation.metrics import HorizonAccumulator
from collision_jepa3.evaluation.operating_point import operating_metrics


def _pair_robustness(batch_meta: list[dict], peaks: np.ndarray, stop_thr: float) -> dict[str, float]:
    """Peak-gap stats for background pairs at matched frames (within eval batch aggregate)."""
    by_key: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for i, m in enumerate(batch_meta):
        pid = str(m.get("background_pair_id") or "")
        if not pid:
            continue
        by_key[(pid, int(m["frame"]))].append((float(peaks[i]), i))
    gaps, flips, n_pairs = [], 0, 0
    for vals in by_key.values():
        if len(vals) != 2:
            continue
        pa, pb = vals[0][0], vals[1][0]
        gaps.append(abs(pa - pb))
        if (pa >= stop_thr) != (pb >= stop_thr):
            flips += 1
        n_pairs += 1
    arr = np.asarray(gaps, dtype=np.float32)
    return {
        "pair_gap_mean": float(arr.mean()) if arr.size else float("nan"),
        "pair_gap_median": float(np.median(arr)) if arr.size else float("nan"),
        "pair_warning_flip_rate": flips / max(n_pairs, 1),
        "n_pairs_evaluated": n_pairs,
    }


def accumulate_pilot_metrics(model, loader, device: str, warning: dict, parent_key: str = "H_teacher") -> dict[str, Any]:
    model.eval()
    acc = HorizonAccumulator("1s", warning=warning)
    preds, gts, valids = [], [], []
    batch_meta: list[dict] = []
    peaks_list: list[float] = []
    parent_acc = HorizonAccumulator("1s", warning=warning)
    stop_thr = float(warning.get("stop_threshold", 0.60))
    with torch.no_grad():
        for batch in loader:
            batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch_t["clip"])
            h = out["Hhat_1s"].float().cpu().numpy()
            hp = batch_t[parent_key].float().cpu().numpy()
            gt = batch_t["gt_1s"].float().cpu().numpy()
            bsz = h.shape[0]
            pre = batch_t.get("pre_fork")
            if torch.is_tensor(pre):
                pre_np = pre.detach().cpu().numpy().astype(bool)
            else:
                pre_np = np.asarray(pre, dtype=bool)
            outcomes = batch_t.get("outcome_type", ["collision"] * bsz)
            if not isinstance(outcomes, list):
                outcomes = [str(outcomes)] * bsz
            for i in range(bsz):
                valid = np.ones_like(gt[i], dtype=bool)
                acc.update(
                    h[i],
                    gt[i],
                    masked=False,
                    outcome=str(outcomes[i] if i < len(outcomes) else "collision"),
                    background="",
                    scenario=str(batch_t.get("scenario_type", [""] * bsz)[i] if isinstance(batch_t.get("scenario_type"), list) else ""),
                    pre_fork=bool(pre_np[i]) if pre_np is not None else False,
                )
                parent_acc.update(
                    hp[i],
                    gt[i],
                    masked=False,
                    outcome=str(outcomes[i] if i < len(outcomes) else "collision"),
                    background="",
                    scenario="",
                    pre_fork=bool(pre_np[i]) if pre_np is not None else False,
                )
                preds.append(h[i])
                gts.append(gt[i])
                valids.append(valid)
                peaks_list.append(float(h[i].max()))
                batch_meta.append(
                    {
                        "background_pair_id": batch_t.get("background_pair_id", [""] * bsz)[i]
                        if isinstance(batch_t.get("background_pair_id"), list)
                        else "",
                        "frame": int(batch_t["frame"][i]) if torch.is_tensor(batch_t["frame"]) else int(batch_t["frame"][i]),
                    }
                )
    if preds:
        pstack = np.stack(preds, axis=0)
        gstack = np.stack(gts, axis=0)
        vstack = np.stack(valids, axis=0)
        op = operating_metrics(pstack, gstack, vstack, hi=0.5)
    else:
        op = {}
    s = acc.summary()
    ps = parent_acc.summary()
    pair = _pair_robustness(batch_meta, np.asarray(peaks_list), stop_thr)
    return {
        "stop_recall": s["stop_recall"],
        "safe_recall": s["safe_recall"],
        "mse_gt": s["mse"],
        "mse_teacher": ps["mse"],
        "recall_at_5fa": op.get("recall_at_5fa", float("nan")),
        "ap_high": op.get("ap_high", float("nan")),
        "pre_fork_stop_recall": s["pre_fork_stop_recall"],
        "post_fork_stop_recall": s["post_fork_stop_recall"],
        "pre_fork_mse": s["pre_fork_mse"],
        "post_fork_mse": s["post_fork_mse"],
        "parent_stop_recall": ps["stop_recall"],
        "parent_safe_recall": ps["safe_recall"],
        "parent_pre_fork_stop_recall": ps["pre_fork_stop_recall"],
        "parent_post_fork_stop_recall": ps["post_fork_stop_recall"],
        "n": int(s["n"]),
        **pair,
    }
