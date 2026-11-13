"""Development evaluation for F_dec pilot."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, HorizonAccumulator, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics


def _tier(hm: np.ndarray, warning: dict) -> int:
    _, s, _ = classify_heatmap(hm, warning)
    return int(s)


@torch.no_grad()
def evaluate_fdec_pilot(
    model,
    loader,
    device: str,
    warning: dict,
    *,
    verified_pair_keys: list[tuple],
    variant: str,
) -> dict[str, Any]:
    model.eval()
    acc = HorizonAccumulator("1s", warning=warning)
    parent_acc = HorizonAccumulator("1s", warning=warning)
    preds, gts, valids = [], [], []
    meta_rows: list[dict] = []
    peaks: list[float] = []
    parent_peaks: list[float] = []
    hm_mae_parent: list[float] = []
    hm_mse_parent: list[float] = []
    tier_agree = stop_agree = safe_agree = 0
    n_tier = 0
    latent_mse: list[float] = []
    latent_cos: list[float] = []
    verified = {tuple(k) for k in verified_pair_keys}
    stop_thr = float(warning.get("stop_threshold", 0.60))

    for batch in loader:
        batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch_t["clip"])
        h = out["Hhat_1s"].float()
        gt = batch_t["gt_1s"]
        hp = batch_t["H_teacher"]
        bsz = h.shape[0]
        h_np = h.cpu().numpy()
        gt_np = gt.cpu().numpy()
        hp_np = hp.float().cpu().numpy()
        pre = batch_t.get("pre_fork", [False] * bsz)
        if torch.is_tensor(pre):
            pre_np = pre.detach().cpu().bool().cpu().numpy()
        else:
            pre_np = np.asarray(pre, dtype=bool)

        if variant in ("S1", "S2") and "F_student" in out:
            f_s = out["F_student"].float()
            f_t = batch_t["F_teacher"].float()
            latent_mse.append(float(F.mse_loss(f_s, f_t).item()))
            latent_cos.append(float(F.cosine_similarity(f_s.flatten(1), f_t.flatten(1)).mean().item()))

        for i in range(bsz):
            acc.update(
                h_np[i],
                gt_np[i],
                masked=False,
                outcome=str(batch_t.get("outcome_type", [""] * bsz)[i]),
                background="",
                scenario=str(batch_t.get("scenario_type", [""] * bsz)[i]),
                pre_fork=bool(pre_np[i]) if i < len(pre_np) else False,
            )
            parent_acc.update(
                hp_np[i],
                gt_np[i],
                masked=False,
                outcome="",
                background="",
                scenario="",
                pre_fork=bool(pre_np[i]) if i < len(pre_np) else False,
            )
            preds.append(h_np[i])
            gts.append(gt_np[i])
            valids.append(np.ones_like(gt_np[i], dtype=bool))
            peaks.append(float(h_np[i].max()))
            parent_peaks.append(float(hp_np[i].max()))
            hm_mae_parent.append(float(np.mean(np.abs(h_np[i] - hp_np[i]))))
            hm_mse_parent.append(float(np.mean((h_np[i] - hp_np[i]) ** 2)))
            ps, gs = _tier(h_np[i], warning), _tier(gt_np[i], warning)
            pt, gt_p = _tier(hp_np[i], warning), _tier(gt_np[i], warning)
            if ps == pt:
                tier_agree += 1
            if (ps >= STOP) == (pt >= STOP):
                stop_agree += 1
            if (ps == SAFE) == (pt == SAFE):
                safe_agree += 1
            n_tier += 1
            pid = batch_t.get("background_pair_id", [""] * bsz)
            pid_s = str(pid[i] if isinstance(pid, list) else "")
            fr = batch_t.get("frame")
            frame_i = int(fr[i]) if torch.is_tensor(fr) else int(fr[i])
            meta_rows.append({"background_pair_id": pid_s, "frame": frame_i, "peak": peaks[-1]})

    s = acc.summary()
    op = operating_metrics(np.stack(preds), np.stack(gts), np.stack(valids), hi=0.5) if preds else {}
    pair_stats = _pair_metrics(meta_rows, verified, stop_thr)
    out: dict[str, Any] = {
        "stop_recall": s["stop_recall"],
        "safe_recall": s["safe_recall"],
        "threat_recall": s["high_threat_recall"],
        "mse_gt": s["mse"],
        "recall_at_5fa": op.get("recall_at_5fa", float("nan")),
        "ap_high": op.get("ap_high", float("nan")),
        "pre_fork_stop_recall": s["pre_fork_stop_recall"],
        "post_fork_stop_recall": s["post_fork_stop_recall"],
        "teacher_hm_mae_mean": float(np.mean(hm_mae_parent)),
        "teacher_hm_mse_mean": float(np.mean(hm_mse_parent)),
        "teacher_tier_agreement": tier_agree / max(n_tier, 1),
        "teacher_stop_agreement": stop_agree / max(n_tier, 1),
        "teacher_safe_agreement": safe_agree / max(n_tier, 1),
        "n": int(s["n"]),
        **pair_stats,
    }
    if latent_mse:
        out["latent_mse_mean"] = float(np.mean(latent_mse))
        out["latent_cosine_mean"] = float(np.mean(latent_cos))
    return out


def _pair_metrics(meta_rows: list[dict], verified: set[tuple], stop_thr: float) -> dict[str, float]:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for m in meta_rows:
        pid = m.get("background_pair_id") or ""
        if not pid:
            continue
        key = (pid, int(m["frame"]))
        by_key[key].append(m)
    gaps, flips, n_pairs = [], 0, 0
    both_stop = both_safe = 0
    for key, vals in by_key.items():
        if len(vals) != 2:
            continue
        if key not in verified:
            continue
        pa, pb = float(vals[0]["peak"]), float(vals[1]["peak"])
        gaps.append(abs(pa - pb))
        if (pa >= stop_thr) != (pb >= stop_thr):
            flips += 1
        n_pairs += 1
    arr = np.asarray(gaps, dtype=np.float32)
    n_bg = sum(1 for m in meta_rows if m.get("background_pair_id"))
    n_keys_with_two = sum(1 for vals in by_key.values() if len(vals) == 2)
    return {
        "pair_gap_mean": float(arr.mean()) if arr.size else float("nan"),
        "pair_gap_median": float(np.median(arr)) if arr.size else float("nan"),
        "pair_gap_p90": float(np.quantile(arr, 0.9)) if arr.size else float("nan"),
        "pair_warning_flip_rate": (flips / n_pairs) if n_pairs else float("nan"),
        "n_pairs_verified_evaluated": n_pairs,
        "n_rows_with_background_pair_id": n_bg,
        "n_pair_anchor_keys_in_split": len(by_key),
        "n_pair_anchor_keys_complete": n_keys_with_two,
        "n_verified_keys_supplied": len(verified),
        "pair_metrics_unavailable_reason": (
            "no verified pair keys matched this split (often train keys used on dev)"
            if n_pairs == 0 and n_keys_with_two > 0 and len(verified) > 0
            else ("no complete pairs in split" if n_pairs == 0 and n_keys_with_two == 0 else None)
        ),
    }
