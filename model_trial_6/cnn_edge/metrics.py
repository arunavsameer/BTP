"""SAFE/STOP + F_dec cosine. Official rank: SAFE≥0.85 then STOP."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

STOP_THR = 0.60
CAUTION_THR = 0.25
SAFE, CAUTION, STOP = 0, 1, 2


def peak_class(h: np.ndarray) -> int:
    p = float(np.max(h))
    if p >= STOP_THR:
        return STOP
    if p >= CAUTION_THR:
        return CAUTION
    return SAFE


def metrics_from_rows(rows: list[dict], h_key: str) -> dict:
    cm = np.zeros((3, 3), dtype=np.int64)
    parent_mae, gt_mse, stop_peaks = [], [], []
    for r in rows:
        hp = np.asarray(r[h_key], dtype=np.float32)
        hg = np.asarray(r["gt_1s"], dtype=np.float32)
        ht = np.asarray(r["H_teacher"], dtype=np.float32)
        gt_c, pr_c = peak_class(hg), peak_class(hp)
        cm[gt_c, pr_c] += 1
        parent_mae.append(float(np.mean(np.abs(hp - ht))))
        gt_mse.append(float(np.mean((hp - hg) ** 2)))
        if gt_c == STOP:
            stop_peaks.append(float(np.max(hp)))

    def rec(g):
        d = cm[g].sum()
        return float(cm[g, g] / d) if d else float("nan")

    stop_col = cm[:, STOP].sum()
    sa, st = rec(SAFE), rec(STOP)
    peaks = np.asarray(stop_peaks) if stop_peaks else np.zeros((0,), dtype=np.float32)
    return {
        "confusion_matrix": cm.tolist(),
        "stop_recall": st,
        "safe_recall": sa,
        "stop_precision": float(cm[STOP, STOP] / stop_col) if stop_col else float("nan"),
        "gt_heatmap_mse": float(np.mean(gt_mse)) if gt_mse else float("nan"),
        "parent_heatmap_mae": float(np.mean(parent_mae)) if parent_mae else float("nan"),
        "collapsed_stop_frac": float(np.mean(peaks < 0.25)) if peaks.size else float("nan"),
        "n": len(rows),
    }


def rank_checkpoint(m: dict) -> tuple:
    sa = float(m.get("safe_recall") or 0)
    return (
        1 if sa + 1e-12 >= 0.85 else 0,
        float(m.get("stop_recall") or -1),
        float(m.get("stop_precision") or -1),
        float(m.get("fdec_cosine") or -1),
    )


@torch.no_grad()
def collect(model, loader, device: str, amp_dtype=None) -> dict:
    model.eval()
    rows, cos, l1 = [], [], []
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if str(device).startswith("cuda") and amp_dtype is not None
        else nullcontext()
    )
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        fs, ft = out["F"].float(), b["F_teacher"].float()
        cos.append(F.cosine_similarity(fs.flatten(1), ft.flatten(1)).cpu().numpy())
        l1.append((fs - ft).abs().mean(dim=(1, 2, 3)).cpu().numpy())
        pref = b["pre_fork"].cpu().numpy()
        for i in range(h.shape[0]):
            rows.append(
                {
                    "H_student": h[i],
                    "H_teacher": ht[i],
                    "gt_1s": gt[i],
                    "pre_fork": bool(pref[i]),
                    "sample_id": b["sample_id"][i],
                    "frame": int(b["frame"][i]),
                    "background_id": b["background_id"][i],
                    "clip_id": b["clip_id"][i],
                }
            )
    m = metrics_from_rows(rows, "H_student")
    m["fdec_cosine"] = float(np.concatenate(cos).mean()) if cos else float("nan")
    m["fdec_l1"] = float(np.concatenate(l1).mean()) if l1 else float("nan")
    return m
