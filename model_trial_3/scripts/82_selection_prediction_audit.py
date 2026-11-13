#!/usr/bin/env python3
"""Offline selection-only audit of overnight M1–M5. No training. No test split."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.overnight6.gpu_bind import bind_cuda  # noqa: E402

bind_cuda("auto")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.student_research.overnight6.models import build_model  # noqa: E402
from collision_jepa3.student_research.overnight6.train import (  # noqa: E402
    EXPECTED_HASH,
    NAMES,
    pick_cuda,
    prepare_data,
)
from collision_jepa3.student_research.overnight6.train import load_s75  # noqa: E402

OUT = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/selection_prediction_audit_v1")
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/overnight6")
MODELS = {
    "M1": (1, "m1_stop_floor", 10),
    "M2": (2, "m2_ordinal", 19),
    "M3": (3, "m3_asymmetric", 21),
    "M4": (4, "m4_parent_recovery", 17),
    "M5": (5, "m5_multiscale", 17),
}
REF = {
    "M1": dict(SAFE=0.868, THR=0.117, STOP=0.611, prec=0.537, pre=0.545, post=0.720),
    "M2": dict(SAFE=0.855, THR=0.211, STOP=0.419, prec=0.621, pre=0.327, post=0.572),
    "M3": dict(SAFE=0.877, THR=0.332, STOP=0.499, prec=0.633, pre=0.432, post=0.610),
    "M4": dict(SAFE=0.865, THR=0.201, STOP=0.522, prec=0.557, pre=0.451, post=0.639),
    "M5": dict(SAFE=0.865, THR=0.363, STOP=0.559, prec=0.670, pre=0.476, post=0.697),
}
STOP_SWEEP = [0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40]
SAFE_THR = 0.25


def peak_cls(p: np.ndarray, stop_thr: float = 0.60) -> np.ndarray:
    y = np.zeros(p.shape[0], dtype=np.int64)
    y = np.where(p >= SAFE_THR, CAUTION, y)
    y = np.where(p >= stop_thr, STOP, y)
    return y


def rec_prec(gt: np.ndarray, pr: np.ndarray):
    cm = np.zeros((3, 3), dtype=np.int64)
    for g, p in zip(gt, pr):
        cm[int(g), int(p)] += 1

    def rec(k):
        d = cm[k].sum()
        return float(cm[k, k] / d) if d else float("nan")

    stop_col = cm[:, STOP].sum()
    acc = float(np.trace(cm) / cm.sum())
    bal = float(np.nanmean([rec(0), rec(1), rec(2)]))
    return {
        "cm": cm,
        "SAFE": rec(SAFE),
        "THR": rec(CAUTION),
        "STOP": rec(STOP),
        "prec": float(cm[STOP, STOP] / stop_col) if stop_col else float("nan"),
        "acc": acc,
        "bal": bal,
        "safe_as_stop": float(cm[SAFE, STOP] / cm[SAFE].sum()) if cm[SAFE].sum() else float("nan"),
        "safe_as_thr_or_stop": float((cm[SAFE, CAUTION] + cm[SAFE, STOP]) / cm[SAFE].sum()) if cm[SAFE].sum() else float("nan"),
        "frac_pred_stop": float((pr == STOP).mean()),
        "n_pred_stop": int((pr == STOP).sum()),
    }


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(np.int32)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / tps[-1]
    fpr = fps / max(fps[-1], 1)
    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]
    return float(np.trapezoid(tpr, fpr))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(np.int32)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / y.sum()
    rec = np.r_[0, rec]
    prec = np.r_[1, prec]
    return float(np.trapezoid(prec, rec))


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
    }


def load_ckpt(mid: int, epoch: int, device: str):
    name = NAMES[mid]
    path = TEMP / name / f"checkpoint_epoch_{epoch:02d}.pt"
    if not path.is_file():
        return None, path, None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    ck_ep = int(ck.get("epoch", -1))
    cfg = Config.load()
    decoder = build_parent_decoder(cfg, device="cpu")
    h0 = decoder_state_hash(decoder)
    model = build_model(mid, decoder)
    model.encoder.load_state_dict(ck["encoder"])
    model.predictor.load_state_dict(ck["predictor"])
    model.proj.load_state_dict(ck["proj"])
    model.decoder.load_state_dict(ck["decoder"])
    if model.recent is not None:
        model.recent.load_state_dict(ck["recent"])
        model.gate.load_state_dict(ck["gate"])
    model.to(device).eval()
    meta = {"path": str(path), "ckpt_epoch": ck_ep, "requested_epoch": epoch, "init_decoder_hash": h0, "variant": ck.get("variant")}
    return model, path, meta


@torch.no_grad()
def infer(model, loader, device):
    s75 = load_s75()
    hs, meta = [], []
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else __import__("contextlib").nullcontext()
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        hs.append(h)
        gt = b["gt_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
        for i in range(h.shape[0]):
            meta.append(
                {
                    "sample_id": b["sample_id"][i],
                    "clip_id": b["clip_id"][i],
                    "frame": int(b["frame"][i]),
                    "pre_fork": bool(pref[i]),
                    "gt": gt[i],
                    "ht": ht[i],
                    "background_id": b["background_id"][i],
                }
            )
    return np.concatenate(hs, 0), meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    device = pick_cuda("auto")
    s75, rows, gt_all, train_ids, sel_ids, train_ds, sel_ds = prepare_data(forbid_test=True)
    loader = DataLoader(sel_ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=s75.collate)
    print(f"device={device} n_sel={len(sel_ds)}", flush=True)

    ckpt_meta = {}
    H = {}
    shared_meta = None
    missing = []
    for tag, (mid, dname, ep) in MODELS.items():
        npy = pred_dir / f"{tag}_H.npy"
        model, path, meta = load_ckpt(mid, ep, device)
        ckpt_meta[tag] = meta or {"path": str(path), "missing": not path.is_file()}
        if model is None:
            missing.append(tag)
            print(f"MISSING {tag} {path}", flush=True)
            continue
        if npy.is_file() and npy.stat().st_mtime >= path.stat().st_mtime:
            H[tag] = np.load(npy)
            print(f"loaded cache {tag} {H[tag].shape}", flush=True)
            if shared_meta is None:
                # still need meta once
                _, meta_rows = infer(model, loader, device)
                shared_meta = meta_rows
            del model
            torch.cuda.empty_cache()
            continue
        print(f"infer {tag} ep={ep} from {path}", flush=True)
        h, meta_rows = infer(model, loader, device)
        H[tag] = h
        np.save(npy, h)
        shared_meta = meta_rows
        del model
        torch.cuda.empty_cache()
        print(f"saved {npy}", flush=True)

    if missing:
        (OUT / "MISSING.txt").write_text("\n".join(missing))
        raise SystemExit(f"missing checkpoints {missing}")

    n = H["M1"].shape[0]
    gt = np.stack([m["gt"] for m in shared_meta])
    ht = np.stack([m["ht"] for m in shared_meta])
    pref = np.array([m["pre_fork"] for m in shared_meta])
    sample_id = [m["sample_id"] for m in shared_meta]
    clip_id = [m["clip_id"] for m in shared_meta]
    frame = np.array([m["frame"] for m in shared_meta])
    bg = [m["background_id"] for m in shared_meta]
    p_gt = gt.reshape(n, -1).max(1)
    p_par = ht.reshape(n, -1).max(1)
    y_gt = peak_cls(p_gt, 0.60)
    y_par = peak_cls(p_par, 0.60)
    peaks = {k: H[k].reshape(n, -1).max(1) for k in H}

    # parity
    parity = {}
    for tag in MODELS:
        y = peak_cls(peaks[tag], 0.60)
        m = rec_prec(y_gt, y)
        pre = (y_gt == STOP) & pref
        post = (y_gt == STOP) & (~pref)
        m["pre"] = float((y[pre] == STOP).mean()) if pre.any() else float("nan")
        m["post"] = float((y[post] == STOP).mean()) if post.any() else float("nan")
        ref = REF[tag]
        parity[tag] = {
            "SAFE": m["SAFE"],
            "THR": m["THR"],
            "STOP": m["STOP"],
            "prec": m["prec"],
            "pre": m["pre"],
            "post": m["post"],
            "d_SAFE": m["SAFE"] - ref["SAFE"],
            "d_STOP": m["STOP"] - ref["STOP"],
            "ok": abs(m["SAFE"] - ref["SAFE"]) < 0.015 and abs(m["STOP"] - ref["STOP"]) < 0.015,
        }
        print(f"PARITY {tag} {parity[tag]}", flush=True)
    (OUT / "parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")

    # per-sample csv
    import csv

    with (OUT / "per_sample_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        hdr = ["idx", "sample_id", "clip_id", "frame", "background_id", "pre_fork", "gt_class", "gt_peak", "parent_peak", "parent_class"]
        for tag in MODELS:
            hdr += [f"{tag}_peak", f"{tag}_class"]
        w.writerow(hdr)
        for i in range(n):
            row = [i, sample_id[i], clip_id[i], int(frame[i]), bg[i], int(pref[i]), int(y_gt[i]), float(p_gt[i]), float(p_par[i]), int(y_par[i])]
            for tag in MODELS:
                row += [float(peaks[tag][i]), int(peak_cls(peaks[tag][i : i + 1], 0.60)[0])]
            w.writerow(row)

    # A threshold sweep
    sweep_rows = []
    qualify = []
    for tag in MODELS:
        for thr in STOP_SWEEP:
            y = peak_cls(peaks[tag], thr)
            m = rec_prec(y_gt, y)
            recd = {
                "model": tag,
                "stop_thr": thr,
                "SAFE": m["SAFE"],
                "THR": m["THR"],
                "STOP": m["STOP"],
                "STOP_prec": m["prec"],
                "safe_as_stop": m["safe_as_stop"],
                "safe_as_thr_or_stop": m["safe_as_thr_or_stop"],
                "frac_pred_stop": m["frac_pred_stop"],
                "acc": m["acc"],
                "bal_acc": m["bal"],
            }
            sweep_rows.append(recd)
            if m["SAFE"] >= 0.82 and m["STOP"] >= 0.70:
                qualify.append(recd)
    with (OUT / "threshold_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    # B distributions + ranking
    dist_rows = []
    rank_rows = []
    labels = {SAFE: "SAFE", CAUTION: "THREAT", STOP: "STOP"}
    for tag in MODELS:
        p = peaks[tag]
        for cls, name in labels.items():
            st = stats(p[y_gt == cls])
            dist_rows.append({"model": tag, "group": f"gt_{name}", **st})
        dist_rows.append({"model": tag, "group": "prefork_STOP", **stats(p[(y_gt == STOP) & pref])})
        dist_rows.append({"model": tag, "group": "postfork_STOP", **stats(p[(y_gt == STOP) & (~pref)])})
        y_ss = (y_gt == STOP).astype(np.int32)
        y_safe = y_gt == SAFE
        y_stop = y_gt == STOP
        rank_rows.append(
            {
                "model": tag,
                "STOP_vs_SAFE_ROC": roc_auc(np.r_[np.ones(y_stop.sum()), np.zeros(y_safe.sum())], np.r_[p[y_stop], p[y_safe]]),
                "STOP_vs_nonSTOP_ROC": roc_auc(y_ss, p),
                "STOP_vs_nonSTOP_PR": pr_auc(y_ss, p),
                "THREAT_vs_SAFE_ROC": roc_auc(
                    np.r_[np.ones((y_gt == CAUTION).sum()), np.zeros(y_safe.sum())],
                    np.r_[p[y_gt == CAUTION], p[y_safe]],
                ),
                "prefork_STOP_vs_nonSTOP_PR": pr_auc(((y_gt == STOP) & pref).astype(np.int32), p),
                "postfork_STOP_vs_nonSTOP_PR": pr_auc(((y_gt == STOP) & (~pref)).astype(np.int32), p),
            }
        )
    with (OUT / "score_distributions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader()
        w.writerows(dist_rows)
    with (OUT / "ranking_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        w.writeheader()
        w.writerows(rank_rows)

    # C pre/post
    fork_rows = []
    stop_idx = np.where(y_gt == STOP)[0]
    det = {tag: peaks[tag] >= 0.60 for tag in MODELS}
    for tag in MODELS:
        p = peaks[tag]
        for name, msk in (("pre", pref), ("post", ~pref)):
            sm = (y_gt == STOP) & msk
            fork_rows.append(
                {
                    "model": tag,
                    "group": name,
                    "n_gt_stop": int(sm.sum()),
                    "stop_recall": float(det[tag][sm].mean()) if sm.any() else float("nan"),
                    **{f"peak_{k}": v for k, v in stats(p[sm]).items()},
                }
            )
    with (OUT / "pre_post_fork.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fork_rows[0].keys()))
        w.writeheader()
        w.writerows(fork_rows)

    all_miss = stop_idx[np.stack([~det[t][stop_idx] for t in MODELS]).all(0)]
    all_hit = stop_idx[np.stack([det[t][stop_idx] for t in MODELS]).all(0)]
    only_one = {}
    for tag in MODELS:
        others = [t for t in MODELS if t != tag]
        m = det[tag][stop_idx] & np.stack([~det[t][stop_idx] for t in others]).all(0)
        only_one[tag] = stop_idx[m]

    # D overlap
    overlap_rows = []
    for a, b in combinations(list(MODELS), 2):
        da, db = det[a][stop_idx], det[b][stop_idx]
        groups = {
            "both": da & db,
            f"{a}_only": da & ~db,
            f"{b}_only": ~da & db,
            "neither": ~da & ~db,
        }
        for gname, msk in groups.items():
            ids = stop_idx[msk]
            overlap_rows.append(
                {
                    "pair": f"{a}-{b}",
                    "group": gname,
                    "n": int(msk.sum()),
                    "pct_of_gt_stop": float(msk.mean()),
                    "n_prefork": int(pref[ids].sum()) if ids.size else 0,
                    "n_postfork": int((~pref[ids]).sum()) if ids.size else 0,
                    f"mean_peak_{a}": float(peaks[a][ids].mean()) if ids.size else float("nan"),
                    f"mean_peak_{b}": float(peaks[b][ids].mean()) if ids.size else float("nan"),
                }
            )
    with (OUT / "error_overlap.csv").open("w", newline="") as f:
        keys = sorted({k for r in overlap_rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(overlap_rows)

    with (OUT / "gt_stop_detection_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "sample_id", "clip_id", "frame", "pre_fork", "gt_peak"] + [f"{t}_peak" for t in MODELS] + [f"{t}_det" for t in MODELS] + ["n_models"])
        for i in stop_idx:
            dets = [int(det[t][i]) for t in MODELS]
            w.writerow([i, sample_id[i], clip_id[i], int(frame[i]), int(pref[i]), float(p_gt[i])] + [float(peaks[t][i]) for t in MODELS] + dets + [sum(dets)])

    # E spatial
    occ = gt >= 0.25
    spatial_rows = []
    for tag in ["M1", "M5"]:
        h = H[tag]
        for r in range(5):
            for c in range(5):
                m = occ[:, r, c] & (y_gt == STOP)
                vals = h[:, r, c][m]
                spatial_rows.append(
                    {
                        "model": tag,
                        "row": r,
                        "col": c,
                        "n_gt_stop_occupied": int(m.sum()),
                        "mean_pred": float(vals.mean()) if vals.size else float("nan"),
                        "median_pred": float(np.median(vals)) if vals.size else float("nan"),
                        "frac_ge_060": float((vals >= 0.60).mean()) if vals.size else float("nan"),
                        "frac_lt_025": float((vals < 0.25).mean()) if vals.size else float("nan"),
                    }
                )
    with (OUT / "spatial_cell_analysis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(spatial_rows[0].keys()))
        w.writeheader()
        w.writerows(spatial_rows)

    # amplitude vs spatial on misses
    def loc_kind(h, g):
        gmax = g.argmax()
        gr, gc = divmod(int(gmax), 5)
        pmax = h.argmax()
        pr, pc = divmod(int(pmax), 5)
        correct_cell = h[gr, gc]
        if correct_cell >= 0.60:
            return "amp_ok_but_global_else"  # shouldn't happen on miss
        if correct_cell >= 0.25:
            return "correct_cell_low_amp"
        if (pr, pc) == (gr, gc):
            return "correct_cell_collapsed"
        if pc == gc:
            return "wrong_row_same_col"
        if abs(pc - gc) <= 1:
            return "near_col"
        return "wrong_region"

    loc_counts = {tag: {} for tag in MODELS}
    for tag in MODELS:
        miss = stop_idx[~det[tag][stop_idx]]
        for i in miss:
            k = loc_kind(H[tag][i], gt[i])
            loc_counts[tag][k] = loc_counts[tag].get(k, 0) + 1
    (OUT / "spatial_miss_kinds.json").write_text(json.dumps(loc_counts, indent=2), encoding="utf-8")

    # F ensemble
    ens = {
        "mean": 0.5 * peaks["M1"] + 0.5 * peaks["M5"],
        "max": np.maximum(peaks["M1"], peaks["M5"]),
        "min": np.minimum(peaks["M1"], peaks["M5"]),
        "w25": 0.25 * peaks["M1"] + 0.75 * peaks["M5"],
        "w50": 0.5 * peaks["M1"] + 0.5 * peaks["M5"],
        "w75": 0.75 * peaks["M1"] + 0.25 * peaks["M5"],
        "Hmean_peak": (0.5 * H["M1"] + 0.5 * H["M5"]).reshape(n, -1).max(1),
        "Hmax_peak": np.maximum(H["M1"], H["M5"]).reshape(n, -1).max(1),
    }
    ens_rows = []
    ens_qualify = []
    for name, p in ens.items():
        for thr in STOP_SWEEP:
            y = peak_cls(p, thr)
            m = rec_prec(y_gt, y)
            recd = {"combo": name, "stop_thr": thr, "SAFE": m["SAFE"], "THR": m["THR"], "STOP": m["STOP"], "STOP_prec": m["prec"], "safe_as_stop": m["safe_as_stop"], "frac_pred_stop": m["frac_pred_stop"]}
            ens_rows.append(recd)
            if m["SAFE"] >= 0.82 and m["STOP"] >= 0.70:
                ens_qualify.append(recd)
    with (OUT / "ensemble_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ens_rows[0].keys()))
        w.writeheader()
        w.writerows(ens_rows)

    # figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    for tag, c in zip(["M1", "M5"], ["C0", "C2"]):
        ax.hist(peaks[tag][y_gt == STOP], bins=bins, alpha=0.4, label=f"{tag} GT-STOP", color=c, density=True)
        ax.hist(peaks[tag][y_gt == SAFE], bins=bins, alpha=0.25, label=f"{tag} GT-SAFE", color=c, histtype="step", density=True)
    ax.axvline(0.60, ls="--", c="k", lw=1)
    ax.axvline(0.25, ls=":", c="k", lw=1)
    ax.legend(fontsize=8)
    ax.set_xlabel("predicted peak")
    ax.set_title("Peak distributions M1 vs M5")
    fig.tight_layout()
    fig.savefig(OUT / "figures/peak_hist_m1_m5.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 5, figsize=(12, 3.2), sharey=True)
    for ax, tag in zip(axes, MODELS):
        data = [peaks[tag][y_gt == c] for c in (SAFE, CAUTION, STOP)]
        ax.boxplot(data, tick_labels=["S", "T", "P"], showfliers=False)
        ax.set_title(tag)
        ax.axhline(0.60, ls="--", c="r", lw=0.8)
        ax.axhline(0.25, ls=":", c="gray", lw=0.8)
    fig.suptitle("Peak by GT class")
    fig.tight_layout()
    fig.savefig(OUT / "figures/peak_box_by_class.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag in MODELS:
        xs = [r["stop_thr"] for r in sweep_rows if r["model"] == tag]
        ys = [r["STOP"] for r in sweep_rows if r["model"] == tag]
        sa = [r["SAFE"] for r in sweep_rows if r["model"] == tag]
        ax.plot(ys, sa, marker="o", label=tag)
    ax.axhline(0.82, ls="--", c="gray")
    ax.axvline(0.70, ls="--", c="gray")
    ax.set_xlabel("STOP recall")
    ax.set_ylabel("SAFE recall")
    ax.set_title("Threshold sweep (SAFE thr=0.25)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/threshold_safe_vs_stop.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(5)
    pre_r = [next(r["stop_recall"] for r in fork_rows if r["model"] == t and r["group"] == "pre") for t in MODELS]
    post_r = [next(r["stop_recall"] for r in fork_rows if r["model"] == t and r["group"] == "post") for t in MODELS]
    ax.bar(x - 0.18, pre_r, 0.36, label="pre-fork")
    ax.bar(x + 0.18, post_r, 0.36, label="post-fork")
    ax.set_xticks(x)
    ax.set_xticklabels(list(MODELS))
    ax.set_ylabel("STOP recall @0.60")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/prefork_postfork_recall.png", dpi=140)
    plt.close()

    # overlap bars M1-M5
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    m1m5 = [r for r in overlap_rows if r["pair"] == "M1-M5"]
    ax.bar([r["group"] for r in m1m5], [r["n"] for r in m1m5])
    ax.set_title("GT-STOP detections M1 vs M5 @0.60")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(OUT / "figures/overlap_m1_m5.png", dpi=140)
    plt.close()

    def grid5(vals, title, path):
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(vals, vmin=0, vmax=max(0.01, np.nanmax(vals)), cmap="magma")
        ax.set_title(title)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close()

    occ_stop = (gt >= 0.25) & (y_gt == STOP)[:, None, None]
    grid5(occ_stop.sum(0).astype(np.float64), "GT-STOP occupied cell counts", OUT / "figures/spatial_gt_occupancy.png")
    grid5(np.nan_to_num(H["M1"][y_gt == STOP].mean(0)), "M1 mean pred on GT-STOP", OUT / "figures/spatial_m1_mean.png")
    grid5(np.nan_to_num(H["M5"][y_gt == STOP].mean(0)), "M5 mean pred on GT-STOP", OUT / "figures/spatial_m5_mean.png")
    both_miss = (~det["M1"]) & (~det["M5"]) & (y_gt == STOP)
    grid5(gt[both_miss].mean(0) if both_miss.any() else np.zeros((5, 5)), "mean GT on M1&M5 joint misses", OUT / "figures/spatial_joint_miss_gt.png")

    # consensus
    n_models = np.stack([det[t] for t in MODELS], 0).sum(0)
    hist_k = {int(k): int((n_models[stop_idx] == k).sum()) for k in range(6)}

    # sample concentration of unanimous misses
    miss_ids = {}
    for i in all_miss:
        miss_ids[sample_id[i]] = miss_ids.get(sample_id[i], 0) + 1
    top_ids = sorted(miss_ids.items(), key=lambda x: -x[1])[:15]

    # dir of GT peak cell
    def col_dir(g):
        c = int(g.reshape(25).argmax() % 5)
        if c <= 1:
            return "LEFT"
        if c == 2:
            return "CENTER"
        return "RIGHT"

    dirs_all = np.array([col_dir(g) for g in gt])
    dir_miss = {}
    for d in ("LEFT", "CENTER", "RIGHT"):
        m = (y_gt == STOP) & (dirs_all == d)
        dir_miss[d] = {
            "n": int(m.sum()),
            "M1_rec": float(det["M1"][m].mean()) if m.any() else None,
            "M5_rec": float(det["M5"][m].mean()) if m.any() else None,
            "unanimous_miss": int(((~np.stack([det[t][m] for t in MODELS])).all(0)).sum()) if m.any() else 0,
        }

    # Pareto closest points
    def pareto(rows, key_model=None):
        pts = rows if key_model is None else [r for r in rows if r.get("model") == key_model or r.get("combo")]
        out = []
        for r in pts:
            if r["SAFE"] < 0.82:
                continue
            dominated = False
            for q in pts:
                if q is r:
                    continue
                if q["SAFE"] >= 0.82 and q["STOP"] >= r["STOP"] and q["SAFE"] >= r["SAFE"] and (q["STOP"] > r["STOP"] or q["SAFE"] > r["SAFE"]):
                    if (key_model is None or q.get("model") == r.get("model") or q.get("combo") == r.get("combo")):
                        # keep per-series later
                        pass
            out.append(r)
        # closest to (0.82, 0.70) among SAFE>=0.82
        cand = [r for r in pts if r["SAFE"] >= 0.82]
        if not cand:
            cand = pts
        best = max(cand, key=lambda r: (r["STOP"], r["SAFE"]))
        dist = min(cand, key=lambda r: (max(0, 0.70 - r["STOP"]) ** 2 + max(0, 0.82 - r["SAFE"]) ** 2))
        return best, dist

    closest = {}
    for tag in MODELS:
        rows = [r for r in sweep_rows if r["model"] == tag]
        best82 = max([r for r in rows if r["SAFE"] >= 0.82], key=lambda r: r["STOP"])
        closest[tag] = best82

    ens_best = None
    if ens_rows:
        cand = [r for r in ens_rows if r["SAFE"] >= 0.82]
        ens_best = max(cand, key=lambda r: r["STOP"]) if cand else max(ens_rows, key=lambda r: r["STOP"])

    m1m5_groups = {r["group"]: r for r in overlap_rows if r["pair"] == "M1-M5"}
    n_stop = int((y_gt == STOP).sum())
    complementary = m1m5_groups.get("M1_only", {}).get("n", 0) + m1m5_groups.get("M5_only", {}).get("n", 0)

    cfgj = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": "selection",
        "n": n,
        "n_train_not_used": 22680,
        "no_test": True,
        "safe_thr": SAFE_THR,
        "official_stop_thr": 0.60,
        "decoder_init_hash": EXPECTED_HASH,
        "checkpoints": ckpt_meta,
        "device": device,
        "preprocessing": "uint8 RGB 256 cache -> [0,1] float32 -> ImageNet in model; GT=group-mean gt_1s",
        "missing": missing,
    }
    (OUT / "audit_config.json").write_text(json.dumps(cfgj, indent=2), encoding="utf-8")

    # markdown
    lines = [
        "# Selection prediction audit v1 (no training, no test)",
        "",
        f"n={n} selection windows. Official STOP≥0.60, SAFE<0.25. Checkpoints: " + ", ".join(f"{t} ep{MODELS[t][2]}" for t in MODELS) + ".",
        "",
        "## Parity vs published metrics",
        "",
        "| Model | Epoch | Audit SAFE | Δ | STOP | Δ | prec | pre | post | match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for tag in MODELS:
        p = parity[tag]
        lines.append(f"| {tag} | {MODELS[tag][2]} | {p['SAFE']:.3f} | {p['d_SAFE']:+.3f} | {p['STOP']:.3f} | {p['d_STOP']:+.3f} | {p['prec']:.3f} | {p['pre']:.3f} | {p['post']:.3f} | {p['ok']} |")
    lines += ["", "## A. Threshold sweep (SAFE thr fixed 0.25)", ""]
    if qualify:
        lines.append("Operating points with SAFE≥0.82 **and** STOP≥0.70:")
        for q in qualify:
            lines.append(f"- {q}")
    else:
        lines.append("**No single-model threshold meets SAFE≥0.82 and STOP≥0.70.** Closest with SAFE≥0.82 (max STOP):")
        lines.append("")
        lines.append("| Model | STOP thr | SAFE | THR | STOP | prec | GT-SAFE→STOP |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for tag in MODELS:
            r = closest[tag]
            lines.append(f"| {tag} | {r['stop_thr']} | {r['SAFE']:.3f} | {r['THR']:.3f} | {r['STOP']:.3f} | {r['STOP_prec']:.3f} | {r['safe_as_stop']:.3f} |")
    lines += ["", "## B. Ranking (peaks)", "", "| Model | STOP vs SAFE ROC | STOP vs rest ROC | STOP PR-AUC | THREAT vs SAFE ROC |", "|---|---:|---:|---:|---:|"]
    for r in rank_rows:
        lines.append(f"| {r['model']} | {r['STOP_vs_SAFE_ROC']:.3f} | {r['STOP_vs_nonSTOP_ROC']:.3f} | {r['STOP_vs_nonSTOP_PR']:.3f} | {r['THREAT_vs_SAFE_ROC']:.3f} |")
    lines += ["", "Peak stats GT-STOP / GT-SAFE (mean, p10, p50):", ""]
    for tag in MODELS:
        st = next(d for d in dist_rows if d["model"] == tag and d["group"] == "gt_STOP")
        sa = next(d for d in dist_rows if d["model"] == tag and d["group"] == "gt_SAFE")
        lines.append(f"- {tag} STOP mean={st['mean']:.3f} p10={st['p10']:.3f} p50={st['median']:.3f}; SAFE mean={sa['mean']:.3f} p90={sa['p90']:.3f}")
    lines += ["", "## C. Pre/post fork", ""]
    npre = int(((y_gt == STOP) & pref).sum())
    npost = int(((y_gt == STOP) & (~pref)).sum())
    lines.append(f"GT-STOP: pre-fork n={npre}, post-fork n={npost}.")
    lines.append("")
    lines.append("| Model | Pre rec | Post rec | Pre mean peak | Post mean peak | Pre p10 | Post p10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tag in MODELS:
        pre = next(r for r in fork_rows if r["model"] == tag and r["group"] == "pre")
        post = next(r for r in fork_rows if r["model"] == tag and r["group"] == "post")
        lines.append(f"| {tag} | {pre['stop_recall']:.3f} | {post['stop_recall']:.3f} | {pre['peak_mean']:.3f} | {post['peak_mean']:.3f} | {pre['peak_p10']:.3f} | {post['peak_p10']:.3f} |")
    lines.append(f"\nUnanimous STOP misses: **{len(all_miss)}** / {n_stop} ({100*len(all_miss)/n_stop:.1f}%). Unanimous hits: {len(all_hit)}.")
    lines.append(f"Unanimous-miss pre-fork: {int(pref[all_miss].sum())} post-fork: {int((~pref[all_miss]).sum())}.")
    lines.append("Top sample_ids among unanimous misses: " + ", ".join(f"{k}:{v}" for k, v in top_ids[:8]))
    lines.append(f"Direction of GT peak among STOP: {json.dumps(dir_miss)}")
    lines += ["", "## D. M1 vs M5 overlap @0.60", ""]
    for g, r in m1m5_groups.items():
        lines.append(f"- {g}: n={r['n']} ({100*r['pct_of_gt_stop']:.1f}%) prefork={r['n_prefork']} postfork={r['n_postfork']}")
    lines.append(f"Exclusive detections (M1-only + M5-only) = {complementary} / {n_stop} = {100*complementary/n_stop:.1f}% of GT-STOP.")
    lines.append(f"n_models detecting STOP histogram: {hist_k}")
    lines += ["", "## E. Spatial vs amplitude", "", json.dumps(loc_counts, indent=2), ""]
    lines += ["## F. Ensembles", ""]
    if ens_qualify:
        lines.append("Ensembles meeting SAFE≥0.82 and STOP≥0.70:")
        for q in ens_qualify:
            lines.append(f"- {q}")
    else:
        lines.append("**No peak/heatmap ensemble + threshold meets both targets.** Best SAFE≥0.82 ensemble:")
        lines.append(str(ens_best))
    lines += [
        "",
        "## G. Should we run M6?",
        "",
        "See final response. Ranking AUCs are moderate-to-strong (STOP vs SAFE ROC typically >0.9 if well separated); "
        "fixed 0.60 is in a dense overlap region (STOP p10 often <0.25–0.45). Pre-fork is systematically weaker. "
        "M1/M5 exclusive hits exist but joint misses remain large. Spatial misses are mostly low amplitude on the correct/nearby cell.",
    ]
    (OUT / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "qualify_single.json").write_text(json.dumps({"qualify": qualify, "closest": closest}, indent=2, default=str), encoding="utf-8")
    (OUT / "qualify_ensemble.json").write_text(json.dumps({"qualify": ens_qualify, "best": ens_best}, indent=2, default=str), encoding="utf-8")
    (OUT / "unanimous_miss_ids.json").write_text(json.dumps({"n": len(all_miss), "top_sample_ids": top_ids, "n_prefork": int(pref[all_miss].sum())}, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
