#!/usr/bin/env python3
"""Read-only selection-split audit of StudentLat-PairCF epochs 12/38/62. No training."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP  # noqa: E402
from collision_jepa3.student_research.cache_access import ParentTargetCache  # noqa: E402
from collision_jepa3.student_research.fdec_student import build_parent_decoder, decoder_state_hash  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

import importlib.util

spec = importlib.util.spec_from_file_location(
    "s75", ROOT / "scripts/75_train_student_lat_paircf.py"
)
s75 = importlib.util.module_from_spec(spec)
# Avoid executing main: load module but guard
sys.modules["s75"] = s75
spec.loader.exec_module(s75)

EXP = TEMP / "student_lat_paircf"
OUT = EXP / "selection_audit"
RGB_ROOT = TEMP / "shared_rgb_cache_256_v1"
PARENT_ROOT = TEMP / "parent_target_cache_v2"
EXPECTED_HASH = "77f78c8714667617"
EPOCHS = (12, 38, 62)
CLS_NAME = {SAFE: "SAFE", CAUTION: "THREAT", STOP: "STOP"}
LEFT, CENTER, RIGHT = {0, 1}, {2}, {3, 4}


def peak_class(h: np.ndarray, stop_thr: float = 0.60, caution_thr: float = 0.25) -> int:
    p = float(np.max(h))
    if p >= stop_thr:
        return STOP
    if p >= caution_thr:
        return CAUTION
    return SAFE


def argmax_rc(h: np.ndarray) -> tuple[int, int]:
    i = int(np.argmax(h))
    return divmod(i, 5)


def cell_side(c: int) -> str:
    if c in LEFT:
        return "LEFT"
    if c in CENTER:
        return "CENTER"
    return "RIGHT"


def summarize_peaks(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"n": 0}
    qs = np.quantile(vals, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "max": float(qs[6]),
        "frac_ge_0.25": float(np.mean(vals >= 0.25)),
        "frac_ge_0.40": float(np.mean(vals >= 0.40)),
        "frac_ge_0.50": float(np.mean(vals >= 0.50)),
        "frac_ge_0.60": float(np.mean(vals >= 0.60)),
        "frac_ge_0.70": float(np.mean(vals >= 0.70)),
    }


def hist_edges():
    return np.round(np.arange(0.0, 1.05, 0.05), 2)


def histogram(vals: np.ndarray) -> dict:
    edges = hist_edges()
    counts, _ = np.histogram(vals, bins=edges)
    return {"edges": edges.tolist(), "counts": [int(x) for x in counts]}


@torch.no_grad()
def collect_epoch(model, loader, device, amp):
    model.eval()
    recs = []
    hs, hp, hg = [], [], []
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp
        else torch.nullcontext()
    )
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with ctx:
            out = model(b["clip"])
        h = out["Hhat_1s"].float().cpu().numpy()
        ht = b["H_teacher"].float().cpu().numpy()
        gt = b["gt_1s"].float().cpu().numpy()
        pref = b["pre_fork"].cpu().numpy()
        frames = b["frame"].cpu().numpy()
        for i in range(h.shape[0]):
            recs.append(
                {
                    "clip_id": b["clip_id"][i],
                    "sample_id": b["sample_id"][i],
                    "anchor_frame": int(frames[i]),
                    "pre_fork": bool(pref[i]),
                    "fork_status": "pre" if bool(pref[i]) else "post",
                    "background_id": b["background_id"][i],
                    "gt_peak": float(gt[i].max()),
                    "parent_peak": float(ht[i].max()),
                    "student_peak": float(h[i].max()),
                    "gt_class": int(peak_class(gt[i])),
                    "parent_class": int(peak_class(ht[i])),
                    "student_class": int(peak_class(h[i])),
                    "gt_class_name": CLS_NAME[peak_class(gt[i])],
                    "parent_class_name": CLS_NAME[peak_class(ht[i])],
                    "student_class_name": CLS_NAME[peak_class(h[i])],
                    "gt_argmax_r": argmax_rc(gt[i])[0],
                    "gt_argmax_c": argmax_rc(gt[i])[1],
                    "parent_argmax_r": argmax_rc(ht[i])[0],
                    "parent_argmax_c": argmax_rc(ht[i])[1],
                    "student_argmax_r": argmax_rc(h[i])[0],
                    "student_argmax_c": argmax_rc(h[i])[1],
                }
            )
            hs.append(h[i])
            hp.append(ht[i])
            hg.append(gt[i])
    return recs, np.stack(hs), np.stack(hp), np.stack(hg)


def error_type(gt, st, rec):
    """Amplitude vs spatial on GT-STOP misses (student not STOP)."""
    gr, gc = rec["gt_argmax_r"], rec["gt_argmax_c"]
    sr, sc = rec["student_argmax_r"], rec["student_argmax_c"]
    dist = max(abs(gr - sr), abs(gc - sc))
    occ_gt = gt >= 0.50
    occ_st = st >= 0.50
    inter = int(np.logical_and(occ_gt, occ_st).sum())
    union = int(np.logical_or(occ_gt, occ_st).sum())
    jac = inter / union if union else 0.0
    peak = rec["student_peak"]
    if peak < 0.25:
        kind = "collapse_low_amplitude"
    elif dist <= 1 and jac >= 0.25:
        kind = "amplitude_underconfident"
    elif dist <= 1 and peak >= 0.25:
        kind = "amplitude_underconfident"
    elif dist >= 2 and peak >= 0.40:
        kind = "spatial_localization"
    elif dist >= 2:
        kind = "spatial_and_weak"
    else:
        kind = "mixed"
    return {
        "error_kind": kind,
        "argmax_chebyshev": dist,
        "occupied_jaccard_0.50": jac,
        "gt_side": cell_side(gc),
        "student_side": cell_side(sc),
        "side_match": cell_side(gc) == cell_side(sc),
    }


def analyze(recs, Hs, Hp, Hg):
    n = len(recs)
    gt_c = np.array([r["gt_class"] for r in recs])
    p_c = np.array([r["parent_class"] for r in recs])
    s_c = np.array([r["student_class"] for r in recs])
    gt_stop = gt_c == STOP
    p_stop = p_c == STOP
    s_stop = s_c == STOP
    breakdown = {
        "n_gt_stop": int(gt_stop.sum()),
        "parent_STOP_student_STOP": int(np.logical_and(gt_stop, np.logical_and(p_stop, s_stop)).sum()),
        "parent_STOP_student_nonSTOP": int(np.logical_and(gt_stop, np.logical_and(p_stop, ~s_stop)).sum()),
        "parent_nonSTOP_student_STOP": int(np.logical_and(gt_stop, np.logical_and(~p_stop, s_stop)).sum()),
        "both_nonSTOP": int(np.logical_and(gt_stop, np.logical_and(~p_stop, ~s_stop)).sum()),
    }
    # also among GT-STOP, parent/student class names
    both_non_detail = Counter()
    for r, m in zip(recs, gt_stop):
        if m and r["parent_class"] != STOP and r["student_class"] != STOP:
            both_non_detail[f"parent={r['parent_class_name']}|student={r['student_class_name']}"] += 1
    breakdown["both_nonSTOP_detail"] = dict(both_non_detail)

    peaks = {}
    for name, mask in (
        ("GT-SAFE", gt_c == SAFE),
        ("GT-THREAT", gt_c == CAUTION),
        ("GT-STOP", gt_stop),
    ):
        peaks[name] = {
            "n": int(mask.sum()),
            "parent": summarize_peaks(np.array([recs[i]["parent_peak"] for i in np.where(mask)[0]])),
            "student": summarize_peaks(np.array([recs[i]["student_peak"] for i in np.where(mask)[0]])),
            "gt": summarize_peaks(np.array([recs[i]["gt_peak"] for i in np.where(mask)[0]])),
            "parent_hist": histogram(np.array([recs[i]["parent_peak"] for i in np.where(mask)[0]])),
            "student_hist": histogram(np.array([recs[i]["student_peak"] for i in np.where(mask)[0]])),
        }

    thr_rows = []
    for st in [round(x, 2) for x in np.arange(0.40, 0.71, 0.05)]:
        pred = np.array([peak_class(Hs[i], stop_thr=st, caution_thr=0.25) for i in range(n)])
        tp = int(np.logical_and(gt_c == STOP, pred == STOP).sum())
        fp = int(np.logical_and(gt_c != STOP, pred == STOP).sum())
        fn = int(np.logical_and(gt_c == STOP, pred != STOP).sum())
        pred_safe = pred == SAFE
        gt_safe = gt_c == SAFE
        thr_rows.append(
            {
                "stop_threshold": st,
                "caution_threshold": 0.25,
                "stop_recall": tp / max(1, int((gt_c == STOP).sum())),
                "stop_precision": tp / max(1, tp + fp),
                "stop_tp": tp,
                "stop_fp": fp,
                "stop_fn": fn,
                "safe_recall": int(np.logical_and(gt_safe, pred_safe).sum()) / max(1, int(gt_safe.sum())),
                "n_pred_stop": int((pred == STOP).sum()),
            }
        )

    misses = [i for i, r in enumerate(recs) if r["gt_class"] == STOP and r["student_class"] != STOP]
    by_sample = Counter()
    by_fork = Counter()
    by_cell = Counter()
    by_side = Counter()
    kinds = Counter()
    miss_rows = []
    for i in misses:
        r = recs[i]
        extra = error_type(Hg[i], Hs[i], r)
        by_sample[r["sample_id"]] += 1
        by_fork[r["fork_status"]] += 1
        by_cell[f"r{r['gt_argmax_r']}_c{r['gt_argmax_c']}"] += 1
        by_side[extra["gt_side"]] += 1
        kinds[extra["error_kind"]] += 1
        miss_rows.append({**{k: r[k] for k in (
            "clip_id", "sample_id", "anchor_frame", "fork_status",
            "gt_peak", "parent_peak", "student_peak",
            "parent_class_name", "student_class_name",
            "gt_argmax_r", "gt_argmax_c", "student_argmax_r", "student_argmax_c",
        )}, **extra})

    n_miss = max(1, len(misses))
    amp_n = kinds["amplitude_underconfident"] + kinds["collapse_low_amplitude"]
    spat_n = kinds["spatial_localization"] + kinds["spatial_and_weak"]
    diagnosis = {
        "n_gt_stop_misses": len(misses),
        "error_kind_counts": dict(kinds),
        "primarily": (
            "amplitude"
            if amp_n > spat_n + kinds["mixed"]
            else ("spatial" if spat_n > amp_n + kinds["mixed"] else "mixed_amplitude_and_spatial")
        ),
        "fraction_argmax_chebyshev_le1": float(np.mean([m["argmax_chebyshev"] <= 1 for m in miss_rows])) if miss_rows else None,
        "fraction_side_match": float(np.mean([m["side_match"] for m in miss_rows])) if miss_rows else None,
        "mean_student_peak_on_miss": float(np.mean([m["student_peak"] for m in miss_rows])) if miss_rows else None,
        "mean_parent_peak_on_miss": float(np.mean([m["parent_peak"] for m in miss_rows])) if miss_rows else None,
        "mean_gt_peak_on_miss": float(np.mean([m["gt_peak"] for m in miss_rows])) if miss_rows else None,
        "note": (
            "Amplitude: student blob near GT argmax but peak < 0.60 (or map collapsed <0.25). "
            "Spatial: argmax Chebyshev distance >= 2. Mixed otherwise."
        ),
    }

    n_stop = int(gt_stop.sum())
    n_safe = int((gt_c == SAFE).sum())
    n_thr = int((gt_c == CAUTION).sum())
    metrics_060 = {
        "stop_recall": int(np.logical_and(gt_stop, s_stop).sum()) / max(1, n_stop),
        "safe_recall": int(np.logical_and(gt_c == SAFE, s_c == SAFE).sum()) / max(1, n_safe),
        "threat_recall": int(np.logical_and(gt_c == CAUTION, s_c == CAUTION).sum()) / max(1, n_thr),
        "stop_precision": int(np.logical_and(gt_stop, s_stop).sum()) / max(1, int(s_stop.sum())),
        "parent_stop_recall": int(np.logical_and(gt_stop, p_stop).sum()) / max(1, n_stop),
        "parent_safe_recall": int(np.logical_and(gt_c == SAFE, p_c == SAFE).sum()) / max(1, n_safe),
        "n": n,
        "n_gt_stop": n_stop,
        "n_gt_safe": n_safe,
        "n_gt_threat": n_thr,
    }
    return {
        "n_windows": n,
        "metrics_thr_0.60_0.25": metrics_060,
        "gt_stop_parent_student_breakdown": breakdown,
        "peak_distributions": peaks,
        "stop_operating_caution_0.25": thr_rows,
        "misses_by_sample_id": dict(by_sample.most_common()),
        "misses_by_fork": dict(by_fork),
        "misses_by_gt_argmax_cell": dict(by_cell.most_common()),
        "misses_by_gt_side": dict(by_side),
        "amplitude_vs_spatial": diagnosis,
        "n_miss_rows": len(miss_rows),
    }, miss_rows


def md_report(all_sum: dict) -> str:
    lines = [
        "# StudentLat-PairCF selection audit",
        "",
        "Read-only. 5,120 official selection windows. Thresholds 0.25/0.60 unless noted.",
        "Frozen Pair-CF H1s decoder. No test split. Epochs 12, 38, 62.",
        "",
        f"- Decoder hash: `{all_sum['decoder_hash']}`",
        f"- Device: `{all_sum['device']}`",
        f"- RGB cache: `{all_sum['rgb']}`",
        f"- Parent cache: `{all_sum['parent']}`",
        "",
    ]
    for ep in EPOCHS:
        s = all_sum["epochs"][str(ep)]
        m = s["metrics_thr_0.60_0.25"]
        b = s["gt_stop_parent_student_breakdown"]
        d = s["amplitude_vs_spatial"]
        lines += [
            f"## Epoch {ep}",
            "",
            f"- n={m['n']}  GT-STOP={m['n_gt_stop']}  GT-THREAT={m['n_gt_threat']}  GT-SAFE={m['n_gt_safe']}",
            f"- Student STOP recall **{m['stop_recall']:.3f}**, SAFE **{m['safe_recall']:.3f}**, THREAT **{m['threat_recall']:.3f}**, STOP precision **{m['stop_precision']:.3f}**",
            f"- Parent STOP recall **{m['parent_stop_recall']:.3f}**, SAFE **{m['parent_safe_recall']:.3f}**",
            "",
            "### 1. GT-STOP confusion (parent vs student at 0.60)",
            "",
            "| | student STOP | student non-STOP |",
            "|---|---:|---:|",
            f"| parent STOP | {b['parent_STOP_student_STOP']} | {b['parent_STOP_student_nonSTOP']} |",
            f"| parent non-STOP | {b['parent_nonSTOP_student_STOP']} | {b['both_nonSTOP']} |",
            "",
            f"GT-STOP n={b['n_gt_stop']}. both-nonSTOP detail: `{b['both_nonSTOP_detail']}`",
            "",
            "### 2. Peak-score distributions",
            "",
            "| GT class | who | n | mean | p10 | p50 | p90 | max | ≥0.25 | ≥0.60 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for gname, block in s["peak_distributions"].items():
            for who in ("gt", "parent", "student"):
                q = block[who]
                if q.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {gname} | {who} | {q['n']} | {q['mean']:.3f} | {q['p10']:.3f} | {q['p50']:.3f} | {q['p90']:.3f} | {q['max']:.3f} | {q['frac_ge_0.25']:.3f} | {q['frac_ge_0.60']:.3f} |"
                )
        lines += ["", "### 3. STOP recall/precision vs threshold (SAFE/caution still 0.25)", "", "| stop_thr | STOP recall | STOP precision | SAFE recall | TP | FP | FN |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for row in s["stop_operating_caution_0.25"]:
            lines.append(
                f"| {row['stop_threshold']:.2f} | {row['stop_recall']:.3f} | {row['stop_precision']:.3f} | {row['safe_recall']:.3f} | {row['stop_tp']} | {row['stop_fp']} | {row['stop_fn']} |"
            )
        lines += [
            "",
            "### 4. Student STOP misses (GT-STOP, student peak < 0.60)",
            "",
            f"- Count: **{d['n_gt_stop_misses']}**",
            f"- By fork: `{s['misses_by_fork']}`",
            f"- By GT side: `{s['misses_by_gt_side']}`",
            f"- By GT argmax cell: `{s['misses_by_gt_argmax_cell']}`",
            "",
            "Top sample_id miss counts:",
            "",
        ]
        for sid, c in list(s["misses_by_sample_id"].items())[:20]:
            lines.append(f"- `{sid}`: {c}")
        lines += [
            "",
            "### 5. Amplitude vs spatial",
            "",
            f"- Primary: **{d['primarily']}**",
            f"- Kind counts: `{d['error_kind_counts']}`",
            f"- Fraction argmax Chebyshev ≤1: {d['fraction_argmax_chebyshev_le1']}",
            f"- Fraction same left/center/right side: {d['fraction_side_match']}",
            f"- Mean peaks on misses: student {d['mean_student_peak_on_miss']}, parent {d['mean_parent_peak_on_miss']}, GT {d['mean_gt_peak_on_miss']}",
            f"- {d['note']}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch", type=int, default=8)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    OUT.mkdir(parents=True, exist_ok=True)
    assert (RGB_ROOT / "COMPLETE").is_file()
    assert (PARENT_ROOT / "COMPLETE").is_file()
    rows = [json.loads(l) for l in (PARENT_ROOT / "index.jsonl").open()]
    gt = np.load(PARENT_ROOT / "gt_1s.npy").astype(np.float32)
    sel_ids = [i for i, r in enumerate(rows) if r["split"] == "selection"]
    test_ids = [i for i, r in enumerate(rows) if r["split"] == "test"]
    assert len(sel_ids) == 5120
    # refuse test
    assert not any(rows[i]["split"] == "test" for i in sel_ids)
    parent = ParentTargetCache(PARENT_ROOT)
    rgb = s75.RGBCache256(RGB_ROOT)
    ds = s75.WinDS(rows, sel_ids, gt, rgb, parent)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=s75.collate)
    device = resolve_device_safe(str(args.device))
    decoder = build_parent_decoder(cfg, device="cpu")
    dec_h = decoder_state_hash(decoder)
    if dec_h != EXPECTED_HASH:
        raise SystemExit(f"decoder hash {dec_h}")
    model = s75.StudentLatPairCF(decoder).to(device)
    model.eval()
    amp = str(device).startswith("cuda")
    all_sum = {
        "decoder_hash": dec_h,
        "device": device,
        "rgb": str(RGB_ROOT),
        "parent": str(PARENT_ROOT),
        "n_selection": 5120,
        "n_test_untouched": len(test_ids),
        "epochs": {},
    }
    print(f"device={device} decoder={dec_h} n_sel={len(ds)}", flush=True)
    for ep in EPOCHS:
        ck = EXP / f"checkpoint_epoch_{ep:02d}.pt"
        blob = torch.load(ck, map_location=device, weights_only=False)
        model.encoder.load_state_dict(blob["encoder"])
        if blob.get("predictor"):
            model.predictor.load_state_dict(blob["predictor"])
        recs, Hs, Hp, Hg = collect_epoch(model, loader, device, amp)
        summary, miss_rows = analyze(recs, Hs, Hp, Hg)
        all_sum["epochs"][str(ep)] = summary
        np.savez_compressed(
            OUT / f"windows_epoch_{ep:02d}.npz",
            H_student=Hs.astype(np.float16),
            H_parent=Hp.astype(np.float16),
            H_gt=Hg.astype(np.float16),
            gt_peak=np.array([r["gt_peak"] for r in recs], np.float32),
            parent_peak=np.array([r["parent_peak"] for r in recs], np.float32),
            student_peak=np.array([r["student_peak"] for r in recs], np.float32),
            gt_class=np.array([r["gt_class"] for r in recs], np.int8),
            parent_class=np.array([r["parent_class"] for r in recs], np.int8),
            student_class=np.array([r["student_class"] for r in recs], np.int8),
            pre_fork=np.array([r["pre_fork"] for r in recs], np.bool_),
            anchor_frame=np.array([r["anchor_frame"] for r in recs], np.int32),
        )
        meta_path = OUT / f"windows_epoch_{ep:02d}.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        (OUT / f"misses_epoch_{ep:02d}.json").write_text(json.dumps(miss_rows, indent=2), encoding="utf-8")
        (OUT / f"summary_epoch_{ep:02d}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"epoch {ep} {summary['metrics_thr_0.60_0.25']} breakdown={summary['gt_stop_parent_student_breakdown']}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(all_sum, indent=2), encoding="utf-8")
    report = md_report(all_sum)
    (OUT / "selection_audit_report.md").write_text(report, encoding="utf-8")
    (TEMP / "reports").mkdir(parents=True, exist_ok=True)
    (TEMP / "reports/student_lat_paircf_selection_audit.md").write_text(report, encoding="utf-8")
    print("WROTE", OUT / "selection_audit_report.md", flush=True)


if __name__ == "__main__":
    main()
