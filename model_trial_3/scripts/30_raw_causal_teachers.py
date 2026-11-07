#!/usr/bin/env python3
"""Decoder-free causal teachers: GT boxes/occlusion, consensus, RGB quality.

Does not use G's ŷ or lighting labels to pick a view.
Does not train or overwrite G.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.cache import load_cached_frames  # noqa: E402
from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.sampling import FRAME_OFFSETS  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.counterfactual import warning_report  # noqa: E402
from collision_jepa3.u1_runtime import load_frozen_g  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402

STOP, CAUTION = 0.60, 0.25
DYN = frozenset({"person", "vehicle", "bicycle"})
SRC_W, SRC_H, DST = 1920, 1080, 256


def _load24():
    p = Path(__file__).with_name("24_cadet_oracles.py")
    spec = importlib.util.spec_from_file_location("cadet24", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pair_lo_hi(mate: np.ndarray):
    i = np.arange(len(mate))
    ok = (mate >= 0) & (i < mate)
    return i[ok], mate[ok]


def mix_pairs(fine, lo, hi, w_lo):
    lo_t, hi_t = torch.from_numpy(lo), torch.from_numpy(hi)
    w = w_lo.view(-1, 1, 1, 1).to(fine.dtype)
    fc = w * fine[lo_t] + (1.0 - w) * fine[hi_t]
    out = fine.clone()
    out[lo_t] = fc
    out[hi_t] = fc
    return out


def pick(score: np.ndarray, lo, hi, *, low=False, eps=1e-8) -> np.ndarray:
    """Hard pick; ties → 0.5. low=True prefers smaller score."""
    a, b = score[lo], score[hi]
    w = np.full(len(lo), 0.5, dtype=np.float32)
    if low:
        w[a < b - eps] = 1.0
        w[b < a - eps] = 0.0
    else:
        w[a > b + eps] = 1.0
        w[b > a + eps] = 0.0
    return w


def stop_pick(gt_peak, lo, hi, score, *, low=False) -> torch.Tensor:
    w = np.full(len(lo), 0.5, dtype=np.float32)
    stop = (gt_peak[lo] >= STOP) | (gt_peak[hi] >= STOP)
    w[stop] = pick(score, lo[stop], hi[stop], low=low)
    return torch.from_numpy(w)


def soft_mix(score: np.ndarray, lo, hi, temp: float) -> np.ndarray:
    a = np.stack([score[lo], score[hi]], axis=1)
    a = a - np.nanmax(a, axis=1, keepdims=True)
    e = np.exp(a / max(temp, 1e-4))
    s = e.sum(axis=1, keepdims=True)
    w = np.where(s[:, 0] > 0, e[:, 0] / np.clip(s[:, 0], 1e-8, None), 0.5)
    return w.astype(np.float32)


def keep_obj(o: dict) -> bool:
    c = str(o.get("class_name", "")).lower()
    return c in DYN or c.startswith("threat_")


def obj_vis(o: dict) -> dict:
    bb = o.get("bounding_box_2d") or {}
    xmin, ymin = int(bb.get("xmin", 0) or 0), int(bb.get("ymin", 0) or 0)
    xmax, ymax = int(bb.get("xmax", 0) or 0), int(bb.get("ymax", 0) or 0)
    area = max(0, xmax - xmin) * max(0, ymax - ymin)
    fl = o.get("flags") or {}
    kin = o.get("kinematics") or {}
    occ = 1.0 if fl.get("occluded") else 0.0
    trunc = 1.0 if fl.get("truncated") else 0.0
    dist = float(kin.get("distance") or 1e6)
    ttc = float(kin.get("ttc") or 9999.0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    center = 1.0 - np.hypot((cx / SRC_W) - 0.5, (cy / SRC_H) - 0.5) / np.hypot(0.5, 0.5)
    vis = area * (1.0 - occ) * (1.0 - 0.5 * trunc) / max(dist, 1e-3)
    return {
        "area": float(area),
        "occ": occ,
        "trunc": trunc,
        "dist": dist,
        "ttc": ttc,
        "center": float(center),
        "vis": float(vis),
        "inv_ttc": float(1.0 / max(ttc, 0.05)) if ttc < 100 else 0.0,
        "bbox": (xmin, ymin, xmax, ymax),
    }


def best_obj(frame: dict) -> dict | None:
    best = None
    best_key = None
    for o in frame.get("objects") or []:
        if not keep_obj(o):
            continue
        v = obj_vis(o)
        key = (v["vis"], v["area"], -v["dist"])
        if best is None or key > best_key:
            best, best_key = v, key
    return best


def load_clip_frames_ann(path: Path) -> list[dict | None]:
    out: list[dict | None] = [None] * 450
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    frames = data.get("frames") if isinstance(data, dict) else data
    if not isinstance(frames, list):
        return out
    for item in frames:
        if not isinstance(item, dict):
            continue
        raw = item.get("frame_id", item.get("id", 0))
        try:
            fid = int(raw) if isinstance(raw, int) else int(str(raw).lstrip("0") or "0")
        except (TypeError, ValueError):
            continue
        if 0 <= fid < 450:
            out[fid] = best_obj(item)
    return out


def letterbox_xy(x: float, y: float) -> tuple[int, int]:
    scale = min(DST / SRC_W, DST / SRC_H)
    nw, nh = int(round(SRC_W * scale)), int(round(SRC_H * scale))
    x0, y0 = (DST - nw) // 2, (DST - nh) // 2
    return int(round(x0 + x * scale)), int(round(y0 + y * scale))


def rgb_stats(img: np.ndarray, bbox) -> dict:
    """img uint8 HxWx3. bbox original 1920×1080 or None → full frame."""
    if bbox is None:
        crop = img
        bg = None
    else:
        x0, y0 = letterbox_xy(bbox[0], bbox[1])
        x1, y1 = letterbox_xy(bbox[2], bbox[3])
        x0, x1 = sorted((max(0, x0), min(DST, x1)))
        y0, y1 = sorted((max(0, y0), min(DST, y1)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            crop = img
            bg = None
        else:
            crop = img[y0:y1, x0:x1]
            bg = np.concatenate(
                [img[:y0].reshape(-1, 3), img[y1:].reshape(-1, 3), img[y0:y1, :x0].reshape(-1, 3), img[y0:y1, x1:].reshape(-1, 3)],
                axis=0,
            ) if True else None
    f = crop.astype(np.float32)
    gray = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    contrast = float(gray.std())
    edges = 0.0
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        edges = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    sharp = 0.0
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        lap = gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2] - 4.0 * gray[1:-1, 1:-1]
        sharp = float(lap.var())
    glare = float((np.max(f, axis=2) >= 250.0).mean())
    obg = 0.0
    if bg is not None and len(bg) > 8:
        obg = float(np.abs(f.mean() - bg.astype(np.float32).mean()))
    q = contrast * (1.0 + edges) * (1.0 + 0.001 * sharp) * (1.0 - glare)
    return {"contrast": contrast, "edges": edges, "sharp": sharp, "glare": glare, "obj_bg": obg, "q": q}


def print_rec(name, row, feas=None):
    h = row["headline"]
    mark = ""
    if feas is not None:
        mark = " FEASIBLE" if feas.get("feasible") else " no"
    print(
        f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} "
        f"pre={h['pre_fork_stop_recall']:.3f} rec5={h['recall_at_5fa']:.3f} "
        f"post={h['post_fork_stop_recall']:.3f} small={h['small_blob_stop_recall']:.3f} "
        f"flip={row['flips']['warn_flip']:.3f}{mark}",
        flush=True,
    )


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/raw_causal_teachers.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    c24 = _load24()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", False)
    cfg.set("training.num_workers", 0)
    device = resolve_device(args.device)
    print(json.dumps(describe_hardware(device), indent=2), flush=True)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    recs = {r.clip_id: r for r in records}
    ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
    model = load_frozen_g(cfg, Path(args.teacher_ckpt), device, variance_head=False)
    print("collect F", flush=True)
    pack = c24.collect(model, ds, cfg, device, recs)
    fine, mate = pack["fine"], pack["mate"]
    lo, hi = pair_lo_hi(mate)
    pred_g = c24.decode(model, fine, pack["z"], device)
    g_rep = warning_report(
        pred_g, pack["gt"].numpy(), pre_fork=pack["pre"], collision=pack["coll"],
        blob_n=pack["blob"], paired=pack["paired"],
    )
    gt = pack["gt"]
    gt_np = gt.numpy()
    gt_peak = gt_np.reshape(len(gt), -1).max(1)
    n = len(gt)
    ids = pack["clip_id"]
    anchors = pack["anchor"]
    cache_root = cfg.path("paths.cache_root")

    print("load GT vis + RGB quality", flush=True)
    ann_cache: dict[str, list] = {}
    frame_cache: dict[str, np.ndarray] = {}
    keys = ["area", "occ", "trunc", "dist", "center", "vis", "inv_ttc"]
    scores = {k: np.zeros(n, np.float64) for k in keys}
    hist_vis = np.zeros(n, np.float64)
    fut_vis = np.zeros(n, np.float64)
    qkeys = ["contrast", "edges", "sharp", "glare", "obj_bg", "q"]
    qsc = {k: np.zeros(n, np.float64) for k in qkeys}
    n_vis_diff = 0
    n_q_diff = 0
    n_has_obj = 0
    for i, cid in enumerate(ids):
        rec = recs[cid]
        ap = rec.annotations_path
        if cid not in ann_cache:
            ann_cache[cid] = load_clip_frames_ann(Path(ap)) if ap else [None] * 450
        t = int(anchors[i])
        now = ann_cache[cid][t] if 0 <= t < 450 else None
        fut_t = min(449, t + 30)
        fut = ann_cache[cid][fut_t]
        if now:
            n_has_obj += 1
            for k in keys:
                scores[k][i] = now[k]
        hist = []
        for o in FRAME_OFFSETS:
            tt = t + int(o)
            if 0 <= tt < 450 and ann_cache[cid][tt]:
                hist.append(ann_cache[cid][tt]["vis"])
        hist_vis[i] = float(np.mean(hist)) if hist else 0.0
        fut_vis[i] = fut["vis"] if fut else 0.0
        if cid not in frame_cache:
            try:
                frame_cache[cid] = load_cached_frames(cache_root, cid, mmap=True)
            except FileNotFoundError:
                frame_cache[cid] = None
        fr = frame_cache[cid]
        if fr is not None and 0 <= t < len(fr):
            st = rgb_stats(np.asarray(fr[t]), now["bbox"] if now else None)
            for k in qkeys:
                qsc[k][i] = st[k]
        if i % 1280 == 0 and i:
            print(f"  vis {i}/{n}", flush=True)
    for a, b in zip(lo, hi):
        if abs(scores["vis"][a] - scores["vis"][b]) > 1e-6:
            n_vis_diff += 1
        if abs(qsc["q"][a] - qsc["q"][b]) > 1e-6:
            n_q_diff += 1
    audit = {
        "pixel_masks": False,
        "has_boxes_occlusion_distance_ttc": True,
        "n_windows": n,
        "n_pairs": int(pack["n_pairs"]),
        "n_with_dyn_obj_at_anchor": n_has_obj,
        "pairs_GT_vis_differs": n_vis_diff,
        "pairs_RGB_Q_differs": n_q_diff,
        "frac_GT_vis_differs": n_vis_diff / max(len(lo), 1),
        "frac_RGB_Q_differs": n_q_diff / max(len(lo), 1),
        "note": "No silhouette masks. flags.occluded is a centre-ray, bounding_box_2d is a projected AABB. Geometry is usually identical across bg_00/bg_01.",
    }
    print("audit", json.dumps(audit, indent=2), flush=True)

    y16 = F.interpolate(gt.unsqueeze(1), size=fine.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    y_dil = F.interpolate(
        F.max_pool2d(gt.unsqueeze(1), 3, stride=1, padding=1),
        size=fine.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)
    idx = torch.from_numpy(np.where(mate >= 0, mate, np.arange(n)))
    mix = 0.5 * (fine + fine[idx])
    energy = fine.norm(dim=1, keepdim=True)
    f_minabs = torch.where(fine.abs() <= fine[idx].abs(), fine, fine[idx])
    f_maxabs = torch.where(fine.abs() >= fine[idx].abs(), fine, fine[idx])
    dgap = (fine.mean(dim=(2, 3)) - fine[idx].mean(dim=(2, 3))).abs()
    drop = dgap > torch.quantile(dgap, 0.75, dim=1, keepdim=True)
    f_clipch = mix.clone()
    f_clipch[drop[:, :, None, None].expand_as(fine)] = 0
    na, nb = energy, energy[idx]
    f_l2 = torch.where(na >= nb, fine, fine[idx])
    f_keep = y16 * fine + (1.0 - y16) * mix
    f_keep_d = y_dil * fine + (1.0 - y_dil) * mix
    f_thr_max = y16 * f_maxabs + (1.0 - y16) * mix
    f_thr_max_d = y_dil * f_maxabs + (1.0 - y_dil) * mix
    tau = 0.5
    delta = fine - fine[idx]
    f_huber = mix + 0.0 * delta
    f_huber = 0.5 * (fine + fine[idx])  # n=2 arithmetic; keep as control
    # damped shared: mean of tanh-shrunk residuals around 0
    f_damp = 0.5 * (fine[idx] + fine[idx] + tau * torch.tanh(delta / tau))  # = F_mate + 0.5 τ tanh
    # symmetric damped toward mean:
    f_damp = mix + 0.5 * tau * torch.tanh((fine - mix) / tau)  # NOT shared
    # shared damped residual (both get the same): just mix. skip.

    w_area = stop_pick(gt_peak, lo, hi, scores["area"])
    w_occ = stop_pick(gt_peak, lo, hi, scores["occ"], low=True)
    w_trunc = stop_pick(gt_peak, lo, hi, scores["trunc"], low=True)
    w_dist = stop_pick(gt_peak, lo, hi, scores["dist"], low=True)
    w_vis = stop_pick(gt_peak, lo, hi, scores["vis"])
    w_cen = stop_pick(gt_peak, lo, hi, scores["center"])
    w_ttc = stop_pick(gt_peak, lo, hi, scores["inv_ttc"])
    w_hist = stop_pick(gt_peak, lo, hi, hist_vis)
    w_fut = stop_pick(gt_peak, lo, hi, fut_vis)
    w_vsoft = torch.from_numpy(soft_mix(scores["vis"], lo, hi, 1.0))
    # vis softmax only on STOP
    w_vsoft_s = np.full(len(lo), 0.5, np.float32)
    stop = (gt_peak[lo] >= STOP) | (gt_peak[hi] >= STOP)
    w_vsoft_s[stop] = soft_mix(scores["vis"], lo[stop], hi[stop], 1.0)
    w_con = stop_pick(gt_peak, lo, hi, qsc["contrast"])
    w_edg = stop_pick(gt_peak, lo, hi, qsc["edges"])
    w_shp = stop_pick(gt_peak, lo, hi, qsc["sharp"])
    w_glr = stop_pick(gt_peak, lo, hi, qsc["glare"], low=True)
    w_obg = stop_pick(gt_peak, lo, hi, qsc["obj_bg"])
    w_q = stop_pick(gt_peak, lo, hi, qsc["q"])
    w_qsoft = np.full(len(lo), 0.5, np.float32)
    w_qsoft[stop] = soft_mix(qsc["q"], lo[stop], hi[stop], qsc["q"][lo].std() + qsc["q"][hi].std() + 1e-3)
    w_vq = stop_pick(gt_peak, lo, hi, scores["vis"] * (1.0 + qsc["q"]))
    e_blob = (fine.norm(dim=1) * y16.squeeze(1)).flatten(1).sum(1) / y16.flatten(1).sum(1).clamp_min(1)
    w_fblob = stop_pick(gt_peak, lo, hi, e_blob.numpy())

    plans = {
        "G": fine,
        "pair-mean (symmetric consensus)": mix,
        "STOP: max projected box area": mix_pairs(fine, lo, hi, w_area),
        "STOP: unoccluded centre-ray": mix_pairs(fine, lo, hi, w_occ),
        "STOP: not truncated": mix_pairs(fine, lo, hi, w_trunc),
        "STOP: closer object": mix_pairs(fine, lo, hi, w_dist),
        "STOP: V_GT area*(1-occ)*(1-trunc/2)/dist": mix_pairs(fine, lo, hi, w_vis),
        "STOP: box nearer image centre": mix_pairs(fine, lo, hi, w_cen),
        "STOP: more imminent (1/ttc)": mix_pairs(fine, lo, hi, w_ttc),
        "STOP: mean V_GT over 16-frame history": mix_pairs(fine, lo, hi, w_hist),
        "STOP: V_GT at 1s target frame": mix_pairs(fine, lo, hi, w_fut),
        "reliability mix V_GT (all pairs)": mix_pairs(fine, lo, hi, w_vsoft),
        "STOP: reliability mix V_GT": mix_pairs(fine, lo, hi, torch.from_numpy(w_vsoft_s)),
        "elementwise min-|F| (conservative)": f_minabs,
        "elementwise max-|F| (liberal)": f_maxabs,
        "mean then zero 25% most pair-varying channels": f_clipch,
        "per-cell L2 winner": f_l2,
        "preserve: keep own F on GT blob, mean elsewhere": f_keep,
        "preserve: keep own F on dilated blob, mean elsewhere": f_keep_d,
        "preserve: max-|F| on GT blob, mean elsewhere": f_thr_max,
        "preserve: max-|F| on dilated blob, mean elsewhere": f_thr_max_d,
        "STOP: RGB contrast in box": mix_pairs(fine, lo, hi, w_con),
        "STOP: RGB edge strength in box": mix_pairs(fine, lo, hi, w_edg),
        "STOP: RGB sharpness (laplacian var) in box": mix_pairs(fine, lo, hi, w_shp),
        "STOP: fewer saturated pixels in box": mix_pairs(fine, lo, hi, w_glr),
        "STOP: object-vs-background RGB contrast": mix_pairs(fine, lo, hi, w_obg),
        "STOP: composite RGB Q": mix_pairs(fine, lo, hi, w_q),
        "STOP: reliability mix RGB Q": mix_pairs(fine, lo, hi, torch.from_numpy(w_qsoft)),
        "STOP: V_GT * RGB Q": mix_pairs(fine, lo, hi, w_vq),
        "STOP: F-energy on GT blob": mix_pairs(fine, lo, hi, w_fblob),
    }

    results = []
    for name, f_star in plans.items():
        print("decode", name, flush=True)
        pred = pred_g if name == "G" else c24.decode(model, f_star, pack["z"], device)
        row = c24.score_plan(name, pred, pack, g_rep if name != "G" else None)
        results.append(row)
        print_rec(name, row, row.get("feasible_vs_g"))

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = []
    for row in results:
        h = row["headline"]
        slim.append({
            "name": row["name"],
            "stop_recall": h["stop_recall"],
            "safe_recall": h["safe_recall"],
            "pre_fork_stop_recall": h["pre_fork_stop_recall"],
            "post_fork_stop_recall": h["post_fork_stop_recall"],
            "recall_at_5fa": h["recall_at_5fa"],
            "small_blob_stop_recall": h["small_blob_stop_recall"],
            "warn_flip": row["flips"]["warn_flip"],
            "feasible_vs_g": (row.get("feasible_vs_g") or {}).get("feasible"),
        })
    path.write_text(json.dumps({"audit": audit, "results": slim}, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)
    print("\nRANK by STOP then rec5", flush=True)
    for r in sorted(slim, key=lambda x: (x["stop_recall"], x["recall_at_5fa"]), reverse=True):
        print(
            f"  {r['stop_recall']:.3f} rec5={r['recall_at_5fa']:.3f} SAFE={r['safe_recall']:.3f} "
            f"pre={r['pre_fork_stop_recall']:.3f}  {r['name']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
