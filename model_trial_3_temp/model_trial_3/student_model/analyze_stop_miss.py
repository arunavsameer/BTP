#!/usr/bin/env python3
"""Compare student_lat STOP→SAFE misses vs true STOP hits on val."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.preprocessing import signature_from_cfg
from collision_jepa3.data.splits import load_splits
from collision_jepa3.evaluation.metrics import classify_heatmap
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state
from collision_jepa3.training.engine import move_batch
from collision_jepa3.utils.device import resolve_device

from student_model.dataset import StudentDataset, build_student_loader
from student_model.evaluate import student_forward
from student_model.student import CollisionStudent, load_frozen_teacher_heads
from student_model.train import build_frozen_teacher
from student_model.trainer import teacher_features

SEV = {0: "SAFE", 1: "THREAT", 2: "STOP"}
OCC = 0.5


def _bucket_scenario(name: str) -> str:
    n = str(name).lower()
    if n.startswith("jaywalker") or "child" in n or "ped" in n:
        return "person"
    if "cyclist" in n:
        return "cyclist"
    if n.startswith("cube") or n.startswith("shape"):
        return "object"
    if n.startswith("same_dir"):
        return "same_dir"
    if n.startswith("periph"):
        return "peripheral"
    if "car" in n or "swerve" in n:
        return "vehicle"
    return "other"


def _family(name: str) -> str:
    n = str(name).lower()
    for p in (
        "jaywalker",
        "child",
        "cyclist",
        "cube",
        "shape",
        "same_dir",
        "periph",
        "car",
        "swerve",
        "crossing",
    ):
        if n.startswith(p) or p in n:
            return p
    return n


def _row_band(r: int) -> str:
    if r <= 1:
        return "far"
    if r == 2:
        return "mid"
    return "near"


def _col_band(c: int) -> str:
    if c <= 1:
        return "left"
    if c == 2:
        return "center"
    return "right"


def feat_row(h_now, h_1s, pred, warn) -> dict:
    gt_dir, gt_sev, _ = classify_heatmap(h_1s, warn)
    p_dir, p_sev, _ = classify_heatmap(pred, warn)
    occ_now = h_now >= OCC
    occ_1s = h_1s >= OCC
    n_now = int(occ_now.sum())
    n_1s = int(occ_1s.sum())
    appear = occ_1s & ~occ_now
    disappear = occ_now & ~occ_1s
    persist = occ_now & occ_1s
    flat = int(np.argmax(h_1s))
    r, c = divmod(flat, 5)
    return {
        "gt": SEV[gt_sev],
        "pred": SEV[p_sev],
        "gt_dir": gt_dir,
        "pred_dir": p_dir,
        "gt_peak": float(h_1s.max()),
        "pred_peak": float(pred.max()),
        "now_peak": float(h_now.max()),
        "n_occ_1s": n_1s,
        "n_occ_now": n_now,
        "n_appear": int(appear.sum()),
        "n_disappear": int(disappear.sum()),
        "n_persist": int(persist.sum()),
        "empty_now": n_now == 0,
        "object_absent_now": float(h_now.max()) < 0.25,
        "small_blob": n_1s <= 2,
        "tiny_blob": n_1s == 1,
        "row": int(r),
        "col": int(c),
        "row_band": _row_band(int(r)),
        "col_band": _col_band(int(c)),
        "map_l1": float(np.abs(h_1s - h_now).mean()),
        "peak_delta": float(h_1s.max() - h_now.max()),
        "growing": float(h_1s.max() - h_now.max()) >= 0.20,
        "shrinking": float(h_now.max() - h_1s.max()) >= 0.20,
        "steady_peak": abs(float(h_1s.max() - h_now.max())) < 0.10,
    }


@torch.no_grad()
def main() -> None:
    cfg = Config.load()
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("cache.cache_frames", True)
    cfg.set("cache.lru_clips_in_ram", 8)
    cfg.set("evaluation.val_subset_fraction", 1.0)
    device = resolve_device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    if device == "cpu":
        cfg.set("training.pin_memory", False)
        cfg.set("training.num_workers", 0)
    warn = cfg.get("warning")
    man = cfg.path("paths.manifest")
    split_file = cfg.path("paths.split_file")
    records = load_manifest(man)
    splits = load_splits(split_file)
    ds = StudentDataset(cfg, records, "val", splits, dummy_backbone=False)
    loader = build_student_loader(ds, cfg, shuffle=False)

    g_ckpt = ROOT / "checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt"
    teacher, blob = build_frozen_teacher(cfg, g_ckpt, device)
    student = CollisionStudent(cfg)
    load_frozen_teacher_heads(student, blob)
    ck = load_checkpoint(ROOT / "checkpoints/student_lat/student_best.pt", map_location=device)
    student.encoder.load_state_dict(ck["encoder"])
    student.predictor.load_state_dict(ck["predictor"])
    if ck.get("bottleneck"):
        student.bottleneck.load_state_dict(ck["bottleneck"])
    if ck.get("norm"):
        student.norm.load_state_dict(ck["norm"])
    if ck.get("decoder"):
        student.decoder.load_state_dict(ck["decoder"])
    if ck.get("close_head"):
        student.close_head.load_state_dict(ck["close_head"])
    student.to(device).eval()
    teacher.eval()

    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        s_out = student_forward(student, batch)
        t_out = teacher_features(teacher, batch["tokens"])
        pred_s = s_out["Hhat_1s"].float().cpu().numpy()
        pred_g = t_out["Hhat_1s"].float().cpu().numpy()
        gt = batch["h_1s"].float().cpu().numpy()
        now = batch["h_now"].float().cpu().numpy()
        bsz = pred_s.shape[0]
        for i in range(bsz):
            if bool(batch["amb_1s"][i].item()):
                continue
            fs = feat_row(now[i], gt[i], pred_s[i], warn)
            _, g_sev, _ = classify_heatmap(pred_g[i], warn)
            fs.update(
                {
                    "g_pred": SEV[g_sev],
                    "g_peak": float(pred_g[i].max()),
                    "clip_id": batch["clip_id"][i],
                    "sample_id": batch["sample_id"][i],
                    "scenario": batch["scenario_type"][i],
                    "outcome": batch["outcome_type"][i],
                    "bg": batch["background_id"][i],
                    "anchor": int(batch["anchor"][i].item()),
                    "fork": int(batch["fork_frame"][i].item()),
                    "pre_fork": bool(batch["pre_fork"][i].item()),
                }
            )
            fs["family"] = _family(fs["scenario"])
            fs["object"] = _bucket_scenario(fs["scenario"])
            fs["frames_to_fork"] = fs["fork"] - fs["anchor"]
            fs["collision"] = fs["outcome"] == "collision"
            rows.append(fs)

    # consecutive GT STOP in the same clip
    by_clip = defaultdict(list)
    for i, r in enumerate(rows):
        by_clip[r["clip_id"]].append(i)
    for idxs in by_clip.values():
        idxs.sort(key=lambda i: rows[i]["anchor"])
        gt_stop = [rows[i]["gt"] == "STOP" for i in idxs]
        for k, i in enumerate(idxs):
            prev_s = gt_stop[k - 1] if k else False
            next_s = gt_stop[k + 1] if k + 1 < len(idxs) else False
            rows[i]["stop_run"] = int(prev_s) + int(gt_stop[k]) + int(next_s)
            rows[i]["isolated_stop"] = bool(gt_stop[k] and not prev_s and not next_s)
            rows[i]["run_edge"] = bool(gt_stop[k] and (not prev_s or not next_s))

    stop = [r for r in rows if r["gt"] == "STOP"]
    miss = [r for r in stop if r["pred"] == "SAFE"]
    hit = [r for r in stop if r["pred"] == "STOP"]
    threat = [r for r in stop if r["pred"] == "THREAT"]

    def rate(xs, key, pred=None):
        if pred is None:
            return float(np.mean([bool(x[key]) for x in xs])) if xs else float("nan")
        return float(np.mean([pred(x) for x in xs])) if xs else float("nan")

    def mean(xs, key):
        return float(np.mean([x[key] for x in xs])) if xs else float("nan")

    def count_key(xs, key):
        c = Counter(x[key] for x in xs)
        n = max(len(xs), 1)
        return {k: {"n": v, "frac": v / n} for k, v in c.most_common()}

    def contrast(label, miss_v, hit_v):
        return {"miss": miss_v, "hit": hit_v, "delta_miss_minus_hit": miss_v - hit_v}

    flags = [
        "pre_fork",
        "collision",
        "empty_now",
        "object_absent_now",
        "small_blob",
        "tiny_blob",
        "growing",
        "shrinking",
        "steady_peak",
        "isolated_stop",
        "run_edge",
    ]
    flag_tab = {k: contrast(k, rate(miss, k), rate(hit, k)) for k in flags}
    flag_tab["far_row"] = contrast("far_row", rate(miss, "row_band", lambda x: x["row_band"] == "far"), rate(hit, "row_band", lambda x: x["row_band"] == "far"))
    flag_tab["near_row"] = contrast("near_row", rate(miss, "row_band", lambda x: x["row_band"] == "near"), rate(hit, "row_band", lambda x: x["row_band"] == "near"))
    flag_tab["center_col"] = contrast("center", rate(miss, "col_band", lambda x: x["col_band"] == "center"), rate(hit, "col_band", lambda x: x["col_band"] == "center"))
    flag_tab["g_also_safe"] = contrast("g_safe", rate(miss, "g_pred", lambda x: x["g_pred"] == "SAFE"), rate(hit, "g_pred", lambda x: x["g_pred"] == "SAFE"))
    flag_tab["g_stop"] = contrast("g_stop", rate(miss, "g_pred", lambda x: x["g_pred"] == "STOP"), rate(hit, "g_pred", lambda x: x["g_pred"] == "STOP"))
    flag_tab["same_dir"] = contrast("same_dir", rate(miss, "object", lambda x: x["object"] == "same_dir"), rate(hit, "object", lambda x: x["object"] == "same_dir"))
    flag_tab["person"] = contrast("person", rate(miss, "object", lambda x: x["object"] == "person"), rate(hit, "object", lambda x: x["object"] == "person"))
    flag_tab["object_prim"] = contrast("object", rate(miss, "object", lambda x: x["object"] == "object"), rate(hit, "object", lambda x: x["object"] == "object"))
    flag_tab["vehicle"] = contrast("vehicle", rate(miss, "object", lambda x: x["object"] == "vehicle"), rate(hit, "object", lambda x: x["object"] == "vehicle"))
    flag_tab["peripheral"] = contrast("periph", rate(miss, "object", lambda x: x["object"] == "peripheral"), rate(hit, "object", lambda x: x["object"] == "peripheral"))
    flag_tab["early_pre"] = contrast(
        "early",
        rate(miss, "pre_fork", lambda x: x["pre_fork"] and x["frames_to_fork"] >= 60),
        rate(hit, "pre_fork", lambda x: x["pre_fork"] and x["frames_to_fork"] >= 60),
    )

    scalars = {}
    for k in ("gt_peak", "pred_peak", "now_peak", "n_occ_1s", "n_occ_now", "n_appear", "map_l1", "peak_delta", "frames_to_fork", "g_peak"):
        scalars[k] = contrast(k, mean(miss, k), mean(hit, k))

    # student-only misses: G got STOP, student said SAFE
    student_only = [r for r in miss if r["g_pred"] == "STOP"]
    both_miss = [r for r in miss if r["g_pred"] == "SAFE"]
    g_caught_as_threat = [r for r in miss if r["g_pred"] == "THREAT"]

    out = {
        "n_windows": len(rows),
        "n_gt_stop": len(stop),
        "confusion_from_stop": {
            "SAFE": len(miss),
            "THREAT": len(threat),
            "STOP": len(hit),
        },
        "frac_from_stop": {
            "SAFE": len(miss) / max(len(stop), 1),
            "THREAT": len(threat) / max(len(stop), 1),
            "STOP": len(hit) / max(len(stop), 1),
        },
        "g_on_student_safe_misses": Counter(r["g_pred"] for r in miss),
        "student_only_miss_n": len(student_only),
        "both_miss_n": len(both_miss),
        "g_threat_on_student_safe": len(g_caught_as_threat),
        "flags_miss_vs_hit": flag_tab,
        "means_miss_vs_hit": scalars,
        "miss_by_family": count_key(miss, "family"),
        "hit_by_family": count_key(hit, "family"),
        "miss_by_object": count_key(miss, "object"),
        "hit_by_object": count_key(hit, "object"),
        "miss_by_scenario": count_key(miss, "scenario"),
        "hit_by_scenario": count_key(hit, "scenario"),
        "miss_row_band": count_key(miss, "row_band"),
        "hit_row_band": count_key(hit, "row_band"),
        "miss_outcome": count_key(miss, "outcome"),
        "hit_outcome": count_key(hit, "outcome"),
        "student_only_by_family": count_key(student_only, "family"),
        "student_only_flags": {
            k: rate(student_only, k)
            for k in ("pre_fork", "empty_now", "object_absent_now", "small_blob", "isolated_stop", "growing")
        },
        "example_miss_clips": [
            {
                "clip": r["clip_id"],
                "t": r["anchor"],
                "scen": r["scenario"],
                "out": r["outcome"],
                "pre": r["pre_fork"],
                "gt_peak": round(r["gt_peak"], 3),
                "pred_peak": round(r["pred_peak"], 3),
                "now_peak": round(r["now_peak"], 3),
                "n_1s": r["n_occ_1s"],
                "n_now": r["n_occ_now"],
                "row": r["row_band"],
                "g": r["g_pred"],
                "isolated": r["isolated_stop"],
                "absent_now": r["object_absent_now"],
            }
            for r in sorted(miss, key=lambda x: x["pred_peak"])[:25]
        ],
    }
    # json-able counters
    out["g_on_student_safe_misses"] = dict(out["g_on_student_safe_misses"])
    dest = ROOT / "reports/student_lat/stop_miss_analysis.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_windows", "n_gt_stop", "confusion_from_stop", "frac_from_stop", "g_on_student_safe_misses", "student_only_miss_n", "both_miss_n")}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
