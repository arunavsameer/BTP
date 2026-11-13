#!/usr/bin/env python3
"""V1–V3 checks, A4/A5 Z vs token probes, B1 latency, B3 Bayes-oracle baselines.

Train probes on train Z (or pooled tokens), test on val. Soft group-mean labels.
Metrics are split pre-fork / post-fork. False alerts use hard negatives (miss / not-within-h).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.cache import TokenCache, token_fingerprint  # noqa: E402
from collision_jepa3.evaluation.judge import recall_at_fa_capped  # noqa: E402
from collision_jepa3.evaluation.labels import DEFAULT_WARNING, row_scalars, side_from_heatmap  # noqa: E402


SIDES = ("LEFT", "CENTER", "RIGHT")
HORIZONS = (1, 2, 3, 4)
FA_RATES = (0.05, 0.10)
CACHE_ANCHORS = list(range(90, 406, 15))


def load_rows(dump: Path) -> list[dict]:
    return [json.loads(l) for l in (dump / "index.jsonl").open()]


def load_gt(dump: Path) -> dict[str, np.ndarray]:
    z = np.load(dump / "gt_heatmaps.npz")
    return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}


def load_pred(dump: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(dump / f"pred_{name}.npz")
    return np.asarray(z["H"], dtype=np.float32), np.asarray(z["Z"], dtype=np.float32)


def subset_cache_stride(rows: list[dict], *arrays: np.ndarray):
    idx = np.array([i for i, r in enumerate(rows) if int(r["frame"]) in CACHE_ANCHORS], dtype=int)
    rows_s = [rows[i] for i in idx]
    outs = [arr[idx] for arr in arrays]
    return (rows_s, *outs, idx)


def bayes_soft(rows: list[dict], values: np.ndarray) -> np.ndarray:
    """Soft label / Bayes-oracle score.

    Pre-fork RGB is identical for collision/miss mates, so the score is the
    group mean (0.5 inside the event window, 0 outside). Post-fork histories
    have diverged, so the score is the clip's own hard label (0 or 1).
    """
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["pre_fork"]:
            buckets[(r["outcome_group_id"], int(r["frame"]))].append(i)
    if values.ndim == 1:
        for ids in buckets.values():
            m = float(np.mean(values[ids]))
            for i in ids:
                out[i] = m
    else:
        for ids in buckets.values():
            m = values[ids].mean(axis=0)
            for i in ids:
                out[i] = m
    return out


def cache_grid_rows(rows: list[dict], frames: list[int]) -> list[dict]:
    """Rebuild one labeled row per clip × cache-stride frame (not dump stride)."""
    clips: dict[str, dict] = {}
    for r in rows:
        clips.setdefault(r["clip_id"], r)
    out: list[dict] = []
    for r in clips.values():
        amb = {"1s": bool(r.get("amb_1s")), "1_5s": bool(r.get("amb_1_5s")), "2s": bool(r.get("amb_2s"))}
        for f in frames:
            out.append(
                row_scalars(
                    clip_id=r["clip_id"],
                    sample_id=r["sample_id"],
                    frame=int(f),
                    fps=float(r["fps"]),
                    outcome_type=r["outcome_raw"],
                    fork_frame=int(r["fork_frame"]),
                    hit_s=r.get("hit_s"),
                    scenario_type=r["scenario_type"],
                    true_side=r["true_side"],
                    background_id=r["background_id"],
                    background_pair_id=r["background_pair_id"],
                    outcome_group_id=r["outcome_group_id"],
                    amb=amb,
                )
            )
    return out


def in_stage_window(rows: list[dict], horizon_s: float) -> np.ndarray:
    """True when this scene-stage would be within `horizon_s` of hit_s (fixed 14.35 s)."""
    out = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        hit = r.get("hit_s")
        if hit is None:
            continue
        tti = float(hit) - int(r["frame"]) / float(r["fps"])
        out[i] = 0.0 < tti <= float(horizon_s)
    return out


def v1_v2_v3(rows: list[dict], gt_now: np.ndarray, gt_1s: np.ndarray | None = None) -> dict:
    clips = {}
    for r in rows:
        clips.setdefault(r["clip_id"], r)
    col = [r for r in clips.values() if r["outcome"] == "collision"]
    lead = np.array([(int(r["collision_frame"]) - int(r["fork_frame"])) / float(r["fps"]) for r in col], dtype=np.float64)
    bins = np.round(np.arange(0.0, 5.01, 0.1), 4)
    counts, edges = np.histogram(lead, bins=bins)
    hist = {f"[{edges[i]:.1f},{edges[i+1]:.1f})": int(counts[i]) for i in range(len(counts)) if counts[i]}
    n4 = sum(bool(r["collision_within_4s"]) for r in rows)
    n4_pre = sum(bool(r["collision_within_4s"] and r["pre_fork"]) for r in rows)

    def _dir_agree(maps: np.ndarray) -> dict:
        agree = n_pf = 0
        cm: dict[str, dict[str, int]] = {a: {b: 0 for b in SIDES} for a in SIDES}
        for i, r in enumerate(rows):
            if r["pre_fork"] or r["outcome"] != "collision":
                continue
            d = side_from_heatmap(maps[i], DEFAULT_WARNING)
            n_pf += 1
            agree += int(d == r["true_side"])
            if d in cm and r["true_side"] in cm[d]:
                cm[d][r["true_side"]] += 1
        return {"n": n_pf, "agree": agree, "accuracy": agree / n_pf if n_pf else None, "confusion_heatmap_row_true_col": cm}

    now_block = _dir_agree(gt_now)

    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in clips.values():
        key = "collision" if r["outcome"] == "collision" else r["outcome_raw"]
        groups[r["outcome_group_id"]][key] = r
    n_miss = differ = 0
    pair_counts: Counter = Counter()
    for g in groups.values():
        c = g.get("collision")
        misses = [v for k, v in g.items() if k != "collision"]
        if not c or not misses:
            continue
        for m in misses:
            n_miss += 1
            pair_counts[(c["true_side"], m.get("miss_side"))] += 1
            differ += int(m.get("miss_side") != c["true_side"])

    samples = sorted({r["sample_id"] for r in clips.values()})
    per_type = defaultdict(set)
    for r in clips.values():
        per_type[r["scenario_type"]].add(r["sample_id"])
    return {
        "V1_fork_to_impact": {
            "n_collision_clips": len(col),
            "lead_s_unique": sorted(set(round(float(x), 6) for x in lead)),
            "lead_s_seconds": float(lead[0]) if len(lead) else None,
            "lead_frames": int(col[0]["collision_frame"]) - int(col[0]["fork_frame"]) if col else None,
            "histogram_0.1s": hist,
            "collision_within_4s": {"n": n4, "pre_fork": n4_pre, "post_fork": n4 - n4_pre, "pre_fork_frac": n4_pre / n4 if n4 else None},
            "inference_check": (
                "Yes: fork is exactly 65 frames = 2.1667 s before impact on every collision clip. "
                "Half of the 4 s event-positive anchors are still pre-fork (coin flip). "
                "Usable distinguishing lead is 2.17 s minus latency, not 4 s."
            ),
        },
        "V2_direction": {
            "gt_now_vs_true_side_postfork_collision": now_block,
            "gt_1s_vs_true_side_postfork_collision": _dir_agree(gt_1s) if gt_1s is not None else None,
            "miss_side_vs_paired_collision_true_side": {
                "n_pairs": n_miss, "n_differ": differ, "frac_differ": differ / n_miss if n_miss else None,
                "pair_counts_true_side_miss_side": {f"{a}|{b}": int(c) for (a, b), c in pair_counts.items()},
            },
            "recommendation": (
                "miss_side is the miss outcome, not the approach side: it differs from the paired "
                "collision true_side on ~75% of pairs. true_side is constant inside an RGB group, "
                "so a group-mean of true_side is just true_side. GT_now direction is only ~61% "
                "aligned with true_side even post-fork, so a heatmap-peak direction head is not "
                "a true_side head. Pre-fork, if the target is miss_side, use a group mean; if the "
                "target is approach true_side, it is already shared."
            ),
        },
        "V3_val_units": {
            "n_clips": len(clips),
            "n_unique_sample_id": len(samples),
            "clips_per_sample": dict(Counter(Counter(r["sample_id"] for r in clips.values()).values())),
            "n_scenario_types": len(per_type),
            "unique_samples_per_scenario_type": {k: len(v) for k, v in sorted(per_type.items())},
            "note": "128 clips = 32 samples × 2 backgrounds × 2 outcomes. Bootstrap by sample_id. Per-type slices have 1–2 samples.",
        },
    }


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s, dtype=np.float64)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _rec_at_fa(y: np.ndarray, s: np.ndarray, fa: float) -> dict:
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if y.size == 0 or not (y.min() < y.max()):
        return {"recall": float("nan"), "fa": float("nan"), "feasible": False}
    return recall_at_fa_capped(s, y, fa_rate=fa, hi=0.5)


def fit_soft_probe(x_tr: np.ndarray, p_tr: np.ndarray, x_va: np.ndarray) -> np.ndarray:
    """Ridge on the soft probability. Ranking score for AUC / recall@FA."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    p = np.clip(np.asarray(p_tr, dtype=np.float64), 0.0, 1.0)
    scaler = StandardScaler()
    m = Ridge(alpha=1.0)
    m.fit(scaler.fit_transform(x_tr), p)
    return m.predict(scaler.transform(x_va))


def fit_multiclass(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.fit_transform(x_tr), y_tr)
    return clf.predict(scaler.transform(x_va))


def fit_ridge(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    m = Ridge(alpha=1.0)
    m.fit(scaler.fit_transform(x_tr), y_tr)
    return m.predict(scaler.transform(x_va))


def mask_split(rows: list[dict], pre_fork: bool | None) -> np.ndarray:
    if pre_fork is None:
        return np.ones(len(rows), dtype=bool)
    return np.array([bool(r["pre_fork"]) is bool(pre_fork) for r in rows], dtype=bool)


def _slice_report(y_hard: np.ndarray, score: np.ndarray, mask: np.ndarray) -> dict:
    yh, sc = y_hard[mask], score[mask]
    block = {"n": int(mask.sum()), "pos_frac": float(yh.mean()) if yh.size else None, "auc": _auc(yh, sc)}
    for fa in FA_RATES:
        rec = _rec_at_fa(yh, sc, fa)
        block[f"recall_at_{int(fa*100)}fa"] = rec.get("recall")
        block[f"fa_at_{int(fa*100)}fa"] = rec.get("fa")
    return block


def binary_report(y_hard: np.ndarray, score: np.ndarray, rows: list[dict], horizon_s: float | None = None) -> dict:
    out = {}
    for name, pre in (("all", None), ("pre_fork", True), ("post_fork", False)):
        out[name] = _slice_report(y_hard, score, mask_split(rows, pre))
    if horizon_s is not None:
        stage = in_stage_window(rows, horizon_s)
        out["pre_fork_in_window"] = _slice_report(y_hard, score, mask_split(rows, True) & stage)
        out["post_fork_in_window"] = _slice_report(y_hard, score, mask_split(rows, False) & stage)
    return out


def pooled_tokens(cfg, rows: list[dict]) -> np.ndarray:
    cache = TokenCache(cfg.path("paths.cache_root"), token_fingerprint(cfg, dummy=False))
    x = np.full((len(rows), 1408), np.nan, dtype=np.float32)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["clip_id"]].append(i)
    missing = 0
    for cid, ids in by.items():
        try:
            arr, mapping = cache.load_clip(cid, mmap=True)
        except FileNotFoundError:
            missing += len(ids)
            continue
        for i in ids:
            t = int(rows[i]["frame"])
            if t not in mapping:
                missing += 1
                continue
            tok = np.asarray(arr[mapping[t]], dtype=np.float32)  # [N,C]
            x[i] = tok.mean(axis=0)
    if missing:
        print(f"token pool missing {missing}/{len(rows)}", flush=True)
    return x


def flatten_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    return np.nan_to_num(z.reshape(z.shape[0], -1), nan=0.0)


MODEL_SPECS = {
    "phase_a": ("checkpoints/teacher_phase_a_best.pt", "residual"),
    "mse17": ("checkpoints/direct_h1s_mse_until/teacher_phase_a_best.pt", "direct_h1s"),
    "beta2": ("checkpoints/direct_h1s_beta2/teacher_phase_a_best.pt", "direct_h1s"),
}


def z_from_token_rows(cfg, ckpt: str, mode: str, rows: list[dict], device: str = "cpu") -> np.ndarray:
    """Adapter Z_t from cached Giant tokens (no encode)."""
    import torch

    from collision_jepa3.models.teacher import CollisionTeacher3
    from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
    from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state

    dummy = DummyVJEPA2Backbone(
        hidden_size=1408, crop_size=256, patch_size=16, tubelet_size=2, num_frames=16, num_layers=1
    )
    cfg.set("heads.mode", mode)
    model = CollisionTeacher3(cfg, backbone=dummy)
    load_teacher_state(model, load_checkpoint(ckpt, map_location="cpu"))
    model.to(device).eval()
    cache = TokenCache(cfg.path("paths.cache_root"), token_fingerprint(cfg, dummy=False))
    z = np.full((len(rows), 32, 5, 5), np.nan, dtype=np.float32)
    by: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["clip_id"]].append(i)
    with torch.no_grad():
        for cid, ids in by.items():
            try:
                arr, mapping = cache.load_clip(cid, mmap=True)
            except FileNotFoundError:
                continue
            keep, toks = [], []
            for i in ids:
                t = int(rows[i]["frame"])
                if t in mapping:
                    keep.append(i)
                    toks.append(np.asarray(arr[mapping[t]], dtype=np.float32))
            if not toks:
                continue
            x = torch.from_numpy(np.stack(toks, axis=0)).to(device)
            for start in range(0, len(keep), 64):
                sl = slice(start, start + 64)
                z[keep[sl]] = model.encode(tokens=x[sl]).float().cpu().numpy()
    del model
    return flatten_z(z)


def run_probes(train_rows, x_tr, val_rows, x_va, tag: str) -> dict:
    out = {"feature": tag, "n_train": len(train_rows), "n_val": len(val_rows), "horizons": {}, "direction": {}, "tti": {}}
    for h in HORIZONS:
        hard_tr = np.array([float(r[f"collision_within_{h}s"]) for r in train_rows])
        hard_va = np.array([float(r[f"collision_within_{h}s"]) for r in val_rows])
        soft_tr = bayes_soft(train_rows, hard_tr)
        score = fit_soft_probe(x_tr, soft_tr, x_va)
        out["horizons"][f"{h}s"] = binary_report(hard_va, score, val_rows, horizon_s=h)

    y_tr = np.array([SIDES.index(r["true_side"]) if r["true_side"] in SIDES else 1 for r in train_rows])
    y_va = np.array([SIDES.index(r["true_side"]) if r["true_side"] in SIDES else 1 for r in val_rows])
    pred = fit_multiclass(x_tr, y_tr, x_va)
    dir_block = {}
    for name, pre in (("all", None), ("pre_fork", True), ("post_fork", False)):
        m = mask_split(val_rows, pre)
        acc = float(np.mean(pred[m] == y_va[m])) if m.any() else float("nan")
        maj = float(np.mean(y_va[m] == Counter(y_va[m]).most_common(1)[0][0])) if m.any() else float("nan")
        dir_block[name] = {"n": int(m.sum()), "accuracy": acc, "majority_baseline": maj}
    out["direction"] = dir_block

    tr_c = [i for i, r in enumerate(train_rows) if r["tti_s"] is not None]
    va_c = [i for i, r in enumerate(val_rows) if r["tti_s"] is not None]
    if tr_c and va_c:
        ytr = np.array([float(train_rows[i]["tti_s"]) for i in tr_c])
        yva = np.array([float(val_rows[i]["tti_s"]) for i in va_c])
        pred_t = fit_ridge(x_tr[tr_c], ytr, x_va[va_c])
        rmse = float(np.sqrt(np.mean((pred_t - yva) ** 2)))
        # frame-only leak baseline
        ftr = np.array([float(train_rows[i]["frame"]) for i in tr_c]).reshape(-1, 1)
        fva = np.array([float(val_rows[i]["frame"]) for i in va_c]).reshape(-1, 1)
        pred_f = fit_ridge(ftr, ytr, fva)
        rmse_f = float(np.sqrt(np.mean((pred_f - yva) ** 2)))
        out["tti"] = {
            "n_val_collision": len(va_c),
            "rmse_s": rmse,
            "rmse_frame_only_baseline_s": rmse_f,
            "note": "hit_s is constant, so TTI is almost a function of frame. Frame-only RMSE is the leak ceiling.",
        }
    return out


def b3_baselines(rows: list[dict], h: int = 4) -> dict:
    y = np.array([float(r[f"collision_within_{h}s"]) for r in rows])
    soft = bayes_soft(rows, y)
    rng = np.random.RandomState(0)
    p = float(y.mean()) if y.size else 0.0
    random = rng.rand(len(rows))
    always = np.ones(len(rows))
    never = np.zeros(len(rows))
    return {
        "horizon_s": h,
        "never_alert": binary_report(y, never, rows, horizon_s=h),
        "always_alert": binary_report(y, always, rows, horizon_s=h),
        "random_score": binary_report(y, random, rows, horizon_s=h),
        "bayes_oracle_group_mean": binary_report(y, soft, rows, horizon_s=h),
        "prevalence": p,
        "note": (
            "Bayes oracle: pre-fork = mean label among identical-RGB mates "
            "(0 outside the hit_s window, 0.5 inside); post-fork = the clip's own 0/1. "
            "pre_fork_in_window is the coin-flip slice (ceiling AUC 0.5, recall@FA = 0). "
            "Post-fork ceiling is AUC 1. Always-alert cannot meet a 5%/10% FA cap "
            "(FA rate = 1 − prevalence), so recall_at_Nfa is 0 by construction. "
            "No safe clips: FA is among miss / not-within-h anchors, not per hour of walking."
        ),
    }


def b1_latency(cfg, device: str, n: int = 64) -> dict:
    import torch

    from collision_jepa3.models.teacher import CollisionTeacher3
    from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
    from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state

    dummy = DummyVJEPA2Backbone(hidden_size=1408, crop_size=256, patch_size=16, tubelet_size=2, num_frames=16, num_layers=1)
    cfg.set("heads.mode", "direct_h1s")
    model = CollisionTeacher3(cfg, backbone=dummy)
    blob = load_checkpoint("checkpoints/direct_h1s_beta2/teacher_phase_a_best.pt", map_location="cpu")
    load_teacher_state(model, blob)
    model.to(device).eval()
    tok = torch.randn(n, 2048, 1408, device=device)
    # warmup
    with torch.no_grad():
        model(tokens=tok[:1])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, n, 1):
            model(tokens=tok[i : i + 1])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt1 = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        model(tokens=tok)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dtb = (time.perf_counter() - t0) / n
    return {
        "device": device,
        "n": n,
        "adapter_plus_H1s_ms_per_window_batch1": dt1 * 1000,
        "adapter_plus_H1s_ms_per_window_batch64": dtb * 1000,
        "audio_or_haptic_delay_ms": None,
        "note": "No wearable audio/haptic device on this box. Giant encode is extra (~50–80 ms/window from the dense dump). Replace the 1.5 s cutoff with adapter time + unmeasured device delay, not 1.5 s.",
    }


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--val-dump", default="reports/dense_dump")
    parser.add_argument("--train-dump", required=True)
    parser.add_argument("--out", default="reports/dense_dump/verify_probes.json")
    parser.add_argument("--models", nargs="+", default=["phase_a", "mse17", "beta2"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_cfg(args)
    val_dump = Path(args.val_dump)
    train_dump = Path(args.train_dump)
    val_rows = load_rows(val_dump)
    train_rows = load_rows(train_dump)
    gt = load_gt(val_dump)
    report = {"checks": v1_v2_v3(val_rows, gt["gt_now"], gt.get("gt_1s"))}

    # Train dump is already the full token-cache grid (stride 15).
    # Val dump is dense stride 6. A5 can only see cache frames, so rebuild that grid
    # (22 frames/clip, including 405 → h=1 s positives). A4 is scored on both grids.
    val_cache_rows = cache_grid_rows(val_rows, CACHE_ANCHORS)
    print(
        f"grids train={len(train_rows)} val_dense={len(val_rows)} val_cache={len(val_cache_rows)}",
        flush=True,
    )

    x_tok_tr = pooled_tokens(cfg, train_rows)
    x_tok_va = pooled_tokens(cfg, val_cache_rows)
    tok_ok_tr = np.isfinite(x_tok_tr).all(axis=1)
    tok_ok_va = np.isfinite(x_tok_va).all(axis=1)
    print(f"A5 token probes train_ok={int(tok_ok_tr.sum())} val_ok={int(tok_ok_va.sum())}", flush=True)
    report["A5_pooled_tokens"] = run_probes(
        [train_rows[i] for i in np.where(tok_ok_tr)[0]],
        x_tok_tr[tok_ok_tr],
        [val_cache_rows[i] for i in np.where(tok_ok_va)[0]],
        x_tok_va[tok_ok_va],
        "mean_pooled_tokens_1408",
    )
    report["A5_pooled_tokens"]["grid"] = "token_cache_stride15"

    report["A4_Z_probes"] = {}
    report["A4_Z_probes_cache_grid"] = {}
    for name in args.models:
        ckpt, mode = MODEL_SPECS[name]
        print(f"A4 {name} dense val", flush=True)
        _h_tr, z_tr = load_pred(train_dump, name)
        _h_va, z_va = load_pred(val_dump, name)
        z_tr_f = flatten_z(z_tr)
        z_va_f = flatten_z(z_va)
        ok_tr = np.isfinite(z_tr_f).all(axis=1)
        ok_va = np.isfinite(z_va_f).all(axis=1)
        report["A4_Z_probes"][name] = run_probes(
            [train_rows[i] for i in np.where(ok_tr)[0]],
            z_tr_f[ok_tr],
            [val_rows[i] for i in np.where(ok_va)[0]],
            z_va_f[ok_va],
            f"Z_{name}_800",
        )
        report["A4_Z_probes"][name]["grid"] = "train_s15_val_dense_s6"

        print(f"A4 {name} cache-grid val (adapter on tokens)", flush=True)
        z_va_c = z_from_token_rows(cfg, ckpt, mode, val_cache_rows, device="cpu")
        report["A4_Z_probes_cache_grid"][name] = run_probes(
            [train_rows[i] for i in np.where(ok_tr)[0]],
            z_tr_f[ok_tr],
            [val_cache_rows[i] for i in np.where(tok_ok_va)[0]],
            z_va_c[tok_ok_va],
            f"Z_{name}_800_cache_grid",
        )
        report["A4_Z_probes_cache_grid"][name]["grid"] = "token_cache_stride15"

    print("B3", flush=True)
    report["B3_baselines"] = b3_baselines(val_rows, h=4)
    report["B3_baselines"]["grid"] = "val_dense_s6"
    print("B1", flush=True)
    lat = {"cpu": b1_latency(cfg, "cpu", n=32)}
    cuda_dev = str(args.device) if str(args.device).startswith("cuda") else "cuda:1"
    try:
        import torch

        if torch.cuda.is_available():
            lat["cuda"] = b1_latency(cfg, cuda_dev, n=32)
    except Exception as exc:
        lat["cuda_error"] = str(exc)
    report["B1_latency"] = lat

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    def _j(o):
        if isinstance(o, dict):
            return {str(k): _j(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_j(v) for v in o]
        if isinstance(o, (np.floating, float)):
            x = float(o)
            return None if not np.isfinite(x) else x
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o
    out.write_text(json.dumps(_j(report), indent=2), encoding="utf-8")
    print("wrote", out)
    print(json.dumps(_j(report["checks"]), indent=2))


if __name__ == "__main__":
    main()
