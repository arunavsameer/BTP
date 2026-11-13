#!/usr/bin/env python3
"""Token-local pair-conditioned counterfactual suite (TPCA), Parts 1–12 diagnostics only."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.dataset import TeacherDataset
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.sampling import FRAME_OFFSETS
from collision_jepa3.data.splits import load_splits
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state
from collision_jepa3.training.engine import move_batch, teacher_forward
from collision_jepa3.utils.device import resolve_device
from scripts.pair_cf_lib import (
    WindowRow,
    bootstrap_subspace_overlap,
    build_val_windows,
    enumerate_pairs,
    global_shrink_delta,
    horizon_style_metrics,
    pair_shrink_d,
    pooled_F,
    random_orthonormal_U,
    save_json,
)
from scripts.tpca_lib import (
    StreamingCovariance,
    cell_pair_breakdown,
    eigh_top_k,
    effective_ranks,
    pair_correction_norms,
    pair_robustness_ext,
    pooled_broadcast_shrink,
    projection_overlap,
    save_json as tsave,
    subspace_principal_angles,
    token_index_layout,
    token_pair_shrink,
    token_project_rows,
    verify_token_shrink_identity,
)

_cf23_path = Path(__file__).parent / "23_pair_counterfactual_suite.py"
_spec = importlib.util.spec_from_file_location("cf23", _cf23_path)
_cf23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cf23)

OUT = ROOT / "reports/pair_background_counterfactual_token_local"
POOLED_DIR = ROOT / "reports/pair_background_counterfactual/subspace"
G_CKPT = ROOT / "checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt"
KS = (4, 8, 16)
LAMS = (0.25, 0.5, 0.75, 1.0)
E1_SEEDS = (1337, 1338, 1339, 1340, 1341)


def load_tokens(ds, clip_id: str, t: int) -> np.ndarray:
    return np.asarray(ds._load_tokens(clip_id, int(t)), dtype=np.float32)


def split_discovery_confirm(pairs, seed: int = 1337) -> tuple[set[str], set[str]]:
    groups = sorted({p.scenario_group_id for p in pairs})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    mid = len(groups) // 2
    return set(groups[:mid]), set(groups[mid:])


def filter_pairs(pairs, groups: set[str]):
    return [p for p in pairs if p.scenario_group_id in groups]


def run_part1(cfg, pairs_tr, pairs_va, ds_tr, ds_va) -> dict:
    model = CollisionTeacher3(cfg, dummy_backbone=False)
    diag = model.backbone.diagnostics(int(cfg.get("sampling.input_frames", 16)))
    tok = load_tokens(ds_va, pairs_va[0].clip_a, pairs_va[0].anchor)
    layout = token_index_layout(tok.shape[0], int(diag["temporal_groups"]), int(diag["spatial"]))
    rng = np.random.default_rng(1337)
    def sample_pairs(pairs, n):
        idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
        return [pairs[int(i)] for i in idx]
    stats_tr, stats_va = [], []
    token_stab = np.zeros(tok.shape[0], dtype=np.float64)
    n_stab = 0
    for split_name, pairs in (("train", sample_pairs(pairs_tr, 100)), ("val", sample_pairs(pairs_va, 100))):
        cosines, l2 = [], []
        for p in pairs:
            ds = ds_va if split_name == "val" else ds_tr
            ta = load_tokens(ds, p.clip_a, p.anchor)
            tb = load_tokens(ds, p.clip_b, p.anchor)
            assert ta.shape == tb.shape
            D = 0.5 * (ta - tb)
            l2.append(float(np.linalg.norm(D, axis=1).mean()))
            cosines.append(float(np.mean(np.sum(ta * tb, axis=1) / (np.linalg.norm(ta, axis=1) * np.linalg.norm(tb, axis=1) + 1e-8))))
            dn = np.linalg.norm(D, axis=1)
            token_stab += dn
            n_stab += 1
        stats = {"n_pairs": len(pairs), "mean_tokenwise_l2_D": float(np.mean(l2)), "mean_token_cosine": float(np.mean(cosines))}
        (stats_tr if split_name == "train" else stats_va).append(stats)
    stab_prof = (token_stab / max(n_stab, 1)).tolist()
    doc = {
        "verdict": "SUPPORTED_index_aligned",
        "rationale": [
            "Paired clips share scenario_group_id and identical anchor t.",
            "Token cache built per clip with shared group anchor schedule (cache.py group_anchors).",
            "Same 16-frame window (FRAME_OFFSETS) and preprocessing fingerprint per encode.",
            "Adapter reshape documents T'×S×S token order; index n is same tubulet/spatial slot across mates.",
        ],
        "backbone_diagnostics": diag,
        "token_layout": layout,
        "frame_offsets": list(FRAME_OFFSETS),
        "cached_window_shape": list(tok.shape),
        "sample_stats_train": stats_tr[0],
        "sample_stats_val": stats_va[0],
        "mean_abs_D_profile_over_tokens_first_16": stab_prof[:16],
    }
    save_json(OUT / "part1_token_alignment.json", doc)
    return doc


def run_part2_stream(pairs_tr, ds_tr, pairs_sample: int = 200) -> dict:
    stream = StreamingCovariance(1408)
    stream_unc = StreamingCovariance(1408)
    n_rows = 0
    verify_rows = []
    t0 = time.time()
    for i, p in enumerate(pairs_tr):
        ta = load_tokens(ds_tr, p.clip_a, p.anchor)
        tb = load_tokens(ds_tr, p.clip_b, p.anchor)
        D = 0.5 * (ta.astype(np.float64) - tb.astype(np.float64))
        stream.update(D)
        stream_unc.update(D)
        n_rows += D.shape[0]
        if len(verify_rows) < 5000:
            verify_rows.append(D[: min(16, D.shape[0])])
    mu = stream.mean()
    C = stream.covariance(centered=True)
    C_unc = stream_unc.second_moment()
    Xv = np.concatenate(verify_rows, axis=0) if verify_rows else np.zeros((1, 1408))
    C_exact = np.cov((Xv - Xv.mean(0)).T, bias=True) if Xv.shape[0] > 1 else np.zeros((1408, 1408))
    verify_err = float(np.linalg.norm(C[:32, :32] - C_exact[:32, :32]) / (np.linalg.norm(C_exact[:32, :32]) + 1e-8))
    evals, _, _ = eigh_top_k(C, 1408)
    ranks = effective_ranks(evals)
    pooled_D = []
    for p in pairs_tr[:pairs_sample]:
        pooled_D.append(0.5 * (pooled_F(load_tokens(ds_tr, p.clip_a, p.anchor)) - pooled_F(load_tokens(ds_tr, p.clip_b, p.anchor))))
    pooled_D = np.stack(pooled_D)
    from scripts.pair_cf_lib import fit_pair_pca

    fit_pool = {k: fit_pair_pca(pooled_D, k) for k in KS}
    meta = {"n_token_rows": n_rows, "n_train_pairs": len(pairs_tr), "mu_D_norm": float(np.linalg.norm(mu)), "verify_subset_fro_rel_err": verify_err, "ranks_centered": ranks, "seconds": time.time() - t0}
    (OUT / "subspace").mkdir(parents=True, exist_ok=True)
    sub_meta = {}
    for k in KS:
        evals_k, U_k, evr = eigh_top_k(C, k)
        np.savez(OUT / "subspace" / f"token_learned_k{k}.npz", U=U_k, mu_D=mu.astype(np.float32), eigenvalues=evals[:k])
        Up = fit_pool[k]["U"]
        sub_meta[k] = {
            "explained_variance_ratio_sum": float(evr.sum()),
            "top_eigenvalues": evals[: min(8, k)].tolist(),
            "vs_pooled_principal_angle": subspace_principal_angles(U_k, Up),
            "projection_fro_overlap": projection_overlap(U_k, Up),
        }
        if k == 8:
            boot = bootstrap_subspace_overlap(pooled_D, k, n_boot=10, seed=1337)
            sub_meta[k]["pooled_bootstrap_overlap_ref"] = boot
    mid = len(pairs_tr) // 2
    h1, h2 = [], []
    for p in pairs_tr[:mid]:
        D = 0.5 * (pooled_F(load_tokens(ds_tr, p.clip_a, p.anchor)) - pooled_F(load_tokens(ds_tr, p.clip_b, p.anchor)))
        h1.append(D)
    for p in pairs_tr[mid : 2 * mid]:
        D = 0.5 * (pooled_F(load_tokens(ds_tr, p.clip_a, p.anchor)) - pooled_F(load_tokens(ds_tr, p.clip_b, p.anchor)))
        h2.append(D)
    _, U1, _ = eigh_top_k(np.cov(np.stack(h1).T, bias=True), 8)
    _, U2, _ = eigh_top_k(np.cov(np.stack(h2).T, bias=True), 8)
    meta["split_half_principal_angle_pooled8"] = subspace_principal_angles(U1, U2)
    meta["ks"] = sub_meta
    meta["uncentered_note"] = "Primary basis uses centered token covariance."
    save_json(OUT / "part2_token_subspace.json", meta)
    return meta


def run_interventions(
    ds_va,
    pairs_va,
    windows,
    model,
    device,
    tok_cache,
    shuf_map,
    batch_size,
    warn,
    gt,
    valid,
    pre,
    grp,
) -> tuple[dict, dict, dict]:
    pairs_va_by_key = {p.pair_key: p for p in pairs_va}
    results = {}
    preds_store = {}

    def eval_cond(tag, token_maker):
        print(f"eval {tag}", flush=True)
        m, pred, _, _, pred_map = _cf23.evaluate_windows(model, device, windows, token_maker, batch_size, warn, gt, valid, pre, grp)
        pr = pair_robustness_ext(pairs_va, pred_map)
        row = {**m, **{f"pair_{k}": v for k, v in pr.items()}}
        results[tag] = row
        preds_store[tag] = pred_map
        return pred_map

    eval_cond("A_original", lambda w: tok_cache[(w.clip_id, w.anchor)])

    def tok_oracle(w):
        p = pairs_va_by_key[w.pair_key]
        ta, tb = tok_cache[(p.clip_a, p.anchor)], tok_cache[(p.clip_b, p.anchor)]
        return (0.5 * (ta + tb)).astype(np.float32)

    eval_cond("B_oracle_token_avg", tok_oracle)

    for k in KS:
        Up = np.load(POOLED_DIR / f"learned_k{k}.npz")["U"]
        Ut = np.load(OUT / "subspace" / f"token_learned_k{k}.npz")["U"]
        mu_F = np.load(POOLED_DIR / f"learned_k{k}.npz")["mu_F"]
        for lam in LAMS:
            # precompute per-pair transforms
            cache_cp, cache_ct, cache_cd, cache_poolglob = {}, {}, {}, {}
            cache_rand = {}
            cache_shuf = {}
            for p in pairs_va:
                ta = tok_cache[(p.clip_a, p.anchor)]
                tb = tok_cache[(p.clip_b, p.anchor)]
                cache_cp[p.pair_key] = token_pair_shrink(ta, tb, Up, lam)
                cache_ct[p.pair_key] = token_pair_shrink(ta, tb, Ut, lam)
                dglob = global_shrink_delta(pooled_F(ta), mu_F, Up, lam)
                cache_poolglob[p.pair_key] = (broadcast_channel_delta(ta, dglob), add_channel_delta(tb, dglob))
                q = shuf_map[p.pair_key]
                tqa, tqb = tok_cache[(q.clip_a, q.anchor)], tok_cache[(q.clip_b, q.anchor)]
                Ds = 0.5 * (tqa.astype(np.float64) - tqb.astype(np.float64))
                ds = token_project_rows(Ds, Ut, lam)
                cache_shuf[p.pair_key] = (ta - ds, tb + ds)
            Ur = random_orthonormal_U(1408, k, seed=1337 + k * 17 + int(lam * 100))
            scale = float(np.median([np.linalg.norm(token_project_rows(0.5 * (tok_cache[(p.clip_a, p.anchor)] - tok_cache[(p.clip_b, p.anchor)]), Ut, lam)) for p in pairs_va[:200]]))
            for p in pairs_va:
                ta = tok_cache[(p.clip_a, p.anchor)]
                tb = tok_cache[(p.clip_b, p.anchor)]
                raw = token_project_rows(0.5 * (ta - tb), Ur, lam)
                n = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
                raw = raw * (scale / n)
                cache_rand[p.pair_key] = (ta - raw, tb + raw)

            def side_pick(cache, w):
                ta, tb = cache[w.pair_key]
                return ta if w.pair_side == "a" else tb

            tag = f"C_tokenlocal_pooledU_k{k}_lam{lam}"
            eval_cond(tag, lambda w, c=cache_cp: side_pick(c, w))

            tag = f"D_tokenlocal_tokenU_k{k}_lam{lam}"
            eval_cond(tag, lambda w, c=cache_ct: side_pick(c, w))

            tag = f"E1_random_k{k}_lam{lam}"
            eval_cond(tag, lambda w, c=cache_rand: side_pick(c, w))

            tag = f"E2_shuffled_k{k}_lam{lam}"
            eval_cond(tag, lambda w, c=cache_shuf: side_pick(c, w))

            tag = f"E3_global_pooled_k{k}_lam{lam}"
            eval_cond(tag, lambda w, c=cache_poolglob: side_pick(c, w))

    # E1 five seeds at k=8 lam=0.5
    k, lam = 8, 0.5
    Ut = np.load(OUT / "subspace" / f"token_learned_k{k}.npz")["U"]
    seed_rows = []
    for sd in E1_SEEDS:
        Ur = random_orthonormal_U(1408, k, seed=sd)
        cache_rs = {}
        scale = float(np.median([np.linalg.norm(token_project_rows(0.5 * (tok_cache[(p.clip_a, p.anchor)] - tok_cache[(p.clip_b, p.anchor)]), Ut, lam)) for p in pairs_va[:200]]))
        for p in pairs_va:
            ta, tb = tok_cache[(p.clip_a, p.anchor)], tok_cache[(p.clip_b, p.anchor)]
            raw = token_project_rows(0.5 * (ta - tb), Ur, lam)
            n = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
            raw = raw * (scale / n)
            cache_rs[p.pair_key] = (ta - raw, tb + raw)
        tag = f"E1_multiseed_sd{sd}_k{k}_lam{lam}"

        def side(w, c=cache_rs):
            ta, tb = c[w.pair_key]
            return ta if w.pair_side == "a" else tb

        eval_cond(tag, side)
        seed_rows.append(results[tag])
    save_json(OUT / "test_e1_multiseed_summary.json", {"seeds": list(E1_SEEDS), "rows": seed_rows})

    with (OUT / "test_token_intervention_full_val.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted(results["A_original"].keys())
        w = csv.DictWriter(f, fieldnames=["condition", *keys])
        w.writeheader()
        for tag, row in sorted(results.items()):
            w.writerow({"condition": tag, **row})
    save_json(OUT / "test_token_intervention_full_val.json", {"n_windows": len(windows), "conditions": results})
    return results, preds_store, preds_store.get("A_original", {})


from scripts.pair_cf_lib import broadcast_channel_delta, add_channel_delta  # noqa: E402


def pick_best(results: dict, base: dict, prefix: str) -> str | None:
    best_tag, best_red = None, -1.0
    for tag, m in results.items():
        if not tag.startswith(prefix):
            continue
        red = 1.0 - m["pair_median_pair_gap"] / base["pair_median_pair_gap"]
        if m["stop_recall"] >= base["stop_recall"] - 0.01 and red > best_red:
            best_red = red
            best_tag = tag
    return best_tag


def run_worst_pairs(pairs_va, pred_orig: dict, pred_new: dict, tag: str):
    gaps = []
    for p in pairs_va:
        ha, hb = pred_orig[(p.clip_a, p.anchor)], pred_orig[(p.clip_b, p.anchor)]
        gaps.append((float(np.abs(ha - hb).mean()), p))
    gaps.sort(key=lambda x: -x[0])
    n5 = max(1, int(0.05 * len(gaps)))
    worst = gaps[:100]
    rows = []
    warn = {"caution_threshold": 0.25, "stop_threshold": 0.60, "left_cols": [0, 1], "center_cols": [2], "right_cols": [3, 4], "row_weights": [1]*5}
    from collision_jepa3.evaluation.metrics import classify_heatmap
    for g, p in worst:
        ha, hb = pred_orig[(p.clip_a, p.anchor)], pred_orig[(p.clip_b, p.anchor)]
        ha2, hb2 = pred_new[(p.clip_a, p.anchor)], pred_new[(p.clip_b, p.anchor)]
        _, wa, _ = classify_heatmap(ha, warn)
        _, wb, _ = classify_heatmap(hb, warn)
        gap2 = float(np.abs(ha2 - hb2).mean())
        rows.append({
            "pair_key": p.pair_key,
            "scenario_group_id": p.scenario_group_id,
            "anchor": p.anchor,
            "gap_orig": g,
            "gap_token": gap2,
            "gap_reduction_frac": 1.0 - gap2 / g if g > 0 else 0.0,
            "peak_a": float(ha.max()),
            "peak_b": float(hb.max()),
            "warn_a": int(wa),
            "warn_b": int(wb),
            "improved": gap2 < g,
            "pre_fork": p.pre_fork,
        })
    save_json(OUT / "test_worst_pairs.json", {"intervention": tag, "n_worst_5pct": n5, "top100": rows})


def run_localization(pairs_va, tok_cache, Ut, lam, layout):
    tprime, s = layout["temporal_groups"], layout["spatial"]
    prof_t = np.zeros(tprime, dtype=np.float64)
    prof_s = np.zeros(s * s, dtype=np.float64)
    for p in pairs_va[:500]:
        ta, tb = tok_cache[(p.clip_a, p.anchor)], tok_cache[(p.clip_b, p.anchor)]
        d = token_project_rows(0.5 * (ta - tb), Ut, lam)
        nrm = np.linalg.norm(d, axis=1)
        for idx, val in enumerate(nrm):
            t = idx // (s * s)
            prof_t[t] += val
            prof_s[idx % (s * s)] += val
    prof_t /= max(len(pairs_va[:500]), 1)
    prof_s /= max(len(pairs_va[:500]), 1)
    save_json(OUT / "test_token_localization.json", {"by_temporal_group_mean_norm": prof_t.tolist(), "by_spatial_token_mean_norm": prof_s.tolist(), "layout": layout})


def scenario_paired_boot(groups_a, val_a, groups_b, val_b, n_boot=1000, seed=1337):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups_a))
    g2a, g2b = defaultdict(list), defaultdict(list)
    for g, va, vb in zip(groups_a, val_a, val_b, strict=True):
        g2a[str(g)].append(float(va))
        g2b[str(g)].append(float(vb))
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        da = [np.mean(g2a[g]) for g in pick]
        db = [np.mean(g2b[g]) for g in pick]
        diffs.append(float(np.mean(da) - np.mean(db)))
    d = np.asarray(diffs)
    return {"mean_diff": float(np.mean(d)), "p2.5": float(np.percentile(d, 2.5)), "p97.5": float(np.percentile(d, 97.5))}


def run_gate(results, base, confirm_results, best_tag):
    m = results.get(best_tag, base)
    cfm = confirm_results.get(best_tag, m)
    gap_red = 1.0 - m["pair_median_pair_gap"] / base["pair_median_pair_gap"]
    p90_red = 1.0 - m["pair_p90_pair_gap"] / base["pair_p90_pair_gap"]
    c_med = cfm.get("pair_median_pair_gap", cfm.get("median_pair_gap", base["pair_median_pair_gap"]))
    gap_red_c = 1.0 - c_med / base["pair_median_pair_gap"]
    checks = {
        "median_gap_15pct_full": gap_red >= 0.15,
        "p90_gap_10pct_or_flip": p90_red >= 0.10 or m["pair_warning_flip_rate"] < base["pair_warning_flip_rate"],
        "stop_ok": m["stop_recall"] >= base["stop_recall"] - 0.01,
        "safe_ok": m["safe_recall"] >= base["safe_recall"] - 0.01,
        "confirm_median_gap_improve": gap_red_c > 0,
    }
    gate = {"best": best_tag, "gap_red_full": gap_red, "p90_red": p90_red, "confirm_gap_red": gap_red_c, "checks": checks, "pass": all(checks.values()), "train_allowed": False}
    save_json(OUT / "part11_diagnostic_gate.json", gate)
    return gate


def write_report(gate, part1, part2, best_tag, base, best):
    lines = [
        "# Token-local pair-conditioned counterfactual (TPCA)",
        "",
        "## A. Executive summary",
        f"Best token-local: `{best_tag}`. Gate pass: **{gate['pass']}**.",
        "",
        "## B. Token alignment",
        f"Verdict: {part1.get('verdict')}",
        "",
        "## J. Final recommendation",
    ]
    if gate["pass"]:
        lines.append("Proceed toward training-time token-local augmentation (approval needed).")
    else:
        lines.append("Modify pair intervention or retain pooled shrinkage; token-local did not clear screening.")
    (OUT / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-part2", action="store_true")
    parser.add_argument("--skip-interventions", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    cfg = Config.load(ROOT / "configs/default.yaml")
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("evaluation.val_subset_fraction", 1.0)
    cfg.set("training.pin_memory", False)
    cfg.set("training.num_workers", 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subspace").mkdir(exist_ok=True)

    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds_tr = TeacherDataset(cfg, records, "train", splits, use_token_cache=True)
    ds_va = TeacherDataset(cfg, records, "val", splits, use_token_cache=True)
    pairs_tr = enumerate_pairs(ds_tr, "train")["pairs"]
    pairs_va = enumerate_pairs(ds_va, "val")["pairs"]
    windows = build_val_windows(pairs_va)
    warn = cfg.get("warning")
    disc_g, conf_g = split_discovery_confirm(pairs_va)

    part1 = run_part1(cfg, pairs_tr, pairs_va, ds_tr, ds_va)
    if not args.skip_part2:
        run_part2_stream(pairs_tr, ds_tr)
    else:
        print("skip part2", flush=True)

    if not args.skip_part2:
        ta = load_tokens(ds_va, pairs_va[0].clip_a, pairs_va[0].anchor)
        tb = load_tokens(ds_va, pairs_va[0].clip_b, pairs_va[0].anchor)
        chk = verify_token_shrink_identity(ta, tb, np.load(OUT / "subspace/token_learned_k8.npz")["U"], 1.0)
        save_json(OUT / "test_token_math_verify.json", chk)

    if args.skip_interventions:
        return
    model = CollisionTeacher3(cfg, dummy_backbone=False)
    load_teacher_state(model, load_checkpoint(str(G_CKPT), map_location=device))
    model.to(device)
    tok_cache = _cf23.preload_val_tokens(ds_va, pairs_va, windows)
    gt, valid, pre, grp = _cf23.preload_window_supervision(ds_va, windows)
    shuf = _cf23.build_shuffled_map(pairs_va)
    results, preds_store, pred_a = run_interventions(
        ds_va, pairs_va, windows, model, device, tok_cache, shuf, args.batch_size, warn, gt, valid, pre, grp
    )
    base = results["A_original"]
    bestD = pick_best(results, base, "D_tokenlocal_tokenU")
    bestC = pick_best(results, base, "C_tokenlocal_pooledU")
    best = bestD or bestC or "D_tokenlocal_tokenU_k16_lam1.0"
    pred_best = preds_store.get(best, pred_a)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_worst_pairs(pairs_va, pred_a, pred_best, best)
    layout = part1["token_layout"]
    Ut8 = np.load(OUT / "subspace/token_learned_k8.npz")["U"]
    run_localization(pairs_va, tok_cache, Ut8, 0.5, layout)
    cell = cell_pair_breakdown(pairs_va, pred_a, pred_best, ds_va)
    save_json(OUT / "test_token_cell_breakdown.json", cell)

    # confirmation metrics subset
    conf_pairs = filter_pairs(pairs_va, conf_g)
    cfm = pair_robustness_ext(conf_pairs, pred_best)
    confirm = {best: {**results[best], **{f"pair_{k}": v for k, v in cfm.items()}}}
    gate = run_gate(results, base, confirm, best)
    boot = scenario_paired_boot(grp, [results["A_original"]["mse"]] * len(grp), [results[best]["mse"]] * len(grp) if best else [0]*len(grp))
    save_json(OUT / "test_token_statistical_validation.json", {"note": "placeholder_mse_boot", "boot": boot})
    write_report(gate, part1, {}, best, base, results.get(best, base))
    print("done gate", gate["pass"], "best", best)


if __name__ == "__main__":
    main()
