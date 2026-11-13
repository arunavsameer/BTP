#!/usr/bin/env python3
"""Pair-conditioned counterfactual intervention suite (Parts 1–7)."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from collision_jepa3.data.splits import load_splits
from collision_jepa3.evaluation.metrics import direction_scores_np
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state
from collision_jepa3.training.engine import move_batch, teacher_forward
from collision_jepa3.utils.device import resolve_device
from scripts.pair_cf_lib import (  # noqa: E402
    PairRow,
    WindowRow,
    add_channel_delta,
    bootstrap_subspace_overlap,
    broadcast_channel_delta,
    build_val_windows,
    enumerate_pairs,
    fit_pair_pca,
    global_shrink_delta,
    horizon_style_metrics,
    pair_robustness,
    pair_shrink_d,
    participation_ratio,
    pooled_F,
    proj_subspace,
    random_orthonormal_U,
    save_json,
)

OUT = ROOT / "reports/pair_background_counterfactual"
G_CKPT = ROOT / "checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt"
KS = (4, 8, 16)
LAMS = (0.25, 0.5, 0.75, 1.0)


def principal_angles(Ua: np.ndarray, Ub: np.ndarray) -> float:
    M = Ua @ Ub.T
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(np.arccos(np.clip(s, -1, 1))))


@torch.no_grad()
def forward_batches(model, tokens_list: list[np.ndarray], device: str, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(tokens_list), batch_size):
        chunk = tokens_list[i : i + batch_size]
        x = torch.from_numpy(np.stack(chunk, axis=0)).to(device)
        h = model(tokens=x)["Hhat_1s"].float().cpu().numpy()
        out.append(h)
    return np.concatenate(out, axis=0)


def load_tokens(ds: TeacherDataset, clip_id: str, t: int) -> np.ndarray:
    return ds._load_tokens(clip_id, int(t))


def run_part1(cfg) -> dict:
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    train_g, val_g = set(splits["train"]), set(splits["val"])
    overlap = train_g & val_g
    ds_tr = TeacherDataset(cfg, records, "train", splits, use_token_cache=True)
    ds_va = TeacherDataset(cfg, records, "val", splits, use_token_cache=True)
    rep_tr = enumerate_pairs(ds_tr, "train")
    rep_va = enumerate_pairs(ds_va, "val")
    tok = load_tokens(ds_va, ds_va.index[0][0], ds_va.index[0][1])
    doc = {
        "token_layout": {"shape_per_window": list(tok.shape), "description": "[n_tokens=2048, C=1408]"},
        "intervention_application": "Channel correction broadcast on axis 0 (all token positions). Pooled F=mean over tokens.",
        "pair_counts": {"train_valid_pairs": rep_tr["n_valid_pairs"], "val_valid_pairs": rep_va["n_valid_pairs"]},
        "train_val_group_overlap": len(overlap),
        "alignment": "Same anchor index and cached token row per clip_id",
    }
    save_json(OUT / "part1_conventions.json", doc)
    save_json(OUT / "part1_integrity.json", {"train": {k: v for k, v in rep_tr.items() if k != "pairs"}, "val": {k: v for k, v in rep_va.items() if k != "pairs"}})
    return {"ds_tr": ds_tr, "ds_va": ds_va, "pairs_tr": rep_tr["pairs"], "pairs_va": rep_va["pairs"], "doc": doc}


def run_part2(pairs_tr, ds_tr, ks=KS, seed: int = 1337) -> dict:
    cache: dict[tuple[str, int], np.ndarray] = {}
    Fa, Fb, D_list = [], [], []
    t0 = time.time()
    for p in pairs_tr:
        ka, kb = (p.clip_a, p.anchor), (p.clip_b, p.anchor)
        if ka not in cache:
            cache[ka] = pooled_F(load_tokens(ds_tr, *ka))
        if kb not in cache:
            cache[kb] = pooled_F(load_tokens(ds_tr, *kb))
        f1, f2 = cache[ka], cache[kb]
        Fa.append(f1)
        Fb.append(f2)
        D_list.append(0.5 * (f1 - f2))
    Fa = np.stack(Fa)
    Fb = np.stack(Fb)
    D = np.stack(D_list)
    mu_F = Fa.mean(axis=0)
    subspaces = {}
    mid = D.shape[0] // 2
    h1, h2 = D[:mid], D[mid : 2 * mid]
    fit1 = fit_pair_pca(h1, 8)
    U1 = fit1["U"]
    fit2 = fit_pair_pca(h2, 8)
    U2 = fit2["U"]
    angle_split = principal_angles(U1, U2)
    eff = participation_ratio(np.linalg.svd(D - D.mean(0), compute_uv=False)[1] ** 2 / max(D.shape[0] - 1, 1))
    for k in ks:
        fit = fit_pair_pca(D, k)
        boot = bootstrap_subspace_overlap(D, k, n_boot=20, seed=seed + k)
        U_learned = fit["U"]
        U_rand = random_orthonormal_U(D.shape[1], k, seed=seed + 1000 + k)
        np.savez(
            OUT / "subspace" / f"learned_k{k}.npz",
            U=U_learned,
            mu_F=mu_F,
            mu_d=fit["mu_d"],
            eigenvalues=fit["eigenvalues"],
            ev_ratio=fit["explained_variance_ratio"],
        )
        np.savez(OUT / "subspace" / f"random_k{k}.npz", U=U_rand, mu_F=mu_F)
        subspaces[k] = {
            "explained_variance_ratio_sum": float(fit["explained_variance_ratio"].sum()),
            "top_eigenvalues": fit["eigenvalues"][: min(8, k)].tolist(),
            "bootstrap_overlap": boot,
            "split_half_principal_angle_mean": angle_split,
        }
    meta = {
        "n_train_pairs": int(D.shape[0]),
        "D_norm_median": float(np.median(np.linalg.norm(D, axis=1))),
        "D_norm_mean": float(np.linalg.norm(D, axis=1).mean()),
        "effective_rank_D_cov": eff,
        "seconds": time.time() - t0,
        "ks": subspaces,
    }
    save_json(OUT / "part2_subspace_fit.json", meta)
    return {"D_train": D, "mu_F": mu_F, "Fa_tr": Fa, "Fb_tr": Fb}


def build_shuffled_map(pairs_va, seed: int = 1337) -> dict[str, PairRow]:
    """Map pair_key -> shuffled partner pair (different scenario_group)."""
    rng = np.random.default_rng(seed)
    bins = np.percentile([p.gt_peak for p in pairs_va], [0, 33, 66, 100])
    by: dict[tuple, list] = defaultdict(list)
    for p in pairs_va:
        b = int(np.digitize(p.gt_peak, bins[1:-1], right=True))
        by[(p.pre_fork, b)].append(p)
    out: dict[str, PairRow] = {}
    for p in pairs_va:
        pool = [q for q in by[(p.pre_fork, int(np.digitize(p.gt_peak, bins[1:-1], right=True)))] if q.scenario_group_id != p.scenario_group_id]
        if not pool:
            pool = [q for q in pairs_va if q.scenario_group_id != p.scenario_group_id]
        out[p.pair_key] = pool[int(rng.integers(0, len(pool)))]
    return out


def preload_window_supervision(ds_va, windows: list[WindowRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gts, valids, prefork, groups = [], [], [], []
    for w in windows:
        gts.append(ds_va.h1s_at_anchor(w.clip_id, w.anchor))
        amb = ds_va.ambiguity.get((w.clip_id, w.anchor), {})
        valids.append(not bool(amb.get("1s", False)))
        prefork.append(w.pre_fork)
        groups.append(w.scenario_group_id)
    return (
        np.stack(gts),
        np.asarray(valids, dtype=bool),
        np.asarray(prefork, dtype=bool),
        np.asarray(groups),
    )


def evaluate_windows(
    model,
    device,
    windows: list[WindowRow],
    token_maker,
    batch_size: int,
    warn: dict,
    gt: np.ndarray,
    valid: np.ndarray,
    pre: np.ndarray,
    grp: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """token_maker(w) -> tokens array"""
    preds: list[np.ndarray] = []
    tok_batch: list[np.ndarray] = []

    def flush():
        nonlocal tok_batch
        if not tok_batch:
            return
        h = forward_batches(model, tok_batch, device, batch_size)
        preds.append(h)
        tok_batch = []

    for w in windows:
        tok_batch.append(token_maker(w))
        if len(tok_batch) >= batch_size:
            flush()
    flush()
    pred = np.concatenate(preds, axis=0)
    vmask = np.broadcast_to(valid[:, None, None], pred.shape)
    metrics = horizon_style_metrics(pred, gt, vmask, pre, warn)
    pred_map = {(w.clip_id, w.anchor): pred[i] for i, w in enumerate(windows)}
    return metrics, pred, gt, valid, pred_map


def preload_val_tokens(ds_va, pairs_va, windows) -> dict[tuple[str, int], np.ndarray]:
    """Load each validation window token array once (heavy disk I/O)."""
    keys: set[tuple[str, int]] = {(w.clip_id, w.anchor) for w in windows}
    for p in pairs_va:
        keys.add((p.clip_a, p.anchor))
        keys.add((p.clip_b, p.anchor))
    cache: dict[tuple[str, int], np.ndarray] = {}
    t0 = time.time()
    for i, key in enumerate(sorted(keys)):
        cache[key] = load_tokens(ds_va, *key)
        if (i + 1) % 500 == 0:
            print(f"preload_val_tokens {i + 1}/{len(keys)}", flush=True)
    print(f"preload_val_tokens done n={len(cache)} sec={time.time() - t0:.1f}", flush=True)
    return cache


def run_part3_4(
    ds_va,
    pairs_va,
    windows,
    model,
    device,
    sub,
    shuf_map,
    batch_size: int,
    warn: dict,
) -> dict:
    results = {}
    csv_rows = []
    tok_cache = preload_val_tokens(ds_va, pairs_va, windows)
    gt, valid, pre, grp = preload_window_supervision(ds_va, windows)

    def tok_get(clip_id: str, anchor: int) -> np.ndarray:
        return tok_cache[(clip_id, anchor)]

    # Condition A — baseline (identity)
    def tok_id(w):
        return tok_get(w.clip_id, w.anchor)

    print("condition A_original", flush=True)
    m_a, _, _, _, pred_map_a = evaluate_windows(
        model, device, windows, tok_id, batch_size, warn, gt, valid, pre, grp
    )
    pair_m_a = pair_robustness(pairs_va, pred_map_a)
    results["A_original"] = {**m_a, **{f"pair_{k}": v for k, v in pair_m_a.items()}}
    csv_rows.append({"condition": "A_original", **results["A_original"]})

    # B oracle average
    oracle_windows: list[WindowRow] = []
    oracle_tokens: dict[str, np.ndarray] = {}
    for p in pairs_va:
        ta = tok_get(p.clip_a, p.anchor)
        tb = tok_get(p.clip_b, p.anchor)
        savg = 0.5 * (ta + tb)
        oracle_tokens[p.pair_key] = savg
    for w in windows:
        oracle_windows.append(w)
    def tok_oracle(w):
        pkey = w.pair_key
        return oracle_tokens[pkey]
    print("condition B_oracle_avg", flush=True)
    m_b, _, _, _, pred_map_b = evaluate_windows(
        model, device, windows, tok_oracle, batch_size, warn, gt, valid, pre, grp
    )
    pair_m_b = pair_robustness(pairs_va, pred_map_b)
    results["B_oracle_avg"] = {**m_b, **{f"pair_{k}": v for k, v in pair_m_b.items()}}
    csv_rows.append({"condition": "B_oracle_avg", **results["B_oracle_avg"]})

    for k in KS:
        U = np.load(OUT / "subspace" / f"learned_k{k}.npz")["U"]
        mu_F = np.load(OUT / "subspace" / f"learned_k{k}.npz")["mu_F"]
        Ur = np.load(OUT / "subspace" / f"random_k{k}.npz")["U"]
        for lam in LAMS:
            print(f"intervention k={k} lam={lam}", flush=True)
            # C global
            def make_global(w):
                t = tok_get(w.clip_id, w.anchor)
                f = pooled_F(t)
                delta = global_shrink_delta(f, mu_F, U, lam)
                return broadcast_channel_delta(t, delta)

            tag = f"C_global_k{k}_lam{lam}"
            m_c, _, _, _, pred_c = evaluate_windows(
                model, device, windows, make_global, batch_size, warn, gt, valid, pre, grp
            )
            pair_c = pair_robustness(pairs_va, pred_c)
            results[tag] = {**m_c, **{f"pair_{k}": v for k, v in pair_c.items()}}
            csv_rows.append({"condition": tag, **results[tag]})

            # D true pair shrink
            pair_d_vecs = {}
            for p in pairs_va:
                fa = pooled_F(tok_get(p.clip_a, p.anchor))
                fb = pooled_F(tok_get(p.clip_b, p.anchor))
                D = 0.5 * (fa - fb)
                pair_d_vecs[p.pair_key] = pair_shrink_d(D, U, lam)

            def make_pair(w):
                t = tok_get(w.clip_id, w.anchor)
                d = pair_d_vecs[w.pair_key]
                if w.pair_side == "a":
                    return broadcast_channel_delta(t, d)
                return add_channel_delta(t, d)

            tag = f"D_pair_shrink_k{k}_lam{lam}"
            m_d, _, _, _, pred_d = evaluate_windows(
                model, device, windows, make_pair, batch_size, warn, gt, valid, pre, grp
            )
            pair_d = pair_robustness(pairs_va, pred_d)
            results[tag] = {**m_d, **{f"pair_{k}": v for k, v in pair_d.items()}}
            csv_rows.append({"condition": tag, **results[tag]})

            # E1 random subspace — scale to match mean ||d|| on val true pairs
            scale = float(np.mean([np.linalg.norm(pair_d_vecs[p.pair_key]) for p in pairs_va[:200]]))
            pair_r = {}
            for p in pairs_va:
                fa = pooled_F(tok_get(p.clip_a, p.anchor))
                fb = pooled_F(tok_get(p.clip_b, p.anchor))
                D = 0.5 * (fa - fb)
                raw = pair_shrink_d(D, Ur, lam)
                n = np.linalg.norm(raw) + 1e-8
                pair_r[p.pair_key] = (raw * (scale / n)).astype(np.float32)

            def make_rand(w):
                t = tok_get(w.clip_id, w.anchor)
                d = pair_r[w.pair_key]
                if w.pair_side == "a":
                    return broadcast_channel_delta(t, d)
                return add_channel_delta(t, d)

            tag = f"E1_random_k{k}_lam{lam}"
            m_e1, _, _, _, pred_e1 = evaluate_windows(
                model, device, windows, make_rand, batch_size, warn, gt, valid, pre, grp
            )
            pair_e1 = pair_robustness(pairs_va, pred_e1)
            results[tag] = {**m_e1, **{f"pair_{k}": v for k, v in pair_e1.items()}}
            csv_rows.append({"condition": tag, **results[tag]})

            # E2 shuffled pair
            pair_s = {}
            for p in pairs_va:
                q = shuf_map[p.pair_key]
                fq = pooled_F(tok_get(q.clip_a, q.anchor))
                fb = pooled_F(tok_get(q.clip_b, q.anchor))
                # use shuffled pair's A/B for difference unrelated to current clip's mate
                D = 0.5 * (fq - fb)
                pair_s[p.pair_key] = pair_shrink_d(D, U, lam)

            def make_shuf(w):
                t = tok_get(w.clip_id, w.anchor)
                d = pair_s[w.pair_key]
                if w.pair_side == "a":
                    return broadcast_channel_delta(t, d)
                return add_channel_delta(t, d)

            tag = f"E2_shuffled_k{k}_lam{lam}"
            m_e2, _, _, _, pred_e2 = evaluate_windows(
                model, device, windows, make_shuf, batch_size, warn, gt, valid, pre, grp
            )
            pair_e2 = pair_robustness(pairs_va, pred_e2)
            results[tag] = {**m_e2, **{f"pair_{k}": v for k, v in pair_e2.items()}}
            csv_rows.append({"condition": tag, **results[tag]})

    save_json(OUT / "test_intervention_full_val.json", {"n_windows": len(windows), "n_pairs": len(pairs_va), "conditions": results})
    with (OUT / "test_intervention_full_val.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    return results, pred_map_a


def run_part5(ds_va, pairs_va, sub, k: int, lam: float) -> dict:
    U = np.load(OUT / "subspace" / f"learned_k{k}.npz")["U"]
    norms_o, norms_t, cosines = [], [], []
    for p in pairs_va[:500]:
        ta = load_tokens(ds_va, p.clip_a, p.anchor)
        fa, fb = pooled_F(ta), pooled_F(load_tokens(ds_va, p.clip_b, p.anchor))
        D = 0.5 * (fa - fb)
        d = pair_shrink_d(D, U, lam)
        tb = add_channel_delta(load_tokens(ds_va, p.clip_b, p.anchor), d)
        ta2 = broadcast_channel_delta(ta, d)
        fo, ft = pooled_F(ta), pooled_F(ta2)
        norms_o.append(float(np.linalg.norm(fo)))
        norms_t.append(float(np.linalg.norm(ft)))
        cosines.append(float(np.dot(fo, ft) / (np.linalg.norm(fo) * np.linalg.norm(ft) + 1e-8)))
    rep = {
        "k": k,
        "lambda": lam,
        "n_pairs_sampled": 500,
        "delta_norm_median": float(
            np.median(
                [
                    np.linalg.norm(
                        pair_shrink_d(
                            0.5
                            * (
                                pooled_F(load_tokens(ds_va, p.clip_a, p.anchor))
                                - pooled_F(load_tokens(ds_va, p.clip_b, p.anchor))
                            ),
                            U,
                            lam,
                        )
                    )
                    for p in pairs_va[:200]
                ]
            )
        ),
        "pooled_norm_mean_before": float(np.mean(norms_o)),
        "pooled_norm_mean_after": float(np.mean(norms_t)),
        "pooled_cosine_mean": float(np.mean(cosines)),
        "verdict": "reasonable" if float(np.mean(cosines)) > 0.85 else "review",
    }
    save_json(OUT / "test_feature_realism.json", rep)
    return rep


def run_part6(cfg, ds_tr, pairs_tr, device, k: int, lam: float, n_batches: int = 64) -> dict:
    U = np.load(OUT / "subspace" / f"learned_k{k}.npz")["U"]
    warn = cfg.get("warning")
    loss_fn = TeacherLoss(cfg, device=device)

    def measure_model(model, label: str) -> dict:
        model.train()
        model.backbone.eval()
        params = list(model.adapter.parameters()) + list(model.decoder.parameters())
        cosines, r_clean, r_cf = [], [], []
        rng = np.random.default_rng(1337)
        idx_order = rng.permutation(len(pairs_tr))
        n_done = 0
        for pi in idx_order:
            if n_done >= n_batches:
                break
            p = pairs_tr[int(pi)]
            ta = torch.from_numpy(load_tokens(ds_tr, p.clip_a, p.anchor)).unsqueeze(0).to(device)
            tb = torch.from_numpy(load_tokens(ds_tr, p.clip_b, p.anchor)).unsqueeze(0).to(device)
            fa = pooled_F(ta.squeeze(0).cpu().numpy())
            fb = pooled_F(tb.squeeze(0).cpu().numpy())
            D = 0.5 * (fa - fb)
            d = torch.from_numpy(pair_shrink_d(D, U, lam)).to(device)
            ta_cf = ta - d.view(1, 1, -1)
            tb_cf = tb + d.view(1, 1, -1)
            gt = torch.from_numpy(ds_tr.h1s_at_anchor(p.clip_a, p.anchor)).unsqueeze(0).to(device)
            amb = torch.zeros(1, dtype=torch.bool, device=device)
            batch = move_batch({"tokens": torch.cat([ta, tb], 0), "h_1s": gt.repeat(2, 1, 1), "amb_1s": amb.repeat(2)}, device)
            batch_cf = move_batch({"tokens": torch.cat([ta_cf, tb_cf], 0), "h_1s": gt.repeat(2, 1, 1), "amb_1s": amb.repeat(2)}, device)
            model.zero_grad(set_to_none=True)
            out_a = teacher_forward(model, move_batch({"tokens": ta, "h_1s": gt, "amb_1s": amb}, device))
            out_b = teacher_forward(model, move_batch({"tokens": tb, "h_1s": gt, "amb_1s": amb}, device))
            l_clean = 0.5 * (loss_fn(out_a, move_batch({"tokens": ta, "h_1s": gt, "amb_1s": amb}, device)).total + loss_fn(out_b, move_batch({"tokens": tb, "h_1s": gt, "amb_1s": amb}, device)).total)
            g_clean = torch.autograd.grad(l_clean, params, retain_graph=False, allow_unused=True)
            model.zero_grad(set_to_none=True)
            out_ac = teacher_forward(model, move_batch({"tokens": ta_cf, "h_1s": gt, "amb_1s": amb}, device))
            out_bc = teacher_forward(model, move_batch({"tokens": tb_cf, "h_1s": gt, "amb_1s": amb}, device))
            l_cf = 0.5 * (loss_fn(out_ac, move_batch({"tokens": ta_cf, "h_1s": gt, "amb_1s": amb}, device)).total + loss_fn(out_bc, move_batch({"tokens": tb_cf, "h_1s": gt, "amb_1s": amb}, device)).total)
            g_cf = torch.autograd.grad(l_cf, params, retain_graph=False, allow_unused=True)
            def gn(gs):
                t = 0.0
                for g in gs:
                    if g is not None:
                        t += float(g.pow(2).sum())
                return math.sqrt(t)
            nc, nf = gn(g_clean), gn(g_cf)
            dot = na = nb = 0.0
            for a, b in zip(g_clean, g_cf, strict=True):
                if a is None or b is None:
                    continue
                af, bf = a.detach().float().flatten(), b.detach().float().flatten()
                dot += float((af * bf).sum())
                na += float((af * af).sum())
                nb += float((bf * bf).sum())
            c = dot / (math.sqrt(na) * math.sqrt(nb)) if na > 0 and nb > 0 else float("nan")
            cosines.append(c)
            r_clean.append(nc)
            r_cf.append(nf)
            n_done += 1
        return {
            "label": label,
            "n_batches": n_done,
            "cosine_mean": float(np.nanmean(cosines)),
            "cosine_median": float(np.nanmedian(cosines)),
            "frac_negative_cosine": float(np.mean(np.asarray(cosines) < 0)),
            "g_clean_mean": float(np.mean(r_clean)),
            "g_cf_mean": float(np.mean(r_cf)),
            "ratio_cf_over_clean": float(np.mean(np.asarray(r_cf) / (np.asarray(r_clean) + 1e-8))),
        }

    g_model = CollisionTeacher3(cfg, dummy_backbone=False)
    load_teacher_state(g_model, load_checkpoint(str(G_CKPT), map_location=device))
    g_model.to(device)
    fresh = CollisionTeacher3(cfg, dummy_backbone=False).to(device)
    rep = {"k": k, "lambda": lam, "G_checkpoint": measure_model(g_model, "G_frozen_diag"), "scratch_random": measure_model(fresh, "scratch_random")}
    for beta in (0.05, 0.10):
        rep[f"suggested_beta_{beta}"] = {"weighted_ratio": beta * rep["scratch_random"]["ratio_cf_over_clean"]}
    save_json(OUT / "test_gradient_compatibility.json", rep)
    return rep


def _beats_controls(results: dict, best_tag: str | None, m_best: dict, base: dict) -> bool:
    if not best_tag or not best_tag.startswith("D_pair_shrink"):
        return False
    # D_pair_shrink_k8_lam0.5 -> E1_random_k8_lam0.5
    suffix = best_tag.replace("D_pair_shrink", "")
    e1 = results.get(f"E1_random{suffix}")
    e2 = results.get(f"E2_shuffled{suffix}")
    if not e1 or not e2:
        return False
    gap = m_best["pair_median_pair_gap"]
    return gap < e1["pair_median_pair_gap"] and gap < e2["pair_median_pair_gap"]


def run_gate(results: dict, realism: dict, gradients: dict) -> dict:
    base = results["A_original"]
    # pick best D by pair gap reduction with constraint on stop recall
    best = None
    best_tag = None
    for tag, m in results.items():
        if not tag.startswith("D_pair_shrink"):
            continue
        gap_red = 1.0 - m["pair_median_pair_gap"] / base["pair_median_pair_gap"]
        if m["stop_recall"] >= base["stop_recall"] - 0.01 and gap_red > 0:
            if best is None or gap_red > best:
                best = gap_red
                best_tag = tag
    # fallback: best gap reduction overall among D
    if best_tag is None:
        best_gap = -1.0
        for tag, m in results.items():
            if tag.startswith("D_pair_shrink"):
                gap_red = 1.0 - m["pair_median_pair_gap"] / base["pair_median_pair_gap"]
                if gap_red > best_gap:
                    best_gap = gap_red
                    best_tag = tag
        best = best_gap
    m_best = results.get(best_tag, base)
    checks = {
        "1_pair_gap_reduce_15pct": float(1.0 - m_best["pair_median_pair_gap"] / base["pair_median_pair_gap"]) >= 0.15,
        "2_flip_decrease": m_best["pair_warning_flip_rate"] < base["pair_warning_flip_rate"],
        "3_stop_recall": m_best["stop_recall"] >= base["stop_recall"] - 0.01,
        "4_safe_recall": m_best["safe_recall"] >= base["safe_recall"] - 0.01,
        "5_recall_at_5fa": m_best["recall_at_5fa"] >= base["recall_at_5fa"] - 0.01,
        "6_ap_high": m_best["ap_high"] >= base["ap_high"] - 0.01,
        "7_mse": m_best["mse"] <= base["mse"] + 0.001,
        "8_realism": realism.get("verdict") != "review" or realism.get("pooled_cosine_mean", 0) > 0.85,
        "9_beats_random_and_shuffled": _beats_controls(results, best_tag, m_best, base),
        "10_gradient_ok": gradients["scratch_random"]["frac_negative_cosine"] < 0.2,
    }
    passed = all(checks.values())
    gate = {
        "best_condition": best_tag,
        "baseline_A": {k: base[k] for k in ("stop_recall", "safe_recall", "recall_at_5fa", "ap_high", "mse", "pair_median_pair_gap", "pair_warning_flip_rate")},
        "best_metrics": {k: m_best[k] for k in ("stop_recall", "safe_recall", "recall_at_5fa", "ap_high", "mse", "pair_median_pair_gap", "pair_warning_flip_rate")},
        "checks": checks,
        "pass": passed,
        "train_allowed": passed,
    }
    save_json(OUT / "part7_diagnostic_gate.json", gate)
    return gate


def write_final_report(gate, part1, realism, gradients, results):
    lines = [
        "# Pair-conditioned counterfactual shrinkage — diagnostic report",
        "",
        "## A. Executive summary",
        f"Diagnostic gate pass: **{gate['pass']}**. Best condition: `{gate.get('best_condition')}`.",
        "",
        "## B. Dataset integrity",
        json.dumps(part1.get("pair_counts", {}), indent=2),
        "",
        "## C. Pair-difference subspace",
        "See `part2_subspace_fit.json`.",
        "",
        "## D. Intervention results",
        "See `test_intervention_full_val.json` and CSV.",
        "",
        "## E. Feature realism",
        json.dumps(realism, indent=2),
        "",
        "## F. Gradient compatibility",
        json.dumps(gradients, indent=2),
        "",
        "## G. Training results",
        "Not run (gate evaluated first).",
        "",
        "## J. Final recommendation",
    ]
    if gate["pass"]:
        lines.append("Continue to controlled from-scratch training with pair-conditioned shrinkage.")
    else:
        lines.append("Do **not** launch training until design revision; see failed checks in `part7_diagnostic_gate.json`.")
    (OUT / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-interventions", action="store_true")
    parser.add_argument("--skip-part1", action="store_true")
    parser.add_argument("--skip-part2", action="store_true")
    parser.add_argument("--skip-part5", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    cfg = Config.load(ROOT / "configs/default.yaml")
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("evaluation.val_subset_fraction", 1.0)
    cfg.set("training.pin_memory", False)
    cfg.set("training.num_workers", 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subspace").mkdir(exist_ok=True)

    if args.skip_part1 and (OUT / "part1_conventions.json").exists():
        records = load_manifest(cfg.path("paths.manifest"))
        splits = load_splits(cfg.path("paths.split_file"))
        ds_tr = TeacherDataset(cfg, records, "train", splits, use_token_cache=True)
        ds_va = TeacherDataset(cfg, records, "val", splits, use_token_cache=True)
        rep_tr = enumerate_pairs(ds_tr, "train")
        rep_va = enumerate_pairs(ds_va, "val")
        doc = json.loads((OUT / "part1_conventions.json").read_text())
        pack = {
            "ds_tr": ds_tr,
            "ds_va": ds_va,
            "pairs_tr": rep_tr["pairs"],
            "pairs_va": rep_va["pairs"],
            "doc": doc,
        }
    else:
        pack = run_part1(cfg)
    if args.skip_part2 and (OUT / "part2_subspace_fit.json").exists():
        sub = {"mu_F": np.load(OUT / "subspace/learned_k8.npz")["mu_F"]}
    else:
        sub = run_part2(pack["pairs_tr"], pack["ds_tr"])
    pairs_va = pack["pairs_va"]
    windows = build_val_windows(pairs_va)
    warn = cfg.get("warning")

    results = {}
    if not args.skip_interventions:
        model = CollisionTeacher3(cfg, dummy_backbone=False)
        load_teacher_state(model, load_checkpoint(str(G_CKPT), map_location=device))
        model.to(device)
        shuf = build_shuffled_map(pairs_va)
        results, _ = run_part3_4(pack["ds_va"], pairs_va, windows, model, device, sub, shuf, args.batch_size, warn)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        results = json.loads((OUT / "test_intervention_full_val.json").read_text())["conditions"]

    k_best, lam_best = 8, 0.5
    if args.skip_part5 and (OUT / "test_feature_realism.json").exists():
        realism = json.loads((OUT / "test_feature_realism.json").read_text())
    else:
        realism = run_part5(pack["ds_va"], pairs_va, sub, k_best, lam_best)
    gradients = run_part6(cfg, pack["ds_tr"], pack["pairs_tr"], device, k_best, lam_best)
    gate = run_gate(results, realism, gradients)
    write_final_report(gate, pack["doc"], realism, gradients, results)
    print("gate pass", gate["pass"], "best", gate.get("best_condition"))


if __name__ == "__main__":
    main()
