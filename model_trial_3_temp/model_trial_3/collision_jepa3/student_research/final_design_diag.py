"""Focused design diagnostics: temporal ablation, F_pre_decoder PCA, decoder budget."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.data.cache import TokenCache, token_cache_id, token_fingerprint
from collision_jepa3.data.sampling import FRAME_OFFSETS, input_indices
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.student_research.frames import FRAME_CANDIDATES
from collision_jepa3.student_research.parent import load_pair_cf_parent, parent_forward, resolve_parent_ckpt
from collision_jepa3.student_research.cnn import StudentRGBNet, preset_architectures


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _tier(gt: np.ndarray, warning: dict) -> str:
    _, s, _ = classify_heatmap(gt, warning)
    if s >= STOP:
        return "STOP"
    if s >= CAUTION:
        return "THREAT"
    return "SAFE"


def _peak_tier(h: np.ndarray, warning: dict) -> str:
    return _tier(h, warning)


def temporal_mapping(cfg) -> dict:
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    tubelet = int(cfg.get("model.vjepa.tubelet_size", 2))
    groups = []
    for g in range(len(offsets) // tubelet):
        frs = offsets[g * tubelet : (g + 1) * tubelet]
        groups.append(
            {
                "group_id": g,
                "input_frame_indices_in_window": frs,
                "timestamp_sec_at_anchor": [o / 30.0 for o in frs],
                "tubelet_size": tubelet,
                "note": "Each group is one V-JEPA tubelet over consecutive parent offsets.",
            }
        )
    return {
        "parent_offsets_frames": offsets,
        "n_input_frames": len(offsets),
        "n_temporal_groups": len(groups),
        "spatial_tokens_per_group": 16 * 16,
        "hidden_size": 1408,
        "token_tensor_layout": "[B, 8, 16, 16, 1408] after tokens_to_grid",
        "groups": groups,
        "adjacent_overlap": "Tubelets share no frames; adjacent groups are temporally adjacent windows.",
    }


def _token_caches(cfg) -> list[TokenCache]:
    fps = [
        token_cache_id(cfg, dummy=False, stride=6),
        token_cache_id(cfg, dummy=False),
        token_fingerprint(cfg, dummy=False),
    ]
    root = cfg.path("paths.cache_root")
    return [TokenCache(root, fp) for fp in fps]


def rows_with_tokens(rows, cfg, row_ids: list[int]) -> list[int]:
    caches = _token_caches(cfg)
    ok = []
    for rid in row_ids:
        row = rows[rid]
        cid, t = row["clip_id"], int(row["frame"])
        found = False
        for cache in caches:
            try:
                arr, mapping = cache.load_clip(cid, mmap=True)
            except FileNotFoundError:
                continue
            if t in mapping:
                found = True
                break
        if found:
            ok.append(rid)
    return ok


def load_token(rows, rid: int, cfg) -> np.ndarray | None:
    row = rows[rid]
    cid, t = row["clip_id"], int(row["frame"])
    for cache in _token_caches(cfg):
        try:
            arr, mapping = cache.load_clip(cid, mmap=True)
        except FileNotFoundError:
            continue
        if t in mapping:
            return np.asarray(arr[mapping[t]], dtype=np.float32)
    return None


def stratified_sample(rows, gt, row_ids: list[int], warning: dict, n: int, seed: int) -> list[int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for rid in row_ids:
        lab = _tier(gt["gt_1s"][rid], warning)
        pre = "pre" if rows[rid].get("pre_fork") else "post"
        buckets[f"{lab}_{pre}"].append(rid)
    rng = np.random.default_rng(seed)
    per = max(1, n // max(len(buckets), 1))
    out = []
    for ids in buckets.values():
        if not ids:
            continue
        pick = rng.choice(ids, size=min(per, len(ids)), replace=False)
        out.extend(int(x) for x in pick)
    if len(out) < n:
        rest = [r for r in row_ids if r not in out]
        rng.shuffle(rest)
        out.extend(rest[: n - len(out)])
    return sorted(set(out))[:n]


def tokens_to_grid(tokens: torch.Tensor, t_groups: int = 8, spatial: int = 16) -> torch.Tensor:
    """[B,2048,C] -> [B,T,C,H,W]"""
    b, n, c = tokens.shape
    return tokens.reshape(b, t_groups, spatial, spatial, c).permute(0, 1, 4, 2, 3).contiguous()


def ablate_group(grid: torch.Tensor, g: int, mode: str) -> torch.Tensor:
    """grid [B,T,C,H,W]"""
    x = grid.clone()
    t = x.shape[1]
    if mode == "neighbor_interp":
        left = max(0, g - 1)
        right = min(t - 1, g + 1)
        if left == right:
            x[:, g] = x[:, left]
        else:
            x[:, g] = 0.5 * (x[:, left] + x[:, right])
    elif mode == "temporal_mean":
        x[:, g] = x.mean(dim=1, keepdim=True).expand_as(x[:, g : g + 1])
    elif mode == "random_control":
        j = (g + 3) % t
        x[:, g] = x[:, j]
    else:
        x[:, g] = 0.0
    return x


def forward_from_grid(model, grid: torch.Tensor) -> dict[str, torch.Tensor]:
    b, t, c, h, w = grid.shape
    flat = grid.permute(0, 1, 3, 4, 2).reshape(b, t * h * w, c)
    return parent_forward(flat, model)


@torch.no_grad()
def temporal_ablation_study(cfg, rows, gt, pred, sample_ids, warning, device) -> dict:
    model, _ = load_pair_cf_parent(cfg, resolve_parent_ckpt(cfg, None), device=device)
    model.eval()
    per_group = defaultdict(list)
    baselines = []
    modes = ("neighbor_interp", "temporal_mean", "zero")
    for rid in sample_ids:
        vec = load_token(rows, rid, cfg)
        if vec is None:
            continue
        x = torch.from_numpy(vec).unsqueeze(0).to(device)
        grid = tokens_to_grid(x)
        base = forward_from_grid(model, grid)
        h0 = base["Hhat_1s"][0].float().cpu().numpy()
        gt0 = np.asarray(gt["gt_1s"][rid], dtype=np.float32)
        baselines.append(
            {
                "row_id": rid,
                "peak": float(h0.max()),
                "tier": _peak_tier(h0, warning),
                "gt_tier": _tier(gt0, warning),
                "pre_fork": bool(rows[rid].get("pre_fork")),
            }
        )
        for g in range(8):
            for mode in modes:
                gg = ablate_group(grid, g, mode)
                out = forward_from_grid(model, gg)
                h = out["Hhat_1s"][0].float().cpu().numpy()
                per_group[(g, mode)].append(
                    {
                        "row_id": rid,
                        "delta_peak": float(h.max() - h0.max()),
                        "tier_changed": int(_peak_tier(h, warning) != _peak_tier(h0, warning)),
                        "mse_vs_gt": float(np.mean((h - gt0) ** 2)),
                        "hm_l1": float(np.abs(h - h0).mean()),
                    }
                )
    summary = {}
    for (g, mode), recs in per_group.items():
        if not recs:
            continue
        summary[f"g{g}_{mode}"] = {
            "n": len(recs),
            "mean_delta_peak": float(np.mean([r["delta_peak"] for r in recs])),
            "tier_flip_rate": float(np.mean([r["tier_changed"] for r in recs])),
            "mean_hm_l1": float(np.mean([r["hm_l1"] for r in recs])),
            "mean_mse_gt": float(np.mean([r["mse_vs_gt"] for r in recs])),
        }
    return {"n_baselines": len(baselines), "baseline_sample": baselines[:5], "ablation_summary": summary, "replacement_modes": list(modes)}


def group_combinations() -> list[dict]:
    """Heuristic retained temporal groups (0=oldest tubelet .. 7=most recent)."""
    return [
        {"name": "long_range_3", "groups": [0, 4, 7]},
        {"name": "mid_recent_3", "groups": [4, 6, 7]},
        {"name": "recent_dense_3", "groups": [5, 6, 7]},
        {"name": "uniform_5", "groups": [0, 2, 4, 6, 7]},
        {"name": "sparse_long_5", "groups": [0, 2, 5, 6, 7]},
    ]


@torch.no_grad()
def combination_eval(cfg, rows, sample_ids, warning, device) -> list[dict]:
    model, _ = load_pair_cf_parent(cfg, resolve_parent_ckpt(cfg, None), device=device)
    model.eval()
    rows_out = []
    for combo in group_combinations():
        keep = set(combo["groups"])
        deltas = []
        for rid in sample_ids:
            vec = load_token(rows, rid, cfg)
            if vec is None:
                continue
            x = torch.from_numpy(vec).unsqueeze(0).to(device)
            grid = tokens_to_grid(x)
            full = forward_from_grid(model, grid)["Hhat_1s"][0].float().cpu().numpy()
            g2 = grid.clone()
            for gi in range(8):
                if gi not in keep:
                    g2 = ablate_group(g2, gi, "neighbor_interp")
            partial = forward_from_grid(model, g2)["Hhat_1s"][0].float().cpu().numpy()
            deltas.append(float(np.abs(partial - full).mean()))
        rows_out.append(
            {
                **combo,
                "n": len(deltas),
                "mean_hm_l1_vs_full": float(np.mean(deltas)) if deltas else None,
            }
        )
    return rows_out


def map_student_frames() -> dict:
    parent_offs = set(FRAME_OFFSETS)
    out = {}
    for fid in ("F1", "F2", "F3", "F4", "F5"):
        offs = FRAME_CANDIDATES[fid]
        mapped = []
        for o in offs:
            exact = o in parent_offs
            nearest = min(FRAME_OFFSETS, key=lambda p: abs(p - o))
            tg = None
            if exact:
                idx = list(FRAME_OFFSETS).index(o)
                tg = idx // 2
            mapped.append({"offset": o, "parent_exact": exact, "nearest_parent_offset": nearest, "gap_frames": abs(nearest - o), "token_group_if_exact": tg})
        out[fid] = {
            "offsets": offs,
            "n_frames": len(offs),
            "span": max(offs) - min(offs),
            "longest_gap": max(abs(offs[i] - offs[i - 1]) for i in range(1, len(offs))),
            "recent_density": sum(1 for o in offs if o >= -15) / len(offs),
            "exact_on_parent_grid": sum(1 for m in mapped if m["parent_exact"]) / len(offs),
            "per_offset": mapped,
        }
    return out


def extract_f_from_tokens(model, vec: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(vec).unsqueeze(0).to(device)
    with torch.no_grad():
        out = parent_forward(x, model)
    return out["F_pre_decoder"][0].float().cpu().numpy()


def paired_f_samples(rows, train_ids, cfg, device, max_pairs: int = 250) -> tuple[list[np.ndarray], list[tuple], list[int]]:
    model, _ = load_pair_cf_parent(cfg, resolve_parent_ckpt(cfg, None), device=device)
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for rid in train_ids:
        pid = str(rows[rid].get("background_pair_id") or "")
        if not pid:
            continue
        by_key[(pid, int(rows[rid]["frame"]))].append(rid)
    diffs = []
    pairs_used = []
    pair_row_ids = []
    for key, ids in by_key.items():
        if len(ids) != 2:
            continue
        a, b = ids[0], ids[1]
        va, vb = load_token(rows, a, cfg), load_token(rows, b, cfg)
        if va is None or vb is None:
            continue
        fa = extract_f_from_tokens(model, va, device).reshape(-1)
        fb = extract_f_from_tokens(model, vb, device).reshape(-1)
        diffs.append(((fa - fb) / 2.0).astype(np.float64))
        pairs_used.append(key)
        pair_row_ids.append((a, b))
        if len(diffs) >= max_pairs:
            break
    return diffs, pairs_used, pair_row_ids


def pca_sample_space(X: np.ndarray, ks: list[int]) -> dict:
    mu = X.mean(0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    evals = (s**2) / max(X.shape[0] - 1, 1)
    total = evals.sum()
    pr = float(evals.sum() ** 2 / max((evals**2).sum(), 1e-12))
    ent = float(np.exp(-np.sum((evals / max(total, 1e-12)) * np.log((evals / max(total, 1e-12)) + 1e-12))))
    out = {"n_samples": int(X.shape[0]), "dim": int(X.shape[1]), "eigenvalues_top32": evals[:32].tolist(), "participation_ratio": pr, "entropy_rank": ent}
    U = {}
    for k in ks:
        kk = min(k, vt.shape[0])
        explained = float(evals[:kk].sum() / max(total, 1e-12))
        U[str(k)] = {"k": kk, "explained_variance_ratio": explained}
    out["k_summaries"] = U
    out["mu_shape"] = list(mu.shape)
    out["_vt"] = vt.astype(np.float32)
    out["_mu"] = mu.astype(np.float32)
    return out


def split_half_angles(X: np.ndarray, seed: int = 1337) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    h = len(X) // 2
    a, b = X[idx[:h]], X[idx[h : 2 * h]]
    if len(a) < 4:
        return {"unavailable": "too few pairs"}
    def top_dirs(mat):
        mat = mat - mat.mean(0)
        _, _, vt = np.linalg.svd(mat, full_matrices=False)
        return vt[:8]
    Ua, Ub = top_dirs(a), top_dirs(b)
    # principal angles via SVD of Ua @ Ub.T
    M = Ua @ Ub.T
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1, 1)
    return {"principal_angles_rad_top8": np.arccos(s).tolist()}


def ridge_probe(xtr, ytr, xva, yva, alpha=1.0):
    from sklearn.linear_model import Ridge

    r = Ridge(alpha=alpha)
    r.fit(xtr, ytr)
    pred = r.predict(xva)
    mse = float(np.mean((pred - yva) ** 2))
    out = {"mse": mse, "n_train": len(xtr), "n_dev": len(xva)}
    if yva.ndim == 1 and len(np.unique(yva)) <= 2:
        out["accuracy"] = float(np.mean((pred >= 0.5) == (yva >= 0.5)))
    return out


def run_all(cfg, report_root: Path, student_root: Path, device: str = "cpu") -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    out_train = student_root / "teacher_cache_train_pilot"
    rows = [json.loads(l) for l in (out_train / "index.jsonl").open()]
    gt = np.load(out_train / "gt_heatmaps.npz", mmap_mode="r")
    pred = np.load(out_train / "teacher_pred.npz", mmap_mode="r")
    warning = cfg.get("warning") or {}
    tr = json.loads((student_root / "part1_train_manifest.json").read_text())["train_row_ids"]
    dv = json.loads((student_root / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    token_tr = rows_with_tokens(rows, cfg, tr)
    token_dv = rows_with_tokens(rows, cfg, dv)
    _write(report_root / "part0_token_availability.json", {"n_train_manifest": len(tr), "n_dev_manifest": len(dv), "n_train_with_giant_tokens": len(token_tr), "n_dev_with_giant_tokens": len(token_dv)})

    pmap = temporal_mapping(cfg)
    pmap["student_frame_candidates"] = map_student_frames()
    _write(report_root / "part1_temporal_mapping.json", pmap)

    if len(token_tr) < 32:
        _write(
            report_root / "part1_temporal_ablation.json",
            {"status": "unavailable", "reason": f"only {len(token_tr)} train rows with cached Giant tokens"},
        )
        combo_csv = report_root / "part1_temporal_combination_comparison.csv"
        combo_csv.write_text("name,n\n", encoding="utf-8")
    else:
        sample = stratified_sample(rows, gt, token_tr, warning, n=min(96, len(token_tr)), seed=1337)
        abl = temporal_ablation_study(cfg, rows, gt, pred, sample, warning, device)
        abl["sample_row_ids"] = sample
        abl["limitation"] = "Ablating Giant token groups ≠ giving a CNN only 3/5 RGB frames."
        _write(report_root / "part1_temporal_ablation.json", abl)
        combos = combination_eval(cfg, rows, sample, warning, device)
        with (report_root / "part1_temporal_combination_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", "groups", "n", "mean_hm_l1_vs_full"])
            w.writeheader()
            for c in combos:
                w.writerow(c)
    # Part 2 F_predecoder PCA
    diffs: list[np.ndarray] = []
    pair_ids: list[tuple] = []
    diffs, pair_keys, pair_ids = paired_f_samples(rows, tr, cfg, device, max_pairs=300)
    if len(diffs) < 20:
        for p in ("part2_fpredecoder_pca.json", "part2_fpredecoder_task_probes.json", "part2_fpredecoder_interventions.json"):
            _write(report_root / p, {"status": "unavailable", "n_pairs_with_tokens": len(diffs)})
    else:
        X = np.stack(diffs, axis=0)
        pca = pca_sample_space(X, [4, 8, 16, 32])
        vt, mu = pca.pop("_vt"), pca.pop("_mu")
        pca["split_half"] = split_half_angles(X)
        pca["n_pairs_used"] = len(diffs)
        _write(report_root / "part2_fpredecoder_pca.json", pca)
        np.savez_compressed(report_root / "pca_fpredecoder_k32_meta.npz", mu=mu, vt=vt[:32])

        # probes on streaming sample with tokens
        train_pairs_a = [p[0] for p in pair_ids]
        dev_single = token_dv[: min(200, len(token_dv))]
        model, _ = load_pair_cf_parent(cfg, resolve_parent_ckpt(cfg, None), device=device)

        def feats_from_cache(rids, tap="F"):
            vecs, zvecs = [], []
            for rid in rids:
                vec = load_token(rows, rid, cfg)
                if vec is None:
                    continue
                with torch.no_grad():
                    t = torch.from_numpy(vec).unsqueeze(0).to(device)
                    o = parent_forward(t, model)
                    vecs.append(o["F_pre_decoder"][0].reshape(-1).float().cpu().numpy())
                    zvecs.append(o["Z_t"][0].reshape(-1).float().cpu().numpy())
            return np.stack(vecs) if vecs else None, np.stack(zvecs) if zvecs else None

        tr_rids = train_pairs_a[:200]
        f_tr, z_tr = feats_from_cache(tr_rids, "F")
        f_dv, z_dv = feats_from_cache(dev_single, "F")
        probes = {"F_pre_decoder_flat": {}, "Z_t_flat": {}}
        if f_tr is not None and f_dv is not None and len(f_tr) >= 10:
            ytr_cells = np.stack([gt["gt_1s"][i].reshape(-1) for i in tr_rids[: len(f_tr)]])
            ydv_cells = np.stack([gt["gt_1s"][i].reshape(-1) for i in dev_single[: len(f_dv)]])
            ytr_stop = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "STOP" else 0.0 for i in tr_rids[: len(f_tr)]])
            ydv_stop = np.array([1.0 if _tier(gt["gt_1s"][i], warning) == "STOP" else 0.0 for i in dev_single[: len(f_dv)]])
            probes["F_pre_decoder_flat"]["cells"] = ridge_probe(f_tr, ytr_cells, f_dv, ydv_cells)
            probes["F_pre_decoder_flat"]["stop"] = ridge_probe(f_tr, ytr_stop, f_dv, ydv_stop)
        if z_tr is not None and z_dv is not None:
            probes["Z_t_flat"]["stop"] = ridge_probe(z_tr, ytr_stop, z_dv, ydv_stop)
        _write(report_root / "part2_fpredecoder_task_probes.json", {"n_f_train": 0 if f_tr is None else len(f_tr), "probes": probes})

        # decoder interventions on dev pairs with tokens
        inter = []
        k = 8
        V = vt[:k]
        for (a, b) in pair_ids[: min(80, len(pair_ids))]:
            if a not in token_dv and b not in token_dv:
                continue
            va, vb = load_token(rows, a, cfg), load_token(rows, b, cfg)
            if va is None or vb is None:
                continue
            with torch.no_grad():
                ta = torch.from_numpy(va).unsqueeze(0).to(device)
                tb = torch.from_numpy(vb).unsqueeze(0).to(device)
                fa = parent_forward(ta, model)["F_pre_decoder"][0].reshape(-1)
                fb = parent_forward(tb, model)["F_pre_decoder"][0].reshape(-1)
                h0a = model.decoder.forward_with_logits(fa.reshape(1, 128, 16, 16))[0][0].float().cpu().numpy()
                flat_a = fa.double().cpu().numpy()
                proj = (flat_a @ V.T) @ V
                f_perp = flat_a - proj
                ha = model.decoder.forward_with_logits(torch.from_numpy(f_perp.astype(np.float32)).reshape(1, 128, 16, 16).to(device))[0][0].float().cpu().numpy()
                inter.append({"delta_peak_remove_pca8": float(ha.max() - h0a.max())})
        _write(
            report_root / "part2_fpredecoder_interventions.json",
            {"n": len(inter), "mean_delta_peak_remove_top8_pairdirs": float(np.mean([x["delta_peak_remove_pca8"] for x in inter])) if inter else None, "note": "Removes top-8 pair-diff PCA dirs from single-side F (approximate)"},
        )

    # Part 3 compare cached tensors (no new giant)
    zc = np.asarray(pred["Z"][tr[:400]], dtype=np.float32).reshape(-1, 32 * 5 * 5)
    hc = np.asarray(pred["H"][tr[:400]], dtype=np.float32).reshape(-1, 25)
    _write(
        report_root / "part3_decoder_input_vs_bottleneck.json",
        {
            "Z_t_dim": 800,
            "F_predecoder_dim": 32768,
            "Z_cached_probe_stop_acc_note": "see part2_parent_task_information.json",
            "F_available_via_cache_F_dec_replay_mse": 2e-9,
            "spatial_resolution": {"Z_t": "5x5", "F_dec": "16x16"},
            "pair_pca_Z_done_in_prior_report": "part3_parent_latent_pca.json in student_architecture_tests",
            "pair_pca_F_done_here": str(report_root / "part2_fpredecoder_pca.json"),
        },
    )

    # Part 4 budgets
    p4 = json.loads((student_root / "part4_architecture_feasibility.json").read_text())
    dec = 47809
    proj_options = [
        {"name": "conv1x1_5to16", "params_estimate": 128 * 128 + 128, "note": "upsample 5->16 then 1x1"},
        {"name": "depthwise_5to16", "params_estimate": 128 * 9 + 128 * 128, "note": "rough"},
    ]
    budgets = []
    for row in p4.get("budget_verification", []):
        backbone = row["total_params"] - dec
        for proj in proj_options:
            total = row["total_params"] + proj["params_estimate"]
            budgets.append({**row, "parent_decoder_included_in_preset": True, "extra_proj_est": proj["params_estimate"], "total_with_extra_proj": total, "fits_700k": total <= 700000, "encoder_headroom": 700000 - total})
    _write(report_root / "part4_decoder_transfer_budget.json", {"decoder_params": dec, "projection_options": proj_options, "configurations": budgets})

    # Part 5 random forward shapes
    shapes = []
    for a in preset_architectures():
        for nf in (3, 5):
            m = StudentRGBNet(a, n_frames=nf)
            with torch.no_grad():
                dummy = torch.zeros(1, nf, 3, 160, 160)
                o = m(dummy)
            shapes.append({"arch": a.name, "n_frames": nf, "params": sum(p.numel() for p in m.parameters()), "H_shape": list(o["Hhat_1s"].shape), "Z_shape": list(o["Z_t"].shape), "F_shape": list(o["F"].shape)})
    _write(report_root / "part5_student_latent_interface.json", {"random_init_forward": shapes, "decoder_input_target": [128, 16, 16], "note": "Random weights; shape feasibility only."})

    # Part 6 strategies
    _write(
        report_root / "part6_student_objective_feasibility.json",
        {
            "strategies": {
                "A_heatmap_kd": {"params_extra": 0, "evidence": "part5 teacher analysis"},
                "B_Zt_kd": {"params_extra": 0, "risk": "5x5 bottleneck may lose spatial decoder path"},
                "C_Fpre_kd": {"params_extra": "projection to 128x16x16", "evidence": "part2 probes"},
                "D_frozen_decoder": {"params_extra": dec, "risk": "resolution gap part4"},
                "E_pca_weighted_F": {"evidence": "part2_fpredecoder_pca.json"},
            },
            "no_winner_selected": True,
        },
    )

    # Final report
    lines = [
        "# Final student design diagnostics",
        "",
        "Focused read-only study. No student CNN training.",
        "",
        f"- Token-train rows with Giant cache: {len(token_tr)} / {len(tr)}",
        f"- F_predecoder pair PCA pairs used: {len(diffs)}",
        "",
        "See part1–part6 JSON in this directory.",
        "**No architecture or training recommendation.**",
    ]
    (report_root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
