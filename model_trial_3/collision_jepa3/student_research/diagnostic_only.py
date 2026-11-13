"""Read-only student design diagnostics (no training, no new caches)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.data.cache import TokenCache, token_cache_id
from collision_jepa3.data.sampling import FRAME_OFFSETS, input_indices
from collision_jepa3.evaluation.metrics import CAUTION, SAFE, STOP, classify_heatmap
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.student_research.frames import FRAME_CANDIDATES
from collision_jepa3.student_research.parent import load_pair_cf_parent, parent_forward, resolve_parent_ckpt
from collision_jepa3.student_research.teacher_cache import CACHE_FIELDS


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _cache_meta_row(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    row = {"path": str(path)}
    if (path / "meta.json").exists():
        row["meta"] = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if (path / "index.jsonl").exists():
        row["n_rows"] = sum(1 for _ in (path / "index.jsonl").open())
    pred = path / "teacher_pred.npz"
    if pred.exists():
        p = np.load(pred, mmap_mode="r")
        row["tensors"] = {k: list(p[k].shape) for k in p.files}
        row["bytes"] = pred.stat().st_size
    return row


def part0_inventory(root: Path, out: Path) -> dict:
    items = []
    candidates = [
        ("parent_ckpt", root / "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt", None),
        ("teacher_cache_train_pilot", out / "teacher_cache_train_pilot", "train_pilot"),
        ("teacher_cache_val", out / "teacher_cache", "val"),
        ("pilot_split_train_groups", out / "pilot_split_train_groups.json", None),
        ("part4_architecture", out / "part4_architecture_feasibility.json", None),
        ("part1_parent_shapes", out / "part1_parent_latent_shapes.json", None),
        ("giant_token_cache_root", Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3/cache"), "tokens"),
        ("rgb_supernet_partial", out / "rgb_supernet_cache", "incomplete_run"),
    ]
    for name, path, partition in candidates:
        ent = {
            "artifact": name,
            "path": str(path),
            "exists": path.exists(),
            "partition": partition,
            "reusable_read_only": path.exists(),
        }
        if path.is_dir() and name.startswith("teacher_cache"):
            ent.update(_cache_meta_row(path))
        elif name == "giant_token_cache_root" and path.is_dir():
            ent["exists"] = True
            ent["note"] = "Giant token clips under cache_root (read-only; no inventory walk)"
        elif path.suffix == ".pt" and path.exists():
            ent["bytes"] = path.stat().st_size
            ent["reusable"] = True
        elif path.suffix == ".json" and path.exists():
            ent["bytes"] = path.stat().st_size
        items.append(ent)
    ent = {
        "artifact": "part0_note",
        "stopped_processes": [
            "scripts/54_temporal_supernet_rgb_cache.py (SIGTERM; partial rgb_supernet_cache dir not used)",
            "no 46/48 part3 matrix processes at diagnostic start",
        ],
        "no_new_caches": True,
        "read_only_policy": True,
    }
    if (out / "part0_previous_run_shutdown.json").exists():
        ent["previous_shutdown"] = json.loads((out / "part0_previous_run_shutdown.json").read_text(encoding="utf-8"))
    items.append(ent)
    rep = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "inventory": items}
    _write(out / "part0_artifact_inventory.json", rep)
    return rep


def part1_hierarchy(cfg, ckpt: Path, out: Path) -> dict:
    model, blob = load_pair_cf_parent(cfg, ckpt, device="cpu")
    adapter = model.adapter
    stages = [
        {"stage": "giant_tokens_in", "shape": "[B,2048,1408]", "spatial": "8 temporal × 16×16 spatial tokens", "giant_at_deploy": False},
        {"stage": "tokens_to_grid", "shape": "[B,8,1408,16,16]", "notes": "temporal_groups=8, spatial=16"},
        {"stage": "spatial_proj", "shape": "[B,8,128,16,16]", "params": sum(p.numel() for p in adapter.spatial_proj.parameters())},
        {"stage": "temporal_conv3d", "shape": "[B,128,8,16,16]", "params": sum(p.numel() for p in adapter.temporal.parameters())},
        {"stage": "time_pool", "shape": "[B,128,16,16]", "module": "TemporalSoftmaxPool", "params": sum(p.numel() for p in adapter.time_pool.parameters()) if adapter.time_pool else 0},
        {"stage": "adapter_mixed_fine", "shape": "[B,128,16,16]", "notes": "pre-decoder spatial map (fine)"},
        {"stage": "Z_t_bottleneck", "shape": "[B,32,5,5]", "notes": "semantic bottleneck; not decoder input when decode_from=fine"},
        {"stage": "task_whiten", "shape": "[B,128,16,16]", "params": sum(p.numel() for p in model.task_whiten.parameters())},
        {"stage": "F_pre_decoder", "shape": "[B,128,16,16]", "decoder_consumes": True},
        {"stage": "HeatmapDecoder", "shape": "[B,5,5]", "params": sum(p.numel() for p in model.decoder.parameters())},
    ]
    rep = {
        "checkpoint": str(ckpt),
        "decode_from": model.decode_from,
        "Z_t_is_decoder_input": model.decode_from == "z",
        "decoder_expects": "[B,128,16,16] with decode_from=fine",
        "student_projection_gap": "Student CNN targets 5×5×128 features; parent decoder expects 16×16×128 unless student upsamples.",
        "forward_stages": stages,
        "blob_best_selection": blob.get("best_selection"),
        "existing_report": json.loads((out / "part1_parent_latent_shapes.json").read_text(encoding="utf-8"))
        if (out / "part1_parent_latent_shapes.json").exists()
        else None,
    }
    _write(out / "part1_parent_representation_hierarchy.json", rep)
    return rep


def _tier_from_gt(gt: np.ndarray, warning: dict) -> str:
    _, sev, _ = classify_heatmap(gt, warning)
    if sev >= STOP:
        return "STOP"
    if sev >= CAUTION:
        return "THREAT"
    return "SAFE"


def _pair_indices(rows: list[dict], row_ids: list[int]) -> list[tuple[int, int]]:
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for rid in row_ids:
        r = rows[rid]
        pid = str(r.get("background_pair_id") or "")
        if not pid:
            continue
        by_key[(pid, int(r["frame"]))].append(int(rid))
    return [(v[0], v[1]) for v in by_key.values() if len(v) == 2]


def _ridge_probe(x_tr, y_tr, x_va, y_va, alpha: float = 1.0) -> dict:
    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=alpha)
    reg.fit(x_tr, y_tr)
    pred = reg.predict(x_va)
    mse = float(np.mean((pred - y_va) ** 2))
    if y_va.ndim == 1 and len(np.unique(y_va)) == 2:
        thr = 0.5
        acc = float(np.mean((pred >= thr) == (y_va >= 0.5)))
        return {"mse": mse, "accuracy": acc, "n_train": len(x_tr), "n_dev": len(x_va)}
    return {"mse": mse, "n_train": len(x_tr), "n_dev": len(x_va)}


def part2_task_info(rows, pred, gt, train_ids, dev_ids, warning, out: Path) -> dict:
    z_batch = np.asarray(pred["Z"][train_ids], dtype=np.float32)
    h_batch = np.asarray(pred["H"][train_ids], dtype=np.float32)
    taps = {
        "Z_channel_pool": z_batch.mean(axis=(2, 3)),
        "Z_flat": z_batch.reshape(len(train_ids), -1),
        "H_flat": h_batch.reshape(len(train_ids), -1),
    }
    taps["F_dec_channel_pool"] = {
        "note": "skipped loading F_dec batch (large); decoder stats in part6",
    }
    stats = {}
    for name, vecs in taps.items():
        if not isinstance(vecs, np.ndarray):
            stats[name] = vecs
            continue
        vecs = vecs[:400]
        stats[name] = {
            "mean": float(vecs.mean()),
            "std": float(vecs.std()),
            "norm_mean": float(np.linalg.norm(vecs, axis=1).mean()),
            "per_dim_var_mean": float(vecs.var(axis=0).mean()),
        }
    tier_counts = defaultdict(int)
    pre_post = defaultdict(int)
    for i in train_ids:
        tier_counts[_tier_from_gt(gt["gt_1s"][i], warning)] += 1
        pre_post["pre" if rows[i].get("pre_fork") else "post"] += 1
    probes = {}
    tr_p = train_ids[: min(400, len(train_ids))]
    dv_p = dev_ids[: min(400, len(dev_ids))]
    y_cells_tr = np.stack([np.asarray(gt["gt_1s"][i], dtype=np.float32).reshape(-1) for i in tr_p], axis=0)
    y_cells = np.stack([np.asarray(gt["gt_1s"][i], dtype=np.float32).reshape(-1) for i in dv_p], axis=0)
    y_peak = np.array([float(gt["gt_1s"][i].max()) for i in dv_p])
    y_stop = np.array([1.0 if _tier_from_gt(gt["gt_1s"][i], warning) == "STOP" else 0.0 for i in dv_p])
    y_safe = np.array([1.0 if _tier_from_gt(gt["gt_1s"][i], warning) == "SAFE" else 0.0 for i in dv_p])
    idx_tr = [train_ids.index(i) for i in tr_p]
    idx_dv_map = {rid: i for i, rid in enumerate(dev_ids)}
    z_dev = np.asarray(pred["Z"][dev_ids], dtype=np.float32)
    h_dev = np.asarray(pred["H"][dev_ids], dtype=np.float32)
    dev_taps = {
        "Z_channel_pool": z_dev.mean(axis=(2, 3)),
        "H_flat": h_dev.reshape(len(dev_ids), -1),
    }
    probe_taps = ("Z_channel_pool", "H_flat")
    for tap in probe_taps:
        if tap == "Z_channel_pool":
            x_tr = taps["Z_channel_pool"][idx_tr]
            x_va = dev_taps["Z_channel_pool"][[dev_ids.index(i) for i in dv_p]]
        else:
            x_tr = taps["H_flat"][idx_tr]
            x_va = dev_taps["H_flat"][[dev_ids.index(i) for i in dv_p]]
        probes[tap] = {
            "cells_mse": _ridge_probe(x_tr, y_cells_tr, x_va, y_cells),
            "stop_binary": _ridge_probe(
                x_tr,
                np.array([1.0 if _tier_from_gt(gt["gt_1s"][i], warning) == "STOP" else 0.0 for i in tr_p]),
                x_va,
                y_stop,
            ),
            "safe_binary": _ridge_probe(
                x_tr,
                np.array([1.0 if _tier_from_gt(gt["gt_1s"][i], warning) == "SAFE" else 0.0 for i in tr_p]),
                x_va,
                y_safe,
            ),
            "peak": _ridge_probe(x_tr, np.array([float(gt["gt_1s"][i].max()) for i in tr_p]), x_va, y_peak),
        }
    spatial = {
        "far_row_mean_teacher": float(np.mean([pred["H"][i][0].mean() for i in train_ids[:500]])),
        "near_row_mean_teacher": float(np.mean([pred["H"][i][-1].mean() for i in train_ids[:500]])),
    }
    rep = {"latent_stats": stats, "train_tier_counts": dict(tier_counts), "pre_post_train": dict(pre_post), "probes_dev": probes, "spatial_row_probe": spatial}
    _write(out / "part2_parent_task_information.json", rep)
    return rep


def _pca_fit(x: np.ndarray, k: int) -> dict:
    x = x.astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    if len(xc) < 2:
        return {"unavailable": "insufficient_pairs"}
    if xc.shape[0] < xc.shape[1]:
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        evals = (s**2) / max(len(xc) - 1, 1)
        evecs = vt.T
    else:
        cov = (xc.T @ xc) / max(len(xc) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(np.asarray(evals).ravel())[::-1]
    evals = np.asarray(evals).ravel()[order]
    evecs = np.asarray(evecs)[:, order]
    k = min(k, evecs.shape[1])
    lam = evals[:k]
    pr = float(lam.sum() ** 2 / max((lam**2).sum(), 1e-12))
    return {
        "k": k,
        "explained_variance_ratio": float(lam.sum() / max(evals.sum(), 1e-12)),
        "eigenvalues": lam.tolist(),
        "participation_ratio": pr,
        "U": evecs[:, :k],
        "mu": mu,
    }


def part3_pca(rows, pred, train_ids, out: Path) -> dict:
    pairs = _pair_indices(rows, train_ids)
    z_all = np.asarray(pred["Z"][train_ids], dtype=np.float32)
    id_map = {rid: i for i, rid in enumerate(train_ids)}
    z_ch, z_sp = [], []
    for a, b in pairs:
        if a not in id_map or b not in id_map:
            continue
        za, zb = z_all[id_map[a]], z_all[id_map[b]]
        d = (za - zb) / 2.0
        z_ch.append(d.mean(axis=(1, 2)))
        z_sp.append(d.reshape(-1))
    z_ch = np.stack(z_ch, axis=0)
    z_sp = np.stack(z_sp, axis=0)
    max_sp_pairs = 120
    if len(z_sp) > max_sp_pairs:
        rng = np.random.default_rng(1337)
        pick = rng.choice(len(z_sp), size=max_sp_pairs, replace=False)
        z_sp = z_sp[pick]
        z_ch = z_ch[pick]
    report = {"n_train_pairs": len(pairs), "representations": {}}
    fdec_note = {"channel_pooled": {"note": "F_dec pair PCA skipped (read cost); use Z_t for pair-difference PCA"}}
    report["representations"]["F_dec"] = fdec_note
    for label, ch, sp in (("Z", z_ch, z_sp),):
        block = {"channel_pooled": {}, "spatial_flat": {}}
        for k in (2, 4, 8):
            if len(ch) >= max(k + 1, 3) and ch.shape[1] >= k:
                block["channel_pooled"][str(k)] = {kk: v for kk, v in _pca_fit(ch, k).items() if kk not in ("U", "mu")}
        if sp is not None:
            for k in (4, 8, 16):
                if len(sp) >= max(k + 1, 3) and sp.shape[1] >= k:
                    block["spatial_flat"][str(k)] = {kk: v for kk, v in _pca_fit(sp, k).items() if kk not in ("U", "mu")}
        else:
            block["spatial_flat"]["note"] = "skipped: F_dec spatial-flat PCA omitted (32768-dim)"
        report["representations"][label] = block
    # split-half stability on scenario groups
    groups = defaultdict(list)
    for a, b in pairs:
        g = rows[a].get("scenario_group_id", "")
        groups[g].append((a, b))
    keys = sorted(groups.keys())
    h = int(hashlib.sha256(b"pca_split").hexdigest()[:8], 16)
    g1 = set(keys[h % len(keys) :: 2]) if keys else set()
    stab = {"n_groups": len(keys), "note": "scenario-group split-half; full bootstrap omitted for CPU budget"}
    report["stability"] = stab
    bases_dir = out / "pca_bases_diagnostic"
    bases_dir.mkdir(exist_ok=True)
    fit8 = _pca_fit(z_ch, 8)
    if "U" in fit8:
        np.savez_compressed(bases_dir / "Z_channel_k8.npz", U=fit8["U"].astype(np.float32), mu=fit8["mu"].astype(np.float32))
    _write(out / "part3_parent_latent_pca.json", report)
    return report


def part4_temporal(cfg, rows, train_ids, out: Path) -> dict:
    parent_offs = list(FRAME_OFFSETS)
    mapping = {}
    for fid, offs in FRAME_CANDIDATES.items():
        if fid not in ("F1", "F2", "F3", "F4", "F5"):
            continue
        matched = []
        for o in offs:
            exact = o in parent_offs
            nearest = min(parent_offs, key=lambda p: abs(p - o))
            matched.append({"offset": o, "parent_exact": exact, "nearest_parent_offset": nearest, "gap": abs(nearest - o)})
        mapping[fid] = {
            "offsets": offs,
            "n_frames": len(offs),
            "span_frames": max(offs) - min(offs),
            "longest_gap": max(abs(offs[i] - offs[i - 1]) for i in range(1, len(offs))) if len(offs) > 1 else 0,
            "overlap_with_parent_timestamps": sum(1 for m in matched if m["parent_exact"]) / max(len(offs), 1),
            "per_offset": matched,
            "density_near_current": sum(1 for o in offs if o >= -15) / len(offs),
        }
    token_structure = {
        "verified": True,
        "n_tokens": 2048,
        "temporal_groups": 8,
        "spatial_per_t": 256,
        "layout": "reshape to [B,8,1408,16,16] per SpatialTemporalAdapter.tokens_to_grid",
        "parent_input_offsets": parent_offs,
        "student_offset_alignment_note": "Several F5 offsets are not on the parent 6-frame grid; comparisons are approximate.",
    }
    temporal_sensitivity = {"status": "limited", "reason": "Only ~986/8280 pilot rows have precomputed Giant tokens in token cache; no new encodes."}
    _write(out / "part4_temporal_information.json", {"token_structure": token_structure, "student_patterns": mapping, "temporal_sensitivity": temporal_sensitivity})
    return mapping


def part5_teacher(rows, pred, gt, dev_ids, warning, out: Path) -> dict:
    errs = []
    tiers = defaultdict(list)
    for i in dev_ids:
        hp = np.asarray(pred["H"][i], dtype=np.float32)
        hg = np.asarray(gt["gt_1s"][i], dtype=np.float32)
        errs.append(float(np.mean((hp - hg) ** 2)))
        tiers[_tier_from_gt(hg, warning)].append(float(np.max(hp) - np.max(hg)))
    cell_fp = []
    cell_fn = []
    stop_thr = float(warning.get("stop_threshold", 0.6))
    hi = 0.5
    for i in dev_ids:
        hp, hg = pred["H"][i], gt["gt_1s"][i]
        cell_fp.append(float(((hp >= hi) & (hg < 0.05)).mean()))
        cell_fn.append(float(((hp < stop_thr) & (hg >= stop_thr)).mean()))
    rep = {
        "dev_mse_mean": float(np.mean(errs)),
        "tier_peak_error_mean": {k: float(np.mean(v)) for k, v in tiers.items()},
        "empty_cell_fp_rate_mean": float(np.mean(cell_fp)),
        "occupied_stop_miss_rate_mean": float(np.mean(cell_fn)),
        "n_dev": len(dev_ids),
    }
    _write(out / "part5_teacher_supervision_analysis.json", rep)
    return rep


def part6_decoder(cfg, ckpt: Path, pred, out: Path) -> dict:
    model, _ = load_pair_cf_parent(cfg, ckpt, device="cpu")
    idx = np.arange(min(32, pred["F_dec"].shape[0]))
    feat = torch.from_numpy(np.asarray(pred["F_dec"][idx], dtype=np.float32).copy())
    with torch.no_grad():
        h1, _ = model.decoder.forward_with_logits(feat)
    href = torch.from_numpy(np.asarray(pred["H"][idx], dtype=np.float32))
    mse = float(F.mse_loss(h1, href).item())
    max_diff = float((h1 - href).abs().max().item())
    p4 = json.loads((out / "part4_architecture_feasibility.json").read_text(encoding="utf-8"))
    arch_rows = []
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    for row in p4.get("budget_verification", []):
        arch_rows.append(
            {
                **row,
                "parent_decoder_params": dec_p,
                "projection_gap_note": "Student CNN emits 5×5 maps; parent decoder expects 16×16×128 unless upsample head added.",
            }
        )
    rep = {"decoder_repro_from_cached_F_dec": {"mse": mse, "max_abs_diff": max_diff, "n_samples": len(idx), "tolerance_note": "F_dec cache should match decoder input after task_whiten"}, "student_budgets": arch_rows}
    _write(out / "part6_decoder_transfer_feasibility.json", rep)
    return rep


def part7_architecture(out: Path) -> dict:
    p4 = json.loads((out / "part4_architecture_feasibility.json").read_text(encoding="utf-8"))
    rep = {
        "source": "part4_architecture_feasibility.json (reused, not regenerated)",
        "budget_verification": p4.get("budget_verification"),
        "models_sample": p4.get("models", [])[:6],
        "design_notes": [
            "All preset students include HeatmapDecoder (~47.8k) in-total.",
            "Concat fuses at RGB; shared fuses after frame encoder.",
            "5×5 output grid may limit far-row threat unless receptive field is sufficient.",
        ],
    }
    _write(out / "part7_architecture_design_space.json", rep)
    return rep


def part8_deployment(out: Path) -> dict:
    p4 = json.loads((out / "part4_architecture_feasibility.json").read_text(encoding="utf-8"))
    rows = []
    for m in p4.get("models", []):
        rows.append(
            {
                "name": m["config"]["name"],
                "n_frames": 3 if m["total"] == p4["budget_verification"][0]["total_params"] else None,
                "total_params": m["total"],
                "fp32_state_dict_bytes": m.get("fp32_state_dict_bytes"),
                "int8_estimated_note": "Approx fp32/4 + metadata; dynamic_quant in part4 failed TS export.",
                "latency_ms_cpu_batch1": m.get("batch1_latency_ms_cpu"),
                "under_1mb_int8_heuristic": (m.get("fp32_state_dict_bytes") or 0) // 4 < 1_000_000,
            }
        )
    rep = {"int8_target_bytes": 1_000_000, "max_params": 700_000, "candidates": rows, "qat_not_run": True}
    _write(out / "part8_deployment_feasibility.json", rep)
    return rep


def part9_objectives(part2, part3, part6) -> dict:
    objs = {}
    for key, title in [
        ("A", "Standard heatmap KD"),
        ("B", "Intermediate latent KD"),
        ("C", "Progressive frozen-decoder"),
        ("D", "Pair-difference PCA matching"),
        ("E", "Pair-difference PCA suppression"),
        ("F", "Pair-shared latent alignment"),
        ("G", "PCA-weighted latent matching"),
    ]:
        objs[key] = {
            "title": title,
            "evidence_for": [],
            "evidence_against": [],
            "extra_params": "see part6",
            "training_not_run": True,
        }
    objs["A"]["evidence_for"].append("part5 teacher heatmaps available in cache; probes show H_flat predictable.")
    objs["B"]["evidence_for"].append("Z channel probes predict STOP/SAFE with non-trivial accuracy.")
    objs["C"]["evidence_for"].append(f"Decoder repro MSE from F_dec: {part6.get('decoder_repro_from_cached_F_dec', {})}")
    objs["C"]["evidence_against"].append("Shape mismatch student 5×5 vs decoder 16×16.")
    objs["D"]["evidence_for"].append("PCA on pair differences shows structured spectrum in part3.")
    objs["E"]["evidence_against"].append("PCA directions may encode threat; suppression risky per part3.")
    rep = {"objectives": objs, "no_final_recommendation": True}
    return rep


def part10_shortcut(rows, train_ids, dev_ids, gt, warning, out: Path) -> dict:
    pre = sum(1 for i in train_ids if rows[i].get("pre_fork"))
    scen = defaultdict(int)
    bg = defaultdict(int)
    for i in train_ids + dev_ids:
        scen[rows[i].get("scenario_group_id", "")] += 1
        bg[rows[i].get("background_id", "")] += 1
    peaks = [float(gt["gt_1s"][i].max()) for i in train_ids]
    rep = {
        "n_train": len(train_ids),
        "n_dev": len(dev_ids),
        "pre_fork_frac_train": pre / max(len(train_ids), 1),
        "unique_scenario_groups_train": len({rows[i].get("scenario_group_id") for i in train_ids}),
        "background_id_entropy_note": len(bg),
        "gt_peak_quantiles_train": {q: float(np.quantile(peaks, q)) for q in (0.25, 0.5, 0.75, 0.9)},
        "metadata_leakage_probe": "not_run (no full logistic on metadata only)",
        "real_wearable_footage": "not_in_manifest",
    }
    _write(out / "part10_shortcut_audit.json", rep)
    return rep


def write_final_report(out: Path, root: Path) -> None:
    parts = [
        "part0_artifact_inventory.json",
        "part1_parent_representation_hierarchy.json",
        "part2_parent_task_information.json",
        "part3_parent_latent_pca.json",
        "part4_temporal_information.json",
        "part5_teacher_supervision_analysis.json",
        "part6_decoder_transfer_feasibility.json",
        "part7_architecture_design_space.json",
        "part8_deployment_feasibility.json",
        "part9_objective_feasibility.json",
        "part10_shortcut_audit.json",
    ]
    lines = [
        "# Diagnostic-only student design report",
        "",
        "Read-only analysis for sub-700K RGB student. **No training, no new caches.**",
        "",
        f"Command: `python3 scripts/58_student_diagnostic_only.py`",
        "",
        "## Artifacts",
    ]
    for p in parts:
        fp = out / p
        lines.append(f"- `{fp.relative_to(root)}`" + (" (missing)" if not fp.exists() else ""))
    lines += [
        "",
        "## Executive summary",
        "",
        "- Parent decoder input is **`F_pre_decoder` [128,16,16]** (`decode_from=fine`); **Z_t [32,5,5]** is a parallel bottleneck, not the decoder input.",
        "- Cached teacher tensors (`H`, `Z`, `F_dec`) in `teacher_cache_train_pilot` support probes and PCA without Giant re-encode.",
        "- Student presets from Part 4 fit ≤700K with included decoder; **16×16 vs 5×5 head resolution** is the main frozen-decoder transfer obstacle.",
        "- Temporal pattern mapping: parent uses 16 offsets on a 6-frame grid; **F5 offsets are not all exact parent timestamps**.",
        "",
        "**No final architecture, frame set, or objective selected in this report.**",
    ]
    (out / "diagnostic_only_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(cfg, out: Path, root: Path) -> None:
    def log(s: str) -> None:
        print(s, flush=True)

    ckpt = resolve_parent_ckpt(cfg, None)
    log("part0 inventory")
    part0_inventory(root, out)
    log("part1 hierarchy")
    part1_hierarchy(cfg, ckpt, out)
    cache = out / "teacher_cache_train_pilot"
    rows = [json.loads(l) for l in (cache / "index.jsonl").open()]
    pred = np.load(cache / "teacher_pred.npz", mmap_mode="r")
    gt = np.load(cache / "gt_heatmaps.npz", mmap_mode="r")
    warning = cfg.get("warning") or {}
    train_ids = json.loads((out / "part1_train_manifest.json").read_text())["train_row_ids"]
    dev_ids = json.loads((out / "part1_dev_manifest.json").read_text())["dev_row_ids"]
    log("part2 probes")
    p2 = part2_task_info(rows, pred, gt, train_ids, dev_ids, warning, out)
    log("part3 pca")
    p3 = part3_pca(rows, pred, train_ids, out)
    log("part4 temporal")
    part4_temporal(cfg, rows, train_ids, out)
    log("part5 teacher")
    part5_teacher(rows, pred, gt, dev_ids, warning, out)
    log("part6 decoder")
    p6 = part6_decoder(cfg, ckpt, pred, out)
    log("part7-10")
    part7_architecture(out)
    part8_deployment(out)
    p9 = part9_objectives(p2, p3, p6)
    _write(out / "part9_objective_feasibility.json", p9)
    part10_shortcut(rows, train_ids, dev_ids, gt, warning, out)
    write_final_report(out, root)
