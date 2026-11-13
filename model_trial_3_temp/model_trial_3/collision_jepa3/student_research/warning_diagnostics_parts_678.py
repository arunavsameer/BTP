"""Parts 6–8 warning diagnostics (read-only)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.evaluation.metrics import STOP, classify_heatmap
from collision_jepa3.student_research.diagnostic_only import _tier_from_gt
from collision_jepa3.student_research.fdec_pilot_eval import evaluate_fdec_pilot
from collision_jepa3.student_research.fdec_pilot_integrity import build_integrity_report
from collision_jepa3.student_research.fdec_pilot_train import _loss_s0, _loss_s1, apply_backbone, prepare_shared_backbone
from collision_jepa3.student_research.fdec_student import (
    FDEC_FRAME_OFFSETS,
    FDecNormStats,
    StudentS0,
    StudentS1,
    build_parent_decoder,
)
from collision_jepa3.student_research.parent import load_pair_cf_parent, resolve_parent_ckpt
from collision_jepa3.student_research.warning_diagnostics_round2 import peak_stats, _class_stats


def _grad_norm_list(params: list[torch.nn.Parameter]) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum())
    return s**0.5


def _cos_grads(a: list[torch.nn.Parameter], b: list[torch.nn.Parameter]) -> float | None:
    va, vb = [], []
    for pa, pb in zip(a, b):
        if pa.grad is None or pb.grad is None:
            continue
        va.append(pa.grad.detach().reshape(-1))
        vb.append(pb.grad.detach().reshape(-1))
    if not va:
        return None
    x, y = torch.cat(va), torch.cat(vb)
    return float(F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())


def run_part6_fdec_alignment(
    dev_ids: list[int],
    f_student: np.ndarray,
    f_teacher: np.ndarray,
    gts: np.ndarray,
    warning: dict,
    decoder: torch.nn.Module,
    device: str,
) -> dict[str, Any]:
    fs = torch.from_numpy(f_student).to(device)
    ft = torch.from_numpy(f_teacher).to(device)
    diff = (fs - ft).float()
    mse = float((diff**2).mean().item())
    mae = float(diff.abs().mean().item())
    cos = float(F.cosine_similarity(fs.flatten(1), ft.flatten(1), dim=1).mean().item())
    ch_mse = (diff**2).mean(dim=(0, 2, 3)).cpu().numpy()
    ch_cos = []
    for c in range(fs.shape[1]):
        ch_cos.append(float(F.cosine_similarity(fs[:, c].flatten(1), ft[:, c].flatten(1), dim=1).mean().item()))
    sp_mse = (diff**2).mean(dim=(0, 1)).cpu().numpy()
    tiers = [_tier_from_gt(gts[i], warning) for i in range(len(gts))]
    by_tier: dict[str, list[float]] = defaultdict(list)
    for i, t in enumerate(tiers):
        by_tier[t].append(float((diff[i] ** 2).mean().item()))
    worst_ch = np.argsort(ch_mse)[-max(1, int(0.1 * len(ch_mse))) :]
    worst_sp = np.argsort(sp_mse.reshape(-1))[-max(1, int(0.1 * sp_mse.size)) :]
    decoder = decoder.to(device)
    sens = []
    decoder.eval()
    with torch.no_grad():
        h0, _ = decoder.forward_with_logits(ft[:32].to(device))
        base_peak = h0.max(dim=-1).values.max(dim=-1).values
    for c in worst_ch[:5].tolist():
        fpert = ft[:32].clone().to(device)
        fpert[:, c] *= 1.1
        with torch.no_grad():
            h1, _ = decoder.forward_with_logits(fpert)
            peak = h1.max(dim=-1).values.max(dim=-1).values
        sens.append({"channel": int(c), "mean_peak_delta": float((peak - base_peak).mean().item())})
    return {
        "overall_mse": mse,
        "overall_mae": mae,
        "overall_cosine": cos,
        "per_channel_mse_mean": float(ch_mse.mean()),
        "per_channel_cosine_mean": float(np.mean(ch_cos)),
        "worst_10pct_channels": worst_ch.tolist(),
        "worst_10pct_spatial_flat": worst_sp.tolist(),
        "by_gt_tier_mse_mean": {k: float(np.mean(v)) for k, v in by_tier.items()},
        "decoder_sensitivity_top_error_channels": sens,
        "n_dev": int(fs.shape[0]),
    }


@torch.no_grad()
def _infer_f_student(s1: StudentS1, loader, device: str) -> np.ndarray:
    s1.eval().to(device)
    chunks = []
    for batch in loader:
        out = s1(batch["clip"].to(device))
        chunks.append(out["F_student"].float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_part7_gradients(
    cfg,
    cache_dir: Path,
    train_ids: list[int],
    report_root: Path,
    baseline_init: Path,
    device: str,
    norm: FDecNormStats,
    *,
    s0_ckpt: Path,
    s1_ckpt: Path,
    num_workers: int,
    n_batches: int = 8,
) -> dict[str, Any]:
    norm_d = norm.to(device)
    from collision_jepa3.student_research.fdec_pilot_dataset import FDecPilotDataset, build_frame_store, make_fdec_loader

    store, _ = build_frame_store(cfg, cache_dir, train_ids, FDEC_FRAME_OFFSETS)
    tr_ds = FDecPilotDataset(cache_dir, train_ids, FDEC_FRAME_OFFSETS, load_f_dec=True)
    tr_ds.attach_frame_store(store, {})
    tr_loader = make_fdec_loader(tr_ds, 8, True, num_workers=num_workers)

    shared = prepare_shared_backbone(report_root, seed=1337, baseline_init=baseline_init)
    s0 = StudentS0(n_frames=5)
    apply_backbone(s0, shared)
    s0.load_state_dict(torch.load(s0_ckpt, map_location="cpu", weights_only=True))
    parent_dec = build_parent_decoder(cfg, "cpu")
    s1 = StudentS1(n_frames=5, parent_decoder=parent_dec)
    apply_backbone(s1, shared)
    s1.load_state_dict(torch.load(s1_ckpt, map_location="cpu", weights_only=True))
    s1.set_decoder_trainable(False)

    def s0_groups(m):
        return {
            "backbone": list(m.net.stem.parameters()) + list(m.net.encoder.parameters()) + list(m.net.head.parameters()),  # type: ignore
            "decoder": list(m.net.decoder.parameters()),
        }

    def s1_groups(m):
        return {
            "backbone": list(m.backbone.stem.parameters()) + list(m.backbone.encoder.parameters()) + list(m.backbone.head.parameters()),  # type: ignore
            "proj": list(m.proj.parameters()),
            "decoder": list(m.decoder.parameters()),
        }

    report: dict[str, Any] = {"S0": {"batches": []}, "S1": {"batches": []}}
    for bi, batch in enumerate(tr_loader):
        if bi >= n_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        s0.eval().to(device)
        s1.eval().to(device)
        s0_g = s0_groups(s0)
        s1_g = s1_groups(s1)

        def record_s0():
            out = s0(batch["clip"])
            losses = {
                "gt": F.mse_loss(out["Hhat_1s"], batch["gt_1s"]),
                "kd": F.mse_loss(out["Hhat_1s"], batch["H_teacher"]),
            }
            all_p = [p for p in s0.parameters() if p.requires_grad]
            rec: dict[str, Any] = {}
            vecs = {}
            for name, loss in losses.items():
                rec[name] = {}
                g = torch.autograd.grad(loss, all_p, retain_graph=True, allow_unused=True)
                for tag, subset in s0_g.items():
                    sq = sum(float(gi.pow(2).sum()) for gi, p in zip(g, all_p) if any(p is q for q in subset) and gi is not None)
                    rec[name][tag] = sq**0.5
                vecs[name] = torch.cat([gi.reshape(-1) for gi in g if gi is not None])
            if "gt" in vecs and "kd" in vecs:
                rec["cos_gt_kd_decoder"] = float(F.cosine_similarity(vecs["gt"].unsqueeze(0), vecs["kd"].unsqueeze(0)).item())
            return rec

        def record_s1():
            out = s1(batch["clip"])
            f_s, f_t = out["F_student"], batch["F_teacher"]
            f_sn = (f_s - norm_d.mu) / (norm_d.sigma + 1e-6)
            f_tn = (f_t - norm_d.mu) / (norm_d.sigma + 1e-6)
            losses = {
                "gt": F.mse_loss(out["Hhat_1s"], batch["gt_1s"]),
                "kd": F.smooth_l1_loss(out["Hhat_1s"], batch["H_teacher"], beta=0.1),
                "fdec_unweighted": F.smooth_l1_loss(f_s, f_t),
                "fdec_weighted": F.smooth_l1_loss(f_sn, f_tn),
            }
            rec = {}
            grad_vecs: dict[str, torch.Tensor] = {}
            all_trainable = [p for p in s1.parameters() if p.requires_grad]
            for name, loss in losses.items():
                rec[name] = {}
                g = torch.autograd.grad(loss, all_trainable, retain_graph=True, allow_unused=True)
                for tag, subset in s1_g.items():
                    sq = sum(
                        float(gi.pow(2).sum())
                        for gi, p in zip(g, all_trainable)
                        if any(p is q for q in subset) and gi is not None
                    )
                    rec[name][tag] = sq**0.5
                grad_vecs[name] = torch.cat([gi.reshape(-1) for gi in g if gi is not None])
            rec["fdec_weighted_eta_effect"] = {k: rec["fdec_weighted"][k] * 3.717 for k in rec["fdec_weighted"]}
            if "gt" in grad_vecs and "kd" in grad_vecs:
                rec["cos_gt_kd_backbone"] = float(
                    F.cosine_similarity(grad_vecs["gt"].unsqueeze(0), grad_vecs["kd"].unsqueeze(0)).item()
                )
            if "gt" in grad_vecs and "fdec_weighted" in grad_vecs:
                rec["cos_gt_fdec_weighted"] = float(
                    F.cosine_similarity(grad_vecs["gt"].unsqueeze(0), grad_vecs["fdec_weighted"].unsqueeze(0)).item()
                )
            return rec

        report["S0"]["batches"].append(record_s0())
        report["S1"]["batches"].append(record_s1())

    return report


def _pair_analysis(meta_rows: list[dict], pks: np.ndarray, verified: set[tuple], warning: dict) -> dict:
    stop_thr = float(warning.get("stop_threshold", 0.6))
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for m in meta_rows:
        pid = m.get("background_pair_id") or ""
        if not pid:
            continue
        key = (pid, int(m["frame"]))
        by_key[key].append(m)
    gaps, flips, n_pairs = [], 0, 0
    both_stop, both_safe = 0, 0
    for key, vals in by_key.items():
        if len(vals) != 2 or key not in verified:
            continue
        ia, ib = vals[0]["idx"], vals[1]["idx"]
        pa, pb = float(pks[ia]), float(pks[ib])
        gaps.append(abs(pa - pb))
        if (pa >= stop_thr) != (pb >= stop_thr):
            flips += 1
        gt_a, gt_b = vals[0]["gt_tier"], vals[1]["gt_tier"]
        if gt_a == "STOP" and gt_b == "STOP" and pa >= stop_thr and pb >= stop_thr:
            both_stop += 1
        if gt_a == "SAFE" and gt_b == "SAFE" and pa < stop_thr and pb < stop_thr:
            both_safe += 1
        n_pairs += 1
    arr = np.asarray(gaps, dtype=np.float32)
    any_warn = bool((pks >= stop_thr).any())
    return {
        "n_verified_pairs_evaluated": n_pairs,
        "n_verified_keys_supplied": len(verified),
        "pair_gap_mean": float(arr.mean()) if arr.size else None,
        "pair_gap_median": float(np.median(arr)) if arr.size else None,
        "pair_gap_p90": float(np.quantile(arr, 0.9)) if arr.size else None,
        "warning_tier_flips": flips if n_pairs else None,
        "warning_flip_rate": (flips / n_pairs) if n_pairs else None,
        "flip_rate_note": (
            "Near-zero flips may reflect no peaks>=stop_threshold, not robustness" if not any_warn else None
        ),
        "both_background_correct_stop": both_stop,
        "both_background_correct_safe": both_safe,
    }


def run_part8(
    rows: list[dict],
    dev_ids: list[int],
    verified_dev: list[tuple],
    warning: dict,
    preds_npz: Path,
    *,
    s0_h: np.ndarray,
    s1_h: np.ndarray,
    parent_h: np.ndarray,
    gts: np.ndarray,
    pre_fork: np.ndarray,
) -> dict[str, Any]:
    meta_rows = []
    peaks = {
        "parent": parent_h.max(axis=(1, 2)),
        "s0": s0_h.max(axis=(1, 2)),
        "s1": s1_h.max(axis=(1, 2)),
    }
    rid_to_i = {int(r): i for i, r in enumerate(np.load(preds_npz)["row_id"])}
    for rid in dev_ids:
        i = rid_to_i[rid]
        row = rows[rid]
        meta_rows.append(
            {
                "idx": i,
                "row_id": rid,
                "background_pair_id": row.get("background_pair_id", ""),
                "frame": int(row["frame"]),
                "outcome_group_id": row.get("outcome_group_id", ""),
                "background_id": row.get("background_id", ""),
                "pre_fork": bool(row.get("pre_fork")),
                "gt_tier": _tier_from_gt(gts[i], warning),
            }
        )
    verified = {tuple(k) for k in verified_dev}
    by_og: dict[str, dict] = defaultdict(dict)
    for m, pks in peaks.items():
        for og in sorted({r.get("outcome_group_id", "") for r in meta_rows}):
            idxs = [i for i, r in enumerate(meta_rows) if r.get("outcome_group_id") == og]
            if idxs:
                by_og[og][m] = peak_stats(pks[[meta_rows[i]["idx"] for i in idxs]])
    out = {
        "pair_metrics": {m: _pair_analysis(meta_rows, peaks[m], verified, warning) for m in peaks},
        "pre_fork_post_fork": {},
        "by_outcome_group_peak_stats": dict(by_og),
        "excluded_pairs_note": "Keys with !=2 rows or GT mismatch excluded via verified_pair_keys_dev",
    }
    for model, pks in peaks.items():
        pre_m = pre_fork.astype(bool)
        pf_stop = []
        for i, tier in enumerate([_tier_from_gt(gts[j], warning) for j in range(len(gts))]):
            if tier != "STOP":
                continue
            if pre_m[i]:
                pf_stop.append(("pre", pks[i]))
            else:
                pf_stop.append(("post", pks[i]))
        out["pre_fork_post_fork"][model] = {
            "pre_fork_n": int(pre_m.sum()),
            "post_fork_n": int((~pre_m).sum()),
            "peak_mean_pre_fork": float(pks[pre_m].mean()) if pre_m.any() else None,
            "peak_mean_post_fork": float(pks[~pre_m].mean()) if (~pre_m).any() else None,
        }
    return out


def write_final_report(report_dir: Path, parts: dict[str, Path]) -> None:
    lines = [
        "# Student warning diagnostics — final report (Parts 1–8)",
        "",
        "## Confirmed",
        "- Students compress peak scores: neither S0 nor S1 crosses operational 0.25/0.60 on dev.",
        "- S1 improves cell-level Recall@5FA/AP vs S0 without operational STOP recall.",
        "- Part 5: weak F_S through D_P; S1 decoder partially compensates; D_S1(F_T) near parent.",
        "",
        "## Part 6–8 (this completion)",
        f"- See `{parts.get('part6', 'part6_fdec_alignment.json')}`",
        f"- See `{parts.get('part7', 'part7_late_gradient_diagnostics.json')}`",
        f"- See `{parts.get('part8', 'part8_group_and_pair_diagnostics.json')}`",
        "",
        "## Uncertain / requires experiment",
        "- Whether permanent decoder freeze (S2) improves decoder-compatible F_S and warning peaks.",
        "- Whether late-stage gradient conflict (Part 7) causally limits representation learning.",
    ]
    (report_dir / "student_warning_diagnostics_final_report.md").write_text("\n".join(lines), encoding="utf-8")
