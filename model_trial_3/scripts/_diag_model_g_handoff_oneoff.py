#!/usr/bin/env python3
"""Model G handoff diagnostics: full-val metrics (CPU), calibration sweeps, gradient conflict.

Run from repo: ``python scripts/_diag_model_g_handoff_oneoff.py``

Writes JSON under ``reports/diag_model_g_handoff/``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.dataset import TeacherDataset, build_dataloader
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.splits import load_splits
from collision_jepa3.evaluation.evaluator import evaluate_loader
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.training.bg_regularization import (
    apply_token_mask,
    clean_anchor_drop_strength,
    deterministic_channel_mask,
    load_channel_sets_from_p4,
)
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state
from collision_jepa3.training.engine import move_batch, teacher_forward


def _stack_batches(loader, model, device: str):
    model.eval()
    pred, gt, valid = [], [], []
    outcomes, backgrounds, scenarios, pre_fork = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.no_grad():
            out = teacher_forward(model, batch)
        p = out["Hhat_1s"].float().cpu().numpy()
        g = batch["h_1s"].float().cpu().numpy()
        amb = batch["amb_1s"].detach().cpu().numpy().astype(bool)
        pred.append(p)
        gt.append(g)
        valid.append(~amb)
        outcomes.extend([str(x) for x in batch["outcome_type"]])
        backgrounds.extend([str(x) for x in batch["background_id"]])
        scenarios.extend([str(x) for x in batch["scenario_type"]])
        pre_fork.append(batch["pre_fork"].detach().cpu().numpy().astype(bool))
    return {
        "pred": np.concatenate(pred, axis=0),
        "gt": np.concatenate(gt, axis=0),
        "valid": np.concatenate(valid, axis=0),
        "outcome": np.asarray(outcomes, dtype=object),
        "background": np.asarray(backgrounds, dtype=object),
        "scenario": np.asarray(scenarios, dtype=object),
        "pre_fork": np.concatenate(pre_fork, axis=0),
    }


def _band_frame(peak: np.ndarray, caution: float, stop: float) -> np.ndarray:
    out = np.zeros(peak.shape, dtype=np.int64)
    out[peak >= caution] = 1
    out[peak >= stop] = 2
    return out


def _peak_stats(pred_peak: np.ndarray, gt_peak: np.ndarray, caution: float, stop: float) -> dict:
    def summ(mask: np.ndarray) -> dict:
        x = pred_peak[mask]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "p10": float(np.quantile(x, 0.10)),
            "p25": float(np.quantile(x, 0.25)),
            "p75": float(np.quantile(x, 0.75)),
            "p90": float(np.quantile(x, 0.90)),
        }

    gf = _band_frame(gt_peak, caution, stop)
    return {
        "GT_SAFE": summ(gf == 0),
        "GT_THREAT": summ(gf == 1),
        "GT_STOP": summ(gf == 2),
    }


def _frame_warning_metrics(pred_peak: np.ndarray, gt_peak: np.ndarray, caution: float, stop: float) -> dict:
    gtf = _band_frame(gt_peak, caution, stop)
    prf = _band_frame(pred_peak, caution, stop)
    n = len(gtf)
    stop_mask = gtf == 2
    safe_mask = gtf == 0
    threat_mask = gtf == 1
    stop_rec = float((prf[stop_mask] == 2).mean()) if stop_mask.any() else float("nan")
    safe_rec = float((prf[safe_mask] == 0).mean()) if safe_mask.any() else float("nan")
    threat_rec = float((prf[threat_mask] >= 1).mean()) if threat_mask.any() else float("nan")
    flip = float((gtf != prf).mean()) if n else float("nan")
    stos = float(((gtf == 2) & (prf == 0)).mean()) if n else float("nan")
    sast = float(((gtf == 0) & (prf == 2)).mean()) if n else float("nan")
    return {
        "stop_recall": stop_rec,
        "safe_recall": safe_rec,
        "threat_recall": threat_rec,
        "warning_flip_rate": flip,
        "stop_to_safe_rate": stos,
        "safe_to_stop_rate": sast,
    }


def _stop_recall_at_peak_threshold(pred_peak: np.ndarray, gt_peak: np.ndarray, thr: float, stop: float) -> float:
    mask = gt_peak >= stop
    if not mask.any():
        return float("nan")
    return float((pred_peak[mask] >= thr).mean())


def _safe_recall_at_peak_threshold(pred_peak: np.ndarray, gt_peak: np.ndarray, thr: float, caution: float) -> float:
    mask = gt_peak < caution
    if not mask.any():
        return float("nan")
    return float((pred_peak[mask] < thr).mean())


def _subgroup_stop_recall(
    pred_peak: np.ndarray, gt_peak: np.ndarray, pre_fork: np.ndarray, stop: float
) -> dict[str, float]:
    stop_gt = gt_peak >= stop
    out = {}
    for name, m in [("pre_fork", pre_fork), ("post_fork", ~pre_fork)]:
        mm = stop_gt & m
        out[f"{name}_stop_recall"] = (
            float((pred_peak[mm] >= stop).mean()) if mm.any() else float("nan")
        )
    return out


def _flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter], retain: bool) -> list[torch.Tensor | None]:
    return torch.autograd.grad(loss, params, retain_graph=retain, create_graph=False, allow_unused=True)


def _grad_norm_from_list(grads: list[torch.Tensor | None]) -> float:
    t = 0.0
    for g in grads:
        if g is not None:
            t += float(g.detach().float().pow(2).sum().item())
    return math.sqrt(t)


def _cosine_from_grads(a: list[torch.Tensor | None], b: list[torch.Tensor | None]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            continue
        xf = x.detach().float().flatten()
        yf = y.detach().float().flatten()
        dot += float((xf * yf).sum().item())
        na += float((xf * xf).sum().item())
        nb += float((yf * yf).sum().item())
    if na <= 0 or nb <= 0:
        return float("nan")
    return dot / (math.sqrt(na) * math.sqrt(nb))


def main() -> None:
    device = "cpu"
    out_dir = ROOT / "reports" / "diag_model_g_handoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.load(ROOT / "configs" / "default.yaml")
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("evaluation.val_subset_fraction", 1.0)
    cfg.set("training.pin_memory", False)
    cfg.set("training.num_workers", 0)

    warn_cfg = cfg.get("warning")
    caution = float(warn_cfg["caution_threshold"])
    stop_thr = float(warn_cfg["stop_threshold"])

    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    val_ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
    val_loader = build_dataloader(val_ds, cfg, shuffle=False)

    ckpts = {
        "original_gr40g": ROOT / "checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
        "clean_fromscratch_best": ROOT / "checkpoints/clean_anchor_fromscratch_s1337/teacher_phase_a_best.pt",
        "clean_fromscratch_last": ROOT / "checkpoints/clean_anchor_fromscratch_s1337/teacher_phase_a_last.pt",
    }

    # --- 1: full val (evaluator horizons + operating + selection) ---
    full_val = {}
    for name, path in ckpts.items():
        model = CollisionTeacher3(cfg, dummy_backbone=False)
        blob = load_checkpoint(str(path), map_location=device)
        load_teacher_state(model, blob)
        model.to(device)
        metrics = evaluate_loader(model, val_loader, cfg, device, loss_fn=TeacherLoss(cfg, device=device))
        full_val[name] = {
            "checkpoint": str(path),
            "epoch_saved": blob.get("epoch"),
            "selection": metrics.get("selection"),
            "operating": metrics.get("operating"),
            "horizon_1s": metrics.get("horizons", {}).get("1s"),
            "loss_means": metrics.get("loss"),
        }
        del model

    (out_dir / "full_val_evaluate_loader.json").write_text(json.dumps(full_val, indent=2, default=str), encoding="utf-8")

    # --- 6: peaks, sweeps (two main checkpoints) ---
    calibration = {}
    for label in ("original_gr40g", "clean_fromscratch_best"):
        path = ckpts[label]
        model = CollisionTeacher3(cfg, dummy_backbone=False)
        blob = load_checkpoint(str(path), map_location=device)
        load_teacher_state(model, blob)
        model.to(device)
        pack = _stack_batches(val_loader, model, device)
        del model
        pred = pack["pred"]
        gt = pack["gt"]
        valid = pack["valid"]
        vcell = np.broadcast_to(valid[:, None, None], pred.shape)
        pred_peak = pred.reshape(len(pred), -1).max(axis=1)
        gt_peak = gt.reshape(len(gt), -1).max(axis=1)
        op = operating_metrics(pred, gt, vcell)
        frame = _frame_warning_metrics(pred_peak, gt_peak, caution, stop_thr)
        fork = _subgroup_stop_recall(pred_peak, gt_peak, pack["pre_fork"], stop_thr)
        frame.update(fork)
        stop_thr_sweep = {
            str(t): _stop_recall_at_peak_threshold(pred_peak, gt_peak, t, stop_thr) for t in np.arange(0.45, 0.76, 0.05)
        }
        safe_thr_sweep = {str(t): _safe_recall_at_peak_threshold(pred_peak, gt_peak, t, caution) for t in [0.15, 0.2, 0.25, 0.3, 0.35]}
        calibration[label] = {
            "peak_by_gt_band": _peak_stats(pred_peak, gt_peak, caution, stop_thr),
            "frame_warning": frame,
            "operating_cells": op,
            "stop_recall_vs_peak_threshold_on_gt_stop_frames": stop_thr_sweep,
            "safe_recall_vs_peak_threshold_on_gt_safe_frames": safe_thr_sweep,
        }

    (out_dir / "calibration_sweeps.json").write_text(json.dumps(calibration, indent=2, default=str), encoding="utf-8")

    # --- 3–4: gradient conflict on train paired batches (epochs 1–5) ---
    clean_ckpt_path = ckpts["clean_fromscratch_best"]
    blob0 = load_checkpoint(str(clean_ckpt_path), map_location=device)
    train_cfg = Config(blob0["config"], source=cfg.source)
    train_cfg.set("evaluation.val_subset_fraction", 1.0)
    train_cfg.set("training.pin_memory", False)
    train_cfg.set("training.num_workers", 0)

    train_ds = TeacherDataset(train_cfg, records, "train", splits, dummy_backbone=False, use_token_cache=True)
    train_loader = build_dataloader(train_ds, train_cfg, shuffle=True)
    p4 = (ROOT / train_cfg.get("training.bg_reg.p4_report")).resolve()
    targeted, n_eligible_p4, _eligible_all = load_channel_sets_from_p4(p4)
    p_drop = float(train_cfg.get("training.bg_reg.dropout_prob", 0.25))
    beta = float(blob0.get("bg_beta_clean_anchor", 0.0996359840494786))

    def _spatial_params(m: CollisionTeacher3) -> list[torch.nn.Parameter]:
        return [p for n, p in m.adapter.named_parameters() if "spatial_proj" in n]

    def _temporal_params(m: CollisionTeacher3) -> list[torch.nn.Parameter]:
        return [p for n, p in m.adapter.named_parameters() if any(k in n for k in ("temporal", "time_pool", "mix"))]

    grad_report = {}
    for ep in [1, 2, 3, 4, 5]:
        ck = ROOT / f"checkpoints/clean_anchor_fromscratch_s1337/teacher_phase_a_epoch_{ep:02d}.pt"
        if not ck.exists():
            continue
        model = CollisionTeacher3(train_cfg, dummy_backbone=False)
        blob = load_checkpoint(str(ck), map_location=device)
        load_teacher_state(model, blob)
        model.to(device)
        model.train()
        loss_fn = TeacherLoss(train_cfg, device=device)
        strength = clean_anchor_drop_strength(ep)

        all_params = list(model.adapter.parameters()) + list(model.decoder.parameters())
        s_par = _spatial_params(model)
        t_par = _temporal_params(model)
        d_par = list(model.decoder.parameters())

        clean_n, drop_n, w_aux_n, ratio, cos_all = [], [], [], [], []
        cos_neg = 0
        cos_sp, cos_t, cos_d = [], [], []

        torch.manual_seed(1337 + ep)
        it = iter(train_loader)
        n_done = 0
        target_batches = 50
        while n_done < target_batches:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            batch = move_batch(batch, device)
            if "tokens" not in batch or not bool(batch.get("bg_reg_paired", False)):
                continue
            bsz = int(batch["tokens"].shape[0])
            if bsz < 2 or bsz % 2 != 0:
                continue
            pair_ids = batch.get("background_pair_id") or []
            pid = str(pair_ids[0]) if pair_ids else ""
            key_pick = f"{int(ep)}|{pid}|clean_anchor_aux"
            h = hashlib.sha256(key_pick.encode("utf-8")).digest()
            thr_pick = min(255, max(0, int(round(0.5 * 256))))
            if not (h[0] < thr_pick):
                continue

            mid = bsz // 2
            tok = batch["tokens"].clone()
            anchors = batch["anchor"]
            for i in range(mid):
                anc = int(anchors[i].item()) if torch.is_tensor(anchors[i]) else int(anchors[i])
                mk = f"{int(ep)}|{pair_ids[i]}|{anc}"
                mask_c = deterministic_channel_mask(
                    targeted,
                    c_dim=int(tok.shape[-1]),
                    device=device,
                    dtype=tok.dtype,
                    p_drop=p_drop,
                    key=mk,
                )
                tok[i] = apply_token_mask(tok[i], mask_c)
                tok[i + mid] = apply_token_mask(tok[i + mid], mask_c)
            drop_batch = {**batch, "tokens": tok}

            model.zero_grad(set_to_none=True)
            out_c = teacher_forward(model, batch)
            l_clean = loss_fn(out_c, batch).total
            g_c = _flat_grad(l_clean, all_params, retain=False)
            n_clean = _grad_norm_from_list(g_c)

            model.zero_grad(set_to_none=True)
            out_d = teacher_forward(model, drop_batch)
            l_drop = loss_fn(out_d, drop_batch).total
            g_d = _flat_grad(l_drop, all_params, retain=False)
            n_drop = _grad_norm_from_list(g_d)

            clean_n.append(n_clean)
            drop_n.append(n_drop)
            w_aux_n.append(beta * strength * n_drop)
            if n_clean > 1e-12:
                ratio.append((beta * strength * n_drop) / n_clean)

            c = _cosine_from_grads(g_c, g_d)
            if not math.isnan(c):
                cos_all.append(c)
                if c < 0:
                    cos_neg += 1

            model.zero_grad(set_to_none=True)
            out_c = teacher_forward(model, batch)
            l_clean = loss_fn(out_c, batch).total
            gc_sp = torch.autograd.grad(
                l_clean, s_par, retain_graph=False, create_graph=False, allow_unused=True
            )
            model.zero_grad(set_to_none=True)
            out_d = teacher_forward(model, drop_batch)
            l_drop = loss_fn(out_d, drop_batch).total
            gd_sp = torch.autograd.grad(
                l_drop, s_par, retain_graph=False, create_graph=False, allow_unused=True
            )
            csp = _cosine_from_grads(list(gc_sp), list(gd_sp))
            if not math.isnan(csp):
                cos_sp.append(csp)

            model.zero_grad(set_to_none=True)
            out_c = teacher_forward(model, batch)
            l_clean = loss_fn(out_c, batch).total
            gc_t = torch.autograd.grad(
                l_clean, t_par, retain_graph=False, create_graph=False, allow_unused=True
            )
            model.zero_grad(set_to_none=True)
            out_d = teacher_forward(model, drop_batch)
            l_drop = loss_fn(out_d, drop_batch).total
            gd_t = torch.autograd.grad(
                l_drop, t_par, retain_graph=False, create_graph=False, allow_unused=True
            )
            ct = _cosine_from_grads(list(gc_t), list(gd_t))
            if not math.isnan(ct):
                cos_t.append(ct)

            model.zero_grad(set_to_none=True)
            out_c = teacher_forward(model, batch)
            l_clean = loss_fn(out_c, batch).total
            gc_d = torch.autograd.grad(
                l_clean, d_par, retain_graph=False, create_graph=False, allow_unused=True
            )
            model.zero_grad(set_to_none=True)
            out_d = teacher_forward(model, drop_batch)
            l_drop = loss_fn(out_d, drop_batch).total
            gd_d = torch.autograd.grad(
                l_drop, d_par, retain_graph=False, create_graph=False, allow_unused=True
            )
            cd = _cosine_from_grads(list(gc_d), list(gd_d))
            if not math.isnan(cd):
                cos_d.append(cd)

            n_done += 1

        grad_report[f"epoch_{ep}"] = {
            "n_batches": len(clean_n),
            "beta": beta,
            "strength_a_e": strength,
            "clean_grad_L2_mean": float(np.mean(clean_n)) if clean_n else None,
            "drop_grad_L2_mean": float(np.mean(drop_n)) if drop_n else None,
            "weighted_aux_grad_L2_mean": float(np.mean(w_aux_n)) if w_aux_n else None,
            "ratio_weighted_aux_over_clean_mean": float(np.mean(ratio)) if ratio else None,
            "cosine_clean_vs_drop_mean": float(np.mean(cos_all)) if cos_all else None,
            "frac_batches_cos_negative": cos_neg / len(cos_all) if cos_all else None,
            "cosine_spatial_adapter_mean": float(np.mean(cos_sp)) if cos_sp else None,
            "cosine_temporal_adapter_mean": float(np.mean(cos_t)) if cos_t else None,
            "cosine_decoder_mean": float(np.mean(cos_d)) if cos_d else None,
        }
        del model

    (out_dir / "gradient_conflict.json").write_text(json.dumps(grad_report, indent=2), encoding="utf-8")

    # --- 4: mask audit (deterministic mask stats for sample keys) ---
    c_dim = 1408
    dtype = torch.float32
    dev = torch.device("cpu")
    dropped_counts = []
    for trial in range(200):
        mk = f"audit|{trial}"
        m = deterministic_channel_mask(targeted, c_dim=c_dim, device=dev, dtype=dtype, p_drop=p_drop, key=mk)
        dropped = int((m == 0.0).sum().item())
        dropped_counts.append(dropped)
    mask_audit = {
        "n_targeted_channels": len(targeted),
        "dropout_prob_config": p_drop,
        "expected_drops_mean": float(np.mean(dropped_counts)),
        "expected_drops_std": float(np.std(dropped_counts)),
        "n_eligible_reported_p4": int(n_eligible_p4),
    }
    (out_dir / "dropout_mask_audit.json").write_text(json.dumps(mask_audit, indent=2), encoding="utf-8")

    print("wrote reports under", out_dir)


if __name__ == "__main__":
    main()
