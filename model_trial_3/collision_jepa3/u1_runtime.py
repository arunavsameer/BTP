"""Dump G predictions, score oracles, train/eval the U1 variance head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.dataset import TeacherDataset, build_dataloader
from .data.pair_variance import pair_targets_for_window
from .evaluation.counterfactual import (
    KAPPA_GRID,
    STOP_SWEEP,
    apply_mu_sigma_lookup,
    bernoulli_var,
    cluster_bootstrap,
    feasible_vs_g,
    fit_mu_sigma_lookup,
    jsonify,
    risk_maps,
    select_kappa,
    twin_accounting,
    upgrade_source,
    variance_diagnostics,
    warning_report,
)
from .models.teacher import CollisionTeacher3
from .models.vjepa_backbone import DummyVJEPA2Backbone
from .losses.variance import weighted_var_loss
from .training.checkpoint import load_checkpoint, load_teacher_state
from .training.engine import move_batch, teacher_forward


def dummy_giant(cfg) -> DummyVJEPA2Backbone:
    return DummyVJEPA2Backbone(
        hidden_size=1408,
        crop_size=int(cfg.get("preprocessing.input_size", 256)),
        patch_size=16,
        tubelet_size=2,
        num_frames=int(cfg.get("sampling.input_frames", 16)),
        num_layers=1,
        mean=tuple(cfg.get("preprocessing.mean")),
        std=tuple(cfg.get("preprocessing.std")),
    )


def load_frozen_g(
    cfg, ckpt: Path, device: str, *, variance_head: bool, freeze_variance_only: bool = True
) -> CollisionTeacher3:
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", bool(variance_head))
    model = CollisionTeacher3(cfg, backbone=dummy_giant(cfg))
    blob = load_checkpoint(ckpt, map_location="cpu")
    load_teacher_state(model, blob)
    model.to(device)
    model.eval()
    if variance_head and freeze_variance_only:
        model.freeze_except_variance()
    return model


@torch.no_grad()
def collect_split(model, dataset: TeacherDataset, cfg, device: str) -> dict[str, Any]:
    loader = build_dataloader(dataset, cfg, shuffle=False)
    model.eval()
    pred, gt_mean = [], []
    gt_clip, v_true, envelope = [], [], []
    pre, paired, canonical, blob_n = [], [], [], []
    collision, anchor = [], []
    clip_id, group, outcome, scenario, ogid = [], [], [], [], []
    v_hat = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = teacher_forward(model, batch)
        mu = out["Hhat_1s"].float().cpu().numpy()
        pred.append(mu)
        if "Hhat_var" in out:
            v_hat.append(out["Hhat_var"].float().cpu().numpy())
        gt_mean.append(batch["h_1s"].float().cpu().numpy())
        ids = list(batch["clip_id"])
        anc = batch["anchor"].detach().cpu().numpy().astype(np.int64)
        for j, cid in enumerate(ids):
            t = int(anc[j])
            pack = pair_targets_for_window(dataset, cid, t)
            gt_clip.append(pack["gt_clip"])
            v_true.append(pack["v_true"])
            envelope.append(pack["envelope"])
            pre.append(pack["pre_fork"])
            paired.append(pack["paired"])
            canonical.append(pack["canonical"])
            blob_n.append(pack["blob_n"])
        collision.extend([str(o) == "collision" for o in batch["outcome_type"]])
        anchor.append(anc)
        clip_id.extend(ids)
        group.extend(list(batch["scenario_group_id"]))
        outcome.extend(list(batch["outcome_type"]))
        scenario.extend(list(batch["scenario_type"]))
        ogid.extend(list(batch["outcome_group_id"]))
    arrays = {
        "pred_mu": np.concatenate(pred, axis=0).astype(np.float32),
        "gt_mean": np.concatenate(gt_mean, axis=0).astype(np.float32),
        "gt_clip": np.stack(gt_clip, axis=0).astype(np.float32),
        "v_true": np.stack(v_true, axis=0).astype(np.float32),
        "envelope": np.stack(envelope, axis=0).astype(np.float32),
        "pre_fork": np.asarray(pre, dtype=np.bool_),
        "paired": np.asarray(paired, dtype=np.bool_),
        "canonical": np.asarray(canonical, dtype=np.bool_),
        "collision": np.asarray(collision, dtype=np.bool_),
        "blob_n": np.asarray(blob_n, dtype=np.int16),
        "anchor": np.concatenate(anchor, axis=0).astype(np.int32),
    }
    if v_hat:
        arrays["v_hat"] = np.concatenate(v_hat, axis=0).astype(np.float32)
    meta = {
        "clip_id": clip_id,
        "scenario_group_id": group,
        "outcome_type": outcome,
        "scenario_type": scenario,
        "outcome_group_id": ogid,
    }
    return {"arrays": arrays, "meta": meta}


@torch.no_grad()
def cache_fine_features(
    model,
    dataset: TeacherDataset,
    cfg,
    device: str,
    *,
    canonical_only: bool = False,
) -> dict[str, torch.Tensor]:
    """One frozen adapter pass. Later epochs train only the 1×1 head on these maps."""
    loader = build_dataloader(dataset, cfg, shuffle=False)
    model.eval()
    feats, v_true, mask, mu = [], [], [], []
    for batch in loader:
        batch_d = move_batch(batch, device)
        if "tokens" in batch_d:
            z, fine = model.encode_pair(tokens=batch_d["tokens"])
        else:
            z, fine = model.encode_pair(clip=batch_d["frames"])
        feat = model._heatmap_feat(z, fine)
        heads = model._heads(z, fine)
        v_np, m_np = [], []
        ids = list(batch["clip_id"])
        anc = batch["anchor"].detach().cpu().numpy().astype(np.int64)
        for j, cid in enumerate(ids):
            pack = pair_targets_for_window(dataset, cid, int(anc[j]))
            v_np.append(pack["v_true"])
            m_np.append(pack["canonical"])
        v_t = torch.from_numpy(np.stack(v_np, axis=0))
        m_t = torch.tensor(m_np, dtype=torch.bool)
        if canonical_only:
            keep = m_t
            if int(keep.sum()) == 0:
                continue
            feat = feat[keep]
            v_t = v_t[keep]
            m_t = m_t[keep]
            mu_t = heads["Hhat_1s"][keep]
        else:
            mu_t = heads["Hhat_1s"]
        feats.append(feat.detach().cpu().half())
        v_true.append(v_t)
        mask.append(m_t)
        mu.append(mu_t.detach().cpu().float())
    return {
        "feat": torch.cat(feats, dim=0),
        "v_true": torch.cat(v_true, dim=0),
        "mask": torch.cat(mask, dim=0),
        "mu": torch.cat(mu, dim=0),
    }


@torch.no_grad()
def collect_variance(model, dataset: TeacherDataset, cfg, device: str) -> np.ndarray:
    loader = build_dataloader(dataset, cfg, shuffle=False)
    model.eval()
    if model.variance_head is not None:
        model.variance_head.eval()
    chunks = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = teacher_forward(model, batch)
        chunks.append(out["Hhat_var"].float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def save_dump(path: Path, dump: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dump["arrays"])
    meta_path = path.with_name(path.stem + "_meta.json")
    meta_path.write_text(json.dumps(dump["meta"]), encoding="utf-8")


def load_dump(path: Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(path.with_name(path.stem + "_meta.json").read_text(encoding="utf-8"))
    return {"arrays": arrays, "meta": meta}


def _ctx(dump: dict[str, Any], gt_key: str = "gt_clip") -> dict[str, Any]:
    a = dump["arrays"]
    return {
        "gt": a[gt_key],
        "pre_fork": a["pre_fork"],
        "collision": a["collision"],
        "blob_n": a["blob_n"],
        "paired": a["paired"],
    }


def score_pred(pred: np.ndarray, dump: dict[str, Any], *, gt_key: str = "gt_clip", caution: float = 0.25, stop: float = 0.60) -> dict[str, Any]:
    ctx = _ctx(dump, gt_key)
    return warning_report(pred, ctx["gt"], caution=caution, stop=stop, **{k: ctx[k] for k in ctx if k != "gt"})


def extra_diagnostics(pred: np.ndarray, dump: dict[str, Any], *, gt_key: str = "gt_clip", mu: np.ndarray | None = None, stop: float = 0.60) -> dict[str, Any]:
    a = dump["arrays"]
    m = dump["meta"]
    peak = np.asarray(pred).reshape(len(pred), -1).max(axis=1)
    twins = twin_accounting(
        peak,
        a[gt_key],
        outcome_group=np.asarray(m["outcome_group_id"]),
        anchor=a["anchor"],
        collision=a["collision"],
        pre_fork=a["pre_fork"],
        paired=a["paired"],
        stop=stop,
    )
    out: dict[str, Any] = {"twins": twins}
    if mu is not None:
        out["upgrade"] = upgrade_source(mu, pred, g_stop=0.60, u_stop=stop)
    receding = np.array(["reced" in str(s).lower() or "hold" in str(s).lower() or "walk_away" in str(s).lower() for s in m["scenario_type"]])
    extra = np.array([_is_extra_id(c) for c in m["clip_id"]])
    quiet = receding | extra
    if np.any(quiet):
        out["quiet_extras"] = score_pred(pred, _mask_dump(dump, quiet), gt_key=gt_key)
    return out


def _is_extra_id(clip_id: str) -> bool:
    # extras are sample_0207..0229 in this trial
    text = str(clip_id)
    for token in text.replace("-", "_").split("_"):
        if token.isdigit() and 207 <= int(token) <= 229:
            return True
    return False


def _mask_dump(dump: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    arrays = {k: (v[m] if hasattr(v, "shape") and len(v) == len(m) else v) for k, v in dump["arrays"].items()}
    meta = {k: [x for x, keep in zip(v, m) if keep] for k, v in dump["meta"].items()}
    return {"arrays": arrays, "meta": meta}


def score_method(name: str, pred: np.ndarray, dump: dict[str, Any], g_clip: dict[str, Any], g_mean: dict[str, Any]) -> dict[str, Any]:
    clip = score_pred(pred, dump, gt_key="gt_clip")
    mean = score_pred(pred, dump, gt_key="gt_mean")
    a = dump["arrays"]
    return {
        "name": name,
        "clip_own": clip,
        "mean_target": mean,
        "vs_g_clip": feasible_vs_g(clip, g_clip),
        "vs_g_mean": feasible_vs_g(mean, g_mean),
        "diagnostics": extra_diagnostics(pred, dump, gt_key="gt_clip", mu=a["pred_mu"]),
    }


def run_oracles(train_dump: dict[str, Any], val_dump: dict[str, Any]) -> dict[str, Any]:
    va = val_dump["arrays"]
    ta = train_dump["arrays"]
    g_clip = score_pred(va["pred_mu"], val_dump, gt_key="gt_clip")
    g_mean = score_pred(va["pred_mu"], val_dump, gt_key="gt_mean")
    methods = [score_method("G", va["pred_mu"], val_dump, g_clip, g_mean)]

    g_thr = []
    for thr in STOP_SWEEP:
        caution = min(0.25, float(thr))
        report = score_pred(va["pred_mu"], val_dump, gt_key="gt_clip", caution=caution, stop=float(thr))
        g_thr.append(
            {
                "stop_threshold": float(thr),
                "caution_threshold": float(caution),
                "headline": report["headline"],
                "vs_g_clip": feasible_vs_g(report, g_clip),
            }
        )
    methods.append({"name": "G-thr", "curve": g_thr})

    lookup = fit_mu_sigma_lookup(ta["pred_mu"], ta["v_true"], ta["canonical"])
    v_mu = apply_mu_sigma_lookup(va["pred_mu"], lookup)
    v_bern = bernoulli_var(va["pred_mu"])
    mu_sigma_curve = []
    bern_curve = []
    for kappa in KAPPA_GRID:
        r = risk_maps(va["pred_mu"], v_mu, kappa)
        mu_sigma_curve.append(
            {
                "kappa": float(kappa),
                "report": score_pred(r, val_dump, gt_key="gt_clip"),
                "vs_g": feasible_vs_g(score_pred(r, val_dump, gt_key="gt_clip"), g_clip),
                "var_diag_vs_true": variance_diagnostics(v_mu, va["v_true"], va["canonical"]),
            }
        )
        rb = risk_maps(va["pred_mu"], v_bern, kappa)
        bern_curve.append(
            {
                "kappa": float(kappa),
                "report": score_pred(rb, val_dump, gt_key="gt_clip"),
                "vs_g": feasible_vs_g(score_pred(rb, val_dump, gt_key="gt_clip"), g_clip),
            }
        )
    methods.append({"name": "mu-sigma", "selection": select_kappa(mu_sigma_curve, g_clip), "curve": mu_sigma_curve})
    methods.append({"name": "bern-sigma", "selection": select_kappa(bern_curve, g_clip), "curve": bern_curve})

    env_clip = score_pred(va["envelope"], val_dump, gt_key="gt_clip")
    env_on_paired = va["pred_mu"].copy()
    paired_pre = va["paired"] & va["pre_fork"]
    env_on_paired[paired_pre] = va["envelope"][paired_pre]
    env_policy = score_pred(env_on_paired, val_dump, gt_key="gt_clip")
    methods.append(score_method("envelope_oracle_all", va["envelope"], val_dump, g_clip, g_mean))
    methods.append(score_method("envelope_oracle_paired", env_on_paired, val_dump, g_clip, g_mean))

    oracle_sigma_curve = []
    for kappa in KAPPA_GRID:
        r = risk_maps(va["pred_mu"], va["v_true"], kappa)
        rep = score_pred(r, val_dump, gt_key="gt_clip")
        oracle_sigma_curve.append({"kappa": float(kappa), "report": rep, "vs_g": feasible_vs_g(rep, g_clip)})
    methods.append(
        {
            "name": "oracle-sigma",
            "selection": select_kappa(oracle_sigma_curve, g_clip),
            "curve": oracle_sigma_curve,
        }
    )

    env_feasible = bool(feasible_vs_g(env_policy, g_clip)["feasible"])
    # kappa=0 is always feasible; treat "worth training" as any kappa>0 feasible with a STOP gain
    worth = False
    for c in oracle_sigma_curve:
        if float(c["kappa"]) > 0 and c["vs_g"]["feasible"] and c["vs_g"]["delta_pre_fork_stop"] > 1e-6:
            worth = True
            break
    if env_feasible and feasible_vs_g(env_policy, g_clip).get("delta_pre_fork_stop", 0) > 1e-6:
        worth = True
    g_thr_worth = any(row["vs_g_clip"]["feasible"] and float(row["stop_threshold"]) < 0.60 and (row["vs_g_clip"]["delta_pre_fork_stop"] or 0) > 1e-6 for row in g_thr)

    return jsonify(
        {
            "g_clip_own": g_clip,
            "g_mean_target": g_mean,
            "methods": methods,
            "mu_sigma_lookup": lookup,
            "verdict": {
                "envelope_feasible": bool(feasible_vs_g(env_policy, g_clip)["feasible"]),
                "oracle_sigma_best_kappa": select_kappa(oracle_sigma_curve, g_clip),
                "g_thr_can_improve_prefork_under_cap": g_thr_worth,
                "policy_worth_training_u1": worth or not g_thr_worth,
                "note": (
                    "If G-thr already matches oracle-sigma, U1 must beat a threshold warp. "
                    "If no kappa>0 is feasible, the conservative policy fails the FA cap."
                ),
            },
            "train_pair_windows": int(ta["canonical"].sum()),
            "val_pair_windows": int(va["canonical"].sum()),
        }
    )


def score_u1(
    val_dump: dict[str, Any],
    v_hat: np.ndarray,
    oracle_report: dict[str, Any],
    *,
    n_boot: int = 200,
    pred_mu: np.ndarray | None = None,
) -> dict[str, Any]:
    va = val_dump["arrays"]
    mu = np.asarray(pred_mu if pred_mu is not None else va["pred_mu"], dtype=np.float32)
    g_clip = oracle_report["g_clip_own"]
    g_mean = oracle_report["g_mean_target"]
    curve = []
    for kappa in KAPPA_GRID:
        r = risk_maps(mu, v_hat, float(kappa))
        rep = score_pred(r, val_dump, gt_key="gt_clip")
        item = {
            "kappa": float(kappa),
            "report": rep,
            "vs_g": feasible_vs_g(rep, g_clip),
            "mean_target": score_pred(r, val_dump, gt_key="gt_mean"),
            "diagnostics": extra_diagnostics(r, val_dump, gt_key="gt_clip", mu=mu),
        }
        if abs(float(kappa) - 0.0) < 1e-12:
            item["mu_lock_peak_mae"] = float(np.mean(np.abs(r - mu)))
        curve.append(item)
    selection = select_kappa(curve, g_clip)
    best_k = float(selection["kappa"])
    best_r = risk_maps(mu, v_hat, best_k)
    groups = np.asarray(val_dump["meta"]["scenario_group_id"])
    boot = cluster_bootstrap(best_r, va["gt_clip"], groups, va["pre_fork"], n_boot=n_boot)
    v_mu = apply_mu_sigma_lookup(mu, oracle_report["mu_sigma_lookup"])
    return jsonify(
        {
            "selection": selection,
            "curve": curve,
            "variance_vs_true": variance_diagnostics(v_hat, va["v_true"], va["canonical"]),
            "variance_vs_mu_lookup": variance_diagnostics(v_hat, v_mu, va["canonical"]),
            "mu_lookup_vs_true": variance_diagnostics(v_mu, va["v_true"], va["canonical"]),
            "bootstrap_best_kappa": boot,
            "beats_mu_lookup_mae": (
                variance_diagnostics(v_hat, va["v_true"], va["canonical"]).get("mae")
                is not None
                and variance_diagnostics(v_mu, va["v_true"], va["canonical"]).get("mae") is not None
                and variance_diagnostics(v_hat, va["v_true"], va["canonical"])["mae"]
                < variance_diagnostics(v_mu, va["v_true"], va["canonical"])["mae"]
            ),
            "g_clip_own_headline": g_clip["headline"],
            "g_mean_target_headline": g_mean["headline"],
        }
    )
