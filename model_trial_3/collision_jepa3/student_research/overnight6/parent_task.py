"""Parent TeacherLoss task terms for RGB students. Does not change parent math."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from collision_jepa3.config import Config
from collision_jepa3.data.heatmap_annotations import load_heatmaps
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.student_research.overnight6.losses import decoder_anchor, latent_terms

TASK_WEIGHTS = {
    "L_1s": 1.0,
    "L_blob": 0.35,
    "L_stop_hinge": 1.0,
    "L_safe_hinge": 0.5,
    "L_FP": 0.10,
    "L_dir": 0.05,
}


def parent_cfg() -> Config:
    return Config.load(
        overrides={
            "heads.mode": "direct_h1s",
            "loss.close_weight": 0.0,
            "loss.stop_hinge_weight": 1.0,
            "loss.safe_hinge_weight": 0.5,
            "loss.small_blob_weight": 2.0,
            "loss.background_consistency_weight": 0.0,
            "loss.beta_nll_weight": 0.0,
            "loss.evidential_weight": 0.0,
            "loss.prob_stop_weight": 0.0,
            "loss.prob_safe_weight": 0.0,
            "loss.var_weight": 0.0,
        }
    )


def build_teacher_loss(device: str) -> TeacherLoss:
    loss = TeacherLoss(parent_cfg(), device=device)
    if loss.heatmap != "bce":
        raise SystemExit(f"parent heatmap mode is {loss.heatmap}, expected bce")
    if loss.logit_hinges:
        raise SystemExit("parent logit_hinges is on; would change hinge math vs Pair-CF treatment")
    if abs(loss.close_w) > 0 or abs(loss.bg_w) > 0:
        raise SystemExit("close/bg weights must be zero for this experiment")
    return loss


def load_h_now_table(rows: list[dict], ids: list[int]) -> dict[tuple[str, int], np.ndarray]:
    cfg = Config.load()
    clip_to_hm = {r.clip_id: r.heatmap_json_path for r in load_manifest(cfg.path("paths.manifest"))}
    needed: set[str] = set()
    keys: list[tuple[str, int]] = []
    for i in ids:
        row = rows[int(i)]
        if str(row.get("split")) == "test":
            raise SystemExit("test split id leaked into h_now table")
        cid, t = str(row["clip_id"]), int(row["frame"])
        needed.add(cid)
        keys.append((cid, t))
    cache: dict[str, np.ndarray] = {}
    missing = []
    for cid in needed:
        p = clip_to_hm.get(cid)
        if not p or not Path(p).is_file():
            missing.append(cid)
            continue
        cache[cid] = load_heatmaps(p, expected_frames=int(cfg.get("data.expected_frames", 450)), grid=5)
    if missing:
        raise SystemExit(
            f"cannot reuse parent appear/disappear weights: missing heatmap json for {len(missing)} clips "
            f"(first={missing[:3]}). Stopping rather than dropping appear×8."
        )
    out: dict[tuple[str, int], np.ndarray] = {}
    for cid, t in keys:
        hm = cache[cid]
        if t < 0 or t >= hm.shape[0]:
            raise SystemExit(f"h_now frame {t} out of range for {cid}")
        out[(cid, t)] = np.asarray(hm[t], dtype=np.float32)
    return out


def task_from_teacher(teacher: TeacherLoss, out: dict, batch: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if "Hhat_1s_logits" not in out:
        raise SystemExit("decoder did not return logits; refusing probability-only BCE")
    if "h_now" not in batch:
        raise SystemExit("h_now missing; appear×8 would silently drop")
    tb = {
        "h_1s": batch["gt_1s"].float(),
        "amb_1s": batch["amb_1s"],
        "h_now": batch["h_now"].float(),
    }
    tout = {
        "Hhat_1s": out["Hhat_1s"].float(),
        "Hhat_1s_logits": out["Hhat_1s_logits"].float(),
        "Z_t": out["F"],
    }
    res = teacher(tout, tb, z_bg_pair=None)
    parts = {k: res.terms[k] for k in TASK_WEIGHTS}
    l_task = (
        TASK_WEIGHTS["L_1s"] * parts["L_1s"]
        + TASK_WEIGHTS["L_blob"] * parts["L_blob"]
        + TASK_WEIGHTS["L_stop_hinge"] * parts["L_stop_hinge"]
        + TASK_WEIGHTS["L_safe_hinge"] * parts["L_safe_hinge"]
        + TASK_WEIGHTS["L_FP"] * parts["L_FP"]
        + TASK_WEIGHTS["L_dir"] * parts["L_dir"]
    )
    return parts, l_task


def m7_losses(teacher, out, batch, mu, sigma, predictor, decoder, dec_init):
    tparts, l_task = task_from_teacher(teacher, out, batch)
    lf, lc, lr, lm = latent_terms(out, batch, mu, sigma, predictor)
    l_lat = lf + 0.2 * lc + 0.5 * lm + 0.2 * lr
    l_reg = decoder_anchor(decoder, dec_init)
    parts = {
        **tparts,
        "L_feat": lf,
        "L_cos": lc,
        "L_rel": lr,
        "L_mgd": lm,
        "L_latent": l_lat,
        "L_task": l_task,
        "L_dec_reg": l_reg,
    }
    return parts


def m8_losses(teacher, out, batch, decoder, dec_init):
    if "F_teacher" in batch:
        raise SystemExit("Model 8 batch still contains F_teacher; refusing silent unused latent load")
    tparts, l_task = task_from_teacher(teacher, out, batch)
    l_reg = decoder_anchor(decoder, dec_init)
    parts = {**tparts, "L_task": l_task, "L_dec_reg": l_reg}
    return parts


M9_WEIGHTS = {"L_1s": 1.0, "L_blob": 0.35, "L_FP": 0.10, "L_dir": 0.05}


def parent_cfg_m9() -> Config:
    return Config.load(
        overrides={
            "heads.mode": "direct_h1s",
            "loss.close_weight": 0.0,
            "loss.stop_hinge_weight": 0.0,
            "loss.safe_hinge_weight": 0.0,
            "loss.small_blob_weight": 2.0,
            "loss.background_consistency_weight": 0.0,
            "loss.beta_nll_weight": 0.0,
            "loss.evidential_weight": 0.0,
            "loss.prob_stop_weight": 0.0,
            "loss.prob_safe_weight": 0.0,
            "loss.var_weight": 0.0,
        }
    )


def build_teacher_loss_m9(device: str) -> TeacherLoss:
    loss = TeacherLoss(parent_cfg_m9(), device=device)
    if loss.heatmap != "bce":
        raise SystemExit(f"parent heatmap mode is {loss.heatmap}, expected bce")
    if float(loss.stop_hinge_w) != 0.0 or float(loss.safe_hinge_w) != 0.0:
        raise SystemExit("M9 hinges must be weight 0 so they are not computed")
    if float(loss.close_w) != 0.0 or float(loss.bg_w) != 0.0:
        raise SystemExit("close/bg must be off")
    return loss


def m9_losses(teacher: TeacherLoss, out: dict, batch: dict) -> dict[str, torch.Tensor]:
    if "Hhat_1s_logits" not in out:
        raise SystemExit("decoder did not return logits")
    if "h_now" not in batch:
        raise SystemExit("h_now missing; appear×8 would silently drop")
    tb = {
        "h_1s": batch["gt_1s"].float(),
        "amb_1s": batch["amb_1s"],
        "h_now": batch["h_now"].float(),
    }
    tout = {
        "Hhat_1s": out["Hhat_1s"].float(),
        "Hhat_1s_logits": out["Hhat_1s_logits"].float(),
        "Z_t": out["F"],
    }
    res = teacher(tout, tb, z_bg_pair=None)
    if float(res.terms["L_stop_hinge"].detach()) != 0.0 or float(res.terms["L_safe_hinge"].detach()) != 0.0:
        raise SystemExit("hinge tensors were nonzero despite weight 0")
    parts = {k: res.terms[k] for k in M9_WEIGHTS}
    l_task = (
        M9_WEIGHTS["L_1s"] * parts["L_1s"]
        + M9_WEIGHTS["L_blob"] * parts["L_blob"]
        + M9_WEIGHTS["L_FP"] * parts["L_FP"]
        + M9_WEIGHTS["L_dir"] * parts["L_dir"]
    )
    parts["L_task"] = l_task
    return parts
