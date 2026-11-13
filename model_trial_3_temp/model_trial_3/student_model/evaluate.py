"""Val metrics for the RGB student (never routed through token-cache teacher_forward)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from collision_jepa3.config import supervised_horizon_names
from collision_jepa3.evaluation.evaluator import _mean_map
from collision_jepa3.evaluation.metrics import HorizonAccumulator
from collision_jepa3.evaluation.operating_point import operating_metrics
from collision_jepa3.training.engine import composite_score, move_batch

from .loss import blob_weights, spatial_l1, weighted_l1


def student_forward(model, batch) -> dict[str, torch.Tensor]:
    if "frames" not in batch:
        raise RuntimeError("student needs RGB frames in the batch")
    return model(batch["frames"])


@torch.no_grad()
def evaluate_student(model, loader, cfg, device: str, loss_fn=None, teacher=None) -> dict[str, Any]:
    model.eval()
    names = supervised_horizon_names(cfg)
    warn = cfg.get("warning")
    acc = {
        "1s": HorizonAccumulator(
            "1s",
            high_threat_threshold=float(cfg.get("evaluation.high_threat_threshold", 0.5)),
            safe_cell_threshold=float(cfg.get("loss.safe_cell_threshold", 0.05)),
            warning=warn,
        )
    }
    pred_1s: list[np.ndarray] = []
    gt_1s: list[np.ndarray] = []
    valid_1s: list[np.ndarray] = []
    loss_sums: dict[str, float] = {}
    loss_n = 0
    latent_l1 = 0.0
    latent_cos = 0.0
    latent_blob_l1 = 0.0
    latent_n = 0
    small_stop_n = 0
    small_stop_tp = 0
    stop_thr = float(cfg.get("warning.stop_threshold", 0.60))
    caution_thr = float(cfg.get("warning.caution_threshold", 0.25))
    energy_alpha = float(cfg.get("student.energy_alpha", 4.0))
    appear_boost = float(cfg.get("student.appear_boost", 4.0))
    stop_boost = float(cfg.get("student.stop_boost", 2.0))
    for batch in loader:
        batch = move_batch(batch, device)
        out = student_forward(model, batch)
        t_out = None
        if teacher is not None and "tokens" in batch:
            z, fine = teacher.encode_pair(tokens=batch["tokens"])
            t_out = {"F": fine, "Z_t": z}
            p = out["F"].float()
            t = fine.float()
            w = blob_weights(
                t,
                batch.get("h_now"),
                batch.get("h_1s"),
                energy_alpha=energy_alpha,
                appear_boost=appear_boost,
                stop_boost=stop_boost,
            )
            latent_l1 += float(spatial_l1(p, t))
            latent_blob_l1 += float(weighted_l1(p, t, w))
            pn = torch.nn.functional.normalize(p.flatten(1), dim=1, eps=1e-6)
            tn = torch.nn.functional.normalize(t.flatten(1), dim=1, eps=1e-6)
            latent_cos += float((pn * tn).sum(dim=1).mean())
            latent_n += 1
        if loss_fn is not None:
            packed = loss_fn(
                out,
                batch,
                teacher_out=t_out,
                epoch=10**9,
                predictor=getattr(model, "predictor", None),
            )
            for k, v in packed.as_floats().items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
            loss_n += 1
        preds = out["Hhat_1s"].float().cpu().numpy()
        trues = batch["h_1s"].float().cpu().numpy()
        bsz = preds.shape[0]
        for i in range(bsz):
            masked = bool(batch["amb_1s"][i].item())
            acc["1s"].update(
                preds[i],
                trues[i],
                masked=masked,
                outcome=batch["outcome_type"][i],
                background=batch["background_id"][i],
                scenario=batch["scenario_type"][i],
                pre_fork=bool(batch["pre_fork"][i].item()),
            )
            pred_1s.append(preds[i])
            gt_1s.append(trues[i])
            valid_1s.append(np.full(preds[i].shape, not masked, dtype=bool))
            if (not masked) and float(trues[i].max()) >= stop_thr:
                n_occ = int((trues[i] >= 0.50).sum())
                empty_now = False
                if "h_now" in batch:
                    empty_now = float(batch["h_now"][i].float().max()) < caution_thr
                if n_occ <= 2 or empty_now:
                    small_stop_n += 1
                    if float(preds[i].max()) >= stop_thr:
                        small_stop_tp += 1
    horizon_summaries = {n: (acc["1s"].summary() if n == "1s" else {}) for n in names}
    result = {
        "horizons": horizon_summaries,
        "selection": composite_score(horizon_summaries, cfg),
    }
    if pred_1s:
        result["operating"] = operating_metrics(
            np.stack(pred_1s, axis=0),
            np.stack(gt_1s, axis=0),
            np.asarray(valid_1s, dtype=bool),
        )
    if loss_n:
        result["loss"] = {k: v / loss_n for k, v in loss_sums.items()}
    result["latent"] = {
        "small_stop_n": small_stop_n,
        "small_stop_tp": small_stop_tp,
        "small_stop_recall": (small_stop_tp / small_stop_n) if small_stop_n else None,
    }
    if latent_n:
        result["latent"].update(
            {
                "l1": latent_l1 / latent_n,
                "cos": latent_cos / latent_n,
                "blob_l1": latent_blob_l1 / latent_n,
            }
        )
    result["subgroups"] = {
        "outcome": _mean_map(acc["1s"].by_outcome),
        "background": _mean_map(acc["1s"].by_background),
        "scenario": _mean_map(acc["1s"].by_scenario),
    }
    return result
