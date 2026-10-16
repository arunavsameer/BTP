"""Run the teacher on a loader and collect per-horizon metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..config import heads_mode, supervised_horizon_names
from ..training.engine import composite_score, move_batch, teacher_forward
from .metrics import HorizonAccumulator
from .operating_point import operating_metrics


@torch.no_grad()
def evaluate_loader(model, loader, cfg, device: str, loss_fn=None) -> dict[str, Any]:
    model.eval()
    names = supervised_horizon_names(cfg)
    warn = cfg.get("warning")
    include_now = heads_mode(cfg) != "direct_h1s"
    acc_keys = list(names) + (["now"] if include_now else [])
    acc = {
        n: HorizonAccumulator(
            n,
            high_threat_threshold=float(cfg.get("evaluation.high_threat_threshold", 0.5)),
            safe_cell_threshold=float(cfg.get("loss.safe_cell_threshold", 0.05)),
            warning=warn,
        )
        for n in acc_keys
    }
    loss_sums: dict[str, float] = {}
    loss_n = 0
    pred_1s: list[np.ndarray] = []
    gt_1s: list[np.ndarray] = []
    valid_1s: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = teacher_forward(model, batch)
        if loss_fn is not None:
            packed = loss_fn(out, batch)
            for k, v in packed.as_floats().items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
            loss_n += 1
        has_now = "Hhat_now" in out
        preds = {"1s": out["Hhat_1s"].float().cpu().numpy()}
        trues = {"1s": batch["h_1s"].float().cpu().numpy()}
        if has_now:
            preds["now"] = out["Hhat_now"].float().cpu().numpy()
            trues["now"] = batch["h_now"].float().cpu().numpy()
        for extra in ("1_5s", "2s"):
            key = f"Hhat_{extra}"
            if key in out:
                preds[extra] = out[key].float().cpu().numpy()
                trues[extra] = batch[f"h_{extra}"].float().cpu().numpy()
        eval_names = [n for n in acc_keys if n in preds]
        bsz = preds["1s"].shape[0]
        for i in range(bsz):
            for n in eval_names:
                masked = False if n == "now" else bool(batch[f"amb_{n}"][i].item())
                acc[n].update(
                    preds[n][i],
                    trues[n][i],
                    masked=masked,
                    outcome=batch["outcome_type"][i],
                    background=batch["background_id"][i],
                    scenario=batch["scenario_type"][i],
                    pre_fork=bool(batch["pre_fork"][i].item()),
                )
            if n == "1s":
                pred_1s.append(preds["1s"][i])
                gt_1s.append(trues["1s"][i])
                valid_1s.append(np.full(preds["1s"][i].shape, not masked, dtype=bool))
    horizon_summaries = {n: acc[n].summary() for n in names}
    result = {
        "horizons": horizon_summaries,
        "selection": composite_score(horizon_summaries, cfg),
    }
    if "now" in acc:
        result["now"] = acc["now"].summary()
    if pred_1s:
        p = np.stack(pred_1s, axis=0)
        g = np.stack(gt_1s, axis=0)
        v = np.asarray(valid_1s, dtype=bool)
        result["operating"] = operating_metrics(p, g, v)
    if loss_n:
        result["loss"] = {k: v / loss_n for k, v in loss_sums.items()}
    # subgroup MSE from 1s horizon as a representative future
    result["subgroups"] = {
        "outcome": _mean_map(acc["1s"].by_outcome),
        "background": _mean_map(acc["1s"].by_background),
        "scenario": _mean_map(acc["1s"].by_scenario),
    }
    return result


def _mean_map(d: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) for k, v in d.items() if v}
