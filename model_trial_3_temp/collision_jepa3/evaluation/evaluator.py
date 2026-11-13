"""Run the teacher on a loader and collect per-horizon metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..config import heads_mode, supervised_horizon_names
from ..training.engine import composite_score, move_batch, teacher_forward
from .judge import HeatmapPack
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


@torch.no_grad()
def collect_h1s_pack(model, loader, device: str) -> HeatmapPack:
    """Run the teacher once and keep H_1s maps plus clip metadata for the judge."""
    model.eval()
    pred, gt, valid = [], [], []
    clip_id, scenario, background, outcome, pre_fork, anchor = [], [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = teacher_forward(model, batch)
        p = out["Hhat_1s"].float().cpu().numpy()
        g = batch["h_1s"].float().cpu().numpy()
        amb = batch["amb_1s"].detach().cpu().numpy().astype(bool)
        pred.append(p)
        gt.append(g)
        valid.append(~amb)
        clip_id.extend(list(batch["clip_id"]))
        scenario.extend(list(batch["scenario_type"]))
        background.extend(list(batch["background_id"]))
        outcome.extend(list(batch["outcome_type"]))
        pre_fork.append(batch["pre_fork"].detach().cpu().numpy().astype(bool))
        anchor.append(batch["anchor"].detach().cpu().numpy().astype(np.int64))
    if not pred:
        raise RuntimeError("empty loader")
    return HeatmapPack(
        pred=np.concatenate(pred, axis=0),
        gt=np.concatenate(gt, axis=0),
        valid=np.concatenate(valid, axis=0),
        clip_id=np.asarray(clip_id, dtype=object),
        scenario=np.asarray(scenario, dtype=object),
        background=np.asarray(background, dtype=object),
        outcome=np.asarray(outcome, dtype=object),
        pre_fork=np.concatenate(pre_fork, axis=0),
        anchor=np.concatenate(anchor, axis=0),
    )
