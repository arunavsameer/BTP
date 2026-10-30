"""Score the now map. Future heatmaps stay in the batch and are not accumulated."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..training.engine import composite_score, move_batch, teacher_forward
from .metrics import HorizonAccumulator


@torch.no_grad()
def evaluate_loader(model, loader, cfg, device: str, loss_fn=None) -> dict[str, Any]:
    model.eval()
    warn = cfg.get("warning")
    acc = HorizonAccumulator(
        "now",
        high_threat_threshold=float(cfg.get("evaluation.high_threat_threshold", 0.5)),
        safe_cell_threshold=float(cfg.get("loss.safe_cell_threshold", 0.05)),
        warning=warn,
    )
    loss_sums: dict[str, float] = {}
    loss_n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = teacher_forward(model, batch)
        if loss_fn is not None:
            packed = loss_fn(out, batch)
            for k, v in packed.as_floats().items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
            loss_n += 1
        preds = out["Hhat_now"].float().cpu().numpy()
        trues = batch["h_now"].float().cpu().numpy()
        bsz = preds.shape[0]
        for i in range(bsz):
            acc.update(
                preds[i],
                trues[i],
                masked=False,
                outcome=batch["outcome_type"][i],
                background=batch["background_id"][i],
                scenario=batch["scenario_type"][i],
                pre_fork=bool(batch["pre_fork"][i].item()),
            )
    summary = acc.summary()
    result = {
        "horizons": {"now": summary},
        "now": summary,
        "selection": composite_score({"now": summary}, cfg),
        "subgroups": {
            "outcome": _mean_map(acc.by_outcome),
            "background": _mean_map(acc.by_background),
            "scenario": _mean_map(acc.by_scenario),
        },
    }
    if loss_n:
        result["loss"] = {k: v / loss_n for k, v in loss_sums.items()}
    return result


def _mean_map(d: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) for k, v in d.items() if v}
