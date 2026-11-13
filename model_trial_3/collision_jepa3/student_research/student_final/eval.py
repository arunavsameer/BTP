"""Evaluation with ordinal cutoffs and reporting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_jepa3.evaluation.metrics import CAUTION, STOP, SAFE

from .targets import peak_tier


def gt_class_from_peak(peak: float) -> int:
  if peak >= 0.60:
    return STOP
  if peak >= 0.25:
    return CAUTION
  return SAFE


def predict_tier(q_warn: float, q_stop: float, c_warn: float, c_stop: float) -> int:
  if q_stop >= c_stop:
    return STOP
  if q_warn >= c_warn:
    return CAUTION
  return SAFE


def argmax_tier(p_safe: float, p_threat: float, p_stop: float) -> int:
  arr = [p_safe, p_threat, p_stop]
  return int(np.argmax(arr))


@torch.no_grad()
def forward_collect(model, loader, device: str) -> dict[str, Any]:
  model.eval()
  rows: list[dict] = []
  for batch in loader:
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    out = model(b["clip"])
    q_w = out["q_warn"].detach().cpu().numpy()
    q_s = out["q_stop"].detach().cpu().numpy()
    p_gt = b["p_gt"].detach().cpu().numpy()
    h = out["Hhat_1s"].detach().cpu().numpy()
    gt = b["gt_1s"].detach().cpu().numpy()
    pref = b["pre_fork"].detach().cpu().numpy()
    amb = b["amb_1s"].detach().cpu().numpy()
    for i in range(len(q_w)):
      rows.append(
        {
          "q_warn": float(q_w[i]),
          "q_stop": float(q_s[i]),
          "p_gt": float(p_gt[i]),
          "heatmap_peak": float(h[i].max()),
          "gt_peak": float(p_gt[i]),
          "pre_fork": bool(pref[i]),
          "amb_1s": bool(amb[i]),
          "gt_class": gt_class_from_peak(float(p_gt[i])),
          "P_SAFE": float(out["P_SAFE"][i].item()),
          "P_THREAT": float(out["P_THREAT"][i].item()),
          "P_STOP": float(out["P_STOP"][i].item()),
          "sample_id": b["sample_id"][i] if isinstance(b["sample_id"], list) else "",
          "h_pred": h[i],
          "h_gt": gt[i],
        }
      )
  return {"rows": rows}


def confusion_and_rates(rows: list[dict], *, c_warn: float, c_stop: float, use_argmax: bool = False) -> dict[str, Any]:
  cm = np.zeros((3, 3), dtype=np.int64)
  hm_low_stop = 0
  hm_low_threat = 0
  n_stop = n_safe = n_threat = 0
  for r in rows:
    gt = int(r["gt_class"])
    if use_argmax:
      pr = argmax_tier(r["P_SAFE"], r["P_THREAT"], r["P_STOP"])
    else:
      pr = predict_tier(r["q_warn"], r["q_stop"], c_warn, c_stop)
    cm[gt, pr] += 1
    if gt == STOP:
      n_stop += 1
    elif gt == SAFE:
      n_safe += 1
    elif gt == CAUTION:
      n_threat += 1
    if pr in (STOP, CAUTION) and r["heatmap_peak"] < 0.10:
      hm_low_threat += 1
    if pr == STOP and r["heatmap_peak"] < 0.10:
      hm_low_stop += 1

  def recall_for(gt_c: int, pr_c: int | None = None) -> float:
    denom = cm[gt_c].sum()
    if denom == 0:
      return float("nan")
    if pr_c is None:
      return float(cm[gt_c, gt_c] / denom)
    return float(cm[gt_c, pr_c] / denom)

  def mask_post(flag: bool) -> list[dict]:
    return [r for r in rows if bool(r["pre_fork"]) != flag]

  def balanced(sa: float, st: float) -> float:
    if np.isnan(sa) or np.isnan(st):
      return float("nan")
    return 0.5 * (sa + st)

  overall = {
    "stop_recall": recall_for(STOP, STOP),
    "safe_recall": recall_for(SAFE, SAFE),
    "threat_recall": recall_for(CAUTION, CAUTION),
    "stop_precision": float(cm[:, STOP].sum() and cm[STOP, STOP] / cm[:, STOP].sum()),
    "safe_false_alarm_rate": float(1.0 - recall_for(SAFE, SAFE)) if cm[SAFE].sum() else float("nan"),
    "gt_stop_predicted_safe": float(cm[STOP, SAFE] / max(1, cm[STOP].sum())),
    "gt_threat_predicted_safe": float(cm[CAUTION, SAFE] / max(1, cm[CAUTION].sum())),
    "balanced_stop_safe_recall": balanced(recall_for(SAFE, SAFE), recall_for(STOP, STOP)),
    "heatmap_disagreement_stop": hm_low_stop / max(1, cm[:, STOP].sum()),
    "heatmap_disagreement_threat_or_stop": hm_low_threat / max(1, len(rows)),
    "n_windows": len(rows),
    "n_gt_stop": int(n_stop),
    "n_gt_safe": int(n_safe),
    "n_gt_threat": int(n_threat),
  }
  def _recall_subset(sub: list[dict], gt_c: int, pr_c: int) -> float:
    if not sub:
      return float("nan")
    hit = miss = 0
    for r in sub:
      gt = int(r["gt_class"])
      if gt != gt_c:
        continue
      miss += 1
      if use_argmax:
        pr = argmax_tier(r["P_SAFE"], r["P_THREAT"], r["P_STOP"])
      else:
        pr = predict_tier(r["q_warn"], r["q_stop"], c_warn, c_stop)
      if pr == pr_c:
        hit += 1
    return float(hit / miss) if miss else float("nan")

  overall["pre_fork_stop_recall"] = _recall_subset(mask_post(True), STOP, STOP)
  overall["post_fork_stop_recall"] = _recall_subset(mask_post(False), STOP, STOP)
  overall["pre_fork_safe_recall"] = _recall_subset(mask_post(True), SAFE, SAFE)
  overall["post_fork_safe_recall"] = _recall_subset(mask_post(False), SAFE, SAFE)
  return {"confusion_matrix": cm.tolist(), "metrics": overall}


def search_cutoffs(rows: list[dict], *, parent_safe_floor: float) -> dict[str, Any]:
  best = None
  grid = np.arange(0.05, 0.96, 0.05)
  for cw in grid:
    for cs in grid:
      if cs < cw:
        continue
      rep = confusion_and_rates(rows, c_warn=float(cw), c_stop=float(cs))
      m = rep["metrics"]
      if m["safe_recall"] + 1e-9 < parent_safe_floor:
        continue
      score = m["balanced_stop_safe_recall"]
      cand = (score, m.get("post_fork_stop_recall", float("nan")), m["stop_recall"], m["threat_recall"], float(cw), float(cs), rep)
      if best is None or cand > best[:4]:
        best = cand
  if best is None:
    return {"feasible": False, "reason": "no cutoff satisfies SAFE floor"}
  _, post_stop, stop_r, threat_r, cw, cs, rep = best
  return {
    "feasible": True,
    "c_warn": cw,
    "c_stop": cs,
    "selection_score": best[0],
    "post_fork_stop_recall": post_stop,
    "stop_recall": stop_r,
    "threat_recall": threat_r,
    "report": rep,
  }


def rank_metrics(rows: list[dict]) -> dict[str, float]:
  y_stop = np.array([1 if r["gt_class"] == STOP else 0 for r in rows])
  y_safe = np.array([1 if r["gt_class"] == SAFE else 0 for r in rows])
  q_s = np.array([r["q_stop"] for r in rows])
  q_w = np.array([r["q_warn"] for r in rows])
  peaks = np.array([r["gt_peak"] for r in rows])
  out = {}
  def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    y = y[order]
    s = s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(1, y.sum())
    fpr = fp / max(1, (1 - y).sum())
    return float(np.trapz(tpr, fpr))

  try:
    out["stop_vs_safe_auc_q_stop"] = _auc(y_stop, q_s) if y_stop.sum() and (1 - y_stop).sum() else float("nan")
  except Exception:
    out["stop_vs_safe_auc_q_stop"] = float("nan")
  try:
    ys = np.array([1 if peak_tier(p) in ("THREAT", "STOP") else 0 for p in peaks])
    out["threat_vs_safe_auc_q_warn"] = _auc(ys, q_w) if ys.sum() and (1 - ys).sum() else float("nan")
  except Exception:
    out["threat_vs_safe_auc_q_warn"] = float("nan")
  out["mse_gt_heatmap"] = float(np.mean([float(np.mean((r["h_pred"] - r["h_gt"]) ** 2)) for r in rows]))
  return out


def bootstrap_by_sample(rows: list[dict], *, seed: int = 1337, n: int = 400) -> dict[str, Any]:
  rng = np.random.default_rng(seed)
  by: dict[str, list[dict]] = {}
  for r in rows:
    by.setdefault(str(r["sample_id"]), []).append(r)
  samples = list(by.keys())
  if len(samples) < 2:
    return {"ok": False, "reason": "too few sample groups"}
  scores = []
  for _ in range(n):
    pick = rng.choice(samples, size=len(samples), replace=True)
    sub = []
    for s in pick:
      sub.extend(by[s])
    m = confusion_and_rates(sub, c_warn=0.5, c_stop=0.55)["metrics"]
    scores.append(m["balanced_stop_safe_recall"])
  arr = np.asarray(scores, dtype=np.float64)
  return {
    "ok": True,
    "balanced_stop_safe_recall_mean": float(arr.mean()),
    "balanced_stop_safe_recall_ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
  }
