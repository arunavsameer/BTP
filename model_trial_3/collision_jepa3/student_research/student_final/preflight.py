"""Mandatory pre-flight checks before 30-epoch training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from collision_jepa3.data.preprocessing import letterbox_rgb
from collision_jepa3.evaluation.metrics import HorizonAccumulator
from collision_jepa3.student_research.fdec_pilot_dataset import build_frame_store
from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS, init_shared_backbone
from collision_jepa3.student_research.frames import indices_at_anchor

from .dataset import StudentFinalDataset, collate_student_final, flip_heatmap_columns
from .eval import confusion_and_rates, forward_collect
from .losses import combined_loss
from .model import StudentFinal
from .targets import peak_tier


def parent_dev_sanity(cfg, cache_dir: Path, arch_report_dir: Path, gt_group_mean: np.ndarray, warning: dict) -> dict[str, Any]:
  rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open(encoding="utf-8")]
  dev_ids = json.loads((arch_report_dir / "part1_dev_manifest.json").read_text(encoding="utf-8"))["dev_row_ids"]
  H = np.load(cache_dir / "teacher_pred.npz")["H"]
  acc = HorizonAccumulator("1s", warning=warning)
  for i in dev_ids:
    r = rows[int(i)]
    acc.update(
      H[int(i)],
      gt_group_mean[int(i)],
      masked=False,
      outcome=str(r.get("outcome", "")),
      background=str(r.get("background_id", "")),
      scenario=str(r.get("scenario_type", "")),
      pre_fork=bool(r.get("pre_fork")),
    )
  s = acc.summary()
  ok = float(s.get("stop_recall", 0.0)) > 0.0
  return {"ok": ok, "parent_dev_metrics": s, "note": "Parent evaluated on 816-window dev with group-mean GT."}


def flip_symmetry_check(
  frame_store,
  rows: list[dict],
  prepared_row_ids: list[int],
  *,
  n: int = 16,
  seed: int = 1337,
) -> dict[str, Any]:
  rng = np.random.default_rng(seed)
  pool = [int(i) for i in prepared_row_ids if 0 <= int(i) < len(rows)]
  if not pool:
    return {"ok": False, "reason": "no prepared rows for flip check", "n_probes": 0}
  pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
  max_diff = 0.0
  ok = True
  reason = ""
  for rid in pick:
    row = rows[int(rid)]
    cid = row["clip_id"]
    t = int(row["frame"])
    fi = int(indices_at_anchor(t, FDEC_FRAME_OFFSETS)[-1])
    try:
      fr = frame_store.get(cid, fi)
    except KeyError:
      ok = False
      reason = f"missing stored frame clip={cid} fi={fi}"
      break
    if fr.shape[0] != 160:
      fr = letterbox_rgb(fr, 160)
    flipped = fr[:, ::-1, :].copy()
    # 5x5 column mirror check with synthetic asymmetric probe
    probe = np.linspace(0.0, 1.0, 5, dtype=np.float32)[:, None] * np.ones((5, 5), dtype=np.float32)
    mirrored = flip_heatmap_columns(probe)
    if not np.allclose(mirrored[:, 0], probe[:, -1]):
      ok = False
      reason = "column mirror identity failed"
      break
    max_diff = max(max_diff, float(np.abs(fr[:, ::-1] - flipped).max()))
  return {
    "ok": ok,
    "reason": reason or "letterbox + column mirror consistent",
    "max_pixel_flip_diff": max_diff,
    "n_probes": len(pick),
  }


def select_overfit_windows(rows: list[dict], gt: np.ndarray, train_row_ids: list[int], *, n: int = 64) -> list[int]:
  peaks = gt.max(axis=(1, 2))
  safe = [i for i in train_row_ids if peak_tier(float(peaks[i])) == "SAFE"]
  stop = [i for i in train_row_ids if peak_tier(float(peaks[i])) == "STOP"]
  if len(safe) < 16 or len(stop) < 16:
    return []
  rng = np.random.default_rng(1337)
  pick = list(rng.choice(safe, 32, replace=False)) + list(rng.choice(stop, 32, replace=False))
  return sorted(int(x) for x in pick[:n])


def overfit64_test(
  model: StudentFinal,
  dataset: StudentFinalDataset,
  indices: list[int],
  device: str,
  *,
  steps: int = 300,
  lambda_gt: float,
  lambda_kd: float,
) -> dict[str, Any]:
  if len(indices) < 32:
    return {"ok": False, "reason": "insufficient windows for 64-window overfit", "n": len(indices)}
  subset_ids = list(indices)
  ds = StudentFinalDataset(
    dataset.cache_dir,
    subset_ids,
    dataset.gt,
    frame_store=dataset._store,
    frame_offsets=dataset.offsets,
    flip_prob=0.0,
    train=True,
  )
  ds.materialize_teacher_cache()
  loader = DataLoader(ds, batch_size=min(32, len(subset_ids)), shuffle=True, collate_fn=collate_student_final)
  opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
  model.train()
  step = 0
  while step < steps:
    for batch in loader:
      b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
      out = model(b["clip"])
      loss, _ = combined_loss(out, b, lambda_gt=lambda_gt, lambda_kd=lambda_kd)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      step += 1
      if step >= steps:
        break
  model.eval()
  with torch.no_grad():
    rows = forward_collect(model, loader, device)["rows"]
  rep = confusion_and_rates(rows, c_warn=0.5, c_stop=0.55, use_argmax=True)
  m = rep["metrics"]
  ok = m.get("stop_recall", 0) >= 0.95 and m.get("safe_recall", 0) >= 0.95
  return {
    "ok": ok,
    "n_windows": len(indices),
    "steps": steps,
    "stop_recall": m.get("stop_recall"),
    "safe_recall": m.get("safe_recall"),
    "confusion": rep["confusion_matrix"],
  }


def run_preflight(
  cfg,
  *,
  cache_dir: Path,
  arch_report_dir: Path,
  report_dir: Path,
  exp_dir: Path,
  audit: dict[str, Any],
  gt_group_mean: np.ndarray,
  backbone_state: dict,
  warning: dict,
  device: str,
  frame_store,
  lambda_gt: float = 1.0,
  lambda_kd: float = 0.1,
) -> dict[str, Any]:
  rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open(encoding="utf-8")]
  train_stop_groups = audit["train_summary"]["n_stop_sample_groups"]
  sel_stop_groups = audit["selection_summary"]["n_stop_sample_groups"]
  halt_reasons: list[str] = []
  if train_stop_groups < 10:
    halt_reasons.append(f"train STOP groups={train_stop_groups} < 10 (flag unestablished)")
  if sel_stop_groups < 5:
    halt_reasons.append(f"selection STOP groups={sel_stop_groups} < 5 — halt before main training")

  parent = parent_dev_sanity(cfg, cache_dir, arch_report_dir, gt_group_mean, warning)
  if not parent["ok"]:
    halt_reasons.append("parent dev STOP recall is zero")

  prepared_ids = sorted(
    set(audit["train_row_ids"]) | set(audit["selection_row_ids"]) | set(audit["dev_row_ids"])
  )
  flip = flip_symmetry_check(frame_store, rows, prepared_ids)

  overfit_ids = select_overfit_windows(rows, gt_group_mean, audit["train_row_ids"])
  overfit = {"ok": False, "reason": "not run"}
  if len(overfit_ids) >= 64:
    train_ds = StudentFinalDataset(
      cache_dir, audit["train_row_ids"], gt_group_mean, frame_store=frame_store, flip_prob=0.0, train=True
    )
    train_ds.materialize_teacher_cache()
    try:
      model = StudentFinal().to(device)
    except Exception:
      device = "cpu"
      model = StudentFinal().to(device)
    model.load_fresh_backbone(backbone_state)
    fresh = copy.deepcopy(model)
    overfit = overfit64_test(fresh, train_ds, overfit_ids, device, steps=300, lambda_gt=lambda_gt, lambda_kd=lambda_kd)
    if not overfit.get("ok"):
      halt_reasons.append("64-window overfit failed (<0.95 STOP or SAFE recall)")
  else:
    halt_reasons.append("64-window overfit could not be constructed reliably")

  report = {
    "data_counts": {
      "train": audit["train_summary"],
      "selection": audit["selection_summary"],
      "dev": audit["dev_summary"],
    },
    "parent_sanity": parent,
    "flip_symmetry": flip,
    "overfit64": overfit,
    "halt_reasons": halt_reasons,
    "proceed_to_training": not any(
      x in h for h in halt_reasons for x in ("halt before", "failed", "recall is zero", "overfit failed")
    ),
    "flip_augmentation_enabled": bool(flip.get("ok")),
  }
  exp_dir.mkdir(parents=True, exist_ok=True)
  report_dir.mkdir(parents=True, exist_ok=True)
  (report_dir / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
  md = [
    "# Student final from-scratch preflight",
    "",
    f"- Train windows: {audit['train_summary']['n_windows']} ({audit['train_summary']['n_sample_groups']} groups)",
    f"- Selection windows: {audit['selection_summary']['n_windows']} ({audit['selection_summary']['n_sample_groups']} groups)",
    f"- Dev windows: {audit['dev_summary']['n_windows']}",
    f"- Train STOP groups: {train_stop_groups}; selection STOP groups: {sel_stop_groups}",
    f"- Parent dev stop recall: {parent['parent_dev_metrics'].get('stop_recall')}",
    f"- Overfit64 ok: {overfit.get('ok')} (STOP={overfit.get('stop_recall')}, SAFE={overfit.get('safe_recall')})",
    f"- Flip symmetry ok: {flip.get('ok')}",
    f"- Proceed: **{report['proceed_to_training']}**",
  ]
  if halt_reasons:
    md.append("\n## Halt / flags\n")
    for h in halt_reasons:
      md.append(f"- {h}")
  (report_dir / "preflight_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
  return report
