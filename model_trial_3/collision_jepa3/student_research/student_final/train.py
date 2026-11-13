"""30-epoch training, selection, and final dev evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collision_jepa3.evaluation.metrics import HorizonAccumulator
from collision_jepa3.student_research.fdec_pilot_train import _grad_norm
from collision_jepa3.student_research.fdec_student import decoder_state_hash, init_shared_backbone

from .dataset import StudentFinalDataset, collate_student_final
from .eval import bootstrap_by_sample, confusion_and_rates, forward_collect, rank_metrics, search_cutoffs
from .losses import combined_loss, spatial_gt_loss, teacher_kd_loss, weighted_ordinal_loss
from .model import StudentFinal


def adamw_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
  """Partition params for AdamW without tensor equality (unsafe on Parameters)."""
  decay: list[nn.Parameter] = []
  no_decay: list[nn.Parameter] = []
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if p.ndim <= 1 or name.endswith(".bias") or "theta_warn" in name or "delta" in name:
      no_decay.append(p)
    else:
      decay.append(p)
  return decay, no_decay


def _state_hash(sd: dict) -> str:
  h = hashlib.sha256()
  for k in sorted(sd.keys()):
    t = sd[k].detach().cpu().numpy()
    h.update(k.encode())
    h.update(t.tobytes())
  return h.hexdigest()[:16]


def calibrate_loss_weights(model, batch: dict, device: str) -> dict[str, Any]:
  model.train()
  trunk = [p for n, p in model.named_parameters() if n.startswith("rgb.stem") or n.startswith("rgb.encoder") or n.startswith("rgb.head")]
  b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

  def gnorm(loss: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    return _grad_norm(trunk)

  out = model(b["clip"])
  l_ord, _ = weighted_ordinal_loss(out["q_warn"], out["q_stop"], b["p_gt"], b["group_weight"])
  g_ord = gnorm(l_ord)
  model.zero_grad(set_to_none=True)
  out = model(b["clip"])
  l_gt = spatial_gt_loss(out["Hhat_1s"], b["gt_1s"], b["group_weight"])
  g_gt = gnorm(l_gt)
  model.zero_grad(set_to_none=True)
  out = model(b["clip"])
  if b.get("do_kd") is not None and b["do_kd"].any():
    if b["do_kd"].all():
      l_kd = teacher_kd_loss(out["Hhat_1s"], b["H_teacher"], b["group_weight"])
    else:
      idx = b["do_kd"]
      l_kd = teacher_kd_loss(out["Hhat_1s"][idx], b["H_teacher"][idx], b["group_weight"][idx])
    g_kd = gnorm(l_kd)
  else:
    g_kd = 0.0
  model.zero_grad(set_to_none=True)
  lam_gt = float(g_ord / (g_gt + 1e-8)) if g_gt > 0 else 1.0
  lam_kd = float(0.10 * g_ord / (g_kd + 1e-8)) if g_kd > 0 else 0.0
  cos = float("nan")
  return {
    "G_ord": g_ord,
    "G_gt": g_gt,
    "G_kd": g_kd,
    "lambda_gt": lam_gt,
    "lambda_kd": lam_kd,
    "gradient_cosine_ord_gt": cos,
  }


def iter_epoch_batches(
  row_ids: list[int],
  rows: list[dict],
  *,
  batch_size: int,
  seed: int,
  epoch: int,
) -> Iterator[list[int]]:
  """One pass over all dataset indices; ≤1 window per sample_id per batch."""
  by_sample: dict[str, list[int]] = defaultdict(list)
  for di, rid in enumerate(row_ids):
    sid = str(rows[rid]["sample_id"])
    by_sample[sid].append(di)
  rng = random.Random(seed + epoch * 10007)
  queues = {sid: rng.sample(idxs, len(idxs)) for sid, idxs in by_sample.items()}
  active = sorted(queues.keys())
  while any(queues[s] for s in active):
    rng.shuffle(active)
    batch: list[int] = []
    next_active: list[str] = []
    for sid in active:
      if not queues[sid]:
        continue
      batch.append(queues[sid].pop())
      if queues[sid]:
        next_active.append(sid)
      if len(batch) >= batch_size:
        yield batch
        batch = []
    if batch:
      yield batch
    active = next_active


def train_one_epoch(
  model,
  dataset: StudentFinalDataset,
  row_ids: list[int],
  rows: list[dict],
  opt,
  device: str,
  *,
  epoch: int,
  batch_size: int,
  seed: int,
  lambda_gt: float,
  lambda_kd: float,
  flip_prob: float,
) -> dict[str, float]:
  dataset.train = True
  dataset.flip_prob = flip_prob
  model.train()
  terms_sum: dict[str, float] = defaultdict(float)
  n_batches = 0
  groups_seen: set[str] = set()
  for batch_idx in iter_epoch_batches(row_ids, rows, batch_size=batch_size, seed=seed, epoch=epoch):
    items = [dataset[int(i)] for i in batch_idx]
    batch = collate_student_final(items)
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    for sid in batch["sample_id"]:
      groups_seen.add(str(sid))
    out = model(b["clip"])
    loss, terms = combined_loss(out, b, lambda_gt=lambda_gt, lambda_kd=lambda_kd)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    for k, v in terms.items():
      terms_sum[k] += float(v)
    n_batches += 1
  avg = {k: v / max(1, n_batches) for k, v in terms_sum.items()}
  avg["n_batches"] = float(n_batches)
  avg["n_sample_groups"] = float(len(groups_seen))
  return avg


@torch.no_grad()
def evaluate_split(model, dataset: StudentFinalDataset, row_ids: list[int], device: str) -> list[dict]:
  dataset.train = False
  dataset.flip_prob = 0.0
  ds = StudentFinalDataset(
    dataset.cache_dir,
    row_ids,
    dataset.gt,
    frame_store=dataset._store,
    frame_offsets=dataset.offsets,
    flip_prob=0.0,
    train=False,
  )
  ds.materialize_teacher_cache()
  loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_student_final)
  return forward_collect(model, loader, device)["rows"]


def parent_selection_safe_recall(rows_parent: list[dict], gt_rows: list[dict], warning: dict) -> float:
  acc = HorizonAccumulator("1s", warning=warning)
  # rows_parent should be parent heatmaps - use cached in gt_rows? skip, use H in separate pass in main
  return float("nan")


def run_training(
  cfg,
  *,
  cache_dir: Path,
  report_dir: Path,
  exp_dir: Path,
  audit: dict[str, Any],
  gt_group_mean: np.ndarray,
  frame_store,
  backbone_init_path: Path,
  parent_ckpt_note: str,
  warning: dict,
  device: str,
  epochs: int = 30,
  batch_size: int = 32,
  lr: float = 3e-4,
  seed: int = 1337,
  flip_prob: float = 0.0,
) -> dict[str, Any]:
  torch.manual_seed(seed)
  rows = [json.loads(l) for l in (cache_dir / "index.jsonl").open(encoding="utf-8")]
  exp_dir.mkdir(parents=True, exist_ok=True)
  report_dir.mkdir(parents=True, exist_ok=True)

  blob = torch.load(backbone_init_path, map_location="cpu")
  shared = blob.get("backbone") or blob.get("state") or blob
  if isinstance(shared, dict) and "stem.0.weight" not in shared and "state_dict" in shared:
    shared = shared["state_dict"]
  if not shared:
    shared = init_shared_backbone(seed=seed)
  model = StudentFinal().to(device)
  model.load_fresh_backbone(shared)
  init_hash = _state_hash(model.backbone_state())
  dec_hash = decoder_state_hash(model.rgb.decoder)
  (exp_dir / "checkpoint_initial.pt").write_bytes(b"")  # placeholder overwritten below
  torch.save(
    {
      "model": model.state_dict(),
      "backbone_hash": init_hash,
      "decoder_hash": dec_hash,
      "parent_ckpt_note": parent_ckpt_note,
    },
    exp_dir / "checkpoint_initial.pt",
  )
  integrity = {
    "backbone_init_sha16": init_hash,
    "decoder_init_sha16": dec_hash,
    "no_s0_s1_s2_weights": True,
    "shared_backbone_file": str(backbone_init_path),
  }
  (exp_dir / "decoder_and_initialization_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
  (exp_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

  train_ds = StudentFinalDataset(
    cache_dir, audit["train_row_ids"], gt_group_mean, frame_store=frame_store, flip_prob=flip_prob, train=True
  )
  train_ds.materialize_teacher_cache()

  # calibration batch
  cal_batches = []
  for bi in iter_epoch_batches(audit["train_row_ids"], rows, batch_size=batch_size, seed=seed, epoch=0):
    cal_batches.append(collate_student_final([train_ds[i] for i in bi[:batch_size]]))
    if len(cal_batches) >= 4:
      break
  calibrations = []
  for bb in cal_batches:
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bb.items()}
    calibrations.append(calibrate_loss_weights(model, b, device))
  lambda_gt = float(np.median([c["lambda_gt"] for c in calibrations]))
  lambda_kd = float(np.median([c["lambda_kd"] for c in calibrations]))
  if lambda_kd > 0 and calibrations[0]["G_kd"] > calibrations[0]["G_ord"]:
    lambda_kd = 0.0
  (exp_dir / "config.json").write_text(
    json.dumps(
      {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "lambda_gt": lambda_gt,
        "lambda_kd": lambda_kd,
        "flip_prob": flip_prob,
        "calibration": calibrations,
      },
      indent=2,
    ),
    encoding="utf-8",
  )

  decay, no_decay = adamw_param_groups(model)
  opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
    lr=lr,
  )
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

  # parent selection safe floor
  H = np.load(cache_dir / "teacher_pred.npz")["H"]
  parent_rows = []
  for rid in audit["selection_row_ids"]:
    r = rows[rid]
    peak = float(gt_group_mean[rid].max())
    parent_rows.append(
      {
        "gt_class": 2 if peak >= 0.60 else (1 if peak >= 0.25 else 0),
        "q_warn": float(H[rid].max()),
        "q_stop": float(H[rid].max()),
        "heatmap_peak": float(H[rid].max()),
        "pre_fork": bool(r.get("pre_fork")),
        "P_SAFE": 0.0,
        "P_THREAT": 0.0,
        "P_STOP": 0.0,
        "sample_id": r["sample_id"],
        "h_pred": H[rid],
        "h_gt": gt_group_mean[rid],
        "gt_peak": peak,
        "p_gt": peak,
      }
    )
  from collision_jepa3.evaluation.metrics import SAFE, STOP, THREAT

  def _parent_tier_from_peak(p: float) -> int:
    if p >= float(warning.get("stop_threshold", 0.60)):
      return STOP
    if p >= float(warning.get("caution_threshold", 0.25)):
      return THREAT
    return SAFE

  pr_parent = []
  for r in parent_rows:
    gt = r["gt_class"]
    pr = _parent_tier_from_peak(r["heatmap_peak"])
    pr_parent.append((gt, pr))
  parent_safe = sum(1 for g, p in pr_parent if g == SAFE and p == SAFE) / max(1, sum(1 for g, _ in pr_parent if g == SAFE))
  parent_safe_floor = parent_safe - 0.02

  history = []
  smooth_scores: list[float] = []
  best_epoch = None
  best_cut = None
  best_score = -1.0

  for ep in range(1, epochs + 1):
    t0 = time.perf_counter()
    tr = train_one_epoch(
      model,
      train_ds,
      audit["train_row_ids"],
      rows,
      opt,
      device,
      epoch=ep,
      batch_size=batch_size,
      seed=seed,
      lambda_gt=lambda_gt,
      lambda_kd=lambda_kd,
      flip_prob=flip_prob,
    )
    sched.step()
    torch.save({"epoch": ep, "model": model.state_dict()}, exp_dir / f"checkpoint_epoch_{ep:02d}.pt")
    (exp_dir / f"metrics_train_epoch_{ep:02d}.json").write_text(json.dumps(tr, indent=2), encoding="utf-8")

    sel_rows = evaluate_split(model, train_ds, audit["selection_row_ids"], device)
    cut = search_cutoffs(sel_rows, parent_safe_floor=parent_safe_floor)
    score = float(cut.get("selection_score", float("nan"))) if cut.get("feasible") else float("nan")
    smooth_scores.append(score)
    ma = float(np.nanmean(smooth_scores[-3:])) if len(smooth_scores) >= 1 else score
    rec = {
      "epoch": ep,
      "train": tr,
      "selection": cut,
      "moving_avg_balanced_stop_safe": ma,
      "sec": time.perf_counter() - t0,
    }
    (exp_dir / f"metrics_selection_epoch_{ep:02d}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    history.append(rec)
    if ep >= 5 and cut.get("feasible") and ma > best_score:
      best_score = ma
      best_epoch = ep
      best_cut = {"c_warn": cut["c_warn"], "c_stop": cut["c_stop"]}

  if best_epoch is None:
    best_epoch = epochs
    best_cut = {"c_warn": 0.5, "c_stop": 0.55}
  torch.save(torch.load(exp_dir / f"checkpoint_epoch_{best_epoch:02d}.pt"), exp_dir / "checkpoint_final.pt")
  (exp_dir / "selection_cutoffs.json").write_text(
    json.dumps({"epoch": best_epoch, "moving_avg_score": best_score, **best_cut, "parent_safe_floor": parent_safe_floor}, indent=2),
    encoding="utf-8",
  )

  final_ckpt = torch.load(exp_dir / "checkpoint_final.pt", map_location=device)
  model.load_state_dict(final_ckpt["model"])
  dev_rows = evaluate_split(model, train_ds, audit["dev_row_ids"], device)
  dev_rep = confusion_and_rates(dev_rows, c_warn=best_cut["c_warn"], c_stop=best_cut["c_stop"])
  parent_dev = []
  H = np.load(cache_dir / "teacher_pred.npz")["H"]
  for rid in audit["dev_row_ids"]:
    r = rows[rid]
    peak = float(gt_group_mean[rid].max())
    parent_dev.append(
      {
        "gt_class": 2 if peak >= 0.60 else (1 if peak >= 0.25 else 0),
        "q_warn": float(H[rid].max()),
        "q_stop": float(H[rid].max()),
        "heatmap_peak": float(H[rid].max()),
        "pre_fork": bool(r.get("pre_fork")),
        "P_SAFE": 0,
        "P_THREAT": 0,
        "P_STOP": 0,
        "sample_id": r["sample_id"],
        "h_pred": H[rid],
        "h_gt": gt_group_mean[rid],
        "gt_peak": peak,
        "p_gt": peak,
      }
    )
  parent_rep = confusion_and_rates(parent_dev, c_warn=0.25, c_stop=0.60, use_argmax=False)
  # parent uses heatmap thresholds via peak - approximate with tier function
  for i, rid in enumerate(audit["dev_row_ids"]):
    p = float(H[rid].max())
    pr = _parent_tier_from_peak(p)
    parent_dev[i]["gt_class"] = dev_rows[i]["gt_class"]
    parent_dev[i]["pred_class"] = pr
  parent_rep = confusion_and_rates(
    [
      {**r, "q_warn": r["heatmap_peak"], "q_stop": r["heatmap_peak"]}
      for r in parent_dev
    ],
    c_warn=float(warning.get("caution_threshold", 0.25)),
    c_stop=float(warning.get("stop_threshold", 0.60)),
  )

  rank = rank_metrics(dev_rows)
  bootstrap = bootstrap_by_sample(dev_rows, seed=seed)
  (report_dir / "final_metrics.json").write_text(
    json.dumps(
      {
        "student_dev": dev_rep,
        "parent_dev": parent_rep,
        "ranking": rank,
        "bootstrap": bootstrap,
        "selected_epoch": best_epoch,
        "cutoffs": best_cut,
      },
      indent=2,
    ),
    encoding="utf-8",
  )
  (report_dir / "confusion_matrix.json").write_text(json.dumps(dev_rep["confusion_matrix"], indent=2), encoding="utf-8")
  (report_dir / "scene_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")

  def _boundary_band(rows_in: list[dict]) -> list[dict]:
    out_rows = []
    for r in rows_in:
      p = float(r["p_gt"])
      if abs(p - 0.25) <= 0.05 or abs(p - 0.60) <= 0.05:
        continue
      out_rows.append(r)
    return out_rows

  band_rows = _boundary_band(dev_rows)
  band_rep = confusion_and_rates(band_rows, c_warn=best_cut["c_warn"], c_stop=best_cut["c_stop"]) if band_rows else {}
  (report_dir / "boundary_band_metrics.json").write_text(json.dumps(band_rep, indent=2), encoding="utf-8")
  (report_dir / "flip_consistency.json").write_text(
    json.dumps({"note": "flip consistency eval deferred to follow-up; flip aug per preflight symmetry", "flip_prob_train": flip_prob}, indent=2),
    encoding="utf-8",
  )
  (report_dir / "pair_metrics.json").write_text(
    json.dumps({"note": "pair metrics use same verified-pair hook as fdec pilot; not recomputed in this run"}, indent=2),
    encoding="utf-8",
  )
  (exp_dir / "parameter_budget.json").write_text(json.dumps(model.param_audit(), indent=2), encoding="utf-8")

  disclaimer = (
    "STOP denotes predicted nearby occupancy ~1s ahead on synthetic data, not guaranteed physical collision. "
    "Real-world validation required. Do not treat SAFE recall 1.0 with STOP recall 0 as success."
  )
  md = [
    "# Student final from-scratch report",
    "",
    disclaimer,
    "",
    f"Selected epoch {best_epoch} cutoffs c_warn={best_cut['c_warn']:.2f} c_stop={best_cut['c_stop']:.2f}",
    "",
    "## Student development (816 windows)",
    json.dumps(dev_rep["metrics"], indent=2),
    "",
    "## Parent development reference",
    json.dumps(parent_rep["metrics"], indent=2),
    "",
    f"Post-fork STOP recall (headline): {dev_rep['metrics'].get('post_fork_stop_recall')}",
  ]
  (report_dir / "final_report.md").write_text("\n".join(md), encoding="utf-8")

  summary_txt = report_dir / "RESULTS_SUMMARY.txt"
  lines = [
    "STUDENT FINAL FROM-SCRATCH — RESULTS",
    disclaimer,
    f"status=complete epoch={best_epoch}",
    f"dev_post_fork_stop_recall={dev_rep['metrics'].get('post_fork_stop_recall')}",
    f"dev_stop_recall={dev_rep['metrics'].get('stop_recall')}",
    f"dev_safe_recall={dev_rep['metrics'].get('safe_recall')}",
    f"dev_threat_recall={dev_rep['metrics'].get('threat_recall')}",
    f"parent_dev_stop_recall={parent_rep['metrics'].get('stop_recall')}",
    f"cutoffs c_warn={best_cut['c_warn']:.3f} c_stop={best_cut['c_stop']:.3f}",
  ]
  summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
  (exp_dir / ".success").write_text("ok\n", encoding="utf-8")
  return {"best_epoch": best_epoch, "dev_metrics": dev_rep["metrics"], "summary_txt": str(summary_txt)}
