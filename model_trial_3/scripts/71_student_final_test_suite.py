#!/usr/bin/env python3
"""Run all student_final checks before a full training launch."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "reports/student_final_from_scratch"
OUT = REPORT / "test_suite_report.json"


def run_smoke(device: str) -> dict:
  p = subprocess.run(
    [sys.executable, str(ROOT / "scripts/70_student_final_smoke.py"), "--config", "configs/default.yaml", "--device", device],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
  )
  ok = p.returncode == 0
  detail = p.stdout[-2000:] if p.stdout else p.stderr[-2000:]
  return {"ok": ok, "returncode": p.returncode, "tail": detail}


def run_mini_overfit64(device: str) -> dict:
  """Same code path as preflight overfit (deepcopy + 300-step loop would be too slow)."""
  import numpy as np
  import torch
  from torch.utils.data import DataLoader

  from scripts._common import load_cfg
  import argparse

  from collision_jepa3.student_research.fdec_pilot_dataset import build_frame_store
  from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS, init_shared_backbone
  from collision_jepa3.student_research.student_final.dataset import StudentFinalDataset, collate_student_final
  from collision_jepa3.student_research.student_final.preflight import overfit64_test, select_overfit_windows
  from collision_jepa3.student_research.student_final.splits import build_data_splits
  from collision_jepa3.student_research.student_final.targets import build_group_mean_gt_array
  from collision_jepa3.student_research.student_final.model import StudentFinal
  from collision_jepa3.utils.device import resolve_device_safe

  CACHE = ROOT / "reports/student_architecture_tests/teacher_cache_train_pilot"
  ARCH = ROOT / "reports/student_architecture_tests"
  cfg = load_cfg(argparse.Namespace(config=str(ROOT / "configs/default.yaml"), dataset_root=None, seed=1337))
  device = resolve_device_safe(device)
  cache_rows = [json.loads(l) for l in (CACHE / "index.jsonl").open(encoding="utf-8")]
  with np.load(CACHE / "gt_heatmaps.npz") as z:
    gt, _ = build_group_mean_gt_array(cfg, cache_rows, raw_gt=np.asarray(z["gt_1s"]))
  audit = build_data_splits(cfg, CACHE, ARCH, gt, seed=1337)
  overfit_ids = select_overfit_windows(cache_rows, gt, audit["train_row_ids"])
  if len(overfit_ids) < 32:
    return {"ok": False, "reason": "not enough overfit windows", "n": len(overfit_ids)}
  row_ids = overfit_ids[:64]
  store, _ = build_frame_store(cfg, CACHE, row_ids, FDEC_FRAME_OFFSETS, prepare_workers=2)
  train_ds = StudentFinalDataset(CACHE, row_ids, gt, frame_store=store, train=True)
  train_ds.materialize_teacher_cache()
  model = StudentFinal()
  try:
    model = model.to(device)
  except Exception:
    device = "cpu"
    model = model.to(device)
  model.load_fresh_backbone(init_shared_backbone(seed=1337))
  fresh = copy.deepcopy(model)
  res = overfit64_test(fresh, train_ds, row_ids, device, steps=30, lambda_gt=1.0, lambda_kd=0.1)
  ok = "stop_recall" in res and res.get("steps") == 30 and "confusion" in res
  return {"ok": ok, "device": device, "overfit": res}


def run_eval_split(device: str) -> dict:
  import numpy as np
  import torch
  from torch.utils.data import DataLoader

  from scripts._common import load_cfg
  import argparse

  from collision_jepa3.student_research.fdec_pilot_dataset import build_frame_store
  from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS, init_shared_backbone
  from collision_jepa3.student_research.student_final.dataset import StudentFinalDataset
  from collision_jepa3.student_research.student_final.train import evaluate_split
  from collision_jepa3.student_research.student_final.targets import build_group_mean_gt_array
  from collision_jepa3.student_research.student_final.model import StudentFinal
  from collision_jepa3.utils.device import resolve_device_safe

  CACHE = ROOT / "reports/student_architecture_tests/teacher_cache_train_pilot"
  cfg = load_cfg(argparse.Namespace(config=str(ROOT / "configs/default.yaml"), dataset_root=None, seed=1337))
  device = resolve_device_safe(device)
  cache_rows = [json.loads(l) for l in (CACHE / "index.jsonl").open(encoding="utf-8")]
  with np.load(CACHE / "gt_heatmaps.npz") as z:
    gt, _ = build_group_mean_gt_array(cfg, cache_rows, raw_gt=np.asarray(z["gt_1s"]))
  dev_ids = json.loads((ROOT / "reports/student_architecture_tests/part1_dev_manifest.json").read_text())["dev_row_ids"][:32]
  store, _ = build_frame_store(cfg, CACHE, dev_ids, FDEC_FRAME_OFFSETS, prepare_workers=2)
  ds = StudentFinalDataset(CACHE, dev_ids[:8], gt, frame_store=store, train=False)
  ds.materialize_teacher_cache()
  model = StudentFinal().to(device)
  model.load_fresh_backbone(init_shared_backbone(1337))
  model.eval()
  rows = evaluate_split(model, ds, dev_ids[:8], device)
  return {"ok": len(rows) == 8, "n_rows": len(rows)}


def run_optimizer_setup() -> dict:
  from collision_jepa3.student_research.student_final.model import StudentFinal
  from collision_jepa3.student_research.student_final.train import adamw_param_groups
  import torch

  m = StudentFinal()
  d, nd = adamw_param_groups(m)
  opt = torch.optim.AdamW(
    [{"params": d, "weight_decay": 0.05}, {"params": nd, "weight_decay": 0.0}],
    lr=3e-4,
  )
  return {"ok": len(d) > 0 and len(nd) > 0, "n_decay": len(d), "n_no_decay": len(nd)}


def main() -> None:
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="auto")
  args = ap.parse_args()
  device = args.device
  tests = [
    ("optimizer_setup", lambda: run_optimizer_setup()),
    ("smoke", lambda: run_smoke(device)),
    ("mini_overfit64", lambda: run_mini_overfit64(device)),
    ("evaluate_split", lambda: run_eval_split(device)),
  ]
  results = []
  all_ok = True
  for name, fn in tests:
    print(f"=== {name} ===", flush=True)
    try:
      r = fn()
      r["name"] = name
      passed = bool(r.get("ok"))
    except Exception as e:
      import traceback

      r = {"name": name, "ok": False, "error": traceback.format_exc()}
      passed = False
    all_ok = all_ok and passed
    results.append(r)
    print(json.dumps(r, indent=2)[:1500], flush=True)
    if not passed:
      print(f"STOP: {name} failed", flush=True)
      break

  REPORT.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps({"ok": all_ok, "tests": results}, indent=2), encoding="utf-8")
  if not all_ok:
    raise SystemExit(1)
  print("ALL TESTS PASSED", flush=True)


if __name__ == "__main__":
  main()
