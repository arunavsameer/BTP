#!/usr/bin/env python3
"""Fast smoke test for student_final pipeline (~2–5 min, small RGB subset)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.fdec_pilot_dataset import build_frame_store  # noqa: E402
from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS, init_shared_backbone  # noqa: E402
from collision_jepa3.student_research.student_final.dataset import StudentFinalDataset, collate_student_final  # noqa: E402
from collision_jepa3.student_research.student_final.losses import combined_loss  # noqa: E402
from collision_jepa3.student_research.student_final.model import StudentFinal  # noqa: E402
from collision_jepa3.student_research.student_final.preflight import (  # noqa: E402
  flip_symmetry_check,
  overfit64_test,
  parent_dev_sanity,
  select_overfit_windows,
)
from collision_jepa3.student_research.student_final.splits import build_data_splits  # noqa: E402
from collision_jepa3.student_research.student_final.targets import build_group_mean_gt_array, peak_tier  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

CACHE = ROOT / "reports/student_architecture_tests/teacher_cache_train_pilot"
ARCH = ROOT / "reports/student_architecture_tests"
OUT = ROOT / "reports/student_final_from_scratch/smoke_report.json"


def _pick_smoke_rows(cache_rows: list[dict], train_ids: list[int], gt: np.ndarray, *, n: int = 24) -> list[int]:
  peaks = gt.max(axis=(1, 2))
  safe = [i for i in train_ids if peak_tier(float(peaks[i])) == "SAFE"]
  stop = [i for i in train_ids if peak_tier(float(peaks[i])) == "STOP"]
  rng = np.random.default_rng(1337)
  want = min(12, len(safe), len(stop))
  if want < 4:
    return sorted(train_ids[:n])
  pick = list(rng.choice(safe, want, replace=False)) + list(rng.choice(stop, want, replace=False))
  return sorted(int(x) for x in pick[:n])


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="auto")
  add_config_args(parser)
  args = parser.parse_args()
  cfg = load_cfg(args)
  device = resolve_device_safe(str(args.device))
  checks: list[dict] = []
  ok = True

  def record(name: str, passed: bool, detail: dict | None = None) -> None:
    nonlocal ok
    ok = ok and passed
    checks.append({"name": name, "ok": passed, **(detail or {})})

  try:
    cache_rows = [json.loads(l) for l in (CACHE / "index.jsonl").open(encoding="utf-8")]
    with np.load(CACHE / "gt_heatmaps.npz") as z:
      raw_gt = np.asarray(z["gt_1s"])
    gt_group_mean, _ = build_group_mean_gt_array(cfg, cache_rows, raw_gt=raw_gt)
    record("group_mean_gt", True, {"shape": list(gt_group_mean.shape)})

    seed = int(args.seed if args.seed is not None else 1337)
    audit = build_data_splits(cfg, CACHE, ARCH, gt_group_mean, seed=seed)
    record("data_splits", bool(audit["train_row_ids"]) and bool(audit["dev_row_ids"]))

    row_ids = _pick_smoke_rows(cache_rows, audit["train_row_ids"], gt_group_mean)
    store, prep = build_frame_store(cfg, CACHE, row_ids, FDEC_FRAME_OFFSETS, prepare_workers=2)
    record("rgb_prepare_subset", prep.get("n_clips_decoded", 0) > 0, prep)

    warning = cfg.get("warning") or {}
    parent = parent_dev_sanity(cfg, CACHE, ARCH, gt_group_mean, warning)
    record("parent_dev_sanity", parent.get("ok", False), {"stop_recall": parent.get("parent_dev_metrics", {}).get("stop_recall")})

    flip = flip_symmetry_check(store, cache_rows, row_ids)
    record("flip_symmetry", flip.get("ok", False), flip)

    ds = StudentFinalDataset(CACHE, row_ids, gt_group_mean, frame_store=store, train=True)
    ds.materialize_teacher_cache()
    loader = DataLoader(ds, batch_size=min(8, len(row_ids)), shuffle=True, collate_fn=collate_student_final)
    record("dataset_teacher_materialized", ds._H_arr is not None)

    model = StudentFinal()
    try:
      model = model.to(device)
    except Exception:
      device = "cpu"
      model = model.to(device)
    model.load_fresh_backbone(init_shared_backbone(seed=seed))
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    steps = 0
    terms: dict = {}
    while steps < 5:
      for batch in loader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(b["clip"])
        loss, terms = combined_loss(out, b, lambda_gt=1.0, lambda_kd=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        if steps >= 5:
          break
    record("train_forward_backward", steps >= 5, {"steps": steps, "L_total": terms.get("L_total"), "device": device})

    with torch.no_grad():
      out = model(b["clip"])
    record("forward_eval", bool(out.get("Hhat_1s") is not None and out.get("q_stop") is not None))

    import copy

    train_ids = audit["train_row_ids"]
    of_ids = select_overfit_windows(cache_rows, gt_group_mean, train_ids)
    if len(of_ids) >= 32:
      sub = of_ids[:32]
      store_of, _ = build_frame_store(cfg, CACHE, sub, FDEC_FRAME_OFFSETS, prepare_workers=2)
      ds_of = StudentFinalDataset(CACHE, sub, gt_group_mean, frame_store=store_of, train=True)
      ds_of.materialize_teacher_cache()
      m2 = StudentFinal().to(device)
      m2.load_fresh_backbone(init_shared_backbone(seed=seed))
      of_res = overfit64_test(copy.deepcopy(m2), ds_of, sub, device, steps=3, lambda_gt=1.0, lambda_kd=0.1)
      record("overfit64_backward", of_res.get("steps") == 3 and "confusion" in of_res, of_res)
  except Exception as e:
    ok = False
    checks.append({"name": "exception", "ok": False, "error": traceback.format_exc()})
    print(traceback.format_exc(), flush=True)

  OUT.parent.mkdir(parents=True, exist_ok=True)
  report = {"ok": ok, "checks": checks}
  OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(json.dumps(report, indent=2), flush=True)
  if not ok:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
