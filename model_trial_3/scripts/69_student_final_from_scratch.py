#!/usr/bin/env python3
"""Final collision-JEPA student from scratch (ordinal + heatmap)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.fdec_pilot_dataset import build_frame_store  # noqa: E402
from collision_jepa3.student_research.fdec_pilot_train import prepare_shared_backbone  # noqa: E402
from collision_jepa3.student_research.student_final.preflight import run_preflight  # noqa: E402
from collision_jepa3.student_research.student_final.splits import build_data_splits  # noqa: E402
from collision_jepa3.student_research.student_final.targets import build_group_mean_gt_array  # noqa: E402
from collision_jepa3.student_research.student_final.train import run_training  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

REPORT = ROOT / "reports/student_final_from_scratch"
CACHE = ROOT / "reports/student_architecture_tests/teacher_cache_train_pilot"
EXP = ROOT / "experiments/student_final_from_scratch"
BASELINE = ROOT / "experiments/student_fdec_unoptimized_baseline/shared_backbone_init.pt"
PARENT_CKPT = ROOT / "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch", type=int, default=32)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--preflight-only", action="store_true")
  parser.add_argument("--skip-preflight", action="store_true", help="use existing preflight_report.json if proceed_to_training")
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--device", default="auto", help="cuda, cpu, or auto (falls back to cpu if GPU busy)")
  add_config_args(parser)
  args = parser.parse_args()
  cfg = load_cfg(args)
  device = resolve_device_safe(str(args.device))
  REPORT.mkdir(parents=True, exist_ok=True)
  EXP.mkdir(parents=True, exist_ok=True)
  log_path = EXP / "run.log"

  def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    print(msg, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
      f.write(line)

  if (EXP / ".success").exists() and not args.force:
    log("Already complete (.success). Use --force to rerun.")
    return

  try:
    rows_cache = CACHE
    cache_rows = _load_cache_rows(rows_cache)
    with _np_load_gt(rows_cache) as raw_gt:
      gt_group_mean, gt_meta = build_group_mean_gt_array(cfg, cache_rows, raw_gt=raw_gt)
    log(f"Built group-mean targets: {gt_meta}")

    seed = int(args.seed if args.seed is not None else 1337)
    audit = build_data_splits(cfg, rows_cache, ROOT / "reports/student_architecture_tests", gt_group_mean, seed=seed)
    (EXP / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    all_ids = sorted(set(audit["train_row_ids"] + audit["selection_row_ids"] + audit["dev_row_ids"]))
    from collision_jepa3.student_research.fdec_student import FDEC_FRAME_OFFSETS

    store, prep = build_frame_store(cfg, rows_cache, all_ids, FDEC_FRAME_OFFSETS, prepare_workers=2)
    log(f"RGB prepare: {prep}")

    backbone_path = BASELINE
    shared = prepare_shared_backbone(EXP, seed=seed, baseline_init=backbone_path)
    warning = cfg.get("warning") or {}

    preflight_path = REPORT / "preflight_report.json"
    if args.skip_preflight and preflight_path.is_file():
      pre = json.loads(preflight_path.read_text(encoding="utf-8"))
      log(f"Skipped preflight (loaded); proceed={pre.get('proceed_to_training')}")
    else:
      pre = run_preflight(
        cfg,
        cache_dir=rows_cache,
        arch_report_dir=ROOT / "reports/student_architecture_tests",
        report_dir=REPORT,
        exp_dir=EXP,
        audit=audit,
        gt_group_mean=gt_group_mean,
        backbone_state=shared,
        warning=warning,
        device=device,
        frame_store=store,
      )
      log(f"Preflight proceed={pre.get('proceed_to_training')} reasons={pre.get('halt_reasons')}")

    if args.preflight_only or not pre.get("proceed_to_training"):
      if not pre.get("proceed_to_training"):
        (EXP / ".failed").write_text("preflight_halt\n", encoding="utf-8")
      return

    flip_prob = 0.5 if pre.get("flip_augmentation_enabled") else 0.0
    out = run_training(
      cfg,
      cache_dir=rows_cache,
      report_dir=REPORT,
      exp_dir=EXP,
      audit=audit,
      gt_group_mean=gt_group_mean,
      frame_store=store,
      backbone_init_path=BASELINE,
      parent_ckpt_note=str(PARENT_CKPT),
      warning=warning,
      device=device,
      epochs=args.epochs,
      batch_size=args.batch,
      lr=args.lr,
      seed=seed,
      flip_prob=flip_prob,
    )
    log(f"Training complete: {out}")
  except Exception:
    traceback.print_exc()
    (EXP / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
    raise


def _load_cache_rows(cache: Path):
  return [json.loads(l) for l in (cache / "index.jsonl").open(encoding="utf-8")]


class _np_load_gt:
  def __init__(self, cache: Path):
    self.cache = cache

  def __enter__(self):
    import numpy as np

    self._f = np.load(self.cache / "gt_heatmaps.npz")
    self.arr = self._f["gt_1s"]
    return self.arr

  def __exit__(self, *a):
    self._f.close()


if __name__ == "__main__":
  main()
