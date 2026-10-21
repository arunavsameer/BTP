#!/usr/bin/env python3
"""Build clip manifest + group splits. Idempotent; does not modify the dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.manifest import build_manifest, manifest_stats, save_manifest, valid_clips  # noqa: E402
from collision_jepa3.data.sampling import planned_anchors_per_clip  # noqa: E402
from collision_jepa3.data.splits import save_splits, split_groups  # noqa: E402
from collision_jepa3.reproducibility import set_seed  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--reset-split", action="store_true", help="Rebuild splits even if the file exists")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit to the first N sample directories")
    args = parser.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg.get("project.seed", 1337)))
    log = setup_logger("manifest", cfg.path("paths.logs_root"))
    records = build_manifest(cfg, max_samples=args.max_samples)
    man_path = cfg.path("paths.manifest")
    fp = save_manifest(records, man_path)
    stats = manifest_stats(records)
    log.info("manifest %s fingerprint=%s stats=%s", man_path, fp, stats)

    split_path = cfg.path("paths.split_file")
    if split_path.exists() and not args.reset_split:
        log.info("keeping existing splits at %s", split_path)
        splits = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        splits = split_groups(
            records,
            seed=int(cfg.get("project.seed", 1337)),
            train_fraction=float(cfg.get("splits.train_fraction", 0.7)),
            val_fraction=float(cfg.get("splits.val_fraction", 0.15)),
            test_fraction=float(cfg.get("splits.test_fraction", 0.15)),
            stratify_by=str(cfg.get("splits.stratify_by", "scenario_type")),
        )
        extra = {
            "seed": int(cfg.get("project.seed", 1337)),
            "manifest_fingerprint": fp,
            "policy": "group_by_scenario_sample",
        }
        sfp = save_splits(splits, split_path, extra)
        log.info("wrote splits %s fingerprint=%s counts=%s", split_path, sfp, {k: len(v) for k, v in splits.items()})
        splits = {"splits": splits}

    n_valid = len(valid_clips(records))
    report = {
        "manifest": str(man_path),
        "manifest_fingerprint": fp,
        "stats": stats,
        "anchors_per_clip": planned_anchors_per_clip(cfg),
        "anchor_mode": cfg.get("sampling.anchor_mode", "stride"),
        "valid_clips": n_valid,
        "split_file": str(split_path),
        "splits": splits.get("splits", splits),
    }
    out = cfg.path("paths.reports_root") / "manifest_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stats", "anchors_per_clip", "valid_clips")}, indent=2))
    print(f"wrote {man_path} and {split_path}")


if __name__ == "__main__":
    main()
