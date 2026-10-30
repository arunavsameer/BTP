#!/usr/bin/env python3
"""Summarize clips on disk: complete vs incomplete, outcomes, backgrounds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa4.data.manifest import build_manifest, manifest_stats  # noqa: E402
from collision_jepa4.data.sampling import planned_anchors_per_clip  # noqa: E402
from collision_jepa4.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = load_cfg(args)
    log = setup_logger("inspect", cfg.path("paths.logs_root"))
    root = cfg.dataset_root()
    log.info("dataset_root=%s exists=%s", root, root.exists())
    records = build_manifest(cfg, root)
    stats = manifest_stats(records)
    n_per = planned_anchors_per_clip(cfg)
    stats["anchors_per_clip"] = n_per
    stats["anchor_mode"] = cfg.get("sampling.anchor_mode", "stride")
    stats["valid_train_anchors"] = n_per * stats["n_valid"]
    out = cfg.path("paths.reports_root") / "dataset_inspect.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}")
    for rec in records:
        if not rec.valid:
            print(f"INVALID {rec.clip_id}: {rec.invalid_reason}")


if __name__ == "__main__":
    main()
