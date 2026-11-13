#!/usr/bin/env python3
"""Resize-and-cache RGB frames. Resumable. Optional --max-gb only if you pass it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.cache import prepare_frame_cache  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--max-gb", type=float, default=None)
    args = parser.parse_args()
    cfg = load_cfg(args)
    log = setup_logger("cache", cfg.path("paths.logs_root"))
    man = cfg.path("paths.manifest")
    if not man.exists():
        raise SystemExit(f"manifest missing: {man}. Run 01_build_manifest.py first.")
    records = load_manifest(man)
    result = prepare_frame_cache(records, cfg, max_gb=args.max_gb)
    log.info("%s", result)
    out = cfg.path("paths.reports_root") / "cache_report.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
