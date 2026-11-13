#!/usr/bin/env python3
"""Encode frozen V-JEPA tokens for every valid clip. Resumable. No frame cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.cache import prepare_token_cache, token_cache_id  # noqa: E402
from collision_jepa3.data.extra_same_dir import extra_same_dir_records  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.reproducibility import configure_runtime, set_seed  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument(
        "--only-extra-same-dir",
        action="store_true",
        help="Encode only data.extra_same_dir sample_min..sample_max (207-229).",
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg.get("project.seed", 1337)))
    configure_runtime(int(cfg.get("training.num_workers", 2)))
    log = setup_logger("token_cache", cfg.path("paths.logs_root"))
    man = cfg.path("paths.manifest")
    if not man.exists():
        raise SystemExit(f"manifest missing: {man}. Run 01_build_manifest.py first.")
    device = resolve_device(args.device)
    log.info("hardware %s", describe_hardware(device, cfg.get("backbone.model_id")))
    log.info("token cache id %s", token_cache_id(cfg, dummy=bool(args.dummy)))
    records = load_manifest(man)
    if args.only_extra_same_dir:
        records = extra_same_dir_records(records, cfg)
        log.info("only extra_same_dir: %d valid clips", len(records))
        if not records:
            raise SystemExit("no extra_same_dir clips in the manifest; run 01_build_manifest.py")
    model = CollisionTeacher3(cfg, dummy_backbone=bool(args.dummy))
    model.to(device)
    result = prepare_token_cache(records, cfg, model, device)
    log.info("%s", result)
    out = cfg.path("paths.reports_root") / "token_cache_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
