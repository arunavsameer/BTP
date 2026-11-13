#!/usr/bin/env python3
"""Part 2: persistent RGB cache for temporal supernet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.student_research.rgb_supernet_cache import (  # noqa: E402
    RGBSupernetCache,
    build_rgb_cache,
    save_cache_meta,
    verify_equivalence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--input-size", type=int, default=160)
    add_config_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    cfg = load_cfg(argparse.Namespace(config=args.config, dataset_root=args.dataset_root, seed=args.seed))
    train_m = json.loads((out / "part1_train_manifest.json").read_text())
    dev_m = json.loads((out / "part1_dev_manifest.json").read_text())
    cache_dir = out / "rgb_supernet_cache"
    teacher = out / "teacher_cache_train_pilot"
    rows = [json.loads(l) for l in (teacher / "index.jsonl").open()]
    row_ids = sorted(set(train_m["train_row_ids"]) | set(dev_m["dev_row_ids"]))
    row_anchors = [(rows[i]["clip_id"], int(rows[i]["frame"])) for i in row_ids]
    clip_ids = {rows[i]["clip_id"] for i in row_ids}
    records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    manifest = build_rgb_cache(
        cfg,
        out_dir=cache_dir,
        clip_ids=clip_ids,
        row_anchors=row_anchors,
        records=records,
        input_size=args.input_size,
    )
    # verify sample rows
    rgb = RGBSupernetCache(cache_dir, manifest)
    checks = []
    for rid in row_ids[:10]:
        row = rows[rid]
        arr = rgb.get_anchor_stack(row["clip_id"], int(row["frame"]))
        checks.append({"row_id": rid, "ok": bool(arr.shape[0])})
    missing = []
    for rid in row_ids:
        row = rows[rid]
        try:
            rgb.get_anchor_stack(row["clip_id"], int(row["frame"]))
        except Exception as exc:
            missing.append({"row_id": rid, "error": str(exc)})
    manifest["retrieval_checks"] = checks
    manifest["missing_entries"] = missing
    manifest["n_manifest_rows"] = len(row_ids)
    save_cache_meta(out, manifest, cache_dir)
    if missing:
        raise SystemExit(f"rgb cache missing {len(missing)} entries")
    print(json.dumps({"n_clips": manifest["n_clips"], "bytes": manifest["bytes_estimated"], "missing": len(missing)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
