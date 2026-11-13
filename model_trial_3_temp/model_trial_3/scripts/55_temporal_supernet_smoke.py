#!/usr/bin/env python3
"""One-batch smoke test for temporal supernet pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.student_research.rgb_supernet_cache import RGBSupernetCache  # noqa: E402
from collision_jepa3.student_research.supernet_dataset import SupernetAnchorDataset, collate_supernet, make_loader  # noqa: E402
from collision_jepa3.student_research.temporal_supernet import build_supernet, pack_pattern_clip  # noqa: E402
from collision_jepa3.student_research.cnn import assert_param_budget, MAX_TOTAL_PARAMS  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "reports/student_architecture_tests"))
    parser.add_argument("--device", default="cuda")
    add_config_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    device = resolve_device(args.device)
    meta = json.loads((out / "shared_rgb_supernet_cache.json").read_text())
    rgb = RGBSupernetCache(out / "rgb_supernet_cache", meta)
    train_m = json.loads((out / "part1_train_manifest.json").read_text())
    ds = SupernetAnchorDataset(out / "teacher_cache_train_pilot", rgb, train_m["train_row_ids"][:16], epoch=0)
    loader = make_loader(ds, 4, True)
    batch = next(iter(loader))
    model = build_supernet().to(device)
    assert_param_budget(model, MAX_TOTAL_PARAMS)
    pattern = batch["train_pattern"][0]
    clip, mask, off = pack_pattern_clip(batch["flat_frames"].to(device), pattern)
    out_fwd = model(clip, mask=mask, offsets_norm=off)
    rep = {
        "ok": True,
        "pattern": pattern,
        "H_shape": list(out_fwd["Hhat_1s"].shape),
        "params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    (out / "temporal_supernet_smoke.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
