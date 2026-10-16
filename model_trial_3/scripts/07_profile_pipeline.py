#!/usr/bin/env python3
"""Time decode / preprocess / backbone / adapter / backward on a tiny subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest, valid_clips  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.losses.composite import TeacherLoss  # noqa: E402
from collision_jepa3.training.engine import teacher_forward  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.engine import move_batch  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402
from collision_jepa3.utils.profiling import StageTimer  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dummy", action="store_true", default=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_cfg(args)
    device = resolve_device(args.device)
    dummy = not args.real
    timer = StageTimer()
    if dummy:
        from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone

        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, num_frames=16)
        model = CollisionTeacher3(cfg, backbone=backbone).to(device)
        loss_fn = TeacherLoss(cfg)
        opt = torch.optim.AdamW(model.adapter_parameters(), lr=1e-3)
        for i in range(args.steps):
            clip = torch.rand(1, 16, 3, 64, 64, device=device)
            batch = {
                "h_now": torch.rand(1, 5, 5, device=device),
                "h_1s": torch.rand(1, 5, 5, device=device),
                "h_1_5s": torch.rand(1, 5, 5, device=device),
                "h_2s": torch.rand(1, 5, 5, device=device),
                "amb_1s": torch.zeros(1, dtype=torch.bool, device=device),
                "amb_1_5s": torch.zeros(1, dtype=torch.bool, device=device),
                "amb_2s": torch.zeros(1, dtype=torch.bool, device=device),
            }
            with timer.track("backbone_adapter_forward"):
                out = model(clip=clip)
            with timer.track("loss"):
                packed = loss_fn(out, batch)
            with timer.track("backward"):
                packed.total.backward()
            with timer.track("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)
    else:
        records = load_manifest(cfg.path("paths.manifest"))
        splits = load_splits(cfg.path("paths.split_file"))
        ds = TeacherDataset(cfg, records, "train", splits)
        loader = build_dataloader(ds, cfg, shuffle=False)
        model = CollisionTeacher3(cfg, dummy_backbone=False).to(device)
        loss_fn = TeacherLoss(cfg)
        opt = torch.optim.AdamW(model.adapter_parameters(), lr=1e-3)
        it = iter(loader)
        for i in range(args.steps):
            with timer.track("dataloader"):
                batch = move_batch(next(it), device)
            with timer.track("forward"):
                out = teacher_forward(model, batch)
            with timer.track("loss"):
                packed = loss_fn(out, batch)
            with timer.track("backward"):
                packed.total.backward()
            with timer.track("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)
    report = {"hardware": describe_hardware(device), "stages": timer.summary(), "dummy": dummy}
    out = cfg.path("paths.reports_root") / "profile.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
