#!/usr/bin/env python3
"""Forward and backward smoke test on a dummy encoder. No Giant download."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa4.losses.composite import TeacherLoss  # noqa: E402
from collision_jepa4.models.teacher import CollisionTeacher4  # noqa: E402
from collision_jepa4.models.trunk import TRUNK_PARAMS_1408  # noqa: E402
from collision_jepa4.models.vjepa_backbone import DummyVJEPA2Backbone  # noqa: E402
from collision_jepa4.reproducibility import set_seed  # noqa: E402
from collision_jepa4.training.checkpoint import extract_teacher_state, load_teacher_state  # noqa: E402
from collision_jepa4.utils.device import describe_hardware, resolve_device  # noqa: E402
from collision_jepa4.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_cfg(args)
    set_seed(int(cfg.get("project.seed", 1337)))
    log = setup_logger("smoke", cfg.path("paths.logs_root"))
    device = resolve_device(args.device)
    log.info("hardware %s", describe_hardware(device, "dummy"))

    backbone = DummyVJEPA2Backbone(
        hidden_size=1408,
        crop_size=256,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        num_layers=1,
    )
    model = CollisionTeacher4(cfg, backbone=backbone).to(device)
    clip = torch.rand(2, 16, 3, 256, 256, device=device)
    out = model(clip)
    assert out["Hhat_now"].shape == (2, 5, 5)
    assert out["Hhat_now_logits"].shape == (2, 5, 5)
    probs = out["Hhat_now"].detach()
    assert torch.isfinite(probs).all()
    assert float(probs.min()) > 0.0
    assert float(probs.max()) < 1.0
    batch = {"h_now": torch.rand(2, 5, 5, device=device)}
    loss = TeacherLoss(cfg)(out, batch)
    loss.total.backward()
    for p in model.backbone.parameters():
        if p.requires_grad:
            raise RuntimeError("backbone should stay frozen")
        if p.grad is not None:
            raise RuntimeError("backbone received a gradient")
    if model.trunk.projection.weight.grad is None:
        raise RuntimeError("projection received no gradient")
    audit = model.parameter_audit()
    if int(audit["trunk"]) != TRUNK_PARAMS_1408:
        raise RuntimeError(f"trunk params {audit['trunk']} != {TRUNK_PARAMS_1408}")
    blob = extract_teacher_state(model)
    load_teacher_state(model, blob)
    report = {
        "device": device,
        "shapes": {k: list(v.shape) for k, v in out.items()},
        "audit": audit,
        "loss": loss.as_floats(),
    }
    out_path = cfg.path("paths.reports_root") / "smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
