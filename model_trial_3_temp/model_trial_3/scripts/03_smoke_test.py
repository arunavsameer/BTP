#!/usr/bin/env python3
"""Tiny forward/backward smoke test. Default uses a dummy encoder (no Giant download)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import build_manifest, load_manifest, valid_clips  # noqa: E402
from collision_jepa3.data.preprocessing import signature_from_cfg  # noqa: E402
from collision_jepa3.data.splits import load_splits, split_groups  # noqa: E402
from collision_jepa3.losses.composite import TeacherLoss  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.reproducibility import set_seed  # noqa: E402
from collision_jepa3.training.checkpoint import extract_teacher_state, load_teacher_state  # noqa: E402
from collision_jepa3.training.engine import move_batch  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402
from collision_jepa3.utils.logging import setup_logger  # noqa: E402


def _records(cfg, n_clips: int):
    man = cfg.path("paths.manifest")
    if man.exists():
        recs = load_manifest(man)
    else:
        recs = build_manifest(cfg)
    valid = valid_clips(recs)
    if not valid:
        return recs, []
    return recs, valid[: max(1, n_clips)]


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--num-clips", type=int, default=2)
    parser.add_argument("--dummy", action="store_true", default=True, help="Use DummyVJEPA (default)")
    parser.add_argument("--real", action="store_true", help="Load the real V-JEPA checkpoint")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    dummy = not args.real
    cfg = load_cfg(args)
    set_seed(int(cfg.get("project.seed", 1337)))
    log = setup_logger("smoke", cfg.path("paths.logs_root"))
    device = resolve_device(args.device)
    log.info("hardware %s", describe_hardware(device, cfg.get("backbone.model_id")))

    recs, subset = _records(cfg, args.num_clips)
    report: dict = {"dummy": dummy, "device": device, "n_subset": len(subset)}

    if dummy:
        from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone

        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        model = CollisionTeacher3(cfg, backbone=backbone)
        clip = torch.rand(2, 16, 3, 64, 64)
        model.to(device)
        out = model(clip.to(device))
        assert out["Z_t"].shape == (2, 32, 5, 5)
        for k in ("Hhat_now", "Hhat_1s", "Hhat_1_5s", "Hhat_2s"):
            assert out[k].shape == (2, 5, 5)
            assert torch.isfinite(out[k]).all()
        batch = {
            "h_now": torch.rand(2, 5, 5, device=device),
            "h_1s": torch.rand(2, 5, 5, device=device),
            "h_1_5s": torch.rand(2, 5, 5, device=device),
            "h_2s": torch.rand(2, 5, 5, device=device),
            "amb_1s": torch.zeros(2, dtype=torch.bool, device=device),
            "amb_1_5s": torch.zeros(2, dtype=torch.bool, device=device),
            "amb_2s": torch.tensor([False, True], device=device),
        }
        loss = TeacherLoss(cfg)(out, batch)
        loss.total.backward()
        for p in model.backbone.parameters():
            if p.requires_grad:
                raise RuntimeError("dummy backbone should stay frozen in frozen mode")
        grads = [p.grad for p in model.adapter_parameters() if p.grad is not None]
        if not grads:
            raise RuntimeError("adapter received no gradients")
        if not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError("non-finite adapter gradients")
        blob = extract_teacher_state(model)
        load_teacher_state(model, blob)
        report["shapes"] = {k: list(v.shape) for k, v in out.items() if torch.is_tensor(v)}
        report["audit"] = model.parameter_audit()
        report["loss"] = loss.as_floats()
    else:
        model = CollisionTeacher3(cfg, dummy_backbone=False)
        model.to(device)
        diag = model.backbone.diagnostics(16)
        log.info("backbone diagnostics %s", diag)
        x = torch.rand(1, 16, 3, 256, 256, device=device)
        with torch.no_grad():
            tokens = model.encode_tokens(x)
        log.info("tokens %s peak_mem_gb=%.2f", tuple(tokens.shape), torch.cuda.max_memory_allocated() / 1024**3 if device.startswith("cuda") else 0)
        out = model(clip=x)
        report["diagnostics"] = diag
        report["token_shape"] = list(tokens.shape)
        report["z_shape"] = list(out["Z_t"].shape)
        if subset:
            splits = {"train": [r.scenario_group_id for r in subset], "val": [], "test": []}
            ds = TeacherDataset(cfg, subset, "train", splits, use_frame_cache=False, use_token_cache=False)
            loader = build_dataloader(ds, cfg, shuffle=False)
            batch = move_batch(next(iter(loader)), device)
            from collision_jepa3.training.engine import teacher_forward

            out = teacher_forward(model, batch)
            loss = TeacherLoss(cfg)(out, batch)
            report["real_batch_loss"] = loss.as_floats()

    out_path = cfg.path("paths.reports_root") / "smoke_test.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
