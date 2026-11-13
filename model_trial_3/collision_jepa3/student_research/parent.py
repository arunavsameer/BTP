"""Fixed Pair-CF parent: inspect shapes, load checkpoint, full forward for caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from collision_jepa3.config import heads_mode
from collision_jepa3.evaluation.metrics import classify_heatmap
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state

from . import PARENT_CKPT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dummy_backbone() -> DummyVJEPA2Backbone:
    return DummyVJEPA2Backbone(
        hidden_size=1408,
        crop_size=256,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        num_layers=1,
    )


def resolve_parent_ckpt(cfg, ckpt: str | Path | None = None) -> Path:
    raw = Path(ckpt or PARENT_CKPT)
    candidates = [raw]
    if not raw.is_absolute():
        root = Path(__file__).resolve().parents[2]
        candidates += [root / raw, cfg.path("paths.checkpoint_root") / raw.name, cfg.path("paths.checkpoint_root").parent / raw]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"parent checkpoint not found: {raw} (tried {candidates})")


def load_pair_cf_parent(cfg, ckpt: str | Path | None = None, device: str = "cpu") -> tuple[CollisionTeacher3, dict[str, Any]]:
    path = resolve_parent_ckpt(cfg, ckpt)
    cfg.set("heads.mode", "direct_h1s")
    model = CollisionTeacher3(cfg, backbone=_dummy_backbone())
    blob = load_checkpoint(path, map_location="cpu")
    load_teacher_state(model, blob)
    model.to(device).eval()
    return model, blob


def count_modules(model: CollisionTeacher3) -> dict[str, int]:
    def n(mod):
        return int(sum(p.numel() for p in mod.parameters()))

    out = {
        "decoder": n(model.decoder),
        "adapter": n(model.adapter),
        "close_head": n(model.close_head),
        "evidence_head": n(model.evidence_head),
        "task_whiten": n(model.task_whiten),
    }
    if model.variance_head is not None:
        out["variance_head"] = n(model.variance_head)
    out["decoder_only_trainable_copy"] = out["decoder"]
    return out


@torch.no_grad()
def parent_forward(tokens: torch.Tensor, model: CollisionTeacher3) -> dict[str, torch.Tensor]:
    z_t, fine = model.encode_pair(tokens=tokens)
    feat = model._heatmap_feat(z_t, fine)
    h, logits = model.decoder.forward_with_logits(feat)
    return {
        "Z_t": z_t,
        "F_pre_decoder": feat,
        "F_adapter_mixed": fine,
        "Hhat_1s": h,
        "Hhat_1s_logits": logits,
    }


def inspect_parent_shapes(cfg, ckpt: Path) -> dict[str, Any]:
    model, blob = load_pair_cf_parent(cfg, ckpt, device="cpu")
    audit = model.parameter_audit()
    dec_in = int(model.heatmap_grid) if model.decode_from == "fine" else model.z_channels
    if model.decode_from == "fine":
        latent_decoder_in = f"[B, {int(cfg.get('adapter.projection_channels', 128))}, {dec_in}, {dec_in}]"
    else:
        latent_decoder_in = f"[B, {model.z_channels}, {model.feature_grid}, {model.feature_grid}]"
    n_tok = model.adapter.expected_tokens()
    hidden = int(model.backbone.hidden_size)
    z_dummy = torch.zeros(1, n_tok, hidden)
    out0 = parent_forward(z_dummy, model)
    z_t = out0["Z_t"]
    fine = out0["F_adapter_mixed"]
    feat = out0["F_pre_decoder"]
    warning = cfg.get("warning") or {}
    _, sev, _ = classify_heatmap(out0["Hhat_1s"][0].cpu().numpy(), warning)
    peak = float(out0["Hhat_1s"][0].max().item())
    return {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "checkpoint_epoch": blob.get("epoch"),
        "best_selection": blob.get("best_selection"),
        "heads_mode": heads_mode(cfg),
        "decode_from": model.decode_from,
        "token_shape": [2048, 1408],
        "adapter_fine_shape": list(fine.shape[1:]),
        "Z_t_shape": list(z_t.shape[1:]),
        "F_pre_decoder_shape": list(feat.shape[1:]),
        "decoder_input_notes": latent_decoder_in,
        "H_1s_shape": [5, 5],
        "z_channels": model.z_channels,
        "feature_grid": model.feature_grid,
        "heatmap_grid": model.heatmap_grid,
        "projection_channels": int(cfg.get("adapter.projection_channels", 128)),
        "parameter_audit": audit,
        "component_param_counts": count_modules(model),
        "warning_thresholds": {
            "stop": warning.get("stop_threshold"),
            "caution": warning.get("caution_threshold"),
            "threat": warning.get("caution_threshold"),
        },
        "dummy_forward_severity": int(sev),
        "dummy_forward_peak": peak,
    }


def write_part1_reports(out_dir: Path, cfg, ckpt: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = inspect_parent_shapes(cfg, ckpt)
    sizes = {
        "component_param_counts": shapes.pop("component_param_counts"),
        "parameter_audit": shapes.pop("parameter_audit"),
        "decoder_copy_practical": shapes["F_pre_decoder_shape"][0] == 128,
        "notes": (
            "Decoder expects task-whitened fine map (decode_from=fine). "
            "Student must project RGB features into this space for Strategy C."
        ),
    }
    (out_dir / "part1_parent_latent_shapes.json").write_text(json.dumps(shapes, indent=2), encoding="utf-8")
    (out_dir / "part1_parent_component_sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")
