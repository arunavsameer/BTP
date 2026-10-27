"""Checkpoint save/load with LoRA reconstruction order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def extract_teacher_state(model) -> dict[str, Any]:
    return {
        "adapter": model.adapter.state_dict(),
        "decoder": model.decoder.state_dict(),
        "predictors": model.predictors.state_dict(),
        "delta_heads": model.delta_heads.state_dict(),
        "close_head": model.close_head.state_dict() if hasattr(model, "close_head") else {},
        "evidence_head": model.evidence_head.state_dict() if hasattr(model, "evidence_head") else {},
        "lora": {k: v.detach().cpu() for k, v in _lora_state(model).items()},
        "lora_enabled": bool(getattr(model.backbone, "lora_enabled", False)),
    }


def _lora_state(model) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            out[name] = p
    return out


def load_teacher_state(model, blob: dict[str, Any], *, strict: bool = True) -> None:
    model.adapter.load_state_dict(blob["adapter"], strict=strict)
    model.decoder.load_state_dict(blob["decoder"], strict=strict)
    if len(model.predictors) or blob.get("predictors"):
        model.predictors.load_state_dict(blob.get("predictors") or {}, strict=strict)
    if len(model.delta_heads) or blob.get("delta_heads"):
        model.delta_heads.load_state_dict(blob.get("delta_heads") or {}, strict=strict)
    if hasattr(model, "close_head") and blob.get("close_head"):
        model.close_head.load_state_dict(blob["close_head"], strict=strict)
    if hasattr(model, "evidence_head") and blob.get("evidence_head"):
        model.evidence_head.load_state_dict(blob["evidence_head"], strict=strict)
    lora_blob = blob.get("lora") or {}
    if lora_blob:
        if not getattr(model.backbone, "lora_enabled", False):
            raise RuntimeError("checkpoint has LoRA weights but the model has not wrapped LoRA yet")
        current = dict(model.named_parameters())
        missing = []
        for name, tensor in lora_blob.items():
            if name not in current:
                missing.append(name)
                continue
            current[name].data.copy_(tensor.to(device=current[name].device, dtype=current[name].dtype))
        if missing and strict:
            raise RuntimeError(f"LoRA keys missing in model: {missing[:8]}")
