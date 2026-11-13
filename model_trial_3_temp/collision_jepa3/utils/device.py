"""Hardware discovery and AMP dtype selection."""

from __future__ import annotations

from typing import Any


def resolve_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def select_amp_dtype(cfg, device: str):
    import torch

    requested = str(cfg.get("training.amp_dtype", cfg.get("backbone.torch_dtype", "bfloat16")))
    use_amp = bool(cfg.get("training.amp", True)) and device.startswith("cuda")
    if not use_amp:
        return None
    if requested in {"bfloat16", "bf16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if requested in {"float16", "fp16"}:
        return torch.float16
    return None


def describe_hardware(device: str, model_id: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {"device": device, "model_id": model_id}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_bytes"] = int(props.total_memory)
            info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            info["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def estimate_clip_bytes(batch: int, frames: int, size: int, dtype_bytes: int = 2) -> int:
    return int(batch) * int(frames) * 3 * int(size) * int(size) * int(dtype_bytes)
