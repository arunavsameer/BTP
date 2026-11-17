"""Weight-only affine int8 pack/unpack (not fused int8 GEMM)."""

from __future__ import annotations

from typing import Any

import torch

QMIN, QMAX = -128, 127


def is_quant_blob(blob: dict) -> bool:
    return isinstance(blob, dict) and "backbone_q" in blob and "decoder_q" in blob


def _channel_axis(name: str, tensor: torch.Tensor) -> int | None:
    """Per-out-channel for conv/linear .weight; per-tensor for biases / GN / LN / gamma."""
    nd = int(tensor.ndim)
    if name.endswith(".weight") and nd in (2, 3, 4, 5):
        return 0
    return None


def quantize_tensor(w: torch.Tensor, axis: int | None) -> dict[str, Any]:
    x = w.detach().float().cpu()
    if axis is None:
        amax = x.detach().abs().amax().clamp(min=1e-12)
        scale = (amax / float(QMAX)).to(torch.float32)
        q = torch.clamp(torch.round(x / scale), QMIN, QMAX).to(torch.int8)
        zp = torch.zeros((), dtype=torch.int32)
        return {"q": q, "scale": scale, "zero_point": zp, "axis": None, "shape": list(x.shape)}
    dims = [i for i in range(x.ndim) if i != axis]
    amax = x.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12)
    scale = (amax / float(QMAX)).to(torch.float32)
    q = torch.clamp(torch.round(x / scale), QMIN, QMAX).to(torch.int8)
    zp = torch.zeros_like(scale, dtype=torch.int32)
    return {"q": q, "scale": scale.squeeze().contiguous(), "zero_point": zp.squeeze().to(torch.int32).contiguous(), "axis": int(axis), "shape": list(x.shape)}


def dequantize_entry(entry: dict[str, Any], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    q = entry["q"].float()
    scale = entry["scale"].float()
    zp = entry["zero_point"].float()
    axis = entry.get("axis")
    if axis is None:
        return ((q - zp) * scale).to(dtype)
    # broadcast scale along channel axis
    view = [1] * q.ndim
    view[int(axis)] = -1
    scale_b = scale.reshape(*view)
    zp_b = zp.reshape(*view)
    return ((q - zp_b) * scale_b).to(dtype)


def quantize_state(sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    packed = {}
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            packed[k] = {"q": v.cpu(), "scale": None, "zero_point": None, "axis": None, "dtype": str(v.dtype)}
            continue
        axis = _channel_axis(k, v)
        packed[k] = quantize_tensor(v, axis)
    return packed


def dequant_state(packed: dict[str, Any], dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    out = {}
    for k, e in packed.items():
        if e.get("scale") is None:
            out[k] = e["q"]
        else:
            out[k] = dequantize_entry(e, dtype=dtype)
    return out
