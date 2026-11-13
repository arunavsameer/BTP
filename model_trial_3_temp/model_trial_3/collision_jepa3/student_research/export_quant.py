"""Measure exported model bytes (FP32 state_dict, torchscript, dynamic quant)."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import torch

from collision_jepa3.student_research.cnn import ArchConfig, StudentRGBNet


def export_metrics(model: StudentRGBNet, n_frames: int, input_size: int = 160) -> dict[str, Any]:
    model.eval()
    x = torch.randn(1, n_frames, 3, input_size, input_size)
    audit = model.parameter_audit()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    fp32_bytes = buf.tell()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            model(x)
    latency_ms = (time.perf_counter() - t0) / 5 * 1000
    ts_bytes = None
    int8_bytes = None
    try:
        traced = torch.jit.trace(model, x)
        tbuf = io.BytesIO()
        torch.jit.save(traced, tbuf)
        ts_bytes = tbuf.tell()
        qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
        qbuf = io.BytesIO()
        torch.save(qmodel.state_dict(), qbuf)
        int8_bytes = qbuf.tell()
    except Exception as exc:
        return {
            **audit,
            "fp32_state_dict_bytes": fp32_bytes,
            "torchscript_bytes": ts_bytes,
            "dynamic_quant_state_dict_bytes": int8_bytes,
            "batch1_latency_ms_cpu": latency_ms,
            "export_error": str(exc),
        }
    return {
        **audit,
        "fp32_state_dict_bytes": fp32_bytes,
        "torchscript_bytes": ts_bytes,
        "dynamic_quant_state_dict_bytes": int8_bytes,
        "under_1mb_fp32": fp32_bytes < 1_000_000,
        "under_1mb_int8": int8_bytes is not None and int8_bytes < 1_000_000,
        "batch1_latency_ms_cpu": latency_ms,
        "input": [1, n_frames, 3, input_size, input_size],
    }


def sweep_architectures(presets, input_size: int = 160) -> list[dict[str, Any]]:
    out = []
    for p in presets:
        for nf in (3, 5):
            m = StudentRGBNet(p, n_frames=nf, input_size=input_size)
            out.append(export_metrics(m, nf, input_size))
    return out
