"""Seeds, deterministic flags, and environment fingerprints."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def configure_runtime(num_threads: int = 2) -> None:
    n = max(1, int(num_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass


def software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in ("torch", "numpy", "cv2", "yaml", "transformers", "PIL"):
        try:
            mod = __import__(name if name != "cv2" else "cv2")
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    try:
        import torch

        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["torch_cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            versions["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            versions["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
            versions["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception:
        pass
    return versions
