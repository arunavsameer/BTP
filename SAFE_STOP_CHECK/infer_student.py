"""Thin inference entry: load_model + predict. No scoring, no rendering.

Default name is m1_stop_floor. Swap with --model on run.py, or:

    from models.registry import load_model
    student = load_model("m1_stop_floor", checkpoint, "cuda")
    h = student.predict(clip_nchw01)  # [B,5,5] or [5,5]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.registry import available_models, load_model

__all__ = ["available_models", "load_model", "predict"]


def predict(model, clip_nchw01: np.ndarray) -> np.ndarray:
    return model.predict(clip_nchw01)


def load_and_predict(name: str, checkpoint: str | Path, device: str, clip_nchw01: np.ndarray) -> np.ndarray:
    return load_model(name, checkpoint, device).predict(clip_nchw01)
