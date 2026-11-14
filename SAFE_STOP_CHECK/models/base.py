"""Student adapter contract. Metrics and render take heatmaps, not a class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class StudentModel(ABC):
    """Thin wrap: load(checkpoint, device) then predict(clip) → Hhat_1s."""

    name: str = "base"

    @abstractmethod
    def load(self, checkpoint: str | Path, device: str) -> None:
        """Load weights. Raise on failure — never invent a heatmap."""

    @abstractmethod
    def predict(self, clip_nchw01: np.ndarray) -> np.ndarray:
        """clip: [B,16,3,256,256] or [16,3,256,256] float32 in [0,1] → Hhat_1s [B,5,5] or [5,5]."""

    def extra(self) -> dict[str, Any]:
        return {}
