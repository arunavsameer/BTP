"""Adapter for overnight M1 (epoch 10 STOP-floor) in model_trial_3.

M1 is StudentOvernight (encoder + 1×1 proj + Pair-CF decoder), not CollisionStudent.
CollisionStudent has a different heatmap path (predictor → bottleneck). This adapter
loads the checkpoint that actually exists: encoder / predictor / proj / decoder.

Giant / DummyVJEPA are not constructed. Decoder weights come from the .pt file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import DEFAULT_CHECKPOINT, TRIAL3
from models.base import StudentModel

if str(TRIAL3) not in sys.path:
    sys.path.insert(0, str(TRIAL3))


class M1StopFloorAdapter(StudentModel):
    name = "m1_stop_floor"
    default_checkpoint = DEFAULT_CHECKPOINT

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.checkpoint: Path | None = None
        self.meta: dict[str, Any] = {}

    def load(self, checkpoint: str | Path, device: str) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {path}")

        from collision_jepa3.models.heatmap_decoder import HeatmapDecoder
        from collision_jepa3.student_research.overnight6.models import build_model

        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"failed to load {path}: {exc}") from exc

        if not isinstance(blob, dict) or "encoder" not in blob or "decoder" not in blob:
            keys = sorted(blob.keys()) if isinstance(blob, dict) else type(blob)
            raise RuntimeError(
                f"{path} is not an overnight M1 checkpoint (need encoder+decoder). keys={keys}"
            )

        decoder = HeatmapDecoder(z_channels=128, dilated=True, out_grid=5, pool="max")
        model = build_model(1, decoder)
        try:
            model.encoder.load_state_dict(blob["encoder"])
            model.decoder.load_state_dict(blob["decoder"])
            if blob.get("proj") is not None:
                model.proj.load_state_dict(blob["proj"])
            if blob.get("predictor") is not None:
                model.predictor.load_state_dict(blob["predictor"])
        except Exception as exc:
            raise RuntimeError(f"state_dict mismatch loading {path}: {exc}") from exc

        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.device = str(device)
        self.checkpoint = path
        self.meta = {
            "epoch": blob.get("epoch"),
            "variant": blob.get("variant"),
            "phase": blob.get("phase"),
            "path": str(path),
        }
        print(
            f"loaded M1 {path} epoch={self.meta['epoch']} variant={self.meta['variant']} device={device}",
            flush=True,
        )

    @torch.no_grad()
    def predict(self, clip_nchw01: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call load() before predict()")
        x = np.asarray(clip_nchw01, dtype=np.float32)
        squeezed = False
        if x.ndim == 4:
            x = x[None]
            squeezed = True
        if x.ndim != 5 or x.shape[1:] != (16, 3, 256, 256):
            raise ValueError(f"expected [B,16,3,256,256] in [0,1], got {tuple(x.shape)}")
        tensor = torch.from_numpy(x).to(self.device)
        amp = str(self.device).startswith("cuda")
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(tensor)
        else:
            out = self.model(tensor)
        h = out["Hhat_1s"].float().cpu().numpy()
        if not np.isfinite(h).all():
            raise RuntimeError("non-finite Hhat_1s — refusing to invent a heatmap")
        return h[0] if squeezed else h

    def extra(self) -> dict[str, Any]:
        return dict(self.meta)
