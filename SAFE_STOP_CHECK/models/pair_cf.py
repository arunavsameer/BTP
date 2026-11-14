"""Adapter for Pair-CF PARENT CollisionTeacher3 (direct_h1s, Giant hidden=1408).

Loads the real frozen V-JEPA2 Giant backbone. DummyVJEPA2Backbone is not used.
ViT-L fallback is disabled by setting fallback_model_id to the Giant id.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import TRIAL3
from models.base import StudentModel

if str(TRIAL3) not in sys.path:
    sys.path.insert(0, str(TRIAL3))

GIANT_ID = "facebook/vjepa2-vitg-fpc64-256"
HF_CACHE = Path("/storage/BTP/model_trial_3/hf_cache")
HIDDEN = 1408


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return False
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


class PairCFAdapter(StudentModel):
    name = "pair_cf"

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.checkpoint: Path | None = None
        self.meta: dict[str, Any] = {}

    def load(self, checkpoint: str | Path, device: str) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {path}")

        from collision_jepa3.config import Config
        from collision_jepa3.models.teacher import CollisionTeacher3
        from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state

        try:
            blob = load_checkpoint(path, map_location="cpu")
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"failed to load {path}: {exc}") from exc

        if not isinstance(blob, dict) or "adapter" not in blob or "decoder" not in blob:
            keys = sorted(blob.keys()) if isinstance(blob, dict) else type(blob)
            raise RuntimeError(
                f"{path} is not a Pair-CF teacher checkpoint (need adapter+decoder). keys={keys}"
            )
        if blob.get("lora_enabled"):
            raise RuntimeError(f"{path} has LoRA enabled; this adapter does not wrap LoRA")

        os.environ["HF_HOME"] = str(HF_CACHE)
        cfg = Config.load()
        saved = blob.get("config") or {}
        saved_ad = saved.get("adapter") or {}
        if not isinstance(saved_ad, dict) or not saved_ad:
            raise RuntimeError(f"{path} has no checkpoint adapter config")
        for key, value in saved_ad.items():
            current = cfg.get(f"adapter.{key}")
            if not _same(current, value):
                print(f"adapter.{key}: default {current!r} -> checkpoint {value!r}", flush=True)
            cfg.set(f"adapter.{key}", value)
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("paths.hf_cache_dir", str(HF_CACHE))
        cfg.set("paths.hf_cache_fallback", str(HF_CACHE))

        mid = (
            blob.get("backbone_model_id")
            or (saved.get("backbone") or {}).get("model_id")
            or GIANT_ID
        )
        mid = str(mid)
        if "vitl" in mid.lower() or "vit-l" in mid.lower():
            raise RuntimeError(f"refusing ViT-L backbone {mid}; need Giant hidden={HIDDEN}")
        # Same id as fallback so a failed Giant load cannot switch to ViT-L.
        cfg.set("backbone.model_id", mid)
        cfg.set("backbone.fallback_model_id", mid)

        model = CollisionTeacher3(cfg, dummy_backbone=False)
        backbone_id = str(getattr(model.backbone, "model_id", ""))
        hidden = int(model.backbone.hidden_size)
        if backbone_id.lower() in {"dummy", "dummyvjepa2"} or hidden != HIDDEN:
            raise RuntimeError(
                f"need Giant {GIANT_ID} hidden={HIDDEN}; got model_id={backbone_id} hidden={hidden}"
            )
        if "vitl" in backbone_id.lower() or "vit-l" in backbone_id.lower():
            raise RuntimeError(f"loaded ViT-L {backbone_id}; refusing to continue")
        try:
            load_teacher_state(model, blob)
        except Exception as exc:
            raise RuntimeError(f"state_dict mismatch loading {path}: {exc}") from exc

        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        epoch = blob.get("epoch")
        self.model = model
        self.device = str(device)
        self.checkpoint = path
        self.meta = {
            "epoch": epoch,
            "backbone": backbone_id,
            "hidden": hidden,
            "heads_mode": "direct_h1s",
            "dummy_backbone": False,
            "path": str(path),
            "best_selection": blob.get("best_selection"),
            "phase": blob.get("phase"),
        }
        print(
            f"loaded {path} epoch={epoch} backbone={backbone_id} hidden={hidden}",
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
        if "Hhat_1s" not in out:
            raise RuntimeError(f"teacher did not return Hhat_1s; keys={sorted(out)}")
        h = out["Hhat_1s"].float().cpu().numpy()
        if h.ndim != 3 or h.shape[1:] != (5, 5):
            raise RuntimeError(f"expected Hhat_1s [B,5,5], got {tuple(h.shape)}")
        if not np.isfinite(h).all():
            raise RuntimeError("non-finite Hhat_1s — refusing to invent a heatmap")
        return h[0] if squeezed else h

    def extra(self) -> dict[str, Any]:
        return dict(self.meta)
