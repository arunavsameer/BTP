"""Adapter for model_trial_4 CollisionStudentV2 / V3 checkpoints.

These are small RGB CNNs (~380k–393k trainable). Giant / DummyVJEPA are not used.
Architecture is chosen from the checkpoint (time_embed / vel_fuse in-channels),
so trial4_epoch_22 and trial4_epoch_28 share this class and differ only by path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import TRIAL4
from models.base import StudentModel

HIDDEN = 128  # StudentV2/V3 motion / context channels


def _infer_ffn_hidden(backbone: dict) -> int:
    w = backbone.get("mhsa_1.ffn.0.weight")
    if w is None:
        return 155
    return int(w.shape[0])


def _detect_arch(backbone: dict) -> str:
    keys = list(backbone)
    if any(str(k).startswith("gated_fusion") for k in keys):
        return "v6"
    if any(str(k).startswith("gated_fuse") for k in keys):
        return "v4"
    if "time_embed" in backbone or any("time_embed" in str(k) for k in keys):
        return "v3"
    vel = backbone.get("vel_fuse.conv.weight")
    if vel is not None and int(vel.shape[1]) == 291:
        return "v3"
    return "v2"


class Trial4Adapter(StudentModel):
    name = "trial4"

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.checkpoint: Path | None = None
        self.meta: dict[str, Any] = {}

    def load(self, checkpoint: str | Path, device: str) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {path}")

        if str(TRIAL4) not in sys.path:
            sys.path.insert(0, str(TRIAL4))

        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"failed to load {path}: {exc}") from exc

        if not isinstance(blob, dict) or "backbone" not in blob or "decoder" not in blob:
            keys = sorted(blob.keys()) if isinstance(blob, dict) else type(blob)
            raise RuntimeError(
                f"{path} is not a trial_4 student checkpoint (need backbone+decoder). keys={keys}"
            )

        from student.decoder import HeatmapDecoder
        from student.v2.model import CollisionStudentV2
        from student.v3.model import CollisionStudentV3
        from student.v4.model import CollisionStudentV4
        from student.v5.model import CollisionStudentV5
        from student.v6.model import CollisionStudentV6

        backbone_sd = blob["backbone"]
        decoder_sd = blob["decoder"]
        arch = _detect_arch(backbone_sd)
        ffn_hidden = _infer_ffn_hidden(backbone_sd)
        z_ch = int(decoder_sd["net.0.weight"].shape[1])
        dec_hidden = int(decoder_sd["net.0.weight"].shape[0])
        dilated = "net.4.weight" in decoder_sd
        if z_ch != HIDDEN:
            raise RuntimeError(f"decoder z_channels={z_ch}, expected hidden={HIDDEN}")

        decoder = HeatmapDecoder(
            z_channels=z_ch, hidden=dec_hidden, dilated=dilated, out_grid=5, pool="max"
        )
        cls = {
            "v2": CollisionStudentV2,
            "v3": CollisionStudentV3,
            "v4": CollisionStudentV4,
            "v5": CollisionStudentV5,
            "v6": CollisionStudentV6,
        }[arch]
        if arch in ("v4", "v5", "v6"):
            model = cls(decoder, ffn_hidden=ffn_hidden)
        else:
            model = cls(decoder, ffn_hidden=ffn_hidden)
        try:
            model.backbone.load_state_dict(backbone_sd, strict=True)
            model.decoder.load_state_dict(decoder_sd, strict=True)
        except Exception as exc:
            raise RuntimeError(f"state_dict mismatch loading {path}: {exc}") from exc

        hidden = int(getattr(model.backbone, "present")[0].in_channels)
        if hidden != HIDDEN:
            raise RuntimeError(f"need hidden={HIDDEN}; got {hidden} arch={arch}")

        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        epoch = blob.get("epoch")
        audit = blob.get("audit") if isinstance(blob.get("audit"), dict) else {}
        self.model = model
        self.device = str(device)
        self.checkpoint = path
        self.meta = {
            "epoch": epoch,
            "arch": arch,
            "architecture": cls.__name__,
            "backbone": type(model.backbone).__name__,
            "hidden": hidden,
            "ffn_hidden": ffn_hidden,
            "dummy_backbone": False,
            "path": str(path),
            "global_step": blob.get("global_step"),
            "lam_feat": blob.get("lam_feat"),
            "audit": audit,
        }
        print(
            f"loaded {path} epoch={epoch} arch={cls.__name__} "
            f"backbone={self.meta['backbone']} hidden={hidden} ffn={ffn_hidden}",
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
            raise RuntimeError(f"student did not return Hhat_1s; keys={sorted(out)}")
        h = out["Hhat_1s"].float().cpu().numpy()
        if h.ndim != 3 or h.shape[1:] != (5, 5):
            raise RuntimeError(f"expected Hhat_1s [B,5,5], got {tuple(h.shape)}")
        if not np.isfinite(h).all():
            raise RuntimeError("non-finite Hhat_1s — refusing to invent a heatmap")
        return h[0] if squeezed else h

    def extra(self) -> dict[str, Any]:
        return dict(self.meta)
