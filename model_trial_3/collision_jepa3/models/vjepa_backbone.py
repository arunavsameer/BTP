"""V-JEPA 2 encoder wrapper. Layout conversion lives only here."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .lora import apply_lora_to_last_blocks, lora_parameters

logger = logging.getLogger("collision_jepa3.backbone")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_hf_cache(cfg) -> str | None:
    candidates = []
    configured = cfg.get("paths.hf_cache_dir")
    if configured:
        candidates.append(Path(str(configured)).expanduser())
    fallback = cfg.get("paths.hf_cache_fallback")
    if fallback:
        from ..config import PACKAGE_ROOT

        p = Path(str(fallback))
        candidates.append(p if p.is_absolute() else PACKAGE_ROOT / p)
    for path in candidates:
        if path.exists():
            return str(path)
    if candidates:
        candidates[0].mkdir(parents=True, exist_ok=True)
        return str(candidates[0])
    return None


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[str(name)]


class DummyVJEPA2Backbone(nn.Module):
    """Deterministic stand-in for tests. Matches the dense token layout of V-JEPA 2."""

    def __init__(
        self,
        hidden_size: int = 64,
        crop_size: int = 64,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 16,
        num_layers: int = 8,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.crop_size = int(crop_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.num_frames = int(num_frames)
        self.model_id = "dummy"
        self.revision = "dummy"
        self.mean_tuple = tuple(float(x) for x in mean)
        self.std_tuple = tuple(float(x) for x in std)
        self.register_buffer("_mean", torch.tensor(self.mean_tuple).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(self.std_tuple).view(1, 1, 3, 1, 1), persistent=False)
        self.spatial = max(1, self.crop_size // self.patch_size)
        self.temporal_groups = max(1, self.num_frames // self.tubelet_size)
        dim = self.hidden_size
        self.proj = nn.Conv3d(3, dim, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                for _ in range(num_layers)
            ]
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self._grad_enabled = False
        self.lora_enabled = False

    def enable_lora(self, last_n: int, rank: int, alpha: float, dropout: float = 0.0) -> int:
        n = apply_lora_to_last_blocks(self, last_n=last_n, rank=rank, alpha=alpha, dropout=dropout)
        self._grad_enabled = True
        self.lora_enabled = True
        return n

    def set_backbone_grad(self, enabled: bool) -> None:
        self._grad_enabled = bool(enabled)

    def normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean.to(device=clip.device, dtype=clip.dtype)) / self._std.to(
            device=clip.device, dtype=clip.dtype
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip [B, T, 3, H, W] in [0,1] → tokens [B, T'*S*S, C]."""
        x = self.normalize(clip)
        x = x.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
        ctx = torch.enable_grad() if self._grad_enabled else torch.no_grad()
        with ctx:
            tokens = self.proj(x).flatten(2).transpose(1, 2)
            for block in self.encoder.layer:
                tokens = block(tokens)
        return tokens.float()


class VJEPA2Backbone(nn.Module):
    def __init__(
        self,
        model_id: str,
        cache_dir: str | Path | None = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        fallback_model_id: str | None = None,
        skip_predictor: bool = True,
        use_get_vision_features: bool = True,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
    ):
        super().__init__()
        from transformers import AutoModel

        self.requested_model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.skip_predictor = bool(skip_predictor)
        self.use_get_vision_features = bool(use_get_vision_features)
        dtype = _dtype_from_name(torch_dtype)
        kwargs: dict[str, Any] = {"dtype": dtype}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
            kwargs["local_files_only"] = Path(cache_dir).exists()
        if attn_implementation and attn_implementation != "auto":
            kwargs["attn_implementation"] = attn_implementation

        self.model_id, self.model = self._load(model_id, fallback_model_id, kwargs)
        cfg = self.model.config
        self.hidden_size = int(getattr(cfg, "hidden_size"))
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        crop = getattr(cfg, "crop_size", None)
        if isinstance(crop, dict):
            crop = crop.get("height") or crop.get("width")
        self.crop_size = int(crop or getattr(cfg, "image_size", 256))
        self.tubelet_size = int(getattr(cfg, "tubelet_size", 2))
        self.frames_per_clip = int(getattr(cfg, "frames_per_clip", 64))
        self.num_hidden_layers = int(getattr(cfg, "num_hidden_layers", 0))
        self.revision = str(getattr(cfg, "_name_or_path", self.model_id))
        mean_t = mean or IMAGENET_MEAN
        std_t = std or IMAGENET_STD
        try:
            from transformers.models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor

            proc = VJEPA2VideoProcessor()
            mean_t = tuple(float(x) for x in proc.image_mean)
            std_t = tuple(float(x) for x in proc.image_std)
        except Exception as exc:
            logger.warning("could not read VJEPA2VideoProcessor stats (%s); using %s / %s", exc, mean_t, std_t)
        self.mean_tuple = tuple(float(x) for x in mean_t)
        self.std_tuple = tuple(float(x) for x in std_t)
        self.register_buffer("_mean", torch.tensor(self.mean_tuple).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(self.std_tuple).view(1, 1, 3, 1, 1), persistent=False)
        self.spatial = max(1, self.crop_size // self.patch_size)
        self.lora_enabled = False
        self._grad_enabled = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def _load(self, model_id: str, fallback: str | None, kwargs: dict[str, Any]):
        from transformers import AutoModel

        try:
            model = AutoModel.from_pretrained(model_id, **kwargs)
            logger.info("loaded backbone %s", model_id)
            return model_id, model
        except Exception as exc:
            logger.error("failed to load %s: %s", model_id, exc)
            if not fallback or fallback == model_id:
                raise
            logger.warning("falling back to %s", fallback)
            model = AutoModel.from_pretrained(fallback, **kwargs)
            return fallback, model

    def expected_tokens(self, num_frames: int) -> tuple[int, int, int]:
        if num_frames % self.tubelet_size != 0:
            raise RuntimeError(
                f"num_frames={num_frames} is not divisible by tubelet_size={self.tubelet_size}"
            )
        t_prime = num_frames // self.tubelet_size
        s = self.spatial
        return t_prime, s, t_prime * s * s

    def diagnostics(self, num_frames: int = 16) -> dict[str, Any]:
        t_prime, spatial, n = self.expected_tokens(num_frames)
        return {
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "crop_size": self.crop_size,
            "frames_per_clip_pretrained": self.frames_per_clip,
            "num_hidden_layers": self.num_hidden_layers,
            "spatial": spatial,
            "temporal_groups": t_prime,
            "expected_tokens": n,
            "mean": self.mean_tuple,
            "std": self.std_tuple,
            "lora_enabled": self.lora_enabled,
        }

    def enable_lora(self, last_n: int, rank: int, alpha: float, dropout: float = 0.0) -> int:
        n = apply_lora_to_last_blocks(self.model, last_n=last_n, rank=rank, alpha=alpha, dropout=dropout)
        self._grad_enabled = True
        self.lora_enabled = True
        try:
            self.model.gradient_checkpointing_enable()
        except Exception as exc:
            logger.warning("gradient checkpointing not enabled: %s", exc)
        return n

    def set_backbone_grad(self, enabled: bool) -> None:
        self._grad_enabled = bool(enabled)

    def lora_parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.model)

    def normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean.to(device=clip.device, dtype=clip.dtype)) / self._std.to(
            device=clip.device, dtype=clip.dtype
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip [B, T, 3, H, W] in [0,1] → dense tokens [B, T'*S*S, C] float32."""
        if clip.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(clip.shape)}")
        b, t, c, h, w = clip.shape
        if c != 3:
            raise ValueError(f"expected 3 RGB channels, got {c}")
        t_prime, spatial, n_expected = self.expected_tokens(t)
        x = self.normalize(clip)
        ctx = torch.enable_grad() if self._grad_enabled else torch.no_grad()
        with ctx:
            tokens = self._encode(x)
        tokens = tokens.float()
        n = int(tokens.shape[1])
        if n != n_expected:
            raise RuntimeError(
                f"encoder returned {n} tokens, expected T'*S*S={t_prime}*{spatial}*{spatial}={n_expected} "
                f"for input {tuple(clip.shape)} model={self.model_id} hidden={self.hidden_size} "
                f"patch={self.patch_size} tubelet={self.tubelet_size}. Refusing to drop tokens."
            )
        return tokens

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_get_vision_features and hasattr(self.model, "get_vision_features"):
            out = self.model.get_vision_features(x)
            if torch.is_tensor(out):
                return out
        try:
            out = self.model(pixel_values_videos=x, skip_predictor=self.skip_predictor)
        except TypeError:
            out = self.model(pixel_values_videos=x)
        if torch.is_tensor(out):
            return out
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return tokens


def build_backbone(cfg, dummy: bool = False) -> VJEPA2Backbone | DummyVJEPA2Backbone:
    if dummy:
        return DummyVJEPA2Backbone(
            hidden_size=64,
            crop_size=int(cfg.get("preprocessing.input_size", 256)),
            patch_size=16,
            tubelet_size=2,
            num_frames=int(cfg.get("sampling.input_frames", 16)),
            mean=tuple(cfg.get("preprocessing.mean", IMAGENET_MEAN)),
            std=tuple(cfg.get("preprocessing.std", IMAGENET_STD)),
        )
    cache = _resolve_hf_cache(cfg)
    mean = tuple(float(x) for x in cfg.get("preprocessing.mean", IMAGENET_MEAN))
    std = tuple(float(x) for x in cfg.get("preprocessing.std", IMAGENET_STD))
    return VJEPA2Backbone(
        model_id=str(cfg.get("backbone.model_id")),
        cache_dir=cache,
        torch_dtype=str(cfg.get("backbone.torch_dtype", "bfloat16")),
        attn_implementation=str(cfg.get("backbone.attention_implementation", "sdpa")),
        fallback_model_id=str(cfg.get("backbone.fallback_model_id")),
        skip_predictor=bool(cfg.get("backbone.skip_predictor", True)),
        use_get_vision_features=bool(cfg.get("backbone.use_get_vision_features", True)),
        mean=mean,
        std=std,
    )
