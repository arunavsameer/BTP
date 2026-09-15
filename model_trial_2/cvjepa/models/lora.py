"""Minimal LoRA on nn.Linear, no extra dependency."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear)}")
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / max(self.rank, 1)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + delta


def _encoder_blocks(model: nn.Module) -> list[nn.Module] | None:
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
    for root in candidates:
        for attr in ("encoder", "vit", "vision_model"):
            enc = getattr(root, attr, None)
            if enc is None:
                continue
            for layers_name in ("layer", "layers", "blocks"):
                layers = getattr(enc, layers_name, None)
                if layers is not None and len(list(layers)) > 0:
                    return list(layers)
    return None


def _replace_linears(module: nn.Module, rank: int, alpha: float) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            n += 1
        else:
            n += _replace_linears(child, rank, alpha)
    return n


def apply_lora_to_last_blocks(
    model: nn.Module,
    last_n: int = 6,
    rank: int = 16,
    alpha: float = 16.0,
) -> int:
    """Replace Linear modules in the last ``last_n`` encoder blocks with LoRALinear."""
    blocks = _encoder_blocks(model)
    if not blocks:
        raise RuntimeError("could not find encoder blocks on the V-JEPA 2 model")
    n = max(1, min(int(last_n), len(blocks)))
    wrapped = 0
    for block in blocks[-n:]:
        wrapped += _replace_linears(block, rank, alpha)
    return wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            params.append(m.lora_A)
            params.append(m.lora_B)
    return params
