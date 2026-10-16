"""AdamW + cosine scheduler helpers."""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


def build_optimizer(params, lr: float, weight_decay: float, name: str = "AdamW"):
    if name.lower() != "adamw":
        raise ValueError(f"unsupported optimizer {name}")
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def build_param_groups(adapter_params, lora_params, adapter_lr: float, lora_lr: float, weight_decay: float):
    groups = [{"params": adapter_params, "lr": adapter_lr, "weight_decay": weight_decay}]
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    return groups


def build_scheduler(optimizer, epochs: int, min_lr: float):
    if epochs <= 0:
        return None
    warmup = max(1, min(1, epochs))
    # Tiny linear warmup of 1 epoch then cosine.
    if epochs == 1:
        return CosineAnnealingLR(optimizer, T_max=1, eta_min=min_lr)
    warm = LinearLR(optimizer, start_factor=0.2, total_iters=1)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - 1, 1), eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[1])
