"""CollisionStudentV5: identical V4 gated backbone; decoder may unfreeze in train."""

from __future__ import annotations

import torch.nn as nn

from student.v4.model import CollisionStudentV4, n_params

__all__ = ["CollisionStudentV5", "n_params"]

DECAY_START_EPOCH = 20
DECODER_LR_MULT = 0.15
UNFREEZE_WARMUP_EPOCHS = 2


def decoder_lr(epoch: int, backbone_lr: float) -> float:
    if int(epoch) <= DECAY_START_EPOCH:
        return 0.0
    ramp = min(1.0, (int(epoch) - DECAY_START_EPOCH) / float(UNFREEZE_WARMUP_EPOCHS))
    return float(backbone_lr) * DECODER_LR_MULT * ramp


class CollisionStudentV5(CollisionStudentV4):
    """Same params/FLOPs as V4. `train()` does not pin the decoder in eval after unfreeze."""

    def __init__(self, decoder: nn.Module, ffn_hidden: int = 155, init_gate_bias: float = -2.0):
        super().__init__(decoder, ffn_hidden=ffn_hidden, init_gate_bias=init_gate_bias)
        self.decoder_unfrozen = False
        self.set_decoder_trainable(False)

    def set_decoder_trainable(self, on: bool) -> None:
        self.decoder_unfrozen = bool(on)
        for p in self.decoder.parameters():
            p.requires_grad_(self.decoder_unfrozen)
        if self.decoder_unfrozen:
            self.decoder.train()
        else:
            self.decoder.eval()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if not self.decoder_unfrozen:
            self.decoder.eval()
        return self
