"""Shared Pair-CF decoder LR multipliers. Do not accelerate if epochs < 26."""

from __future__ import annotations

LAMBDA_DECODER_REG = 0.01

# Inclusive epoch ranges. Applied as-is; never stretched if the run is shorter.
DECODER_LR_TABLE: list[tuple[int, int, float]] = [
    (1, 8, 0.0),
    (9, 10, 0.001),
    (11, 12, 0.002),
    (13, 14, 0.005),
    (15, 16, 0.01),
    (17, 20, 0.02),
    (21, 25, 0.03),
    (26, 10_000, 0.05),
]


def decoder_lr_multiplier(epoch: int) -> float:
    ep = int(epoch)
    for lo, hi, m in DECODER_LR_TABLE:
        if lo <= ep <= hi:
            return float(m)
    return 0.05 if ep >= 26 else 0.0
