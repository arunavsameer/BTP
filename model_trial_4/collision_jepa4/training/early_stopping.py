"""Patience-based early stopping on selection score or consecutive val-MSE drops."""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 4):
        self.patience = int(patience)
        self.best = None
        self.bad = 0

    def step(self, better: bool) -> bool:
        """Return True if training should stop."""
        if better:
            self.bad = 0
            self.best = True
            return False
        self.bad += 1
        return self.bad >= self.patience
