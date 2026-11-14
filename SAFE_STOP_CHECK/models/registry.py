"""name → adapter class. CLI is --model NAME --checkpoint PATH."""

from __future__ import annotations

from pathlib import Path

from models.base import StudentModel
from models.m1_stop_floor import M1StopFloorAdapter
from models.pair_cf import PairCFAdapter
from models.trial4 import Trial4Adapter

_ADAPTERS: dict[str, type[StudentModel]] = {
    "m1_stop_floor": M1StopFloorAdapter,
    "pair_cf": PairCFAdapter,
    "trial4_epoch_22": Trial4Adapter,
    "trial4_epoch_28": Trial4Adapter,
}


def available_models() -> list[str]:
    return sorted(_ADAPTERS)


def register(name: str, cls: type[StudentModel]) -> None:
    _ADAPTERS[str(name)] = cls


def load_model(name: str, checkpoint: str | Path, device: str) -> StudentModel:
    if name not in _ADAPTERS:
        known = ", ".join(available_models()) or "(none)"
        raise KeyError(f"unknown model {name!r}. registered: {known}")
    adapter = _ADAPTERS[name]()
    adapter.load(Path(checkpoint), device)
    return adapter
