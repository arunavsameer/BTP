"""Student temporal frame offset presets (30 FPS)."""

from __future__ import annotations

FRAME_CANDIDATES: dict[str, list[int]] = {
    "F1": [-90, -42, 0],
    "F2": [-30, -15, 0],
    "F3": [-12, -6, 0],
    "F4": [-90, -60, -30, -15, 0],
    "F5": [-90, -67, -45, -22, 0],
    "single_current": [0],
    "repeat_current_x3": [0, 0, 0],
    "repeat_current_x5": [0, 0, 0, 0, 0],
}


def indices_at_anchor(anchor_t: int, offsets: list[int]) -> list[int]:
    return [int(anchor_t) + int(o) for o in offsets]
