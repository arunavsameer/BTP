"""Knobs for the FOMO walkway demo.

FOMO classifies each cell of a coarse grid. A lit cell is an object centroid,
not a box. The box used by the collision fit is the block of neighboring
cells of the same class, mapped back onto the original frame.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

CLASSES = ["person", "pole", "fire hydrant", "dustbin", "cone", "bicycle"]

# Names we ask the one-time labeler for, and the FOMO class they belong to.
# Lamp posts and bare poles share a class. Dustbins are "trash can" in the
# open-vocabulary model.
LABEL_PROMPTS = {
    "person": "person",
    "pole": "pole",
    "street light": "pole",
    "fire hydrant": "fire hydrant",
    "trash can": "dustbin",
    "traffic cone": "cone",
    "bicycle": "bicycle",
}


@dataclass
class Config:
    input_size: int = 192
    stride: int = 8
    # Cell probability above this becomes a detection. Overwritten by the
    # value chosen on the labeled frames after training, if that is better.
    conf_threshold: float = 0.45
    # One-time labeler confidence. Lower, so small hydrants and bins are kept.
    label_conf: float = 0.25

    train_frames: int = 240
    train_every_seconds: float = 10.0
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-2
    # Background cells outnumber object cells. This scales their loss.
    background_loss_weight: float = 0.05

    # Match a centroid to a track of the same class inside this many pixels.
    match_px: float = 48.0
    max_missed: int = 15
    # Heatmap memory. Weight on the previous frame. 0.4 forgets a false
    # cell in about two frames. A cell turns on at the trained threshold
    # and stays on only while it remains within 0.06 of that threshold.
    heatmap_momentum: float = 0.4
    conf_off: float = 0.62
    # A box may change size by at most this factor in one frame, then we
    # blend that clipped measurement in. Stops a one-cell flicker from
    # doubling the width the collision fit sees.
    size_max_ratio: float = 1.08
    size_blend: float = 0.35
    size_loss_weight: float = 0.5

    # Same three windows as the color-blob demo.
    clips: tuple = (
        ("clip_01_753s-770s.mp4", 752.5494252392793, 769.6522831434336),
        ("clip_02_1177s-1194s.mp4", 1176.7697669163908, 1194.3261405232338),
        ("clip_03_2201s-2221s.mp4", 2201.411274984765, 2220.63338424239),
    )

    video: str = str(WORKSPACE / "impl" / "input" / "street_walking_480p.mp4")
    yolo_weights: str = str(WORKSPACE / "impl" / "weights" / "yoloe-26s-seg.pt")
    cache: str = str(ROOT / "cache")
    output: str = str(ROOT / "output")
    torch_home: str = str(ROOT / "cache")
