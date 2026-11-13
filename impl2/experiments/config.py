"""Knobs for the three cheap box detectors.

The collision forecast (growth, hit zone, confirm streak) is not here.
It stays in impl/collision.py, same math as the YOLO demo.
"""

from dataclasses import dataclass


@dataclass
class TrackConfig:
    # Minimum overlap to treat two boxes as the same thing.
    iou_match: float = 0.20
    # Drop a track after this many frames with no match.
    max_missed: int = 8


@dataclass
class FelzenszwalbConfig:
    # FH scale. Higher lets a bigger color step stay inside one region.
    k: float = 280.0
    # Merge graph components smaller than this many working-frame pixels.
    min_size: int = 50
    # Gaussian blur before the graph. 0 turns it off.
    sigma: float = 0.8
    # Drop a region smaller than this many full-frame pixels.
    min_area: float = 900.0
    # Drop a region bigger than this fraction of the frame.
    max_area_frac: float = 0.15
    # Colored pixels must cover at least this fraction of their own box.
    min_fill: float = 0.40
    # After the filters, keep this many regions, largest first.
    max_boxes: int = 16
    # Segment on a frame this many pixels wide, then scale boxes back. 0 keeps full size.
    process_width: int = 360


@dataclass
class PavementConfig(FelzenszwalbConfig):
    # Top of the road trapezoid, as a fraction of the frame height.
    trap_top: float = 0.68
    # Width of that top edge, as a fraction of the frame. The bottom edge is full width.
    trap_top_width: float = 0.55
    # Bins along hue, and the same number along intensity.
    hist_bins: int = 12
    # A pixel matches pavement when its bin holds at least this fraction of the trapezoid.
    match_threshold: float = 0.03
    # Drop a box when at least this fraction of its interior matches pavement.
    reject_fraction: float = 0.55
    # Below this saturation, hue is meaningless, so the pixel uses one shared gray bin.
    gray_saturation: int = 30


@dataclass
class FlowConfig:
    # How many corners Shi-Tomasi may return.
    max_corners: int = 400
    # Corner quality relative to the strongest corner. Lower keeps weaker corners.
    quality: float = 0.01
    # Minimum spacing between corners, in pixels.
    min_distance: float = 8.0
    # Shi-Tomasi block size. Odd.
    corner_block: int = 7
    # Lucas-Kanade window. Odd. Larger survives blur, worse on a small object.
    lk_window: int = 21
    # Extra pyramid levels above the full-resolution image.
    lk_levels: int = 3
    # Drop a corner whose tracker error is above this, in pixels.
    max_lk_error: float = 20.0
    # Below this many successful tracks, emit no boxes.
    min_tracks: int = 15
    # RANSAC tolerance for the partial-affine camera model, in pixels.
    ransac_px: float = 3.0
    # Neighbors must move apart by at least this many residual pixels per frame.
    expand_threshold: float = 0.25
    # A point needs this many neighbors inside the radius to count as expanding.
    min_neighbors: int = 2
    # Points farther apart than this stay in different clusters, in pixels.
    cluster_radius: float = 32.0
    # Ignore a cluster with fewer corners than this.
    min_points: int = 4
    # Grow the box around the corners by this many pixels.
    box_pad: float = 12.0
    # Drop a box smaller than this many pixels.
    min_area: float = 900.0
    # Drop a box bigger than this fraction of the frame.
    max_area_frac: float = 0.15
    # After the filters, keep this many boxes, largest first.
    max_boxes: int = 12


def validate_track(cfg: TrackConfig) -> None:
    if not 0.0 <= cfg.iou_match <= 1.0:
        raise ValueError("iou_match must be between 0 and 1")
    if cfg.max_missed < 0:
        raise ValueError("max_missed must be >= 0")


def validate_felzenszwalb(cfg: FelzenszwalbConfig) -> None:
    if cfg.k <= 0:
        raise ValueError("k must be > 0")
    if cfg.min_size < 1:
        raise ValueError("min_size must be >= 1")
    if cfg.sigma < 0:
        raise ValueError("sigma must be >= 0")
    if cfg.min_area < 0:
        raise ValueError("min_area must be >= 0")
    if not 0.0 < cfg.max_area_frac <= 1.0:
        raise ValueError("max_area_frac must be in (0, 1]")
    if not 0.0 <= cfg.min_fill <= 1.0:
        raise ValueError("min_fill must be between 0 and 1")
    if cfg.max_boxes < 1:
        raise ValueError("max_boxes must be >= 1")
    if cfg.process_width < 0:
        raise ValueError("process_width must be >= 0")


def validate_pavement(cfg: PavementConfig) -> None:
    validate_felzenszwalb(cfg)
    if not 0.0 < cfg.trap_top < 1.0:
        raise ValueError("trap_top must be between 0 and 1")
    if not 0.0 < cfg.trap_top_width <= 1.0:
        raise ValueError("trap_top_width must be in (0, 1]")
    if cfg.hist_bins < 2:
        raise ValueError("hist_bins must be >= 2")
    if not 0.0 <= cfg.match_threshold <= 1.0:
        raise ValueError("match_threshold must be between 0 and 1")
    if not 0.0 <= cfg.reject_fraction <= 1.0:
        raise ValueError("reject_fraction must be between 0 and 1")
    if not 0 <= cfg.gray_saturation <= 255:
        raise ValueError("gray_saturation must be between 0 and 255")


def validate_flow(cfg: FlowConfig) -> None:
    if cfg.max_corners < 1:
        raise ValueError("max_corners must be >= 1")
    if not 0.0 < cfg.quality <= 1.0:
        raise ValueError("quality must be in (0, 1]")
    if cfg.min_distance < 1:
        raise ValueError("min_distance must be >= 1")
    if cfg.corner_block < 3 or cfg.corner_block % 2 == 0:
        raise ValueError("corner_block must be odd and >= 3")
    if cfg.lk_window < 3 or cfg.lk_window % 2 == 0:
        raise ValueError("lk_window must be odd and >= 3")
    if cfg.lk_levels < 0:
        raise ValueError("lk_levels must be >= 0")
    if cfg.max_lk_error <= 0:
        raise ValueError("max_lk_error must be > 0")
    if cfg.min_tracks < 3:
        raise ValueError("min_tracks must be >= 3")
    if cfg.ransac_px <= 0:
        raise ValueError("ransac_px must be > 0")
    if cfg.expand_threshold < 0:
        raise ValueError("expand_threshold must be >= 0")
    if cfg.min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1")
    if cfg.cluster_radius <= 1:
        raise ValueError("cluster_radius must be > 1")
    if cfg.min_points < 1:
        raise ValueError("min_points must be >= 1")
    if cfg.box_pad < 0:
        raise ValueError("box_pad must be >= 0")
    if cfg.min_area < 0:
        raise ValueError("min_area must be >= 0")
    if not 0.0 < cfg.max_area_frac <= 1.0:
        raise ValueError("max_area_frac must be in (0, 1]")
    if cfg.max_boxes < 1:
        raise ValueError("max_boxes must be >= 1")
