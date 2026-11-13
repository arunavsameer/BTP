"""Knobs for the color-blob collision demo.

Edit the defaults here. run.py also accepts the usual ones as flags.
The collision forecast itself (growth, hit zone, confirm streak) still lives
in impl/collision.py — same math as the YOLO demo.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- which pieces of the long video to render ---
    num_clips: int = 3
    clip_seconds_min: float = 15.0
    clip_seconds_max: float = 20.0
    seed: int = 0

    # --- color blobs ---
    # 0.0 = a blob accepts only the seed pixel's exact RGB.
    # 1.0 = a blob accepts every RGB value, so the first seed swallows the frame.
    # The allowed gap on each channel is round(color_tolerance * 255).
    color_tolerance: float = 0.14

    # Drop blobs smaller than this, measured in pixels of the original frame.
    # Leaves and specks sit under it; a shirt or a face usually sits above it.
    min_area: float = 600.0

    # Drop a blob bigger than this fraction of the frame.
    # 1.0 keeps every size. 0.25 drops a blob that covers more than a quarter
    # of the picture (a whole wall or a merged sidewalk), and still keeps a
    # shirt, a face, or a door.
    max_area_frac: float = 0.25

    # Colored pixels must cover at least this fraction of their own box,
    # measured after nearby shades have been joined. 0.30 keeps a sweater
    # whose folds leave a ragged box. A window frame still scores lower.
    # 0 keeps every shape.
    min_fill: float = 0.30

    # After the filters, keep this many blobs, largest first.
    max_boxes: int = 20

    # 4 = up/down/left/right. 8 = diagonals too.
    connectivity: int = 4

    # True: every pixel in the blob is compared to the starting pixel.
    # False: each pixel is compared to its neighbor. Gradients can then
    # chain from a shirt into the wall and eat the frame.
    compare_to_seed: bool = True

    # Median blur on the small frame before grouping. 0 or 1 turns it off.
    # An odd size. Smooths camera noise so a flat shirt stays one blob.
    blur_ksize: int = 5

    # Segment on a frame this many pixels wide, then scale the boxes back up.
    # 0 uses the full frame. 360 is enough for a shirt-sized box and stays quick.
    process_width: int = 360

    # "channel": each RGB channel must sit within tolerance * 255 of the seed.
    # "euclidean": the straight-line RGB distance must sit within
    # tolerance * 255 * sqrt(3). Same dial, round acceptance region.
    color_metric: str = "channel"

    # --- optional filters. 0 / false leaves the picture unchanged ---
    # Drop a box when the long side is more than this many times the short side.
    max_aspect: float = 0.0
    # Drop a box that touches the frame edge (sky, walls, the ground).
    drop_border: bool = False
    border_px: float = 8.0
    # Drop a seed whose color is this gray. 0 keeps black, white, and gray.
    # Saturation is (max(R,G,B) - min(R,G,B)) / max(R,G,B).
    min_saturation: float = 0.0
    # Join blobs whose boxes touch or overlap (gap 0) when their colors
    # are within merge_color (Euclidean RGB). That puts the light and dark
    # parts of one sweater back together before the fill test. A negative
    # gap turns merging off. A larger gap also joins boxes that sit apart.
    merge_gap: float = 0.0
    merge_color: float = 40.0

    # --- keeping the same blob across frames ---
    # Minimum overlap to treat two boxes as the same thing.
    iou_match: float = 0.20
    # When overlap fails, still match if the centers are this close.
    # The limit is the larger of (this fraction of the box's long side)
    # and a pixel floor, so a small shirt and a big one both get some slack.
    center_match_frac: float = 0.75
    center_match_px: float = 40.0
    # That second chance also requires a similar box area, within this
    # factor either way. A speck cannot inherit a shirt's id.
    size_ratio: float = 2.5
    # A second piece of the same color that still overlaps a live box
    # stays on that id. Color is Euclidean RGB. 80 joins shirt shades
    # and leaves a blue pair of jeans alone.
    color_absorb: float = 80.0
    absorb_iou: float = 0.15
    # Drop a track from the live list after this many frames with no match.
    # A blink of one or two frames stays under this and keeps the id.
    max_missed: int = 15
    # After that, remember the color and the last place until this many
    # frames have passed since the last sighting. 90 is about 3 seconds.
    memory_frames: int = 90
    # Euclidean RGB. A later blob farther than this from the remembered
    # color does not inherit the id.
    memory_color: float = 50.0
    # How far, in pixels, the blob may be from the last center or the
    # predicted center. The allowance grows a little while the id is gone,
    # and stops at memory_radius_cap.
    memory_radius_px: float = 160.0
    memory_radius_cap: float = 240.0
    # Area may change by this factor while the id was in memory. People
    # grow as they walk closer, so this is looser than size_ratio.
    memory_size_ratio: float = 4.0
    # Hide a track until it has been seen this many times.
    min_track_frames: int = 1
    # Average the box over this many sightings. 1 keeps the raw box.
    smooth_frames: int = 1

    # --- extra gates on top of the collision forecast ---
    # Both width and height have to grow, not just one of them.
    require_both_axes: bool = False
    # Reject a track whose area bounces off the straight line between the
    # first and last sample. 0 turns it off. 0.15 means the average miss
    # is 15% of the larger area.
    max_size_jitter: float = 0.0
    # A track must grow faster than the median track in the frame, times this.
    use_scene_growth: bool = False
    scene_growth_margin: float = 1.10
    # Plain rectangles for ordinary blobs. Paths and HIT text only on a hit.
    quiet_draw: bool = False

    # Boxes that start at the top edge are sky and upper floors. The bottom
    # edge is kept: a person close to the camera sits there.
    drop_top_edge: bool = True
    top_edge_px: float = 2.0
    # A track is background when its motion is within this many robust steps
    # of the median track. 2.5 ignores the clump and keeps a real departure.
    outlier_mad: float = 2.5
    # Smallest step, in pixels per frame, that counts as a departure. Stops a
    # near-zero MAD from flagging jitter. Applied to vx, vy, and to g.
    motion_floor: float = 1.5
    # Do not estimate the camera from fewer tracks than this.
    motion_min_tracks: int = 5


def validate(cfg: Config) -> None:
    if not 0.0 <= cfg.color_tolerance <= 1.0:
        raise ValueError("color_tolerance must be between 0 and 1")
    if cfg.min_area < 0:
        raise ValueError("min_area must be >= 0")
    if not 0.0 <= cfg.max_area_frac <= 1.0:
        raise ValueError("max_area_frac must be between 0 and 1")
    if not 0.0 <= cfg.min_fill <= 1.0:
        raise ValueError("min_fill must be between 0 and 1")
    if cfg.max_boxes < 1:
        raise ValueError("max_boxes must be >= 1")
    if cfg.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if cfg.num_clips < 1:
        raise ValueError("num_clips must be >= 1")
    if cfg.clip_seconds_min <= 0 or cfg.clip_seconds_max < cfg.clip_seconds_min:
        raise ValueError("clip length range is invalid")
    if not 0.0 <= cfg.iou_match <= 1.0:
        raise ValueError("iou_match must be between 0 and 1")
    if cfg.center_match_frac < 0 or cfg.center_match_px < 0:
        raise ValueError("center match distance must be >= 0")
    if cfg.size_ratio < 1.0:
        raise ValueError("size_ratio must be >= 1")
    if cfg.max_missed < 0:
        raise ValueError("max_missed must be >= 0")
    if cfg.color_metric not in ("channel", "euclidean"):
        raise ValueError("color_metric must be 'channel' or 'euclidean'")
    if cfg.max_aspect < 0:
        raise ValueError("max_aspect must be >= 0")
    if not 0.0 <= cfg.min_saturation <= 1.0:
        raise ValueError("min_saturation must be between 0 and 1")
    if cfg.min_track_frames < 1 or cfg.smooth_frames < 1:
        raise ValueError("min_track_frames and smooth_frames must be >= 1")
    if cfg.max_size_jitter < 0:
        raise ValueError("max_size_jitter must be >= 0")
