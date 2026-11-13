"""Knobs for the edge-region collision demo.

Edit the defaults here. run.py accepts the same ones as flags.
The collision forecast (growth, hit zone, confirm streak) is
CollisionConfig in impl/collision.py, the same math as the YOLO demo.

Checked on frames from the three street clips. Lowered so weaker
boundaries still split regions.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Gaussian blur before Canny, on the working frame. An odd size.
    # 0 or 1 turns it off. An even size is rounded up to the next odd number.
    # 3 keeps a faint clothing or door edge that 5 would smooth away.
    # Min area still drops the brick and leaf specks this lets through.
    blur_ksize: int = 3

    # Canny hysteresis, 0–255. The high threshold starts an edge. The low
    # threshold keeps it going. High stays about 3x low.
    # Lowered so weaker boundaries still split regions.
    canny_low: float = 35.0
    canny_high: float = 105.0

    # Sobel kernel inside Canny. 3, 5, or 7. 3 is the sharp one.
    canny_aperture: int = 3

    # Grow every edge by this many pixels on the working frame.
    # A gap smaller than this still acts as a wall, so a broken outline
    # keeps its inside and its outside apart. 0 leaves the Canny lines alone.
    edge_dilate: int = 1

    # 4: a region continues only up, down, left, and right.
    # 8: diagonal pixels count as touching too.
    connectivity: int = 4

    # Drop a region with fewer non-edge pixels than this.
    # Counted on the original frame: a box found on the small frame is scaled back up.
    # Leaves and specks sit under it. A door panel or a shirt usually sits above it.
    min_area: float = 1000.0

    # Drop a region whose non-edge pixels cover more than this fraction of the frame.
    # 1.0 keeps every size. A blank frame has no edges, so it is one region the
    # size of the picture, and anything below 1.0 drops that region.
    max_area_frac: float = 0.25

    # Non-edge pixels must cover at least this fraction of their own box.
    # A flat wall fills its box. A ring of edges around a window does not:
    # the box is huge and mostly empty. 0 keeps every shape.
    min_fill: float = 0.50

    # After the filters, keep this many regions, largest first.
    max_boxes: int = 15

    # Run Canny on a frame this many pixels wide, then scale the boxes back.
    # 0 uses the full frame. The source is 854 wide; 480 is enough for a
    # window-sized region and stays quick. Blur and dilation are in these pixels.
    process_width: int = 480

    # Minimum overlap to treat two boxes as the same region from frame to frame.
    iou_match: float = 0.20

    # Drop a track after this many frames with no match.
    max_missed: int = 8


def validate(cfg: Config) -> None:
    if cfg.blur_ksize < 0:
        raise ValueError("blur_ksize must be >= 0")
    if cfg.canny_low < 0 or cfg.canny_high < 0:
        raise ValueError("Canny thresholds must be >= 0")
    if cfg.canny_low > cfg.canny_high:
        raise ValueError("canny_low must be <= canny_high")
    if cfg.canny_aperture not in (3, 5, 7):
        raise ValueError("canny_aperture must be 3, 5, or 7")
    if cfg.edge_dilate < 0:
        raise ValueError("edge_dilate must be >= 0")
    if cfg.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if cfg.min_area < 0:
        raise ValueError("min_area must be >= 0")
    if not 0.0 <= cfg.max_area_frac <= 1.0:
        raise ValueError("max_area_frac must be between 0 and 1")
    if not 0.0 <= cfg.min_fill <= 1.0:
        raise ValueError("min_fill must be between 0 and 1")
    if cfg.max_boxes < 1:
        raise ValueError("max_boxes must be >= 1")
    if cfg.process_width < 0:
        raise ValueError("process_width must be >= 0")
    if not 0.0 <= cfg.iou_match <= 1.0:
        raise ValueError("iou_match must be between 0 and 1")
    if cfg.max_missed < 0:
        raise ValueError("max_missed must be >= 0")
