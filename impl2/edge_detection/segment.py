"""Turn a frame into boxes by treating edges as walls.

Canny marks the outlines. Each connected patch of pixels that is not an
edge becomes one component, and that component's box is what the tracker
follows.

A blank frame has no edges, so the whole picture is one component.
max_area_frac below 1 drops it.
"""

from dataclasses import dataclass

import cv2

from config import Config


@dataclass
class Region:
    x1: float
    y1: float
    x2: float
    y2: float
    area: float
    name: str = "region"


def _working_gray(frame, cfg: Config):
    """Downscale, then blur. Returns gray pixels and the scale back to full size."""
    height, width = frame.shape[:2]
    if cfg.process_width <= 0 or cfg.process_width >= width:
        small = frame
    else:
        small_w = int(cfg.process_width)
        small_h = max(1, int(round(height * small_w / width)))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = int(cfg.blur_ksize)
    if blur >= 3:
        if blur % 2 == 0:
            blur += 1
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    small_h, small_w = gray.shape[:2]
    return gray, width / small_w, height / small_h


def _edge_walls(gray, cfg: Config):
    edges = cv2.Canny(
        gray,
        float(cfg.canny_low),
        float(cfg.canny_high),
        apertureSize=int(cfg.canny_aperture),
    )
    dilate = int(cfg.edge_dilate)
    if dilate <= 0:
        return edges
    # A square of width 2*n+1 grows the wall by n pixels each way.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate + 1, 2 * dilate + 1))
    return cv2.dilate(edges, kernel)


def find_regions(frame, cfg: Config) -> list[Region]:
    """Boxes of the kept open regions, largest first."""
    gray, scale_x, scale_y = _working_gray(frame, cfg)
    edges = _edge_walls(gray, cfg)
    # Canny writes 255 on an edge. Flip that so the open pixels are the foreground.
    open_pixels = cv2.bitwise_not(edges)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        open_pixels, connectivity=int(cfg.connectivity)
    )

    frame_h, frame_w = frame.shape[:2]
    full_area = float(frame_h * frame_w)
    regions = []
    # Label 0 is the edge pixels themselves. They are the walls, not a region.
    for label in range(1, count):
        painted = int(stats[label, cv2.CC_STAT_AREA])
        rx = int(stats[label, cv2.CC_STAT_LEFT])
        ry = int(stats[label, cv2.CC_STAT_TOP])
        rw = int(stats[label, cv2.CC_STAT_WIDTH])
        rh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if painted <= 0 or rw <= 0 or rh <= 0:
            continue
        # painted is on the working frame. Scale it back to original pixels.
        area = painted * scale_x * scale_y
        fill = painted / float(rw * rh)
        if area < cfg.min_area or area > cfg.max_area_frac * full_area or fill < cfg.min_fill:
            continue
        x1 = max(0.0, rx * scale_x)
        y1 = max(0.0, ry * scale_y)
        x2 = min(float(frame_w), (rx + rw) * scale_x)
        y2 = min(float(frame_h), (ry + rh) * scale_y)
        if x2 <= x1 or y2 <= y1:
            continue
        regions.append(Region(x1, y1, x2, y2, area))

    regions.sort(key=lambda region: region.area, reverse=True)
    return regions[: cfg.max_boxes]
