"""Group similar colors into blobs, and return each blob's box.

This replaces YOLO. We do not look for a whole person. A red shirt, a pair
of jeans, a face, or a truck door can each become its own box, as long as
the pixels are connected and close enough in color.

How a blob is grown (the paint-bucket version of a connected component):

  Pick the next pixel that is not already in a blob. That pixel is the seed.
  Spread to neighboring pixels whose color is still inside the tolerance.
  Then move on to the next unpainted pixel.

color_tolerance is the whole dial, from 0 to 1:

  0  →  only the seed's exact RGB is accepted
  1  →  every RGB value is accepted, so one blob covers the frame

Two ways to measure "close enough" (config.color_metric):

  channel     each channel stays within round(tolerance * 255) of the seed
  euclidean   sqrt((dB)^2 + (dG)^2 + (dR)^2) <= tolerance * 255 * sqrt(3)

Channel is a cube around the seed color. Euclidean is a ball. A shift of
40 on one channel and 0 on the others fails the cube at tolerance 0.10
(the cube only allows 26) and still fits in the ball (the ball allows 44).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


@dataclass
class Blob:
    x1: float
    y1: float
    x2: float
    y2: float
    area: float
    name: str
    color: tuple[int, int, int] = (0, 0, 0)


def color_name(bgr) -> str:
    """A short label from the seed color, so a box can say 'red' instead of a class."""
    b, g, r = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    hi = max(r, g, b)
    lo = min(r, g, b)
    if hi < 45:
        return "black"
    if lo > 205 and hi - lo < 40:
        return "white"
    if hi - lo < 28:
        return "gray"
    if r >= g and r >= b:
        if g > 140 and b < 110:
            return "yellow"
        if g > 80 and b < 90:
            return "orange"
        return "red"
    if g >= r and g >= b:
        return "green"
    if r > 140 and b > 140 and g < 160:
        return "pink"
    return "blue"


def _prepare(frame, cfg: Config):
    """Downscale and blur. Returns the small BGR image and the scale back to full size."""
    height, width = frame.shape[:2]
    if cfg.process_width <= 0 or cfg.process_width >= width:
        small = frame
    else:
        small_w = int(cfg.process_width)
        small_h = max(1, int(round(height * small_w / width)))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

    blur = int(cfg.blur_ksize)
    if blur >= 3:
        if blur % 2 == 0:
            blur += 1
        small = cv2.medianBlur(small, blur)

    small_h, small_w = small.shape[:2]
    scale_x = width / small_w
    scale_y = height / small_h
    return np.ascontiguousarray(small), scale_x, scale_y


def _make_blob(painted, rx, ry, rw, rh, seed_bgr, scale_x, scale_y, cfg: Config, full_area: float):
    if painted <= 0 or rw <= 0 or rh <= 0:
        return None
    area = painted * scale_x * scale_y
    # Fill is applied later, after nearby shades of the same color are joined.
    # A sweater lit unevenly fails the fill test piece by piece and disappears.
    if area < cfg.min_area or area > cfg.max_area_frac * full_area:
        return None
    b, g, r = int(seed_bgr[0]), int(seed_bgr[1]), int(seed_bgr[2])
    return Blob(
        x1=rx * scale_x,
        y1=ry * scale_y,
        x2=(rx + rw) * scale_x,
        y2=(ry + rh) * scale_y,
        area=area,
        name=color_name((b, g, r)),
        color=(b, g, r),
    )


def _blobs_channel(small, scale_x, scale_y, cfg: Config, full_area: float) -> list[Blob]:
    """Per-channel cube around the seed. OpenCV's flood fill."""
    small_h, small_w = small.shape[:2]
    delta = int(round(float(cfg.color_tolerance) * 255))
    diff = (delta, delta, delta)
    flags = int(cfg.connectivity) | cv2.FLOODFILL_MASK_ONLY
    if cfg.compare_to_seed:
        flags |= cv2.FLOODFILL_FIXED_RANGE

    # Non-zero mask pixels are walls, so this mask is also the claimed map.
    mask = np.zeros((small_h + 2, small_w + 2), np.uint8)
    claimed = mask[1 : small_h + 1, 1 : small_w + 1]
    blobs = []
    ys, xs = np.nonzero(claimed == 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if claimed[y, x] != 0:
            continue
        seed_bgr = small[y, x].copy()
        painted, _, _, rect = cv2.floodFill(
            small, mask, (int(x), int(y)), (0, 0, 0), diff, diff, flags
        )
        blob = _make_blob(painted, rect[0], rect[1], rect[2], rect[3], seed_bgr, scale_x, scale_y, cfg, full_area)
        if blob is not None:
            blobs.append(blob)
    return blobs


def _steps(connectivity: int) -> tuple[tuple[int, int], ...]:
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        steps = steps + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return steps


def _fill_euclidean(pix, claimed, stamp, fill_id, x0, y0, r2, width, height, steps, fixed: bool):
    """Paint one Euclidean blob. pix is indexed [y][x] = (b, g, r)."""
    sb, sg, sr = pix[y0][x0]
    stack = [(x0, y0, sb, sg, sr)]
    stamp[y0][x0] = fill_id
    count = 0
    minx = maxx = x0
    miny = maxy = y0
    while stack:
        x, y, rb, rg, rr = stack.pop()
        if claimed[y][x]:
            continue
        b, g, r = pix[y][x]
        db = b - rb
        dg = g - rg
        dr = r - rr
        if db * db + dg * dg + dr * dr > r2:
            continue
        claimed[y][x] = 1
        count += 1
        if x < minx:
            minx = x
        elif x > maxx:
            maxx = x
        if y < miny:
            miny = y
        elif y > maxy:
            maxy = y
        ref = (sb, sg, sr) if fixed else (b, g, r)
        for dy, dx in steps:
            ny = y + dy
            nx = x + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width:
                continue
            if stamp[ny][nx] == fill_id or claimed[ny][nx]:
                continue
            stamp[ny][nx] = fill_id
            stack.append((nx, ny, ref[0], ref[1], ref[2]))
    if count == 0:
        return 0, 0, 0, 0, 0
    return count, minx, miny, maxx - minx + 1, maxy - miny + 1


def _blobs_euclidean(small, scale_x, scale_y, cfg: Config, full_area: float) -> list[Blob]:
    """Ball around the seed color. Same tolerance dial as the channel cube."""
    height, width = small.shape[:2]
    # tolerance 1 reaches the far corner of the RGB cube, distance 255*sqrt(3).
    radius2 = (float(cfg.color_tolerance) * 255.0) ** 2 * 3.0
    pix = [[(int(b), int(g), int(r)) for b, g, r in row] for row in small.tolist()]
    claimed = [[0] * width for _ in range(height)]
    stamp = [[0] * width for _ in range(height)]
    steps = _steps(int(cfg.connectivity))
    blobs = []
    fill_id = 0
    for y in range(height):
        for x in range(width):
            if claimed[y][x]:
                continue
            fill_id += 1
            painted, rx, ry, rw, rh = _fill_euclidean(
                pix, claimed, stamp, fill_id, x, y, radius2, width, height, steps, cfg.compare_to_seed
            )
            blob = _make_blob(painted, rx, ry, rw, rh, pix[y][x], scale_x, scale_y, cfg, full_area)
            if blob is not None:
                blobs.append(blob)
    return blobs


def _saturation(color: tuple[int, int, int]) -> float:
    hi = max(color)
    lo = min(color)
    if hi == 0:
        return 0.0
    return (hi - lo) / hi


def _fill_ratio(blob: Blob) -> float:
    bw = max(1.0, blob.x2 - blob.x1)
    bh = max(1.0, blob.y2 - blob.y1)
    return blob.area / (bw * bh)


def _shape_ok(blob: Blob, width: int, height: int, cfg: Config, check_saturation: bool, check_fill: bool = True) -> bool:
    if blob.area > cfg.max_area_frac * width * height:
        return False
    if cfg.max_aspect > 0:
        bw = max(1.0, blob.x2 - blob.x1)
        bh = max(1.0, blob.y2 - blob.y1)
        if max(bw / bh, bh / bw) > cfg.max_aspect:
            return False
    if cfg.drop_border:
        margin = cfg.border_px
        if blob.x1 <= margin or blob.y1 <= margin or blob.x2 >= width - margin or blob.y2 >= height - margin:
            return False
    if check_saturation and cfg.min_saturation > 0 and _saturation(blob.color) < cfg.min_saturation:
        return False
    if check_fill and cfg.min_fill > 0 and _fill_ratio(blob) < cfg.min_fill:
        return False
    if blob.area < cfg.min_area:
        return False
    return True


def _box_gap(a: Blob, b: Blob) -> float:
    if a.x2 < b.x1:
        dx = b.x1 - a.x2
    elif b.x2 < a.x1:
        dx = a.x1 - b.x2
    else:
        dx = 0.0
    if a.y2 < b.y1:
        dy = b.y1 - a.y2
    elif b.y2 < a.y1:
        dy = a.y1 - b.y2
    else:
        dy = 0.0
    return max(dx, dy)


def _merge(blobs: list[Blob], gap: float, max_color: float) -> list[Blob]:
    """Union boxes that sit near each other and started from a similar color."""
    if len(blobs) < 2:
        return blobs
    limit = max_color * max_color
    parent = list(range(len(blobs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            if _box_gap(blobs[i], blobs[j]) > gap:
                continue
            db = blobs[i].color[0] - blobs[j].color[0]
            dg = blobs[i].color[1] - blobs[j].color[1]
            dr = blobs[i].color[2] - blobs[j].color[2]
            if db * db + dg * dg + dr * dr > limit:
                continue
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pj] = pi

    groups: dict[int, list[Blob]] = {}
    for i, blob in enumerate(blobs):
        groups.setdefault(find(i), []).append(blob)

    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        biggest = max(group, key=lambda blob: blob.area)
        merged.append(
            Blob(
                x1=min(blob.x1 for blob in group),
                y1=min(blob.y1 for blob in group),
                x2=max(blob.x2 for blob in group),
                y2=max(blob.y2 for blob in group),
                area=sum(blob.area for blob in group),
                name=biggest.name,
                color=biggest.color,
            )
        )
    return merged


def find_blobs(frame, cfg: Config) -> list[Blob]:
    """Boxes of the kept color blobs, largest first."""
    small, scale_x, scale_y = _prepare(frame, cfg)
    full_area = float(frame.shape[0] * frame.shape[1])
    if cfg.color_metric == "euclidean":
        blobs = _blobs_euclidean(small, scale_x, scale_y, cfg, full_area)
    else:
        blobs = _blobs_channel(small, scale_x, scale_y, cfg, full_area)

    height, width = frame.shape[:2]
    # Keep ragged pieces of one color through this pass. The fill test runs
    # after they have been joined, otherwise a shaded shirt is deleted in parts.
    blobs = [blob for blob in blobs if _shape_ok(blob, width, height, cfg, check_saturation=True, check_fill=False)]
    if cfg.merge_gap >= 0:
        blobs = _merge(blobs, cfg.merge_gap, cfg.merge_color)
    blobs = [blob for blob in blobs if _shape_ok(blob, width, height, cfg, check_saturation=False, check_fill=True)]
    blobs.sort(key=lambda blob: blob.area, reverse=True)
    return blobs[: cfg.max_boxes]
