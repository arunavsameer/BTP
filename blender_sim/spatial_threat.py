"""k×k spatial threat matrix — continuous [0, 1] heat on the image plane.

Pure Python, no ``bpy``. A cell is hot when a human walker would **slow or
stop** for that object **and** the object currently covers the cell.

Score (when)
------------
Not image size. A lamp beside a seated walker is huge and scores 0. Four
channels, class-aware, ``S = max(S_hit, S_path, S_stat, S_prox)``.

``S_hit`` — hulls on a collision course. Disk intercept × class TTC
window. A pedestrian ~5–8 m on a collision / cut course is already
amber (time to act); 11 m stays cold. A person at ~2 m is red.

``S_path`` — a **mover** occupies or is about to sweep the space I would
walk into. A car crossing 5 m in front is a stop even if the disk centres
miss by metres. The same car in the next lane, parallel, stays cold. A
person occupying or cutting the gait ~5–8 m ahead is already warm;
~2 m is red.

``S_stat`` — I am walking **toward** a world-static body on my gait
(parked car, tree, pole, hole). Inverse in hull gap. An on-gait tree
warms from ~5–6 m. Seated or a miss to the side → 0.

``S_prox`` — already inside personal space in front and closing.

Both still (seated + furniture) → 0. Matching-speed walker ahead → 0
(never catch up). An approaching car vs a seated ego is ``S_hit`` because
``V_rel = V_obj``. A pedestrian ~5–8 m on a geometric / cut course is
amber; 11 m is still cold. A person filling the gait at ~2 m is red.
Urgency is a class TTC window (no closing-speed floor). Behind the
camera, fully off-screen, or occluded → overlay heat 0.

Paint (where)
-------------
``splat_bbox`` fills only cells the 2-D box actually overlaps. No Gaussian
bleed into empty neighbours — the model never sees those bounding boxes.
Empty inner boxes do **not** fall back to a leftover truncated AABB.

``python spatial_threat.py`` runs the numeric self-test.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from threat_math import vdot, vnorm, vscale, vsub


# ---------------------------------------------------------------------------
# Score knobs (not CLI — only k is a command-line hyperparameter)
# ---------------------------------------------------------------------------

EGO_HALF_M = 0.30  # walker half-width (shoulders + a little sway)
HIT_CLEAR_M = 0.45  # hulls this close *now* still count as a meet when diverging
HORIZON_S = 16.0
TIME_SLOP_S = 0.12
REL_STATIC_M_S = 0.12  # |V_rel| below this: intercept is undefined
OBJ_MOVE_M_S = 0.20  # |V_obj| above this: the other body is a mover
ON_GAIT_LAT_M = 0.60
OVERLAP_PX = 1.5


# Per-family envelopes. A human yields further for a car than a person,
# and almost not at all for a pole unless they will walk into it.
#   cpa_soft   — S_hit gaussian width (m)
#   path_s     — how far ahead a *mover* in my gait slab still counts (m)
#   path_pad   — extra lateral personal space beyond the body (m)
#   yield_t    — horizon to a gait-line crossing (s)
#   s_soft     — along-track kernel width when they occupy/cross (m)
#   s_pow      — kernel exponent; 4 = sharp person cutoff, 2 = long car tail
#   t_red      — TTC at or below this is full urgency (s)
#   t_cool     — TTC at or above this is 0 (s). No closing-speed floor.
#   prox       — personal-space radius for the too-close flinch (m)
#   stat_lat   — static bodies farther than this from the gait are ignored (m)
#   stat_soft  — gaussian on hull gap for walking-into-static (m)
_KIND: dict[str, dict[str, float]] = {
    "vehicle": {
        "cpa_soft": 0.32, "path_s": 14.0, "path_pad": 0.90, "yield_t": 5.0,
        "s_soft": 14.0, "s_pow": 2.0, "t_red": 0.70, "t_cool": 4.80, "prox": 2.20,
        "stat_lat": 1.50, "stat_soft": 3.6,
    },
    "agent": {
        "cpa_soft": 0.28, "path_s": 12.0, "path_pad": 0.50, "yield_t": 3.6,
        "s_soft": 8.0, "s_pow": 3.0, "t_red": 0.80, "t_cool": 5.00, "prox": 1.70,
        "stat_lat": 0.70, "stat_soft": 3.6,
    },
    "hole": {
        "cpa_soft": 0.28, "path_s": 0.0, "path_pad": 0.0, "yield_t": 0.0,
        "s_soft": 1.0, "s_pow": 2.0, "t_red": 0.70, "t_cool": 3.00, "prox": 0.90,
        "stat_lat": 0.50, "stat_soft": 2.0,
    },
    "static": {
        "cpa_soft": 0.22, "path_s": 0.0, "path_pad": 0.0, "yield_t": 0.0,
        "s_soft": 1.0, "s_pow": 2.0, "t_red": 1.00, "t_cool": 4.50, "prox": 1.10,
        "stat_lat": 0.62, "stat_soft": 3.4,
    },
}

_VEHICLE_NAMES = frozenset({"vehicle", "bicycle", "scooter"})
_AGENT_NAMES = frozenset({"person"})
_HOLE_NAMES = frozenset({"pothole", "crater", "broken_slab", "debris"})


def _clamp01(x: float) -> float:
    # ``v != v`` is the NaN test. Without it a NaN would slip through both
    # comparisons unchanged and be written to JSON as the invalid token NaN.
    v = float(x)
    if v != v:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _xy(v: Sequence[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), 0.0)


def heading_frame(heading: Sequence[float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Horizontal unit tangent ŝ and road-right r̂ = (ŝ_y, −ŝ_x).

    Matches Frenet ``right = tangent × world_up`` in Blender Z-up, so a
    positive ``path_lat`` is to the walker's right, not mirrored left.
    """
    h = _xy(heading)
    n = vnorm(h)
    if n < 1e-8:
        s_hat = (0.0, 1.0, 0.0)
    else:
        s_hat = vscale(h, 1.0 / n)
    r_hat = (s_hat[1], -s_hat[0], 0.0)
    return s_hat, r_hat


def horizontal_extents(
    corners: Sequence[Sequence[float]],
    heading: Sequence[float],
) -> tuple[float, float]:
    """AABB half-length along ŝ and half-width along r̂ (metres)."""
    s_hat, r_hat = heading_frame(heading)
    if not corners:
        return 0.35, 0.35
    ss = [vdot(_xy(c), s_hat) for c in corners]
    ls = [vdot(_xy(c), r_hat) for c in corners]
    half_s = 0.5 * (max(ss) - min(ss))
    half_l = 0.5 * (max(ls) - min(ls))
    return max(0.12, half_s), max(0.12, half_l)


def kind_for(class_name: str | None) -> str:
    """vehicle / agent / hole / static — drives the yield envelope."""
    c = str(class_name or "").strip().lower()
    if c in _VEHICLE_NAMES or c.startswith("vehicle"):
        return "vehicle"
    if c in _AGENT_NAMES:
        return "agent"
    if c in _HOLE_NAMES:
        return "hole"
    if c.startswith("threat_") or c in {"projectile", "truck_door"}:
        return "agent"
    return "static"


def _urgency(t_hit: float, kn: dict[str, float]) -> float:
    """1 if TTC ≤ t_red, 0 if TTC ≥ t_cool, smoothstep in between.

    No closing-speed floor. A walker 6 s away on a geometric collision
    course is not a vest buzz; the same body at 0.5 s is.
    """
    t = max(float(t_hit), 0.0)
    t_red = max(float(kn["t_red"]), 0.05)
    t_cool = max(float(kn["t_cool"]), t_red + 0.05)
    if t <= t_red:
        return 1.0
    if t >= t_cool:
        return 0.0
    x = (t - t_red) / (t_cool - t_red)
    return 1.0 - (3.0 * x * x - 2.0 * x * x * x)


def _path_kernel(s_at: float, kn: dict[str, float]) -> float:
    s_at = max(float(s_at), 0.0)
    soft = max(float(kn["s_soft"]), 0.2)
    pwr = max(float(kn.get("s_pow", 2.0)), 1.0)
    return math.exp(-((s_at / soft) ** pwr))


def object_threat_score(
    p_cam: Sequence[float],
    p_obj: Sequence[float],
    v_obj: Sequence[float],
    heading: Sequence[float],
    walk_speed: float,
    half_s: float,
    half_lat: float,
    ego_half: float = EGO_HALF_M,
    path_s: float | None = None,
    path_lat: float | None = None,
    class_name: str | None = None,
    occluded: bool = False,
    behind_camera: bool = False,
    z_cam: float | None = None,
) -> float:
    """[0, 1] would-I-stop score. Paint site is separate.

    Bodies are disks in the ground plane. Ego velocity is ``walk_speed``
    along the *body* heading (gait tangent), never the shaking look.

    Off-gait actors (``|path_lat| > ON_GAIT_LAT_M``) keep world XY so a
    crosswalk turn cannot reconstruct a right-side person as if they were
    on the left walking away.

    Invisible bodies (behind the lens, already passed, occluded) score 0
    so leftover 2-D boxes cannot keep painting red.
    """
    if occluded or behind_camera:
        return 0.0
    # Blender camera looks down local −Z; Z ≥ 0 is behind the lens.
    if z_cam is not None and float(z_cam) >= -1e-3:
        return 0.0

    kind = kind_for(class_name)
    kn = _KIND[kind]
    s_hat, r_hat = heading_frame(heading)
    v_ego = float(walk_speed)
    v_ego = 0.0 if not math.isfinite(v_ego) else max(0.0, v_ego)
    r_lat = max(0.12, float(half_lat))
    r_long = max(0.12, float(half_s))
    radius = float(ego_half) + min(1.35, max(r_lat, r_long * 0.45))

    p_world = _xy(vsub(p_obj, p_cam))
    v_o = _xy(v_obj)
    p = p_world
    if path_s is not None and path_lat is not None:
        p_path = (
            float(path_s) * s_hat[0] + float(path_lat) * r_hat[0],
            float(path_s) * s_hat[1] + float(path_lat) * r_hat[1],
            0.0,
        )
        if abs(float(path_lat)) <= ON_GAIT_LAT_M:
            p = p_path
        else:
            p = p_world
    s = vdot(p, s_hat)
    lat = vdot(p, r_hat)
    v_s = vdot(v_o, s_hat)
    v_lat = vdot(v_o, r_hat)
    v_close = v_ego - v_s  # >0 if the along-track gap is shrinking
    s_world = vdot(p_world, s_hat)

    # Behind the look plane, or a static already passed on the ribbon.
    if s_world < -0.05:
        return 0.0
    if kind in ("static", "hole"):
        if path_s is not None and float(path_s) < 0.0:
            return 0.0
        if s < 0.0:
            return 0.0

    if s < -(r_long + 0.6) and v_close <= 0.05:
        return 0.0

    v_e = (s_hat[0] * v_ego, s_hat[1] * v_ego, 0.0)
    v_rel = (v_o[0] - v_e[0], v_o[1] - v_e[1], 0.0)
    speed2 = vdot(v_rel, v_rel)
    dist_now = max(0.0, vnorm(p) - radius)
    obj_speed = vnorm(v_o)

    # Both still: seated lamp, parked bole, two bodies at rest.
    if obj_speed < OBJ_MOVE_M_S and v_ego < 0.12:
        return 0.0

    s_hit = _score_hit(
        p, v_rel, speed2, dist_now, v_close, radius, kn,
    )
    s_path = _score_path(
        s, lat, v_close, v_lat, obj_speed, r_lat, kn,
    )
    s_stat = _score_static(
        s, lat, v_ego, v_close, obj_speed, dist_now, r_lat, kn,
    )
    s_prox = _score_prox(s, lat, dist_now, v_close, r_lat, kn)
    return _clamp01(max(s_hit, s_path, s_stat, s_prox))


def _score_hit(
    p: tuple[float, float, float],
    v_rel: tuple[float, float, float],
    speed2: float,
    dist_now: float,
    v_close: float,
    radius: float,
    kn: dict[str, float],
) -> float:
    """Disk intercept × class TTC window. True collision course only."""
    t_star = math.inf
    d_clear = dist_now
    have_meet = False
    if speed2 >= REL_STATIC_M_S * REL_STATIC_M_S:
        t_raw = -vdot(p, v_rel) / speed2
        if t_raw >= -TIME_SLOP_S:
            t_star = max(0.0, t_raw)
            if t_star > HORIZON_S:
                t_star = math.inf
            else:
                closest = (
                    p[0] + t_star * v_rel[0],
                    p[1] + t_star * v_rel[1],
                    0.0,
                )
                d_clear = max(0.0, vnorm(closest) - radius)
                have_meet = True
        elif dist_now <= HIT_CLEAR_M:
            have_meet = True
            t_star = 0.0
            d_clear = dist_now
    elif dist_now <= HIT_CLEAR_M:
        have_meet = True
        t_star = 0.0
        d_clear = dist_now

    if not have_meet or not math.isfinite(t_star):
        return 0.0
    w_hit = math.exp(-((max(d_clear, 0.0) / max(kn["cpa_soft"], 0.05)) ** 2))
    if w_hit < 0.02:
        return 0.0
    t_hit = t_star if t_star > 1e-4 else max(dist_now, 0.0) / max(v_close, 0.35)
    u = _urgency(t_hit, kn)
    if u < 0.02:
        return 0.0
    return _clamp01(w_hit * u)


def _score_path(
    s: float,
    lat: float,
    v_close: float,
    v_lat: float,
    obj_speed: float,
    r_lat: float,
    kn: dict[str, float],
) -> float:
    """Mover occupying or about to sweep the gait slab ahead of me.

    Disk-CPA of two *centres* treats a car crossing 5 m in front as a 5 m
    miss. A human still stops: that vehicle is using the space they need.
    Parallel adjacent-lane traffic never crosses the gait → 0.

    Occupancy is a *distance* kernel (cars stay hot at 10 m, people from
    ~5–8 m). Time-to-enter the slab still gates darting from the side.
    A person already in the slab and closing counts even if they are
    walking straight at us (``v_lat≈0``); that used to wait on ``S_hit``
    until the last second. Adjacent-lane parallel traffic is still 0.
    """
    if obj_speed < OBJ_MOVE_M_S or kn["path_s"] <= 0.05:
        return 0.0
    gap_lat = max(0.0, abs(float(lat)) - float(r_lat))
    pad = float(kn["path_pad"])
    in_slab = gap_lat <= pad
    toward_gait = (abs(lat) > 1e-3) and (float(lat) * float(v_lat) < 0.0)
    cutting = abs(float(v_lat)) >= 0.45
    t_line = math.inf
    if in_slab:
        # Occupying the gait and closing: warn from the distance kernel.
        # Matching-speed / receding occupants are not a stop.
        if not cutting and v_close <= 0.05:
            return 0.0
        t_line = 0.0
    elif cutting and toward_gait:
        t_line = (gap_lat - pad) / max(abs(float(v_lat)), 1e-3)
    else:
        # Parallel adjacent-lane traffic (v_lat ≈ 0, not in the slab)
        # is S_hit's problem if the hulls truly meet, else 0.
        return 0.0
    if t_line > float(kn["yield_t"]):
        return 0.0
    s_at = float(s) - float(v_close) * t_line
    if s_at <= -0.40 or s_at >= float(kn["path_s"]):
        return 0.0
    w_s = _path_kernel(s_at, kn)
    if w_s < 0.02:
        return 0.0
    # Already in the slab: the kernel *is* the "close enough" test.
    # Still approaching: fold in time-to-enter so a 2 s dart is ignored.
    mix = 1.0 if in_slab else _urgency(t_line, kn)
    if mix < 0.02:
        return 0.0
    return _clamp01(w_s * mix)


def _score_static(
    s: float,
    lat: float,
    v_ego: float,
    v_close: float,
    obj_speed: float,
    dist_now: float,
    r_lat: float,
    kn: dict[str, float],
) -> float:
    """Walking toward a world-static body on the gait. Inverse in gap."""
    if obj_speed >= OBJ_MOVE_M_S:
        return 0.0
    if v_ego < 0.12 or s < 0.10 or v_close <= 0.05:
        return 0.0
    if abs(float(lat)) > float(kn["stat_lat"]) + float(r_lat):
        return 0.0
    w = math.exp(-((max(dist_now, 0.0) / max(kn["stat_soft"], 0.2)) ** 2))
    if w < 0.02:
        return 0.0
    # Inverse in gap. TTC is the same information as distance at walking
    # speed; folding the moving-body window here double-penalizes a bole.
    return _clamp01(w)


def _score_prox(
    s: float,
    lat: float,
    dist_now: float,
    v_close: float,
    r_lat: float,
    kn: dict[str, float],
) -> float:
    """Flinch: something is already inside personal space in front."""
    prox = float(kn["prox"])
    if prox <= 0.05 or s < 0.05 or v_close <= 0.05:
        return 0.0
    if dist_now > prox:
        return 0.0
    if abs(float(lat)) > prox + float(r_lat) + 0.15:
        return 0.0
    w = math.exp(-((max(dist_now, 0.0) / prox) ** 2))
    if w < 0.02:
        return 0.0
    t_arr = max(dist_now, 0.0) / max(v_close, 0.35)
    return _clamp01(w * _urgency(t_arr, kn))


def empty_grid(k: int) -> list[list[float]]:
    n = max(1, int(k))
    return [[0.0] * n for _ in range(n)]


def clip_box_to_image(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    res_x: int,
    res_y: int,
) -> tuple[float, float, float, float] | None:
    """On-screen intersection of a pixel AABB. None if it misses the frame."""
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    rx = float(res_x)
    ry = float(res_y)
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(rx, xmax)
    ymax = min(ry, ymax)
    if xmax - xmin <= OVERLAP_PX or ymax - ymin <= OVERLAP_PX:
        return None
    return (xmin, ymin, xmax, ymax)


def clip_boxes_to_image(
    boxes: Sequence[Sequence[float]],
    res_x: int,
    res_y: int,
) -> list[tuple[float, float, float, float]]:
    """Keep only the on-screen remainder of each inner box."""
    out: list[tuple[float, float, float, float]] = []
    for box in boxes:
        if len(box) < 4:
            continue
        clipped = clip_box_to_image(
            float(box[0]), float(box[1]), float(box[2]), float(box[3]),
            res_x, res_y,
        )
        if clipped is not None:
            out.append(clipped)
    return out


def splat_bbox(
    grid: list[list[float]],
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    score: float,
    res_x: int,
    res_y: int,
) -> None:
    """Paint ``score`` onto cells the 2-D box actually overlaps. No bleed."""
    s = _clamp01(score)
    if s <= 1e-8:
        return
    k = len(grid)
    if k <= 0:
        return
    clipped = clip_box_to_image(xmin, ymin, xmax, ymax, res_x, res_y)
    if clipped is None:
        return
    xmin, ymin, xmax, ymax = clipped
    cw = float(res_x) / k
    ch = float(res_y) / k
    j0 = int(math.floor(xmin / cw))
    j1 = int(math.ceil(xmax / cw))
    i0 = int(math.floor(ymin / ch))
    i1 = int(math.ceil(ymax / ch))
    i0, i1 = max(0, i0), min(k, i1)
    j0, j1 = max(0, j0), min(k, j1)
    for i in range(i0, i1):
        cell_y0 = i * ch
        cell_y1 = cell_y0 + ch
        oy = min(ymax, cell_y1) - max(ymin, cell_y0)
        if oy <= OVERLAP_PX:
            continue
        row = grid[i]
        for j in range(j0, j1):
            cell_x0 = j * cw
            cell_x1 = cell_x0 + cw
            ox = min(xmax, cell_x1) - max(xmin, cell_x0)
            if ox <= OVERLAP_PX:
                continue
            if s > row[j]:
                row[j] = s


def splat_boxes(
    grid: list[list[float]],
    boxes: Sequence[Sequence[float]],
    score: float,
    res_x: int,
    res_y: int,
) -> None:
    """Splat every inscribed box; per-cell max."""
    for box in boxes:
        if len(box) < 4:
            continue
        splat_bbox(
            grid,
            xmin=float(box[0]),
            ymin=float(box[1]),
            xmax=float(box[2]),
            ymax=float(box[3]),
            score=score,
            res_x=res_x,
            res_y=res_y,
        )


def splat_object(
    grid: list[list[float]],
    *,
    p_cam: Sequence[float],
    p_obj: Sequence[float],
    v_obj: Sequence[float],
    heading: Sequence[float],
    walk_speed: float,
    half_s: float,
    half_lat: float,
    bbox: Sequence[float],
    res_x: int,
    res_y: int,
    path_s: float | None = None,
    path_lat: float | None = None,
    boxes: Sequence[Sequence[float]] | None = None,
    class_name: str | None = None,
    occluded: bool = False,
    behind_camera: bool = False,
    z_cam: float | None = None,
) -> float:
    """Score + splat. ``bbox`` is fallback only when ``boxes is None``.

    Pass ``boxes=[]`` (visible inner projection found nothing) to skip
    paint — never a leftover truncated outer AABB. Occluded / behind the
    camera score and paint 0.
    """
    if occluded or behind_camera:
        return 0.0
    s = object_threat_score(
        p_cam, p_obj, v_obj, heading, walk_speed, half_s, half_lat,
        path_s=path_s, path_lat=path_lat, class_name=class_name,
        occluded=occluded, behind_camera=behind_camera, z_cam=z_cam,
    )
    if boxes is not None:
        visible = clip_boxes_to_image(boxes, res_x, res_y)
        if not visible:
            return s
        splat_boxes(grid, visible, s, res_x, res_y)
    else:
        splat_bbox(
            grid,
            xmin=float(bbox[0]),
            ymin=float(bbox[1]),
            xmax=float(bbox[2]),
            ymax=float(bbox[3]),
            score=s,
            res_x=res_x,
            res_y=res_y,
        )
    return s


def finalize_grid(grid: list[list[float]], ndigits: int = 4) -> list[list[float]]:
    return [[round(_clamp01(v), ndigits) for v in row] for row in grid]


def frame_spatial_entry(frame_id: str, grid: Iterable[Iterable[float]]) -> dict:
    """One frame inside the episode-level spatial_annotations JSON."""
    return {
        "frame_id": str(frame_id),
        "matrix": finalize_grid([list(r) for r in grid]),
    }


def episode_spatial_payload(k: int, frames: Sequence[dict]) -> dict:
    """Single-file dump: k plus every frame's matrix, in order."""
    return {
        "k": int(k),
        "frames": list(frames),
    }


def _self_test() -> None:
    # Blender Z-up, walk along +Y.
    cam = (0.0, 0.0, 1.6)
    heading = (0.0, 1.0, 0.0)
    walk = 1.2

    def score(p, v, hs=0.35, hl=0.35, cls=None) -> float:
        return object_threat_score(
            cam, p, v, heading, walk, hs, hl, class_name=cls,
        )

    # Fast head-on car, ~1 s closing.
    s_fast = score((0.0, 16.0, 1.6), (0.0, -14.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_fast > 0.85, s_fast

    # Farther head-on, TTC ≈ 4.5 s: a car you still have time for — not amber.
    s_far_car = score((0.0, 36.0, 1.6), (0.0, -8.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_far_car < 0.20, s_far_car
    assert s_far_car < s_fast, (s_far_car, s_fast)
    # 10 s clip spawn (~6 s TTC): still cold. Heat is a now-scorer.
    s_approach = score((0.0, 42.0, 1.6), (0.0, -7.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_approach < 0.08, s_approach
    s_mid = score((0.0, 16.0, 1.6), (0.0, -7.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_mid > 0.25, s_mid

    # On-gait pothole 8 m away: you will step around, not stop yet.
    s_hole_far = score((0.0, 7.8, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50, "pothole")
    assert s_hole_far < 0.12, s_hole_far
    # Close hole you will step in in ~1.7 s → act.
    s_hole = score((0.0, 2.0, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50, "pothole")
    assert 0.40 <= s_hole <= 1.0, s_hole
    s_hole2 = score((0.0, 2.0, 1.55), (0.0, 0.0, 0.0), 0.50, 0.50, "pothole")
    assert abs(s_hole2 - s_hole) < 1e-6, (s_hole, s_hole2)

    # Offset pothole — will not step in.
    s_off = score((1.80, 8.0, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50, "pothole")
    assert s_off < 0.12, s_off

    # Frenet on-gait hole: world XY can look offset on a curve; path_lat=0 wins.
    s_fr = object_threat_score(
        cam, (1.80, 7.8, 1.6), (0.0, 0.0, 0.0), heading, walk, 0.50, 0.50,
        path_s=2.0, path_lat=0.0, class_name="pothole",
    )
    assert s_fr >= 0.40, s_fr

    # Close jaywalker / child cutting in — red, not shy yellow.
    s_jw = score((2.2, 3.0, 1.6), (-1.6, -0.85, 0.0), 0.28, 0.38, "person")
    assert s_jw >= 0.80, s_jw
    # Filling the frame ~2 m, about to meet (episode_0002 pose).
    s_jw_face = score((1.02, 1.80, 1.6), (-1.28, 0.0, 0.0), 0.28, 0.38, "person")
    assert s_jw_face >= 0.80, s_jw_face

    # Same person far away — they clear long before we arrive.
    s_far = score((0.25, 14.0, 1.6), (1.20, 0.0, 0.0), 0.28, 0.38, "person")
    assert s_far < 0.12, s_far

    # Oncoming person ~5.5 m: amber so there is time to act. 11 m stays cold.
    s_oncoming_5 = score((0.55, 5.50, 1.6), (0.0, -1.05, 0.0), 0.28, 0.38, "person")
    s_oncoming_11 = score((0.50, 11.0, 1.6), (0.0, -1.00, 0.0), 0.28, 0.38, "person")
    assert 0.22 <= s_oncoming_5 < 0.85, s_oncoming_5
    assert s_oncoming_11 < 0.12, s_oncoming_11
    # Occupying / cutting the gait ~5–8 m: already warm, not last-instant.
    s_gait_6 = score((0.25, 6.5, 1.6), (0.0, -1.00, 0.0), 0.28, 0.38, "person")
    s_gait_8 = score((0.20, 7.8, 1.6), (0.0, -1.00, 0.0), 0.28, 0.38, "person")
    assert s_gait_6 >= 0.35, s_gait_6
    assert s_gait_8 >= 0.22, s_gait_8
    assert s_gait_6 > s_oncoming_11, (s_gait_6, s_oncoming_11)

    # On-gait tree: warm from ~4–5 m, hot at ~2 m, still cold at 9 m.
    s_tree_mid = score((0.0, 4.5, 1.6), (0.0, 0.0, 0.0), 0.22, 0.22, "tree")
    s_tree_near = score((0.0, 2.0, 1.6), (0.0, 0.0, 0.0), 0.22, 0.22, "tree")
    s_tree_far = score((0.0, 9.0, 1.6), (0.0, 0.0, 0.0), 0.22, 0.22, "tree")
    assert 0.20 <= s_tree_mid < 0.85, s_tree_mid
    assert s_tree_near > 0.55, s_tree_near
    assert s_tree_far < 0.12, s_tree_far
    assert s_tree_near > s_tree_mid > s_tree_far, (s_tree_near, s_tree_mid, s_tree_far)

    # Fast adjacent miss (far lane): must stay cold, not a 0.5 neighbour glow.
    s_adj = score((2.80, 18.0, 1.6), (0.0, -10.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_adj < 0.16, s_adj

    # Receding → ~0.
    s_away = score((0.0, 8.0, 1.6), (0.0, 5.0, 0.0), 0.35, 0.35, "person")
    assert s_away < 0.08, s_away

    # Pedestrian 4 m ahead walking at exactly our speed: V_rel ≈ 0.
    s_match = score((0.30, 4.0, 1.6), (0.0, walk, 0.0), 0.28, 0.38, "person")
    assert s_match < 0.05, s_match

    # Walking into a static body on the gait is a hit. A 3 m moving
    # person with a glancing CPA still warns (S_path). Offset static no.
    s_touch = score((0.0, 0.6, 1.6), (0.0, 0.0, 0.0), 0.28, 0.38, "person")
    s_3m = score((1.34, 2.70, 1.6), (-1.92, 0.0, 0.0), 0.28, 0.38, "person")
    assert s_touch > 0.85, s_touch
    assert s_3m >= 0.50, s_3m
    assert s_touch > s_3m > s_off, (s_touch, s_3m, s_off)

    # World-static furniture you will miss: S_hit too tight, S_stat off-gait.
    s_lamp_walk = score((0.90, 2.4, 1.6), (0.0, 0.0, 0.0), 0.18, 0.18, "streetlamp")
    assert s_lamp_walk < 0.12, s_lamp_walk
    # Screenshot pose: sidewalk lamp ~1.1 m left, ~1.2 m ahead — pole, not arm.
    s_side_lamp = score((-1.12, 1.18, 1.6), (0.0, 0.0, 0.0), 0.05, 0.05, "streetlamp")
    assert s_side_lamp < 0.05, s_side_lamp
    # Already passed (behind camera / s_obj < s_cam): static heat 0.
    s_passed_lamp = score((0.90, -0.80, 1.6), (0.0, 0.0, 0.0), 0.18, 0.18, "streetlamp")
    assert s_passed_lamp < 0.02, s_passed_lamp
    s_passed_path = object_threat_score(
        cam, (0.90, 8.0, 1.6), (0.0, 0.0, 0.0), heading, walk, 0.18, 0.18,
        path_s=-0.60, path_lat=0.90, class_name="streetlamp",
    )
    assert s_passed_path < 0.02, s_passed_path
    assert object_threat_score(
        cam, (0.0, 4.0, 1.6), (0.0, 0.0, 0.0), heading, walk, 0.22, 0.22,
        class_name="tree", occluded=True,
    ) < 0.02
    assert object_threat_score(
        cam, (0.0, 4.0, 1.6), (0.0, 0.0, 0.0), heading, walk, 0.22, 0.22,
        class_name="tree", behind_camera=True,
    ) < 0.02
    assert object_threat_score(
        cam, (0.0, 4.0, 1.6), (0.0, 0.0, 0.0), heading, walk, 0.22, 0.22,
        class_name="tree", z_cam=0.15,
    ) < 0.02

    # Car crossing ~6 m in front (episode_0003 pose): disk CPA is metres,
    # but a human would not walk into that moving vehicle.
    s_cross_car = score((3.49, 6.60, 1.6), (-4.54, 0.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_cross_car >= 0.50, s_cross_car
    s_cross_ped = score((3.49, 6.60, 1.6), (-4.54, 0.0, 0.0), 0.28, 0.38, "person")
    assert s_cross_ped >= 0.18, s_cross_ped
    assert s_cross_ped < s_cross_car, (s_cross_ped, s_cross_car)
    # Same car after it stops on the gait: walking into a static body.
    s_stopped = score((-0.72, 5.10, 1.6), (0.0, 0.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_stopped >= 0.30, s_stopped

    # After a ~27° crosswalk turn: person filling the camera, glancing CPA.
    look = (0.45, 0.89, 0.0)
    s_turn = object_threat_score(
        cam, (1.34, 2.70, 1.6), (-1.92, 0.0, 0.0), look, 1.87, 0.28, 0.38,
        path_s=2.70, path_lat=1.34, class_name="person",
    )
    assert s_turn >= 0.70, s_turn

    # Right-side through-crosser must not be mirrored to the left (Frenet sign).
    s_right = object_threat_score(
        cam, (2.51, 3.63, 1.6), (-1.42, 0.0, 0.0), heading, 1.87, 0.28, 0.38,
        path_s=3.63, path_lat=2.51, class_name="person",
    )
    s_right_world = object_threat_score(
        cam, (2.51, 3.63, 1.6), (-1.42, 0.0, 0.0), heading, 1.87, 0.28, 0.38,
        class_name="person",
    )
    assert s_right >= 0.50, s_right
    assert abs(s_right - s_right_world) < 0.08, (s_right, s_right_world)

    # --- stationary ego (bench / hesitation): walk_speed = 0 ---
    def score0(p, v, hs=0.35, hl=0.35, cls=None) -> float:
        return object_threat_score(
            cam, p, v, heading, 0.0, hs, hl, class_name=cls,
        )

    s_sit_hole = score0((0.0, 7.8, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50, "pothole")
    assert s_sit_hole < 0.05, s_sit_hole
    s_sit_2m = score0((0.0, 2.0, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50, "pothole")
    assert s_sit_2m < 0.05, s_sit_2m
    # Overlapping lamp / bole while seated: both still, not a danger.
    s_sit_under = score0((0.0, 0.4, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50, "streetlamp")
    assert s_sit_under < 0.05, s_sit_under
    s_sit_lamp = object_threat_score(
        cam, (-0.43, 0.55, 1.6), (0.0, 0.0, 0.0), heading, 0.0, 0.35, 0.55,
        path_s=0.55, path_lat=-0.43, class_name="streetlamp",
    )
    assert s_sit_lamp < 0.05, s_sit_lamp
    s_sit_car = score0((0.0, 14.0, 1.6), (0.0, -12.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_sit_car > 0.80, s_sit_car
    s_sit_pass = score0((3.4, 14.0, 1.6), (0.0, -12.0, 0.0), 2.2, 0.95, "vehicle")
    assert s_sit_pass < s_sit_car, (s_sit_pass, s_sit_car)

    for w in (0.0, 0.15, 1.2, 3.0):
        for pv in (
            ((0.0, 5.0, 1.6), (0.0, 0.0, 0.0)),
            ((0.0, 5.0, 1.6), (0.0, -w, 0.0)),
            ((0.0, 5.0, 1.6), (0.0, w, 0.0)),
            ((0.0, 0.0, 1.6), (0.0, 0.0, 0.0)),
            ((0.0, -5.0, 1.6), (0.0, 0.0, 0.0)),
        ):
            val = object_threat_score(cam, pv[0], pv[1], heading, w, 0.4, 0.4)
            assert 0.0 <= val <= 1.0 and val == val, (w, pv, val)

    g = empty_grid(3)
    splat_bbox(g, xmin=640, ymin=360, xmax=1280, ymax=720, score=0.9, res_x=1920, res_y=1080)
    splat_bbox(g, xmin=640, ymin=360, xmax=1280, ymax=720, score=0.4, res_x=1920, res_y=1080)
    assert g[1][1] >= 0.85, g
    # No bleed: a box in the centre cell must not light a far corner.
    assert g[0][0] < 0.02 and g[2][2] < 0.02, g

    # Child on the right of a 5×5: right cells hot, left/centre stay cold.
    g5 = empty_grid(5)
    splat_bbox(g5, xmin=1450, ymin=420, xmax=1880, ymax=860, score=0.8, res_x=1920, res_y=1080)
    right = max(g5[i][4] for i in range(5))
    left = max(g5[i][0] for i in range(5))
    centre = g5[2][2]
    assert right >= 0.75, g5
    assert left < 0.02, g5
    assert centre < 0.02, g5

    # Empty inner boxes: do not paint a leftover truncated AABB (passed lamp).
    g_left = empty_grid(5)
    splat_object(
        g_left,
        p_cam=cam, p_obj=(-1.12, 1.18, 1.6), v_obj=(0.0, 0.0, 0.0),
        heading=heading, walk_speed=walk, half_s=0.55, half_lat=0.55,
        bbox=(0.0, 0.0, 768.0, 1080.0), res_x=1920, res_y=1080,
        boxes=[], class_name="streetlamp",
    )
    assert max(max(row) for row in g_left) < 0.02, g_left
    # Occluded / behind: skip paint even with a huge on-screen box.
    g_occ = empty_grid(5)
    splat_object(
        g_occ,
        p_cam=cam, p_obj=(0.25, 2.0, 1.6), v_obj=(-1.2, 0.0, 0.0),
        heading=heading, walk_speed=walk, half_s=0.28, half_lat=0.38,
        bbox=(0.0, 0.0, 800.0, 1080.0), res_x=1920, res_y=1080,
        boxes=[(0.0, 0.0, 800.0, 1080.0)], class_name="person",
        occluded=True,
    )
    assert max(max(row) for row in g_occ) < 0.02, g_occ
    g_behind = empty_grid(5)
    splat_object(
        g_behind,
        p_cam=cam, p_obj=(0.25, -1.0, 1.6), v_obj=(0.0, 0.0, 0.0),
        heading=heading, walk_speed=walk, half_s=0.18, half_lat=0.18,
        bbox=(0.0, 0.0, 800.0, 1080.0), res_x=1920, res_y=1080,
        boxes=[(0.0, 0.0, 800.0, 1080.0)], class_name="streetlamp",
        behind_camera=True,
    )
    assert max(max(row) for row in g_behind) < 0.02, g_behind
    miss = clip_box_to_image(-80.0, 10.0, -4.0, 400.0, 1920, 1080)
    assert miss is None

    out = finalize_grid(g)
    assert out[1][1] <= 1.0
    packed = episode_spatial_payload(3, [frame_spatial_entry("000000", g)])
    assert packed["k"] == 3 and packed["frames"][0]["frame_id"] == "000000"
    print("spatial_threat self-test: OK")


if __name__ == "__main__":
    _self_test()
