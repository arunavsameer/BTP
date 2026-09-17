"""k×k spatial threat matrix — continuous [0, 1] heat on the image plane.

Pure Python, no ``bpy``. A cell is hot when something is a **potential
collision** **and** that object currently covers the cell.

Score (when)
------------
Heat is not "how close in the image." A lamp next to a seated walker is
near and huge; it is not a danger. Distance alone is a false indicator.

Relative motion is required. If ``V_rel ≈ 0`` (both still, or co-moving)
the score is 0 — nothing is approaching. Then two channels, ``S = max(S_hit, S_pass)``:

``S_hit`` — hulls on a collision course (static hole you will step in,
head-on car, child cutting through the chest).

    S_hit = P(hit) × urgency(TTC, closing speed)

``S_pass`` — the *other* body is moving, and will pass close even if the
hulls miss. A person filling the camera at ~3 m with a 1.5 m glance still
warns. A parked car or lamp you walk past does **not** (they are
world-static; only ``S_hit`` can light them, and only if you will strike).

    S_pass = P(tight miss) × urgency    if ||V_obj|| is a walk/drive
           = 0                          if the object is at rest in the world

Gait bounce and look-jitter are ignored in this solve so a pothole on the
path does not flicker.

Paint (where)
-------------
``splat_bbox`` fills only cells the 2-D box actually overlaps. No Gaussian
bleed into empty neighbours — the model never sees those bounding boxes.

Stationary ego
--------------
``walk_speed`` may be 0. A lamp or bole at rest then has ``V_rel = 0`` and
scores 0, overlapping or not. An approaching car still produces a TTC
because ``V_rel = V_obj``.

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
HIT_CLEAR_M = 0.45  # hulls this close at CPA count as a hit
CPA_SOFT_M = 0.28  # S_hit: miss this wide is already ~e^-1 (a graze, not a hit)
PASS_SOFT_M = 1.15  # S_pass: a ~1.5 m glance still has weight
PASS_MAX_M = 1.20  # wider CPA than this is not a near-pass (far lane)
HORIZON_S = 12.0  # ignore intercepts past a long episode
HIT_FLOOR = 0.12  # likely hit, far TTC
LAMBDA_V = 0.16
LAMBDA_T = 1.85  # urgency ≈ 1 − exp(−Λ / TTC); TTC=1 s → ~0.84, 4 s → ~0.37
TIME_SLOP_S = 0.12  # intercept times within this still count as a meet
REL_STATIC_M_S = 0.12  # |V_rel| below this: nothing is approaching
OBJ_MOVE_M_S = 0.20  # |V_obj| above this: the other body is a mover
# Frenet |path_lat| inside this uses the spline frame (curved on-gait).
# Wider offsets keep world XY so a look-turn cannot flip left/right.
ON_GAIT_LAT_M = 0.60

# Occupied-cell overlap in pixels. Neighbours with no overlap stay 0.
OVERLAP_PX = 1.5


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
) -> float:
    """[0, 1] potential collision. Paint site is separate.

    Bodies are disks in the ground plane. Ego velocity is ``walk_speed``
    along the *body* heading (gait tangent), never the shaking look.

    Off-gait actors (``|path_lat| > ON_GAIT_LAT_M``) keep world XY so a
    crosswalk turn cannot reconstruct a right-side person as if they were
    on the left walking away.
    """
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
    v_s = vdot(v_o, s_hat)
    v_close = v_ego - v_s  # >0 if the along-track gap is shrinking

    if s < -(r_long + 0.6) and v_close <= 0.05:
        return 0.0

    v_e = (s_hat[0] * v_ego, s_hat[1] * v_ego, 0.0)
    v_rel = (v_o[0] - v_e[0], v_o[1] - v_e[1], 0.0)
    speed2 = vdot(v_rel, v_rel)
    dist_now = max(0.0, vnorm(p) - radius)

    # Nothing is approaching: seated lamp, matching-speed walker, both at rest.
    if speed2 < REL_STATIC_M_S * REL_STATIC_M_S:
        return 0.0

    t_star = math.inf
    d_clear = dist_now
    have_meet = False
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

    if not have_meet or not math.isfinite(t_star):
        return 0.0

    t_hit = t_star if t_star > 1e-4 else max(dist_now, 0.0) / max(v_close, 0.35)
    t_hit = max(t_hit, 0.05)
    u = 1.0 - math.exp(-LAMBDA_V * max(v_close, 0.0) - LAMBDA_T / t_hit)
    mix = HIT_FLOOR + (1.0 - HIT_FLOOR) * u

    s_hit = 0.0
    w_hit = math.exp(-((max(d_clear, 0.0) / CPA_SOFT_M) ** 2))
    if w_hit >= 0.02:
        s_hit = _clamp01(w_hit * mix)

    s_pass = 0.0
    obj_speed = vnorm(v_o)
    # Near-pass only if the *other* body is moving. A lamp you walk past is
    # world-static: S_hit may still fire if the hulls actually collide.
    if obj_speed >= OBJ_MOVE_M_S and d_clear <= PASS_MAX_M:
        w_pass = math.exp(-((max(d_clear, 0.0) / PASS_SOFT_M) ** 2))
        if w_pass >= 0.02:
            s_pass = _clamp01(w_pass * mix)

    return _clamp01(max(s_hit, s_pass))


def empty_grid(k: int) -> list[list[float]]:
    n = max(1, int(k))
    return [[0.0] * n for _ in range(n)]


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
    cw = float(res_x) / k
    ch = float(res_y) / k
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
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
) -> float:
    """Score + splat. ``bbox`` is the fallback outer box. Returns S."""
    s = object_threat_score(
        p_cam, p_obj, v_obj, heading, walk_speed, half_s, half_lat,
        path_s=path_s, path_lat=path_lat,
    )
    if boxes:
        splat_boxes(grid, boxes, s, res_x, res_y)
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

    def score(p, v, hs=0.35, hl=0.35) -> float:
        return object_threat_score(cam, p, v, heading, walk, hs, hl)

    # Fast head-on car, ~1 s closing.
    s_fast = score((0.0, 16.0, 1.6), (0.0, -14.0, 0.0), 2.2, 0.95)
    assert s_fast > 0.85, s_fast

    # Farther head-on, TTC ≈ 4 s: on a collision course but not urgent.
    s_far_car = score((0.0, 36.0, 1.6), (0.0, -8.0, 0.0), 2.2, 0.95)
    assert 0.12 <= s_far_car < s_fast, (s_far_car, s_fast)

    # On-gait pothole: will step in, TTC is several seconds → mid, not full red.
    s_hole = score((0.0, 7.8, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert 0.18 <= s_hole <= 0.75, s_hole
    s_hole2 = score((0.0, 7.8, 1.55), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert abs(s_hole2 - s_hole) < 1e-6, (s_hole, s_hole2)

    # Offset pothole — will not step in.
    s_off = score((1.80, 8.0, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_off < 0.12, s_off

    # Frenet on-gait hole: world XY can look offset on a curve; path_lat=0 wins.
    s_fr = object_threat_score(
        cam, (1.80, 7.8, 1.6), (0.0, 0.0, 0.0), heading, walk, 0.50, 0.50,
        path_s=7.8, path_lat=0.0,
    )
    assert 0.18 <= s_fr <= 0.75, s_fr

    # Close jaywalker / child cutting in from the right — lights before the kerb.
    s_jw = score((2.2, 3.0, 1.6), (-1.6, -0.85, 0.0), 0.28, 0.38)
    assert s_jw >= 0.50, s_jw

    # Same person far away — they clear long before we arrive.
    s_far = score((0.25, 14.0, 1.6), (1.20, 0.0, 0.0), 0.28, 0.38)
    assert s_far < 0.22, s_far

    # Fast adjacent miss (far lane): must stay cold, not a 0.5 neighbour glow.
    s_adj = score((2.80, 18.0, 1.6), (0.0, -10.0, 0.0), 2.2, 0.95)
    assert s_adj < 0.16, s_adj

    # Receding → ~0.
    s_away = score((0.0, 8.0, 1.6), (0.0, 5.0, 0.0), 0.35, 0.35)
    assert s_away < 0.08, s_away

    # Pedestrian 4 m ahead walking at exactly our speed: V_rel ≈ 0.
    s_match = score((0.30, 4.0, 1.6), (0.0, walk, 0.0), 0.28, 0.38)
    assert s_match < 0.05, s_match

    # Walking into a static body on the gait is a hit. A 3 m moving
    # crosser with a glancing CPA still warns (S_pass). Offset static no.
    s_touch = score((0.0, 0.6, 1.6), (0.0, 0.0, 0.0), 0.28, 0.38)
    s_3m = score((1.34, 2.70, 1.6), (-1.92, 0.0, 0.0), 0.28, 0.38)
    assert s_touch > 0.85, s_touch
    assert s_3m >= 0.35, s_3m
    assert s_touch > s_3m > s_off, (s_touch, s_3m, s_off)

    # World-static furniture you will miss: no S_pass, S_hit too wide.
    s_lamp_walk = score((0.90, 2.4, 1.6), (0.0, 0.0, 0.0), 0.18, 0.18)
    assert s_lamp_walk < 0.12, s_lamp_walk

    # After a ~27° crosswalk turn: person filling the camera, glancing CPA.
    look = (0.45, 0.89, 0.0)
    s_turn = object_threat_score(
        cam, (1.34, 2.70, 1.6), (-1.92, 0.0, 0.0), look, 1.87, 0.28, 0.38,
        path_s=2.70, path_lat=1.34,
    )
    assert s_turn >= 0.35, s_turn

    # Right-side through-crosser must not be mirrored to the left (Frenet sign).
    s_right = object_threat_score(
        cam, (2.51, 3.63, 1.6), (-1.42, 0.0, 0.0), heading, 1.87, 0.28, 0.38,
        path_s=3.63, path_lat=2.51,
    )
    s_right_world = object_threat_score(
        cam, (2.51, 3.63, 1.6), (-1.42, 0.0, 0.0), heading, 1.87, 0.28, 0.38,
    )
    assert s_right >= 0.35, s_right
    assert abs(s_right - s_right_world) < 0.08, (s_right, s_right_world)

    # --- stationary ego (bench / hesitation): walk_speed = 0 ---
    def score0(p, v, hs=0.35, hl=0.35) -> float:
        return object_threat_score(cam, p, v, heading, 0.0, hs, hl)

    s_sit_hole = score0((0.0, 7.8, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_sit_hole < 0.05, s_sit_hole
    s_sit_2m = score0((0.0, 2.0, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_sit_2m < 0.05, s_sit_2m
    # Overlapping lamp / bole while seated: V_rel = 0, not a danger.
    s_sit_under = score0((0.0, 0.4, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_sit_under < 0.05, s_sit_under
    s_sit_lamp = object_threat_score(
        cam, (-0.43, 0.55, 1.6), (0.0, 0.0, 0.0), heading, 0.0, 0.35, 0.55,
        path_s=0.55, path_lat=-0.43,
    )
    assert s_sit_lamp < 0.05, s_sit_lamp
    s_sit_car = score0((0.0, 14.0, 1.6), (0.0, -12.0, 0.0), 2.2, 0.95)
    assert s_sit_car > 0.80, s_sit_car
    s_sit_pass = score0((3.4, 14.0, 1.6), (0.0, -12.0, 0.0), 2.2, 0.95)
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

    out = finalize_grid(g)
    assert out[1][1] <= 1.0
    packed = episode_spatial_payload(3, [frame_spatial_entry("000000", g)])
    assert packed["k"] == 3 and packed["frames"][0]["frame_id"] == "000000"
    print("spatial_threat self-test: OK")


if __name__ == "__main__":
    _self_test()
