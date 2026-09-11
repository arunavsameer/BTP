"""k×k spatial threat matrix — continuous [0, 1] heat on the image plane.

Pure Python, no ``bpy``. A cell is hot when the walker would *collide* with
(or step into) the object: body-aware path occupancy, not a point-mass CPA.

Ego motion is the sidewalk tangent × walk speed. Gait bounce and look-jitter
are ignored so a pothole on the gait does not flicker.

Stationary ego
--------------
``walk_speed`` may legitimately be **0** (seated on a bench, or a hesitation
step in the erratic ego mode). The score is written so that this is a real
physical state rather than a special case:

* The stopping rectangle is the volume the walker *sweeps* before they could
  react. At rest that volume collapses to a personal-space buffer, and it is
  gated by the closing rate so a static object at rest scores 0.
* Arrival times divide by the **closing** rate ``v_ego − v_s``, never by
  ``v_ego``, so an approaching car still produces a correct time-to-hit for
  a walker who is not moving.
* Every division is floored, so no ego state can emit ``NaN`` or ``inf``
  into the exported matrix.

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
REACT_S = 2.2  # time the walker still occupies if they keep going
REACT_BUF_M = 0.55
HIT_FLOOR = 0.58  # definite collision maps to at least this before urgency
LAMBDA_V = 0.22
LAMBDA_T = 1.25
ADJ_PEAK = 0.50
ADJ_LANE_M = 1.70
CPA_SOFT_M = 0.42
# Closing rate at which the swept-volume term reaches full weight. Below it
# the gap is barely shrinking, so the walker is not going to walk into this.
CLOSE_GATE_MPS = 0.30

# Gaussian extra width, in grid-cell units, beyond the 2-D box half-size.
BLEED_CELLS = 0.55
BLEED_WEIGHT_MIN = 0.02


def _clamp01(x: float) -> float:
    # ``v != v`` is the NaN test. Without it a NaN would slip through both
    # comparisons unchanged and be written to JSON as the invalid token NaN.
    v = float(x)
    if v != v:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _xy(v: Sequence[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), 0.0)


def _sigmoid(x: float) -> float:
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _soft_unit(abs_x: float, half: float, edge: float) -> float:
    """1 inside ``|x|<half``, 0 outside ``|x|>half+edge``, linear in between."""
    e = max(1e-4, float(edge))
    return _clamp01(1.0 - (max(0.0, abs(float(abs_x)) - float(half)) / e))


def heading_frame(heading: Sequence[float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Horizontal unit tangent ŝ and road-right r̂ = (−ŝ_y, ŝ_x)."""
    h = _xy(heading)
    n = vnorm(h)
    if n < 1e-8:
        s_hat = (0.0, 1.0, 0.0)
    else:
        s_hat = vscale(h, 1.0 / n)
    r_hat = (-s_hat[1], s_hat[0], 0.0)
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


def _time_to_exit(lat: float, v_lat: float, half: float) -> float:
    """t ≥ 0 when |lat + t v| reaches ``half``. 0 if already out. inf if stuck inside."""
    if abs(lat) >= half - 1e-9:
        return 0.0
    if abs(v_lat) < 1e-5:
        return math.inf
    if v_lat >= 0.0:
        return (half - lat) / v_lat
    return (-half - lat) / v_lat


def _time_to_enter(lat: float, v_lat: float, half: float) -> float:
    """t ≥ 0 when the object first reaches |ℓ| ≤ half. 0 if already in."""
    if abs(lat) <= half:
        return 0.0
    if abs(v_lat) < 1e-5:
        return math.inf
    if lat > half and v_lat < 0.0:
        return (lat - half) / (-v_lat)
    if lat < -half and v_lat > 0.0:
        return (-half - lat) / v_lat
    return math.inf


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
    """Probability-like [0, 1] that this object is a collision / fall-in.

    Ground-plane Frenet of the walker (ŝ along the sidewalk, r̂ to the right):

        s = (P_obj − P_cam)_xy · ŝ
        ℓ = (P_obj − P_cam)_xy · r̂
        R = r_ego + r_obj_lat

    Three ways to hit, then a soft adjacent-lane term:

    1. **Stopping volume** — the disk of the object intersects the forward
       rectangle of length ``v_ego T_react + buf`` and half-width ``R``,
       weighted by how fast the gap is actually closing. A jaywalker filling
       the frame at 2 m is inside this box even if a point-mass CPA says they
       will have stepped aside; a bollard in front of a *seated* walker is
       not, because nothing is closing.
    2. **Guaranteed path hit** — they occupy the gait tube now and
       ``t_leave > t_arrive`` (static pothole: t_leave = ∞). Independent of
       camera bob, so the score does not oscillate.
    3. **Crossing intercept** — they *enter* the tube at t_enter and are
       still at the walker's s then (oncoming car, late cut-in).
    4. **Adjacent high speed** — not a hit, but a fast neighbour.

    Definite hits map through ``S = W (HIT_FLOOR + (1−HIT_FLOOR) U)``.
    """
    s_hat, r_hat = heading_frame(heading)
    p = _xy(vsub(p_obj, p_cam))
    v_o = _xy(v_obj)
    v_ego = float(walk_speed)
    # 0 is a valid ego speed (bench / hesitation); NaN is not.
    v_ego = 0.0 if not math.isfinite(v_ego) else max(0.0, v_ego)
    s = float(path_s) if path_s is not None else vdot(p, s_hat)
    lat = float(path_lat) if path_lat is not None else vdot(p, r_hat)
    v_s = vdot(v_o, s_hat)
    v_l = vdot(v_o, r_hat)
    r_lat = max(0.12, float(half_lat))
    r_long = max(0.12, float(half_s))
    radius = ego_half + r_lat
    v_close = v_ego - v_s  # >0 if the gap in s is shrinking

    if s < -(r_long + 0.6) and v_close <= 0.05:
        return 0.0  # already behind and not closing

    # --- 1. stopping rectangle (soft edges so it does not pop) ---
    # Gated by the closing rate: the rectangle is the volume the walker will
    # sweep, so a walker at rest facing a static object sweeps nothing.
    L_stop = v_ego * REACT_S + REACT_BUF_M
    w_s = _soft_unit(s - 0.5 * L_stop, 0.5 * L_stop + r_long, 0.45)
    w_l = _soft_unit(lat, radius, 0.18)
    w_stop = w_s * w_l * _clamp01(v_close / CLOSE_GATE_MPS)

    # --- 2. still in the tube when the walker arrives ---
    if v_close > 0.08 and s > -r_long:
        t_arr = max(s, 0.0) / v_close
    elif s <= r_long and abs(lat) < radius:
        t_arr = 0.0
    else:
        t_arr = math.inf
    if abs(lat) < radius and s > -r_long:
        t_leave = _time_to_exit(lat, v_l, radius)
        # Both times can be +inf: a static object inside the tube never
        # leaves, and a walker who is not closing never arrives. ``inf - inf``
        # is NaN, which would poison max() and reach the exported matrix, so
        # resolve the two infinite cases before subtracting.
        if not math.isfinite(t_arr):
            w_path = 0.0  # we never get there
        elif math.isinf(t_leave):
            w_path = 1.0  # it is still there when we do
        else:
            w_path = _sigmoid((t_leave - t_arr) / 0.22)
    else:
        w_path = 0.0

    # --- 3. they enter the tube at the walker's s ---
    t_in = _time_to_enter(lat, v_l, radius)
    if math.isfinite(t_in) and t_in < 8.0:
        s_ego = v_ego * t_in
        s_obj = s + v_s * t_in
        gap = abs(s_obj - s_ego) - (r_long + ego_half)
        w_x = math.exp(-((max(gap, 0.0) / 0.50) ** 2)) * math.exp(-t_in / 3.2)
    else:
        w_x = 0.0

    # --- body-aware CPA in the horizontal plane (head-on cars) ---
    v_rel = (v_o[0] - s_hat[0] * v_ego, v_o[1] - s_hat[1] * v_ego, 0.0)
    speed2 = vdot(v_rel, v_rel)
    t_star = math.inf
    w_cpa = 0.0
    if speed2 > 1e-6:
        t_star = -vdot(p, v_rel) / speed2
        if t_star >= -0.05:
            closest = (
                p[0] + max(t_star, 0.0) * v_rel[0],
                p[1] + max(t_star, 0.0) * v_rel[1],
                0.0,
            )
            d_clear = max(0.0, vnorm(closest) - radius)
        else:
            d_clear = max(0.0, vnorm(p) - radius)
        w_cpa = math.exp(-((d_clear / CPA_SOFT_M) ** 2))
    # speed2 ~ 0 means nothing is moving relative to the walker, so there is
    # no approach to solve — proximity alone is not a closest-point-of-approach.

    w = max(w_stop, w_path, w_x, w_cpa)

    t_hit = math.inf
    if w_stop > 0.04:
        # Closing rate, not ego speed: correct for a stationary walker with
        # something driving at them, and identical when the object is static.
        t_hit = min(t_hit, max(s, 0.0) / max(v_close, 0.3))
    if w_path > 0.04 and math.isfinite(t_arr):
        t_hit = min(t_hit, t_arr)
    if w_x > 0.04 and math.isfinite(t_in):
        t_hit = min(t_hit, t_in)
    if w_cpa > 0.04 and math.isfinite(t_star) and t_star > 0.0:
        t_hit = min(t_hit, t_star)
    u = 1.0 - math.exp(-LAMBDA_V * max(v_close, 0.0) - LAMBDA_T / max(t_hit, 0.08))
    s_hit = w * (HIT_FLOOR + (1.0 - HIT_FLOOR) * u)

    # Fast neighbour that will miss: mid band, not a collision.
    lat_gap = abs(lat) - radius
    adj_lat = math.exp(-(((lat_gap - ADJ_LANE_M) / 1.15) ** 2)) if lat_gap > 0.05 else 0.0
    v_rel_n = math.sqrt(speed2)
    adj_spd = 1.0 - math.exp(-max(v_rel_n - v_ego, 0.0) / 6.0)
    s_adj = ADJ_PEAK * adj_lat * adj_spd * _clamp01(v_close / max(v_rel_n, 1e-3))

    return _clamp01(s_hit + (1.0 - w) * s_adj)


def empty_grid(k: int) -> list[list[float]]:
    n = max(1, int(k))
    return [[0.0] * n for _ in range(n)]


def _cell_window(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    sig_x: float,
    sig_y: float,
    res_x: int,
    res_y: int,
    k: int,
) -> tuple[int, int, int, int]:
    """Inclusive-exclusive cell index box covering the Gaussian support."""
    pad_x = 2.2 * sig_x
    pad_y = 2.2 * sig_y
    cw = float(res_x) / k
    ch = float(res_y) / k
    j0 = int(math.floor((xmin - pad_x) / cw))
    j1 = int(math.ceil((xmax + pad_x) / cw))
    i0 = int(math.floor((ymin - pad_y) / ch))
    i1 = int(math.ceil((ymax + pad_y) / ch))
    return max(0, i0), min(k, i1), max(0, j0), min(k, j1)


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
    """Paint ``score`` onto every cell the 2-D box overlaps; bleed neighbours.

    Occupied cells keep the full threat (this region *contains* the hazard).
    Immediate neighbours get a size-aware Gaussian so the block is not a hard
    pixel. Per-cell ``max``.
    """
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
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)
    sig_x = 0.5 * bw + BLEED_CELLS * cw
    sig_y = 0.5 * bh + BLEED_CELLS * ch
    i0, i1, j0, j1 = _cell_window(xmin, ymin, xmax, ymax, sig_x, sig_y, res_x, res_y, k)
    inv_sx2 = 1.0 / max(sig_x * sig_x, 1e-8)
    inv_sy2 = 1.0 / max(sig_y * sig_y, 1e-8)
    for i in range(i0, i1):
        cell_y0 = i * ch
        cell_y1 = cell_y0 + ch
        oy = min(ymax, cell_y1) - max(ymin, cell_y0)
        vc = (i + 0.5) * ch
        dy = vc - cy
        row = grid[i]
        for j in range(j0, j1):
            cell_x0 = j * cw
            cell_x1 = cell_x0 + cw
            ox = min(xmax, cell_x1) - max(xmin, cell_x0)
            occupied = ox > 1.5 and oy > 1.5
            if occupied:
                val = s
            else:
                uc = (j + 0.5) * cw
                dx = uc - cx
                w = math.exp(-0.5 * (dx * dx * inv_sx2 + dy * dy * inv_sy2))
                if w < BLEED_WEIGHT_MIN:
                    continue
                val = s * w
            if val > row[j]:
                row[j] = val


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
) -> float:
    """Score + splat. ``bbox`` is (xmin, ymin, xmax, ymax) in pixels. Returns S."""
    s = object_threat_score(
        p_cam, p_obj, v_obj, heading, walk_speed, half_s, half_lat,
        path_s=path_s, path_lat=path_lat,
    )
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
    assert s_fast > 0.90, s_fast

    # On-gait pothole at the injector lead. Static ⇒ t_leave=∞ ⇒ stable hit.
    s_hole = score((0.0, 7.8, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert 0.55 <= s_hole <= 0.85, s_hole
    s_hole2 = score((0.0, 7.8, 1.55), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert abs(s_hole2 - s_hole) < 1e-6, (s_hole, s_hole2)

    # Offset pothole — will not step in.
    s_off = score((1.80, 8.0, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_off < 0.22, s_off

    # Frenet on-gait hole: world XY can look offset on a curve; path_lat=0 wins.
    s_fr = object_threat_score(
        cam, (1.80, 7.8, 1.6), (0.0, 0.0, 0.0), heading, walk, 0.50, 0.50,
        path_s=7.8, path_lat=0.0,
    )
    assert 0.55 <= s_fr <= 0.85, s_fr

    # Close jaywalker crossing the gait (the case the point-CPA model missed).
    s_jw = score((0.25, 2.0, 1.6), (1.20, 0.0, 0.0), 0.28, 0.38)
    assert s_jw >= 0.62, s_jw

    # Same person far away — they clear long before we arrive.
    s_far = score((0.25, 14.0, 1.6), (1.20, 0.0, 0.0), 0.28, 0.38)
    assert s_far < 0.28, s_far

    # Fast adjacent miss.
    s_adj = score((2.80, 18.0, 1.6), (0.0, -10.0, 0.0), 2.2, 0.95)
    assert 0.28 <= s_adj <= 0.62, s_adj

    # Receding → ~0.
    s_away = score((0.0, 8.0, 1.6), (0.0, 5.0, 0.0), 0.35, 0.35)
    assert s_away < 0.08, s_away

    # --- objects inside the gait tube that are not closing ---
    # Pedestrian 4 m ahead on the gait line walking at exactly our speed:
    # t_arrive and t_leave are both +inf, whose difference is NaN. Must be a
    # finite, low score (we never catch up), not NaN.
    s_match = score((0.30, 4.0, 1.6), (0.0, walk, 0.0), 0.28, 0.38)
    assert 0.0 <= s_match < 0.30 and s_match == s_match, s_match

    # --- stationary ego (bench / hesitation): walk_speed = 0 ---
    def score0(p, v, hs=0.35, hl=0.35) -> float:
        return object_threat_score(cam, p, v, heading, 0.0, hs, hl)

    # Static pothole ahead of a seated walker: nothing is closing ⇒ ~0.
    s_sit_hole = score0((0.0, 7.8, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_sit_hole < 0.05, s_sit_hole
    # Same hole 2 m away: still not a threat to someone who is not moving,
    # where the old ego-speed floor of 0.15 m/s invented an approach.
    s_sit_2m = score0((0.0, 2.0, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_sit_2m < 0.05, s_sit_2m
    # A hazard whose own footprint overlaps the walker is a hit at any speed:
    # a 1.0 m hole centred 0.4 m ahead is the one they are standing in.
    s_sit_under = score0((0.0, 0.4, 1.6), (0.0, 0.0, 0.0), 0.50, 0.50)
    assert s_sit_under > 0.55, s_sit_under
    # But a car driving at a seated walker is still critical.
    s_sit_car = score0((0.0, 14.0, 1.6), (0.0, -12.0, 0.0), 2.2, 0.95)
    assert s_sit_car > 0.85, s_sit_car
    # ...and one that will pass 3 m away is not.
    s_sit_pass = score0((3.4, 14.0, 1.6), (0.0, -12.0, 0.0), 2.2, 0.95)
    assert s_sit_pass < s_sit_car, (s_sit_pass, s_sit_car)

    # No ego state may produce a non-finite cell.
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
    assert g[1][1] >= 0.85, g  # centre cell keeps the max
    # Bleed reaches a neighbour.
    neigh = max(g[0][1], g[1][0], g[1][2], g[2][1])
    assert 0.05 < neigh < g[1][1], (neigh, g)
    out = finalize_grid(g)
    assert out[1][1] <= 1.0
    packed = episode_spatial_payload(3, [frame_spatial_entry("000000", g)])
    assert packed["k"] == 3 and packed["frames"][0]["frame_id"] == "000000"
    print("spatial_threat self-test: OK")


if __name__ == "__main__":
    _self_test()
