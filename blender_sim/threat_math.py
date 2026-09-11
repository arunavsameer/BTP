"""Pure-Python relative kinematics, TTC / CPA, and threat taxonomy.

Coordinate convention
---------------------
These functions are convention-agnostic as long as both the camera and the
object live in the *same* Euclidean frame. The pipeline evaluates them in
Blender Z-up world space, then converts the resulting vectors to the spec's
Y-up frame only when writing JSON.

Time-To-Collision (constant-velocity point model)
-------------------------------------------------
Let P_rel = P_obj - P_cam and V_rel = V_obj - V_cam.

The objects are *converging* iff P_rel · V_rel < 0 (the range rate is negative).

The time at which the relative trajectory is closest to the origin is the
critical point of  f(t) = ||P_rel + t V_rel||^2:

    df/dt = 2 (P_rel + t V_rel) · V_rel = 0
    t*    = -(P_rel · V_rel) / ||V_rel||^2

which is the TTC used by the spec. Substituting t* back into the relative
trajectory gives the closest-point-of-approach distance:

    D_cpa = ||P_rel + TTC * V_rel||

If V_rel ~ 0 the trajectories are instantaneously parallel / coincident and
TTC is undefined (reported as +inf). If the pair is diverging, TTC is also
undefined — a negative algebraic t* is *not* a future collision.

Stationary ego (bench / standing)
---------------------------------
Nothing above assumes ``V_cam != 0``. With ``V_cam = 0`` the relative state
degenerates to the object's own state, which is exactly right: a parked car
20 m away has ``V_rel = 0`` ⇒ ``TTC = inf`` ⇒ SAFE_STATIC, and a ball thrown
at a seated walker still solves normally. The only failure mode would be a
division by ``||V_rel||^2``; every solver below guards that explicitly, so
no ego state can produce a ``ZeroDivisionError`` or a ``NaN``.

Non-finite inputs
-----------------
A ``NaN`` anywhere in a position or velocity would silently poison a label
(``NaN < 2.5`` is ``False``, so a real collision would be reported SAFE).
Every entry point runs :func:`_finite3`, which replaces a non-finite vector
with the zero vector, and the classifier refuses to promote a non-finite
TTC / CPA into a threat class.

Planar mode
-----------
Park and plaza episodes move on an open ground plane where the only
meaningful separation is horizontal: a 1.6 m eye-height offset must not be
counted as clearance from a bollard the walker is about to hit. Passing
``planar=True`` zeroes Z on both the relative position and the relative
velocity before solving, which turns the 3-D CPA into a ground-plane miss
distance without changing any of the algebra above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Vector helpers (no numpy — Blender's embedded interpreter is enough)
# ---------------------------------------------------------------------------

def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def vscale(a: Sequence[float], s: float) -> Vec3:
    return (float(a[0]) * s, float(a[1]) * s, float(a[2]) * s)


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(vdot(a, a))


def vdist(a: Sequence[float], b: Sequence[float]) -> float:
    return vnorm(vsub(a, b))


def vnormalize(a: Sequence[float], eps: float = 1e-12) -> Vec3:
    n = vnorm(a)
    if n < eps:
        return (0.0, 0.0, 0.0)
    return vscale(a, 1.0 / n)


def as_vec3(a: Iterable[float]) -> Vec3:
    x, y, z = a
    return (float(x), float(y), float(z))


def _finite3(a: Sequence[float]) -> Vec3:
    """Coerce to a finite Vec3. A non-finite component collapses the vector.

    Zeroing the whole vector (rather than the bad component) is deliberate:
    a half-valid position is a silently wrong geometry, whereas the origin
    is an obviously wrong one that the distance cull will drop.
    """
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return (0.0, 0.0, 0.0)
    return (x, y, z)


def vflat(a: Sequence[float]) -> Vec3:
    """Drop the vertical component (Blender Z-up ground plane)."""
    return (float(a[0]), float(a[1]), 0.0)


def blender_zup_to_yup(v: Sequence[float]) -> Vec3:
    """Blender (X, Y_forward, Z_up) → spec (X, Y_up, Z_forward)."""
    return (float(v[0]), float(v[2]), float(v[1]))


def yup_to_blender_zup(v: Sequence[float]) -> Vec3:
    """Spec (X, Y_up, Z_forward) → Blender (X, Y_forward, Z_up)."""
    return (float(v[0]), float(v[2]), float(v[1]))


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelativeKinematics:
    """All fields are in the *same* world frame that was passed in."""

    p_rel: Vec3
    v_rel: Vec3
    distance: float
    converging: bool
    ttc: float  # seconds; math.inf if undefined
    cpa: float  # metres; current distance if TTC is undefined

    def as_json_dict(self, *, to_yup: bool = True) -> dict:
        """Serialize vectors. `to_yup=True` when `self` was computed in Z-up."""
        v = blender_zup_to_yup(self.v_rel) if to_yup else self.v_rel
        return {
            "relative_velocity": [round(v[0], 4), round(v[1], 4), round(v[2], 4)],
            "distance": round(self.distance, 4),
            "ttc": round(self.ttc, 4) if math.isfinite(self.ttc) else 9999.0,
            "cpa": round(self.cpa, 4) if math.isfinite(self.cpa) else round(self.distance, 4),
        }


def time_to_collision(
    p_rel: Sequence[float],
    v_rel: Sequence[float],
    rel_speed_eps: float = 1e-4,
    converging_eps: float = 0.0,
) -> float:
    """Return TTC in seconds, or +inf if the pair is not a future approach.

    TTC = -(P_rel · V_rel) / ||V_rel||^2     iff P_rel · V_rel < 0

    `||V_rel|| ~ 0` covers both a genuinely parallel pair and the stationary
    ego watching a stationary object; both are "no future approach", so the
    guard below is the same branch and can never divide by zero.
    """
    pr = _finite3(p_rel)
    vr = _finite3(v_rel)
    speed2 = vdot(vr, vr)
    eps2 = max(float(rel_speed_eps), 1e-9) ** 2
    if speed2 < eps2:
        return math.inf
    range_rate = vdot(pr, vr)  # d/dt (0.5 ||P||^2); negative ⇒ closing
    if range_rate >= -converging_eps:
        return math.inf
    ttc = -range_rate / speed2
    return ttc if math.isfinite(ttc) else math.inf


def closest_point_of_approach(
    p_rel: Sequence[float],
    v_rel: Sequence[float],
    ttc: Optional[float] = None,
    rel_speed_eps: float = 1e-4,
    converging_eps: float = 0.0,
) -> float:
    """Euclidean miss distance at the TTC instant.

    D_cpa = ||P_rel + V_rel * TTC||
    """
    pr = _finite3(p_rel)
    vr = _finite3(v_rel)
    if ttc is None:
        ttc = time_to_collision(pr, vr, rel_speed_eps, converging_eps)
    if not math.isfinite(ttc):
        return vnorm(pr)
    return vnorm(vadd(pr, vscale(vr, ttc)))


def relative_kinematics(
    p_cam: Sequence[float],
    v_cam: Sequence[float],
    p_obj: Sequence[float],
    v_obj: Sequence[float],
    rel_speed_eps: float = 1e-4,
    converging_eps: float = 0.0,
    planar: bool = False,
) -> RelativeKinematics:
    """Full relative-state packet used by the annotator every frame.

    `planar=True` solves on the ground plane (Blender XY). Use it in open
    biomes where a vertical offset is not clearance: a seated walker and a
    bollard differ by ~1 m in Z, and counting that as miss distance would
    label an imminent shin-strike SAFE.

    `v_cam = (0, 0, 0)` is fully supported — see the module docstring.
    """
    p_cam = _finite3(p_cam)
    v_cam = _finite3(v_cam)
    p_obj = _finite3(p_obj)
    v_obj = _finite3(v_obj)
    p_rel = vsub(p_obj, p_cam)
    v_rel = vsub(v_obj, v_cam)
    if planar:
        p_rel = vflat(p_rel)
        v_rel = vflat(v_rel)
    ttc = time_to_collision(p_rel, v_rel, rel_speed_eps, converging_eps)
    cpa = closest_point_of_approach(p_rel, v_rel, ttc, rel_speed_eps, converging_eps)
    return RelativeKinematics(
        p_rel=p_rel,
        v_rel=v_rel,
        distance=vnorm(p_rel),
        converging=math.isfinite(ttc),
        ttc=ttc,
        cpa=cpa,
    )


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

SAFE_STATIC = "SAFE_STATIC"
SAFE_DYNAMIC = "SAFE_DYNAMIC"
NEAR_MISS = "NEAR_MISS"
CRITICAL_THREAT = "CRITICAL_THREAT"


def _is_static(speed: float, static_eps: float) -> bool:
    return speed <= static_eps


def classify_threat(
    kin: RelativeKinematics,
    obj_speed: float,
    threat_cfg: dict,
) -> str:
    """Map (TTC, CPA, speed, range) onto the four-class tactile taxonomy.

    Spec (Part 4.2), applied in this priority order so a true collision can
    never be labelled SAFE:

      4. CRITICAL_THREAT  TTC < 2.5 s  AND  D_cpa < 0.5 m
      3. NEAR_MISS        TTC < 4.0 s  AND  0.5 ≤ D_cpa ≤ 1.5 m
      1. SAFE_STATIC      ||V_obj|| ~ 0 AND distance > 5 m
                          (also: static but will miss by > 1.5 m)
      2. SAFE_DYNAMIC     moving, D_cpa > 1.5 m, or not converging

    Static objects the walker is about to strike (low branch, pothole on the
    gait line) have V_obj = 0 ⇒ V_rel = -V_cam. They *do* converge, so they
    correctly fall into CRITICAL / NEAR_MISS rather than SAFE_STATIC.

    A **stationary ego** (V_cam = 0, bench or hesitation) inverts that: a
    static object now has V_rel = 0, never converges, and is SAFE_STATIC at
    any range — correct, because neither body is moving. Objects that move
    toward a seated walker keep their normal TTC / CPA and still promote to
    NEAR_MISS / CRITICAL.
    """
    t_crit = float(threat_cfg["critical_ttc"])
    c_crit = float(threat_cfg["critical_cpa"])
    t_nm = float(threat_cfg["near_miss_ttc"])
    c_nm_lo = float(threat_cfg["near_miss_cpa_min"])
    c_nm_hi = float(threat_cfg["near_miss_cpa_max"])
    d_static = float(threat_cfg["safe_static_distance"])
    c_safe = float(threat_cfg["safe_dynamic_cpa"])
    static_eps = float(threat_cfg["static_speed_eps"])

    ttc, cpa, dist = kin.ttc, kin.cpa, kin.distance
    speed = float(obj_speed)
    if not math.isfinite(speed):
        speed = 0.0
    static = _is_static(speed, static_eps)
    # A non-finite TTC / CPA must never satisfy a "<" threshold by accident.
    solved = kin.converging and math.isfinite(ttc) and math.isfinite(cpa)

    if solved and ttc < t_crit and cpa < c_crit:
        return CRITICAL_THREAT
    if solved and ttc < t_nm and c_nm_lo <= cpa <= c_nm_hi:
        return NEAR_MISS

    if static:
        # Nearby static clutter that the gait will miss, or far-away furniture.
        if dist > d_static or (math.isfinite(cpa) and cpa > c_safe) or not solved:
            return SAFE_STATIC
        # Closing static with an in-between CPA (e.g. 0.3 m at TTC = 3.2 s).
        if cpa < c_crit:
            return CRITICAL_THREAT
        return SAFE_STATIC

    # Dynamic, not a labelled threat: parallel traffic, receding walkers, …
    return SAFE_DYNAMIC


def intercept_velocity(
    p_cam: Sequence[float],
    v_cam: Sequence[float],
    p_obj: Sequence[float],
    tau: float,
    lateral_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> Vec3:
    """Constant velocity that makes the object meet the camera in `tau` seconds.

    Exact intercept (D_cpa = 0, TTC = tau) follows from

        P_obj + V_obj τ  =  P_cam + V_cam τ  +  offset
        V_obj            =  V_cam + (P_cam - P_obj)/τ  +  offset/τ

    A non-zero `lateral_offset` (world metres) yields a controlled near-miss
    whose CPA equals ||offset|| under the constant-velocity assumption.
    """
    if not math.isfinite(tau) or tau <= 1e-6:
        raise ValueError("intercept tau must be positive and finite")
    # V_obj = V_cam - P_rel / tau + offset / tau
    p_rel = vsub(_finite3(p_obj), _finite3(p_cam))
    correction = vscale(vadd(p_rel, vscale(_finite3(lateral_offset), -1.0)), -1.0 / tau)
    return vadd(_finite3(v_cam), correction)


def vec_to_list(v: Sequence[float], ndigits: int = 4) -> list[float]:
    return [round(float(v[0]), ndigits), round(float(v[1]), ndigits), round(float(v[2]), ndigits)]


if __name__ == "__main__":
    # Closed-form sanity checks — run with system Python, no Blender required.
    cfg = {
        "critical_ttc": 2.5,
        "critical_cpa": 0.5,
        "near_miss_ttc": 4.0,
        "near_miss_cpa_min": 0.5,
        "near_miss_cpa_max": 1.5,
        "safe_static_distance": 5.0,
        "safe_dynamic_cpa": 1.5,
        "static_speed_eps": 0.05,
    }
    # Head-on: 4 m apart, closing at 2 m/s → TTC = 2 s, CPA = 0.
    head_on = relative_kinematics((0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 0))
    assert abs(head_on.ttc - 2.0) < 1e-9, head_on
    assert head_on.cpa < 1e-9, head_on
    assert classify_threat(head_on, 0.0, cfg) == CRITICAL_THREAT

    # Parallel miss of 1.0 m.
    miss = relative_kinematics((0, 0, 0), (0, 0, 2), (1.0, 0, 4), (0, 0, 0))
    assert abs(miss.ttc - 2.0) < 1e-9
    assert abs(miss.cpa - 1.0) < 1e-9
    assert classify_threat(miss, 1.2, cfg) == NEAR_MISS

    # Far static dustbin.
    far = relative_kinematics((0, 0, 0), (0, 0, 1.2), (6.0, 0, 20.0), (0, 0, 0))
    assert classify_threat(far, 0.0, cfg) == SAFE_STATIC

    # Safe parallel traffic, CPA = 3 m.
    para = relative_kinematics((0, 0, 0), (0, 0, 1.2), (3.0, 0, 10.0), (0, 0, 15.0))
    assert para.cpa > 1.5
    assert classify_threat(para, 15.0, cfg) == SAFE_DYNAMIC

    vo = intercept_velocity((0, 1.6, 0), (0, 0, 1.2), (4, 1.6, 8), 2.0)
    kin = relative_kinematics((0, 1.6, 0), (0, 0, 1.2), (4, 1.6, 8), vo)
    assert abs(kin.ttc - 2.0) < 1e-6
    assert kin.cpa < 1e-6

    # Seated intercept: V_cam = 0, object still meets the camera in tau seconds.
    vo_sit = intercept_velocity((0, 1.0, 0), (0, 0, 0), (0, 1.0, 8), 2.0)
    kin_sit = relative_kinematics((0, 1.0, 0), (0, 0, 0), (0, 1.0, 8), vo_sit)
    assert abs(kin_sit.ttc - 2.0) < 1e-6, kin_sit
    assert kin_sit.cpa < 1e-6

    # ---- stationary ego (bench / hesitation): V_cam = 0 must not blow up ----
    # Both at rest: no relative motion at all ⇒ never converging, never NaN.
    frozen = relative_kinematics((0, 0, 0), (0, 0, 0), (0, 0, 3.0), (0, 0, 0))
    assert not frozen.converging and math.isinf(frozen.ttc)
    assert abs(frozen.cpa - 3.0) < 1e-9
    assert classify_threat(frozen, 0.0, cfg) == SAFE_STATIC

    # A ball thrown at a seated walker still solves exactly.
    thrown = relative_kinematics((0, 0, 0), (0, 0, 0), (0, 0, 6.0), (0, 0, -4.0))
    assert abs(thrown.ttc - 1.5) < 1e-9, thrown
    assert thrown.cpa < 1e-9
    assert classify_threat(thrown, 4.0, cfg) == CRITICAL_THREAT

    # Seated walker, car passing 3 m away: solves, misses, SAFE_DYNAMIC.
    passing = relative_kinematics((0, 0, 0), (0, 0, 0), (3.0, 0, 12.0), (0, 0, -9.0))
    assert abs(passing.cpa - 3.0) < 1e-9, passing
    assert classify_threat(passing, 9.0, cfg) == SAFE_DYNAMIC

    # ---- planar mode: vertical offset is not clearance ----
    # Bollard 0.4 m ahead but 1.2 m below the eye. 3-D CPA calls that 1.26 m
    # of miss distance; on the ground plane it is a 0.4 m shin strike.
    walk = (0.0, 1.3, 0.0)
    solid = relative_kinematics((0, 0, 1.6), walk, (0.4, 4.0, 0.4), (0, 0, 0))
    flat = relative_kinematics((0, 0, 1.6), walk, (0.4, 4.0, 0.4), (0, 0, 0), planar=True)
    assert solid.cpa > 1.2, solid.cpa
    assert abs(flat.cpa - 0.4) < 1e-9, flat.cpa
    assert classify_threat(flat, 0.0, cfg) == CRITICAL_THREAT

    # ---- non-finite inputs are neutralised, never labelled a threat ----
    nan = float("nan")
    poisoned = relative_kinematics((0, 0, 0), (0, 0, 1.2), (nan, nan, nan), (0, 0, 0))
    assert math.isfinite(poisoned.distance) and math.isfinite(poisoned.cpa)
    assert classify_threat(poisoned, nan, cfg) in (SAFE_STATIC, SAFE_DYNAMIC)

    # Zero relative speed can never divide by zero, whatever the epsilon.
    assert math.isinf(time_to_collision((0, 0, 5), (0, 0, 0)))
    assert math.isinf(time_to_collision((0, 0, 0), (0, 0, 0)))
    assert closest_point_of_approach((0, 0, 5), (0, 0, 0)) == 5.0

    print("threat_math self-test: OK")
