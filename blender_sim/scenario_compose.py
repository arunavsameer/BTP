"""Compound scenario composition: CLI tokens, aliases, Frenet occupancy.

Injector bodies stay in ``world_generator.py``. This module is pure Python
(no ``bpy``) so the occupancy math can be unit-tested with system Python.

Spatial model
-------------
Actors live in the road Frenet frame ``(s, lateral)``. A *capsule* is an
axis-aligned rectangle in that plane (half-extents along-track and across)
that moves with the same piecewise-linear ``(ds/dt, d(lateral)/dt)`` the
integrator uses. Two capsules *conflict* if those rectangles overlap at any
sampled time in the episode horizon — the window that is actually on camera.

Composition never runs a per-frame N-body solver (that would fight the gait
and the two-phase render). Separation is computed **once** at inject time:

1. Static hazards claim cells first.
2. Along-track movers (cars, cyclists, oncoming peds) claim lanes / ``s``.
3. Lateral crossers (jaywalkers, crossing cars) take the remaining depth
   bands and alternate entry side.
4. Each new spawn is nudged along ``+s`` (then a short ``−s`` search, then
   a lane / direction flip) until its predicted tube is clear.

Ground-plane ``Z`` is unchanged: corridor ``ground_z(lateral)`` still puts
feet on asphalt or the curb. Depth ordering is the existing camera + AABB
path; extra actors are just more entries in that vectorized loop.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# Lateral travellers that need a building gap so they are not born in a facade.
CROSS_GAP_SCENARIOS = frozenset(
    {
        "jaywalker",
        "jaywalker_offset",
        "jaywalker_from_left",
        "jaywalker_from_right",
        "jaywalker_turn_toward",
        "jaywalker_turn_away",
        "distant_jaywalk",
        "car_cross_front",
        "car_cross_critical",
        "cube_from_left",
        "cube_from_right",
        "shape_from_left",
        "shape_from_right",
        "cyclist_near_miss",
        "child_darting",
        "group_crossing",
        "crossing_car_side",
        "scooter_from_sidewalk",
        "periph_ped_cut",
        "periph_child_cut",
        "crowd",
        "crowd_path",
    }
)

# Ego turns onto a crosswalk (Frenet lateral + heading). Injectors still
# run; prepare_scenario sets WorldGenerator.force_ego_mode = "crosswalk".
CROSS_EGO_SCENARIOS = frozenset(
    {
        "crossing_street",
        "crossing_car_side",
        "group_crossing",
        "crossing_head_on",
    }
)

# Through-ribbon motion: occupies every lane at a fixed ``s`` over a few seconds.
THROUGH_CROSSERS = frozenset(
    {
        "jaywalker",
        "jaywalker_offset",
        "jaywalker_from_left",
        "jaywalker_from_right",
        "jaywalker_turn_toward",
        "jaywalker_turn_away",
        "distant_jaywalk",
        "car_cross_front",
        "car_cross_critical",
        "cube_from_left",
        "cube_from_right",
        "shape_from_left",
        "shape_from_right",
        "child_darting",
        "group_crossing",
        "crossing_car_side",
        "scooter_from_sidewalk",
        "periph_ped_cut",
        "periph_child_cut",
        "crowd",
        "crowd_path",
    }
)

# Same-direction walker ahead (backs of heads, +s). Catch-up is not an
# intercept you can fork, so these stay mixed-pack only. Ego is forced
# to walk: a seated camera cannot close on someone walking away without
# turning them around.
SAME_DIR_FOLLOW_SCENARIOS = frozenset(
    {
        "same_dir_closing",
        "same_dir_hold",
        "same_dir_pulling",
    }
)

# Pair-pack parallel walker. Collision on gait ± CPA; miss is sideline shift
# from t=0 (same heading / speed — no groups RGB prefix). Not in auto mix.
PARALLEL_PERSON_PAIR_SCENARIOS = frozenset({"parallel_person"})

# Pair-pack static primitive (sibling ``--pairs`` runner). Collision is on
# the gait from t=0; the miss clip is the *same* mesh shifted laterally.
# Not a ``--groups`` peel: RGB prefixes are not shared. Not in auto mix.
STATIC_SHAPE_PAIR_SCENARIOS = frozenset({"static_shape"})

# Pair-pack crowd clips. Files sit at ``crowd/sample_XXXX_crowd/`` (no
# collision/miss, no bg_YY wrap). 2–5 people, occupancy-separated.
CROWD_PACK_SCENARIOS = frozenset({"crowd", "crowd_path"})
PAIRS_FAMILY_CROWD = "crowd"
CROWD_OWNED_INJECTOR_NAMES: tuple[str, ...] = ("crowd", "crowd_path")

# Names the static-shape workstream owns for the sibling rotate loop
# (parallel → static_shape → crowd → …). Sibling owns parallel_person.
PAIRS_OWNED_INJECTOR_NAMES: tuple[str, ...] = ("static_shape",)
PAIR_RENDER_BG_ID = "bg_00"
PAIR_EMPTY_BG_ID = "bg_01"

# Empty sidewalk ahead: no oncoming gait traffic, threats stay in the
# periphery until they cut in. prepare_scenario zeroes crowd/clutter,
# locks ego to walk, and keeps the path mostly straight.
CLEAR_CENTER_SCENARIOS = frozenset(
    {
        "periph_empty",
        "periph_car_side",
        "periph_parked",
        "periph_ped_side",
        "periph_car_turn",
        "periph_car_runoff",
        "periph_ped_cut",
        "periph_child_cut",
    }
)

# Biomes that still have a carriageway next to a sidewalk. Park / plaza
# fill the frame with path or paving, so auto-biome skips them here.
CLEAR_CENTER_BIOMES = frozenset({"street", "residential", "alley", "avenue"})

SPARSE_SCENARIOS = frozenset({"empty_street"}) | CLEAR_CENTER_SCENARIOS

# These injectors *must* keep the near lane (cut-in / graze / swerve).
NEAR_LANE_LOCKED = frozenset(
    {
        "car_near_miss_lane",
        "swerve_vehicle",
        "car_cut_in",
        "car_erratic_swerve",
        "car_runs_off_road",
        "cyclist_weaving",
        "periph_car_turn",
        "periph_car_runoff",
    }
)

# Short names for compound CLI strings such as ``jaywalker,car,pothole``.
SCENARIO_ALIASES: dict[str, str] = {
    "safe": "safe_walk",
    "near_miss": "near_miss_pass",
    "swerve": "swerve_vehicle",
    "projectile": "head_level_projectile",
    "jaywalk": "jaywalker",
    "empty": "empty_street",
    "turn_toward": "jaywalker_turn_toward",
    "turn_away": "jaywalker_turn_away",
    "car": "car_approaching",
    "cars": "car_approaching",
    "vehicle": "car_approaching",
    "oncoming_car": "car_approaching",
    "pothole": "pothole_on_path",
    "hole": "pothole_on_path",
    "tree": "tree_on_path",
    "lamp": "lamp_on_path",
    "streetlamp": "lamp_on_path",
    "hasty": "hasty_look",
    "person": "oncoming_pedestrian",
    "ped": "oncoming_pedestrian",
    "cyclist": "cyclist_same_way",
    "bike": "cyclist_same_way",
    "cube": "cube_near_miss",
    "cubes": "cube_on_path",
    "shape": "shape_near_miss",
    "shapes": "shapes_on_path",
    "sphere": "shape_head_on",
    "pyramid": "shape_head_on",
    "parked": "parked_car_opposite",
    "cut_in": "car_cut_in",
    "cutin": "car_cut_in",
    "weave": "cyclist_weaving",
    "weaving": "cyclist_weaving",
    "erratic": "car_erratic_swerve",
    "runoff": "car_runs_off_road",
    "run_off_road": "car_runs_off_road",
    "child": "child_darting",
    "kid": "child_darting",
    "cross": "crossing_street",
    "crossing": "crossing_street",
    "crosswalk": "crossing_street",
    "side_car": "crossing_car_side",
    "group": "group_crossing",
    "scooter": "scooter_from_sidewalk",
    "overtake": "cyclist_overtake",
    "follow_closing": "same_dir_closing",
    "follow_hold": "same_dir_hold",
    "follow_pulling": "same_dir_pulling",
    "same_dir": "same_dir_hold",
    "door": "parked_car_door",
    "backing": "backing_vehicle",
    "reverse": "backing_vehicle",
    "periph": "periph_empty",
    "clear_center": "periph_empty",
    "side_empty": "periph_empty",
    "side_pass": "periph_car_side",
    "side_parked": "periph_parked",
    "side_ped": "periph_ped_side",
    "side_turn": "periph_car_turn",
    "side_cut": "periph_ped_cut",
    "side_runoff": "periph_car_runoff",
    "side_child": "periph_child_cut",
    "parallel_person": "parallel_person",
    "pair_person": "parallel_person",
    "parallel_pair": "parallel_person",
    "static_shape": "static_shape",
    "shape_pair": "static_shape",
    "static_pair": "static_shape",
    "crowd": "crowd",
    "crowds": "crowd",
    "crowd_path": "crowd_path",
}

_NOOP = frozenset({"safe_walk", "empty_street", "periph_empty", "hasty_look"})
_STATIC_CLASSES = frozenset({
    "pothole", "crater", "broken_slab", "debris", "tree", "streetlamp",
    "threat_cube", "threat_sphere", "threat_cylinder", "threat_pyramid",
    "threat_cone", "threat_capsule", "threat_lump",
})


def _is_static_class(class_name: str) -> bool:
    cls = str(class_name or "")
    return cls in _STATIC_CLASSES or cls.startswith("threat_")

# Lower runs first. User order is preserved within a bucket (stable sort).
_INJECT_PRIORITY: dict[str, int] = {
    "pothole_on_path": 0,
    "pothole_near": 0,
    "pothole_offset": 0,
    "tree_on_path": 0,
    "tree_near": 0,
    "lamp_on_path": 0,
    "lamp_near": 0,
    "parked_car_opposite": 1,
    "empty_street": 2,
    "safe_walk": 2,
    "hasty_look": 2,
    "oncoming_pedestrian": 10,
    "parallel_pedestrian": 10,
    "same_dir_closing": 10,
    "same_dir_hold": 10,
    "same_dir_pulling": 10,
    "cyclist_same_way": 10,
    "car_pass_far": 10,
    "car_approaching": 10,
    "near_miss_pass": 10,
    "cyclist_near_miss": 10,
    "car_near_miss_lane": 10,
    "cube_near_miss": 10,
    "shape_near_miss": 10,
    "sudden_stop": 10,
    "swerve_vehicle": 10,
    "cube_head_on": 10,
    "shape_head_on": 10,
    "cube_on_path": 0,
    "shapes_on_path": 0,
    "car_cut_in": 10,
    "cyclist_head_on": 10,
    "head_level_projectile": 10,
    "cyclist_weaving": 10,
    "car_erratic_swerve": 10,
    "car_runs_off_road": 10,
    "child_darting": 20,
    "crossing_street": 2,
    "cyclist_overtake": 10,
    "group_crossing": 20,
    "scooter_from_sidewalk": 20,
    "parked_car_door": 1,
    "crossing_car_side": 20,
    "crossing_head_on": 10,
    "backing_vehicle": 10,
    "distant_jaywalk": 20,
    "jaywalker_offset": 20,
    "jaywalker_from_left": 20,
    "jaywalker_from_right": 20,
    "jaywalker_turn_away": 20,
    "jaywalker": 20,
    "jaywalker_turn_toward": 20,
    "car_cross_front": 20,
    "cube_from_left": 20,
    "cube_from_right": 20,
    "shape_from_left": 20,
    "shape_from_right": 20,
    "car_cross_critical": 20,
    "periph_empty": 2,
    "periph_parked": 1,
    "periph_car_side": 10,
    "periph_ped_side": 10,
    "periph_car_turn": 10,
    "periph_car_runoff": 10,
    "periph_ped_cut": 20,
    "periph_child_cut": 20,
    "parallel_person": 10,
    "static_shape": 0,
    "crowd": 12,
    "crowd_path": 12,
}

_TOKEN_SPLIT = re.compile(r"[,+\s]+")

# ---------------------------------------------------------------------------
# Same-direction follow kinematics (no bpy)
# ---------------------------------------------------------------------------
#
# Ego walks +s at v_ego. The other person walks the same heading, same
# sidewalk, starting ahead (s_other(0) > s_ego(0)):
#
#     d(t) = s_other(t) - s_ego(t) = d0 + (v_other - v_ego) t
#
# closing: v_other < v_ego, d decreases, d(T) stays ≳ 1.2 m (person ahead,
# not a rear-end smash). hold: v_other ≈ v_ego, |d-d0| ≲ 0.5 m. pulling:
# v_other > v_ego, start close so they do not vanish in 2 s. Never
# oncoming (v_other > 0). Do not speed the ego to fill 15 s.


@dataclass(frozen=True)
class SameDirFollowPlan:
    mode: str
    v_ego: float
    v_other: float
    d0: float
    lat_offset: float
    duration_s: float = 15.0

    def gap(self, t: float) -> float:
        """Along-path gap d(t) = s_other(t) - s_ego(t)."""
        return float(self.d0) + (float(self.v_other) - float(self.v_ego)) * float(t)


_FOLLOW_MODE_ALIASES = {
    "same_dir_closing": "closing",
    "same_dir_hold": "hold",
    "same_dir_pulling": "pulling",
    "follow_closing": "closing",
    "follow_hold": "hold",
    "follow_pulling": "pulling",
    "closing": "closing",
    "hold": "hold",
    "pulling": "pulling",
}

# Other person's walk band. Closing may go a little slower than a stroll;
# pulling may go a little faster than a hurry. Never a statue, never a sprint.
_FOLLOW_V_MIN = 0.50
_FOLLOW_V_MAX = 2.55
_FOLLOW_T = 15.0
_FOLLOW_CLOSING_DV = 0.37
_FOLLOW_CLOSING_D_END = 1.40
_FOLLOW_CLOSING_D0_MIN = 4.80
_FOLLOW_CLOSING_D0_MAX = 8.50
_FOLLOW_HOLD_D0 = 5.20
_FOLLOW_PULL_D0 = 3.60
_FOLLOW_PULL_DV = 0.38
_FOLLOW_LAT = 0.55


def same_dir_follow_mode(name: str) -> str:
    tok = str(name or "").strip().lower().replace("-", "_")
    if tok in _FOLLOW_MODE_ALIASES:
        return _FOLLOW_MODE_ALIASES[tok]
    raise ValueError(f"unknown same-dir follow mode {name!r}")


def same_dir_follow_plan(
    v_ego: float,
    mode: str,
    *,
    duration_s: float = _FOLLOW_T,
    rng: Optional[random.Random] = None,
) -> SameDirFollowPlan:
    """Plan ``(v_other, d0, lat_offset)`` for a same-direction walker ahead.

    ``v_ego`` is an input. A near-zero ego cannot catch up without the other
    walking toward the camera, so closing/hold degrade to pulling (still +s).
    """
    v_ego = max(0.0, float(v_ego))
    T = max(1.0, float(duration_s))
    key = same_dir_follow_mode(mode)
    lat = float(_FOLLOW_LAT)
    if rng is not None:
        lat = float(rng.uniform(0.40, 0.70))

    # Seated / stopped: a statue is forbidden, oncoming is forbidden.
    if v_ego < 0.45:
        key = "pulling"

    if key == "closing":
        dv = float(_FOLLOW_CLOSING_DV)
        if rng is not None:
            dv = float(rng.uniform(0.30, 0.44))
        v_other = min(v_ego - 0.12, v_ego - dv)
        v_other = max(0.55, v_other)
        dv = v_ego - v_other
        d_end = float(_FOLLOW_CLOSING_D_END)
        d0 = d_end + dv * T
        if rng is not None:
            d0 = d0 + float(rng.uniform(-0.30, 0.30))
        d0 = min(max(d0, float(_FOLLOW_CLOSING_D0_MIN)), float(_FOLLOW_CLOSING_D0_MAX))
        if d0 - dv * T < 1.20:
            dv = (d0 - 1.25) / T
            v_other = v_ego - dv
            v_other = max(0.55, min(v_other, v_ego - 0.12))
            dv = v_ego - v_other
        return SameDirFollowPlan(
            mode="closing", v_ego=v_ego, v_other=float(v_other),
            d0=float(d0), lat_offset=lat, duration_s=T,
        )

    if key == "hold":
        d0 = float(_FOLLOW_HOLD_D0)
        v_other = v_ego
        if rng is not None:
            d0 = float(rng.uniform(4.70, 5.80))
            # |Δv|·15 ≤ 0.27 m, inside the 0.5 m hold band.
            v_other = v_ego + float(rng.uniform(-0.018, 0.018))
        v_other = max(_FOLLOW_V_MIN, min(_FOLLOW_V_MAX, v_other))
        return SameDirFollowPlan(
            mode="hold", v_ego=v_ego, v_other=float(v_other),
            d0=float(d0), lat_offset=lat, duration_s=T,
        )

    d0 = float(_FOLLOW_PULL_D0)
    dv = float(_FOLLOW_PULL_DV)
    if rng is not None:
        d0 = float(rng.uniform(3.20, 4.20))
        dv = float(rng.uniform(0.32, 0.46))
    v_other = min(float(_FOLLOW_V_MAX), v_ego + dv)
    if v_other - v_ego < 0.16:
        v_other = v_ego + 0.22
    v_other = max(v_ego + 0.16, min(_FOLLOW_V_MAX, v_other))
    # Start close: at t=2 s they must still be a person, not a speck.
    if d0 + (v_other - v_ego) * 2.0 > 8.0:
        d0 = min(d0, 4.00)
    return SameDirFollowPlan(
        mode="pulling", v_ego=v_ego, v_other=float(v_other),
        d0=float(d0), lat_offset=lat, duration_s=T,
    )


def same_dir_follow_table(
    v_ego: float,
    *,
    duration_s: float = _FOLLOW_T,
    times: Sequence[float] = (0.0, 5.0, 10.0, 15.0),
    rng: Optional[random.Random] = None,
) -> list[dict[str, Any]]:
    """d(t) rows for closing / hold / pulling. Used by the self-test."""
    rows: list[dict[str, Any]] = []
    for mode in ("closing", "hold", "pulling"):
        plan = same_dir_follow_plan(v_ego, mode, duration_s=duration_s, rng=rng)
        gaps = [round(plan.gap(t), 4) for t in times]
        rows.append(
            {
                "mode": plan.mode,
                "v_ego": round(plan.v_ego, 4),
                "v_other": round(plan.v_other, 4),
                "d0": round(plan.d0, 4),
                "lat_offset": round(plan.lat_offset, 4),
                "times": [float(t) for t in times],
                "d": gaps,
            }
        )
    return rows


_EXTENT_S = {
    "vehicle": 2.45,
    "bicycle": 1.15,
    "pothole": 0.90,
    "threat_cube": 0.48,
    "person": 0.70,
    "tree": 0.55,
    "streetlamp": 0.22,
    "crater": 1.10,
    "broken_slab": 0.70,
    "debris": 0.65,
    "puddle": 1.00,
}
_EXTENT_LAT = {
    "vehicle": 1.05,
    "bicycle": 0.48,
    "pothole": 0.55,
    "threat_cube": 0.36,
    "person": 0.42,
    "tree": 0.55,
    "streetlamp": 0.22,
    "crater": 0.75,
    "broken_slab": 0.50,
    "debris": 0.45,
    "puddle": 0.70,
}


# ---------------------------------------------------------------------------
# Parsing / picking
# ---------------------------------------------------------------------------

def split_scenario_tokens(text: str) -> list[str]:
    """Split a CLI blob on commas, plus signs, or whitespace."""
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [p.strip().lower().replace("-", "_") for p in _TOKEN_SPLIT.split(raw)]
    return [p for p in parts if p]


def all_scenario_names(cfg: dict) -> tuple[str, ...]:
    sc = cfg["scenarios"]
    names: list[str] = []
    for key in (
        "safe_pool",
        "near_miss_pool",
        "critical_pool",
        "peripheral_safe",
        "peripheral_near",
        "peripheral_critical",
        "pair_pool",
        "crowd_pool",
    ):
        names.extend(list(sc.get(key) or ()))
    return tuple(dict.fromkeys(names))


def peripheral_pools(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Dedicated empty-center / side-threat catalog (not in ``--scenario auto``)."""
    sc = cfg["scenarios"]
    return (
        list(sc.get("peripheral_safe") or ()),
        list(sc.get("peripheral_near") or ()),
        list(sc.get("peripheral_critical") or ()),
    )


def canonical_name_set(cfg: dict) -> set[str]:
    return set(all_scenario_names(cfg))


def resolve_scenario_name(token: str, rng: random.Random, cfg: dict) -> str:
    """Map one token (alias or canonical) onto a handler key."""
    tok = str(token).strip().lower().replace("-", "_")
    if tok in ("auto", "random", ""):
        return pick_one_auto(rng, cfg)
    if tok == "critical":
        return str(rng.choice(list(cfg["scenarios"]["critical_pool"])))
    if tok in SCENARIO_ALIASES:
        return SCENARIO_ALIASES[tok]
    return tok


def pick_one_auto(rng: random.Random, cfg: dict) -> str:
    """``--scenario auto``: 40 / 30 / 30 then a name from that pool."""
    ratios = cfg["scenarios"]["ratios"]
    x = rng.random()
    if x < float(ratios["safe"]):
        pool = cfg["scenarios"]["safe_pool"]
    elif x < float(ratios["safe"]) + float(ratios["near_miss"]):
        pool = cfg["scenarios"]["near_miss_pool"]
    else:
        pool = cfg["scenarios"]["critical_pool"]
    return str(rng.choice(list(pool)))


def pick_scenarios(rng: random.Random, cfg: dict, requested: str) -> list[str]:
    """Resolve a CLI request into one or more canonical injector names.

    ``auto`` / ``random`` / empty → one draw from the 40/30/30 mix.
    ``jaywalker,car,pothole`` or ``jaywalker+car+pothole`` → three injectors.
    A bare ``auto`` inside a list is itself a random draw (so ``pothole,auto``
    is a pothole plus one extra event).
    """
    tokens = split_scenario_tokens(requested)
    if not tokens or tokens == ["auto"] or tokens == ["random"]:
        return [pick_one_auto(rng, cfg)]

    known = canonical_name_set(cfg)
    alias_keys = set(SCENARIO_ALIASES)
    out: list[str] = []
    for tok in tokens:
        name = resolve_scenario_name(tok, rng, cfg)
        if name not in known:
            extra = ", ".join(sorted(alias_keys))
            raise ValueError(
                f"unknown scenario {tok!r} (resolved to {name!r}). "
                f"Canonical names are the named injectors; short aliases: {extra}. "
                f"Use --list-scenarios."
            )
        out.append(name)
    if not out:
        return [pick_one_auto(rng, cfg)]
    return out


def pick_scenario(rng: random.Random, cfg: dict, requested: str) -> str:
    """Back-compat: first (or only) name from ``pick_scenarios``."""
    return pick_scenarios(rng, cfg, requested)[0]


def compose_slug(names: Sequence[str], max_len: int = 72) -> str:
    """Filesystem-safe compound id. Single names are unchanged."""
    clean = [str(n).strip() for n in names if str(n).strip()]
    if not clean:
        return "auto"
    if len(clean) == 1:
        return clean[0]
    slug = "__".join(clean)
    if len(slug) <= max_len:
        return slug
    return slug[: max(8, max_len - 4)].rstrip("_") + "_etc"


def compose_display(names: Sequence[str]) -> str:
    clean = [str(n).strip() for n in names if str(n).strip()]
    return "+".join(clean) if clean else "auto"


def sort_for_inject(names: Sequence[str]) -> list[str]:
    """Static → along-track → lateral. User order is preserved within a bucket."""
    indexed = list(enumerate(names))
    indexed.sort(key=lambda it: (_INJECT_PRIORITY.get(it[1], 15), it[0]))
    return [n for _i, n in indexed]


def is_noop(name: str) -> bool:
    return name in _NOOP


# ---------------------------------------------------------------------------
# Frenet occupancy
# ---------------------------------------------------------------------------

def extents_for(class_name: str, pad: float = 0.35) -> tuple[float, float]:
    """(half_s, half_lat) metres for a reservation capsule."""
    cls = str(class_name or "person")
    if cls.startswith("threat_"):
        cls = "threat_cube"
    hs = float(_EXTENT_S.get(cls, 0.70))
    hl = float(_EXTENT_LAT.get(cls, max(0.40, float(pad))))
    return hs, hl


def _lat_after(lat0: float, target: Optional[float], speed: float, t: float) -> float:
    if t <= 0.0:
        return lat0
    if target is None:
        return lat0 + speed * t
    if speed <= 1e-8:
        return lat0
    delta = float(target) - lat0
    travel = abs(speed) * t
    if travel >= abs(delta):
        return float(target)
    return lat0 + math.copysign(travel, delta)


@dataclass
class FrenetCapsule:
    """Predicted occupancy of one actor in ``(s, lateral)``."""

    s: float
    lat: float
    half_s: float
    half_lat: float
    ds: float = 0.0
    lat_speed: float = 0.0
    lat_target: Optional[float] = None
    turn_t: Optional[float] = None
    turn_dt: float = 0.85
    post_speed: Optional[float] = None
    post_lat_speed: float = 0.0
    post_lat_target: Optional[float] = None
    swerve_t: Optional[float] = None
    class_name: str = ""
    behavior: str = ""
    tag: str = ""

    def pose_at(self, t: float) -> tuple[float, float]:
        """Closed-form Frenet pose matching ``Actor.update`` piecewise rates."""
        t = max(0.0, float(t))
        ds = float(self.ds)
        s0 = float(self.s)
        lat0 = float(self.lat)

        # Swerve: no lateral motion until swerve_t.
        t_lat = t
        if self.swerve_t is not None:
            t_sw = float(self.swerve_t)
            if t < t_sw:
                t_lat = 0.0
            else:
                t_lat = t - t_sw

        if self.turn_t is None or self.post_speed is None or t < float(self.turn_t):
            s = s0 + ds * t
            lat = _lat_after(lat0, self.lat_target, abs(self.lat_speed), t_lat)
            return s, lat

        t_turn = float(self.turn_t)
        s = s0 + ds * t_turn + float(self.post_speed) * (t - t_turn)
        lat_at_turn = _lat_after(lat0, self.lat_target, abs(self.lat_speed), min(t_lat, t_turn))
        tgt = self.post_lat_target if self.post_lat_target is not None else self.lat_target
        lat = _lat_after(lat_at_turn, tgt, abs(self.post_lat_speed), max(0.0, t - t_turn))
        return s, lat


@dataclass
class Placement:
    s: float
    lat: float
    lat_target: Optional[float]


@dataclass
class ComposeSession:
    """Per-episode reservation board. Cheap: O(n_actors × samples × nudges) once."""

    names: list[str]
    s_lo: float
    s_hi: float
    nudge_s: float = 2.6
    max_nudges: int = 18
    horizon: float = 10.0
    dt_sample: float = 0.12
    clearance_s: float = 0.20
    clearance_lat: float = 0.18
    cross_stride: float = 4.0
    along_stride: float = 3.2
    static_stride: float = 2.4
    capsules: list[FrenetCapsule] = field(default_factory=list)
    current_key: str = ""
    _stagger: dict[str, int] = field(default_factory=dict)
    _first_cross_left: Optional[bool] = None

    @classmethod
    def from_cfg(
        cls,
        names: Sequence[str],
        *,
        s_lo: float,
        s_hi: float,
        cfg: Optional[dict] = None,
    ) -> "ComposeSession":
        block = (cfg or {}).get("scenarios", {}).get("compose", {}) or {}
        return cls(
            names=list(names),
            s_lo=float(s_lo),
            s_hi=float(s_hi),
            nudge_s=float(block.get("nudge_s_m", 2.6)),
            max_nudges=int(block.get("max_nudges", 18)),
            horizon=float(block.get("horizon_s", 15.0)),
            dt_sample=float(block.get("dt_sample", 0.12)),
            clearance_s=float(block.get("clearance_s", 0.20)),
            clearance_lat=float(block.get("clearance_lat", 0.18)),
            cross_stride=float(block.get("cross_stride_m", 4.0)),
            along_stride=float(block.get("along_stride_m", 3.2)),
            static_stride=float(block.get("static_stride_m", 2.4)),
        )

    def begin(self, key: str) -> None:
        self.current_key = str(key)

    def has_through_crosser(self) -> bool:
        return any(n in THROUGH_CROSSERS for n in self.names)

    def prefer_far_lane(self, far: bool) -> bool:
        """Keep a generic oncoming car out of a jaywalker's ribbon.

        Cut-in / swerve / near-miss-lane stay in the near lane; occupancy
        then staggers ``s`` so they do not occupy the same cell as a crosser.
        """
        if far:
            return True
        if self.current_key in NEAR_LANE_LOCKED:
            return False
        if self.has_through_crosser() and self.current_key in ("car_approaching", "car_pass_far"):
            return True
        return False

    def take_group_offset(self, group: str) -> float:
        n = int(self._stagger.get(group, 0))
        self._stagger[group] = n + 1
        stride = {
            "cross": self.cross_stride,
            "along": self.along_stride,
            "static": self.static_stride,
        }.get(group, 0.0)
        return float(n) * float(stride)

    def take_cross_layout(self, from_left: bool) -> tuple[float, bool]:
        """Depth stride + alternate entry side for a through-crosser."""
        n = int(self._stagger.get("cross", 0))
        self._stagger["cross"] = n + 1
        if n == 0 or self._first_cross_left is None:
            self._first_cross_left = bool(from_left)
            side = bool(from_left)
        elif n % 2:
            side = not self._first_cross_left
        else:
            side = bool(self._first_cross_left)
        return float(n) * self.cross_stride, side

    def seed_from_actors(self, actors: Iterable[Any], s0: float) -> None:
        """Reserve background traffic already in the camera-relevant band."""
        lo = float(s0) - 2.0
        hi = float(self.s_hi) + 4.0
        for actor in actors:
            try:
                s = float(getattr(actor, "s", 0.0))
            except (TypeError, ValueError):
                continue
            if s < lo or s > hi:
                continue
            cls = str(getattr(actor, "class_name", "") or "person")
            pad = float(getattr(actor, "corridor_pad", 0.35) or 0.35)
            hs, hl = extents_for(cls, pad)
            self.capsules.append(
                FrenetCapsule(
                    s=s,
                    lat=float(getattr(actor, "lateral", 0.0) or 0.0),
                    half_s=hs,
                    half_lat=hl,
                    ds=float(getattr(actor, "speed", 0.0) or 0.0),
                    lat_speed=abs(float(getattr(actor, "lat_speed", 0.0) or 0.0)),
                    lat_target=getattr(actor, "lat_target", None),
                    class_name=cls,
                    behavior=str(getattr(actor, "behavior", "") or ""),
                    tag="background",
                )
            )

    def _conflicts(self, cap: FrenetCapsule) -> bool:
        n = max(2, int(math.ceil(self.horizon / max(self.dt_sample, 0.05))) + 1)
        full = [self.horizon * i / (n - 1) for i in range(n)]
        hs_pad = self.clearance_s
        hl_pad = self.clearance_lat
        for other in self.capsules:
            need_s = cap.half_s + other.half_s + hs_pad
            need_lat = cap.half_lat + other.half_lat + hl_pad
            # Background Poisson traffic only blocks the spawn cell. A far-lane
            # cruiser would otherwise push every through-crosser to its own s.
            # Ground hazards (potholes) also only block spawn: walking past a
            # hole is realistic; treating the 5 s tube as solid shoved jaywalkers
            # ~10 m down the road.
            if (
                other.tag == "background"
                or cap.tag == "background"
                or _is_static_class(other.class_name)
                or _is_static_class(cap.class_name)
            ):
                times = (0.0,)
            else:
                times = full
            for t in times:
                s1, l1 = cap.pose_at(t)
                s2, l2 = other.pose_at(t)
                if abs(s1 - s2) < need_s and abs(l1 - l2) < need_lat:
                    return True
        return False

    def _try_commit(self, cap: FrenetCapsule) -> bool:
        if self._conflicts(cap):
            return False
        self.capsules.append(cap)
        return True

    def reserve(
        self,
        *,
        s: float,
        lat: float,
        half_s: float,
        half_lat: float,
        ds: float = 0.0,
        lat_speed: float = 0.0,
        lat_target: Optional[float] = None,
        class_name: str = "",
        behavior: str = "",
        allow_flip_lat: bool = False,
        allow_lane_flip: bool = False,
        turn_t: Optional[float] = None,
        turn_dt: float = 0.85,
        post_speed: Optional[float] = None,
        post_lat_speed: float = 0.0,
        post_lat_target: Optional[float] = None,
        swerve_t: Optional[float] = None,
        s_lo: Optional[float] = None,
        s_hi: Optional[float] = None,
    ) -> Placement:
        """Nudge ``(s, lat)`` until the predicted tube is free, then commit."""
        lo = float(self.s_lo if s_lo is None else s_lo)
        hi = float(self.s_hi if s_hi is None else s_hi)
        if hi <= lo + 0.5:
            hi = lo + 8.0

        def make(s_v: float, lat_v: float, tgt: Optional[float]) -> FrenetCapsule:
            s_c = min(max(lo, float(s_v)), hi)
            return FrenetCapsule(
                s=s_c,
                lat=float(lat_v),
                half_s=float(half_s),
                half_lat=float(half_lat),
                ds=float(ds),
                lat_speed=abs(float(lat_speed)),
                lat_target=tgt,
                turn_t=turn_t,
                turn_dt=float(turn_dt),
                post_speed=post_speed,
                post_lat_speed=abs(float(post_lat_speed)),
                post_lat_target=post_lat_target,
                swerve_t=swerve_t,
                class_name=class_name,
                behavior=behavior,
                tag=self.current_key,
            )

        layouts: list[tuple[float, Optional[float]]] = [(lat, lat_target)]
        if allow_flip_lat and lat_target is not None and abs(float(lat_target) - float(lat)) > 0.4:
            layouts.append((float(lat_target), float(lat)))
        if allow_lane_flip and abs(float(lat)) > 0.35:
            layouts.append((-float(lat), lat_target))

        # Prefer the requested (s, lat), then a heading/lane flip at that s,
        # then walk +s / −s. Flipping first keeps the intended depth band.
        candidates: list[tuple[float, float, Optional[float]]] = []
        s0 = float(s)
        for lat_v, tgt in layouts:
            candidates.append((s0, lat_v, tgt))
        for lat_v, tgt in layouts:
            s_try = s0 + self.nudge_s
            for _ in range(max(0, self.max_nudges - 1)):
                if s_try > hi + 0.05:
                    break
                candidates.append((s_try, lat_v, tgt))
                s_try += self.nudge_s
        for lat_v, tgt in layouts:
            s_try = s0 - self.nudge_s
            for _ in range(3):
                if s_try < lo - 0.05:
                    break
                candidates.append((s_try, lat_v, tgt))
                s_try -= self.nudge_s

        for s_v, lat_v, tgt in candidates:
            cap = make(s_v, lat_v, tgt)
            if self._try_commit(cap):
                return Placement(s=cap.s, lat=cap.lat, lat_target=cap.lat_target)

        cap = make(min(max(lo, float(s)), hi), lat, lat_target)
        self.capsules.append(cap)
        return Placement(s=cap.s, lat=cap.lat, lat_target=cap.lat_target)

    def try_reserve(
        self,
        *,
        s: float,
        lat: float,
        half_s: float,
        half_lat: float,
        ds: float = 0.0,
        lat_speed: float = 0.0,
        lat_target: Optional[float] = None,
        class_name: str = "",
        behavior: str = "",
        allow_flip_lat: bool = False,
        s_lo: Optional[float] = None,
        s_hi: Optional[float] = None,
    ) -> Optional[Placement]:
        """Like ``reserve`` but returns None instead of force-committing a clash."""
        before = len(self.capsules)
        placed = self.reserve(
            s=s,
            lat=lat,
            half_s=half_s,
            half_lat=half_lat,
            ds=ds,
            lat_speed=lat_speed,
            lat_target=lat_target,
            class_name=class_name,
            behavior=behavior,
            allow_flip_lat=allow_flip_lat,
            s_lo=s_lo,
            s_hi=s_hi,
        )
        if len(self.capsules) != before + 1:
            return placed
        cap = self.capsules[-1]
        # Force-commit path: last capsule still conflicts with earlier ones.
        probe = FrenetCapsule(
            s=cap.s,
            lat=cap.lat,
            half_s=cap.half_s,
            half_lat=cap.half_lat,
            ds=cap.ds,
            lat_speed=cap.lat_speed,
            lat_target=cap.lat_target,
            class_name=cap.class_name,
            behavior=cap.behavior,
            tag=cap.tag,
        )
        others = self.capsules[:-1]
        saved = self.capsules
        self.capsules = others
        clash = self._conflicts(probe)
        self.capsules = saved
        if clash:
            self.capsules.pop()
            return None
        return placed


# ---------------------------------------------------------------------------
# Static-shape pair + crowd occupancy (pair pack; not --groups)
# ---------------------------------------------------------------------------
#
# Static pair: one primitive, same kind/size/heading in both clips. Collision
# sits on the gait; the miss clip is a pure lateral shift (sideline, not a
# peel after diverge_s). Hit time t_hit ~ U(12.2, 14.5); lead = v_ego * t_hit.
# Ego speed is an input — never raised to fill 15 s.
#
# Crowd: 2–5 people, junction / cross / far head-on variety — no leader on
# ego gait at s≈s0+3 m. Occupancy is an AABB in Frenet (s, lateral) sampled
# over the 15 s horizon, then a Euclidean check
#
#     d_ij(t) = hypot(s_i(t) - s_j(t), lat_i(t) - lat_j(t))
#
# must stay ≥ CROWD_MIN_SEP_M. pose_at is the closed form of Actor.update
# for constant ds plus linear lat_target (no wander, no lat_ease). A matching
# Euler integrator is compared in the self-test so a pose_at drift cannot
# hide a phase-through.

SHAPE_PAIR_KINDS: tuple[str, ...] = (
    "cube", "sphere", "cylinder", "pyramid", "cone", "capsule", "lump",
)
STATIC_SHAPE_HIT_S = (12.2, 14.5)
STATIC_SHAPE_SIZE_M = (0.55, 0.95)
STATIC_SHAPE_MISS_CLEAR_M = 1.85
CROWD_MIN_SEP_M = 0.50
CROWD_PAIRWISE_FLOOR_M = 0.45
CROWD_N_PEOPLE = (2, 5)
_CROWD_ON_PATH_LAT_M = 0.60
_CROWD_WALK_LO_M_S = 0.75


def crowd_person_extents() -> tuple[float, float]:
    """Half-(s, lat) used for crowd occupancy capsules (shoulder + gait sway)."""
    return 0.42, 0.32


def is_static_shape_pair(name: str) -> bool:
    return str(name or "").strip().lower() in STATIC_SHAPE_PAIR_SCENARIOS


def is_parallel_person_pair(name: str) -> bool:
    return str(name or "").strip().lower() in PARALLEL_PERSON_PAIR_SCENARIOS


def is_crowd_pack_scenario(name: str) -> bool:
    return str(name or "").strip().lower() in CROWD_PACK_SCENARIOS


def pair_shares_rgb_prefix(name: str) -> bool:
    """RGB prefix cache is invalid when the object differs from t=0.

    Static-shape and parallel-person pairs move the actor laterally for the
    miss clip, so pixels differ from frame 0. Crowd clips are not pairs.
    """
    tok = str(name or "").strip().lower()
    return (
        not is_static_shape_pair(tok)
        and not is_crowd_pack_scenario(tok)
        and not is_parallel_person_pair(tok)
    )


def pair_clip_relpath(scenario: str, outcome: str, bg_id: str) -> str:
    """Clip path under ``sample_XXXX_<name>/``. Crowd files sit at that root."""
    if is_crowd_pack_scenario(scenario):
        return ""
    out = str(outcome or "collision").strip() or "collision"
    bg = str(bg_id or PAIR_RENDER_BG_ID).strip() or PAIR_RENDER_BG_ID
    return f"{out}/{bg}"


def crowd_pack_relpath() -> str:
    """Crowd files live in the sample folder: no ``collision/``, no ``bg_YY``."""
    return ""


@dataclass(frozen=True)
class PairsCrowdClip:
    """``--pairs`` rotator payload for family ``crowd``.

    Layout (user-confirmed)::

        crowd/sample_XXXX_<scenario>/preview.mp4
        …/spatial_overlay.mp4
        …/annotations/
        …/spatial_annotations/
        …/episode.json

    No ``bg_00`` wrap, no ``collision/`` / ``miss_*``, no empty ``bg_01``.
    """

    family: str
    scenario: str
    relpath: str
    outcomes: tuple[str, ...]
    bg_ids: tuple[str, ...]
    duration_s: float
    frames: int
    fps: int
    injector: str
    layout: str = "flat"

    def sample_dir(self, sample_id: int) -> str:
        slug = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(self.scenario)
        )
        slug = slug or "crowd"
        return f"sample_{int(sample_id):04d}_{slug}"

    def pack_relpath(self, sample_id: int) -> str:
        return f"{self.family}/{self.sample_dir(sample_id)}"

    def as_rotator_dict(self, sample_id: int = 0) -> dict[str, Any]:
        return {
            "family": self.family,
            "scenario": self.scenario,
            "relpath": self.relpath,
            "outcomes": list(self.outcomes),
            "bg_ids": list(self.bg_ids),
            "layout": self.layout,
            "duration_s": self.duration_s,
            "frames": self.frames,
            "fps": self.fps,
            "injector": self.injector,
            "sample_dir": self.sample_dir(sample_id),
            "pack_relpath": self.pack_relpath(sample_id),
        }


def pairs_family_crowd(
    rng: Optional[random.Random] = None,
    *,
    index: int = 0,
    prefer: Optional[str] = None,
    duration_s: float = 15.0,
    fps: float = 30.0,
) -> PairsCrowdClip:
    """Hook the ``--pairs`` rotator calls for family ``crowd``.

    Draws ``crowd`` (mixed junction / cross / head-on) or rare ``crowd_path``
    (far-ahead facade parallel only, no gait leader).
    ``index`` is accepted so the rotate loop can pass the sample slot.
    """
    rng = rng or random.Random(int(index) + 1)
    tok = str(prefer or "").strip().lower().replace("-", "_")
    if tok in ("path", "crowd_path"):
        name = "crowd_path"
    elif tok in ("crowd", "mixed", "crowds"):
        name = "crowd"
    else:
        name = "crowd" if rng.random() < 0.82 else "crowd_path"
    fps_i = max(1, int(round(float(fps))))
    span = max(3.0, float(duration_s))
    frames = int(round(span * fps_i))
    _ = index
    return PairsCrowdClip(
        family=PAIRS_FAMILY_CROWD,
        scenario=name,
        relpath=crowd_pack_relpath(),
        outcomes=(),
        bg_ids=(),
        duration_s=span,
        frames=frames,
        fps=fps_i,
        injector="_inject_crowd",
        layout="flat",
    )


def pairs_crowd_clip(
    rng: Optional[random.Random] = None,
    *,
    index: int = 0,
    prefer: Optional[str] = None,
    duration_s: float = 15.0,
    fps: float = 30.0,
) -> PairsCrowdClip:
    """Alias of ``pairs_family_crowd`` (rotator import either name)."""
    return pairs_family_crowd(
        rng, index=index, prefer=prefer, duration_s=duration_s, fps=fps,
    )


def static_shape_pair_relpaths(miss_side: str) -> dict[str, str | tuple[str, ...]]:
    """Confirmed pair-pack folders. Files only in ``bg_00``; ``bg_01`` is empty mkdir."""
    miss = "miss_left" if "left" in str(miss_side or "") else "miss_right"
    return {
        "collision_files": f"collision/{PAIR_RENDER_BG_ID}",
        "miss_files": f"{miss}/{PAIR_RENDER_BG_ID}",
        "empty": (f"collision/{PAIR_EMPTY_BG_ID}", f"{miss}/{PAIR_EMPTY_BG_ID}"),
    }


def static_shape_pair_miss_lat(
    gait_lat: float,
    side: str,
    *,
    clearance_m: float = STATIC_SHAPE_MISS_CLEAR_M,
    corridor_lim: float = 5.0,
) -> float:
    """Sideline lateral for a miss clip. Keeps the folder's left/right sign.

    Unlike ``miss_target_lat`` this never flips to the far lane: the folder
    name *is* the side of the object.
    """
    sign = -1.0 if "left" in str(side or "") else 1.0
    gait = float(gait_lat)
    lim = max(1.60, float(corridor_lim))
    pad = max(1.25, float(clearance_m))
    target = gait + sign * pad
    clipped = max(-lim, min(lim, target))
    if abs(clipped - gait) >= 1.20:
        return float(clipped)
    if sign >= 0.0:
        room = max(0.0, lim - gait)
    else:
        room = max(0.0, gait + lim)
    bound = lim if sign >= 0.0 else -lim
    if abs(bound - gait) >= 1.20:
        return float(bound)
    use = max(1.20, min(pad, room * 0.90 if room > 1.20 else room))
    out = max(-lim, min(lim, gait + sign * use))
    if abs(out - gait) < 1.20:
        out = max(-lim, min(lim, gait + sign * max(1.20, use)))
    return float(out)


@dataclass(frozen=True)
class StaticShapePairPlan:
    kind: str
    size: float
    t_hit: float
    v_ego: float
    lead_m: float
    gait_lat: float
    collision_lat: float
    miss_side: str
    miss_lat: float
    duration_s: float = 15.0

    def lat_for(self, outcome: str) -> float:
        name = str(outcome or "collision").strip().lower()
        if name in ("collision", "hit", "collide", ""):
            return float(self.collision_lat)
        return float(self.miss_lat)

    def cpa_m(self, outcome: str) -> float:
        return abs(self.lat_for(outcome) - float(self.gait_lat))


def static_shape_pair_plan(
    v_ego: float,
    gait_lat: float,
    *,
    outcome: str = "collision",
    miss_side: Optional[str] = None,
    rng: Optional[random.Random] = None,
    corridor_lim: float = 5.0,
    duration_s: float = 15.0,
    t_hit: Optional[float] = None,
    kind: Optional[str] = None,
    size: Optional[float] = None,
) -> StaticShapePairPlan:
    """Closed-form pair: lead = v_ego * t_hit, t_hit ≥ 12 s. Ego is not sped."""
    rng = rng or random.Random(0)
    v = max(0.0, float(v_ego))
    span = max(3.0, float(duration_s))
    lo, hi = STATIC_SHAPE_HIT_S
    t = float(t_hit) if t_hit is not None else rng.uniform(lo, hi)
    t = min(max(t, lo), min(hi, span - 0.40))
    lead = v * t if v >= 0.20 else 16.0
    kind_s = str(kind or rng.choice(SHAPE_PAIR_KINDS)).strip().lower()
    if kind_s not in SHAPE_PAIR_KINDS:
        kind_s = "cube"
    sz_lo, sz_hi = STATIC_SHAPE_SIZE_M
    sz = float(size) if size is not None else rng.uniform(sz_lo, sz_hi)
    sz = min(max(sz, sz_lo), 1.15)
    side = str(miss_side or "").strip().lower()
    if "left" in side:
        side = "miss_left"
    elif "right" in side:
        side = "miss_right"
    else:
        side = "miss_left" if rng.random() < 0.5 else "miss_right"
    gait = float(gait_lat)
    miss_lat = static_shape_pair_miss_lat(
        gait, side, clearance_m=STATIC_SHAPE_MISS_CLEAR_M, corridor_lim=corridor_lim,
    )
    if abs(miss_lat - gait) < 1.20:
        alt = "miss_right" if side == "miss_left" else "miss_left"
        lat_alt = static_shape_pair_miss_lat(
            gait, alt, clearance_m=STATIC_SHAPE_MISS_CLEAR_M, corridor_lim=corridor_lim,
        )
        if abs(lat_alt - gait) >= 1.20:
            side, miss_lat = alt, lat_alt
    plan = StaticShapePairPlan(
        kind=kind_s,
        size=float(sz),
        t_hit=float(t),
        v_ego=float(v),
        lead_m=float(lead),
        gait_lat=gait,
        collision_lat=gait,
        miss_side=side,
        miss_lat=float(miss_lat),
        duration_s=span,
    )
    # outcome is recorded by the caller; plan always holds both laterals.
    _ = outcome
    return plan


def crowd_min_pairwise_m(
    capsules: Sequence[FrenetCapsule],
    *,
    horizon: float = 15.0,
    dt: float = 0.05,
) -> float:
    """Min Frenet hypot(Δs, Δlat) over the horizon using closed-form pose_at."""
    caps = list(capsules)
    if len(caps) < 2:
        return float("inf")
    n = max(2, int(math.ceil(float(horizon) / max(float(dt), 1e-3))) + 1)
    mind = float("inf")
    for i in range(n):
        t = float(horizon) * i / (n - 1)
        poses = [c.pose_at(t) for c in caps]
        for a in range(len(poses)):
            sa, la = poses[a]
            for b in range(a + 1, len(poses)):
                sb, lb = poses[b]
                mind = min(mind, math.hypot(sa - sb, la - lb))
    return float(mind)


def crowd_euler_min_pairwise_m(
    capsules: Sequence[FrenetCapsule],
    *,
    horizon: float = 15.0,
    dt: float = 0.05,
) -> float:
    """Forward Euler of constant ds + linear lat_target (Actor.update subset)."""
    caps = list(capsules)
    if len(caps) < 2:
        return float("inf")
    states = [[float(c.s), float(c.lat)] for c in caps]
    mind = float("inf")
    nstep = int(math.ceil(float(horizon) / max(float(dt), 1e-3)))
    dt = float(dt)

    def _pair_min() -> float:
        local = float("inf")
        for a in range(len(states)):
            for b in range(a + 1, len(states)):
                local = min(
                    local,
                    math.hypot(states[a][0] - states[b][0], states[a][1] - states[b][1]),
                )
        return local

    mind = min(mind, _pair_min())
    for _ in range(nstep):
        for i, c in enumerate(caps):
            s, lat = states[i]
            s = s + float(c.ds) * dt
            if c.lat_target is None:
                states[i] = [s, lat]
                continue
            delta = float(c.lat_target) - lat
            step = abs(float(c.lat_speed)) * dt
            if abs(delta) <= step:
                lat = float(c.lat_target)
            else:
                lat = lat + math.copysign(step, delta)
            states[i] = [s, lat]
        mind = min(mind, _pair_min())
    return float(mind)


@dataclass(frozen=True)
class CrowdPersonSpec:
    s: float
    lat: float
    ds: float
    lat_speed: float
    lat_target: Optional[float]
    role: str
    heading_sign: float

    def on_path_ish(self, gait_lat: float, tol: float = _CROWD_ON_PATH_LAT_M) -> bool:
        if self.lat_target is not None:
            return False
        if float(self.ds) <= 0.20:
            return False
        return abs(float(self.lat) - float(gait_lat)) <= float(tol)


@dataclass(frozen=True)
class CrowdPeoplePlan:
    n: int
    mode: str
    duration_s: float
    v_ego: float
    gait_lat: float
    people: tuple[CrowdPersonSpec, ...]
    min_pairwise_m: float

    def capsules(self) -> list[FrenetCapsule]:
        hs, hl = crowd_person_extents()
        out: list[FrenetCapsule] = []
        for p in self.people:
            out.append(
                FrenetCapsule(
                    s=float(p.s),
                    lat=float(p.lat),
                    half_s=hs,
                    half_lat=hl,
                    ds=float(p.ds),
                    lat_speed=abs(float(p.lat_speed)),
                    lat_target=p.lat_target,
                    class_name="person",
                    behavior=p.role,
                    tag="crowd",
                )
            )
        return out

    def n_on_path(self) -> int:
        return sum(1 for p in self.people if p.on_path_ish(self.gait_lat))

    def n_same_way(self) -> int:
        """Walking +s on ego's sidewalk, including a slight facade offset."""
        gait = float(self.gait_lat)
        return sum(
            1
            for p in self.people
            if p.lat_target is None
            and float(p.ds) > 0.20
            and float(p.heading_sign) > 0.0
            and (p.on_path_ish(gait) or float(p.lat) * gait >= 0.0)
        )

    def n_off_axis(self) -> int:
        return int(self.n) - self.n_on_path()

    def heading_buckets(self) -> set[str]:
        return crowd_heading_buckets(self)


def crowd_motion_m_s(spec: CrowdPersonSpec) -> float:
    """Ground speed used by the walk cycle (along-track or lateral)."""
    if spec.lat_target is not None:
        return abs(float(spec.lat_speed))
    return abs(float(spec.ds))


def crowd_heading_buckets(plan: CrowdPeoplePlan) -> set[str]:
    """Coarse heading groups: along / oncoming / side / cross / junction."""
    out: set[str] = set()
    gait = float(plan.gait_lat)
    for p in plan.people:
        tok = str(p.role)
        if tok in ("junction", "near_miss"):
            out.add("junction" if tok == "junction" else "cross")
        elif p.lat_target is not None:
            out.add("cross")
        elif float(p.ds) < -0.20:
            out.add("oncoming" if tok != "intercept" else "intercept")
        elif (
            float(p.ds) > 0.20
            and float(p.heading_sign) > 0.0
            and (p.on_path_ish(gait) or float(p.lat) * float(gait) >= 0.0)
        ):
            out.add("along")
        else:
            out.add("side")
    return out


def _crowd_has_gait_leader(
    people: Sequence[CrowdPersonSpec],
    *,
    s0: float,
    gait_lat: float,
    ahead_m: float = 7.5,
) -> bool:
    """True if someone walks +s on ego gait just ahead of the camera."""
    s0 = float(s0)
    gait = float(gait_lat)
    for p in people:
        if float(p.ds) <= 0.20:
            continue
        if p.lat_target is not None:
            continue
        if float(p.s) > s0 + float(ahead_m):
            continue
        if p.on_path_ish(gait, tol=0.42):
            return True
    return False


_CROWD_MIXED_ROLES: tuple[str, ...] = (
    "oncoming", "cross", "side", "junction", "intercept", "near_miss",
)
_CROWD_PATH_ROLES: tuple[str, ...] = (
    "oncoming", "cross", "side", "junction", "parallel_facade",
)


def _crowd_roles(n: int, mode: str, rng: random.Random) -> list[str]:
    n = max(2, min(5, int(n)))
    if str(mode) == "path":
        core = list(_CROWD_PATH_ROLES[:4])
        rng.shuffle(core)
        roles: list[str] = [core[0], core[1]]
        pool = list(_CROWD_PATH_ROLES) + ["intercept", "near_miss"]
        rng.shuffle(pool)
        facade_used = False
        while len(roles) < n:
            pick = pool.pop() if pool else rng.choice(_CROWD_PATH_ROLES)
            if pick == "parallel_facade":
                if facade_used or n < 3:
                    pick = rng.choice(("cross", "oncoming", "junction"))
                else:
                    facade_used = True
            roles.append(pick)
        return roles[:n]
    core = list(_CROWD_MIXED_ROLES[:4])
    rng.shuffle(core)
    roles = [core[0], core[1]]
    pool = list(_CROWD_MIXED_ROLES)
    rng.shuffle(pool)
    while len(roles) < n:
        roles.append(pool.pop() if pool else rng.choice(_CROWD_MIXED_ROLES))
    return roles[:n]


def _crowd_s_band(role: str, s0: float) -> tuple[float, float]:
    """Keep titled people in the walking FOV (not a 30 m speck)."""
    s0 = float(s0)
    if role in ("same_path", "same_path_hold"):
        return s0 + 9.0, s0 + 22.0
    if role == "parallel_facade":
        return s0 + 9.5, s0 + 24.0
    if role in ("junction", "near_miss"):
        return s0 + 4.0, s0 + 16.5
    if role == "intercept":
        return s0 + 6.0, s0 + 28.0
    if role == "cross":
        return s0 + 5.5, s0 + 18.0
    if role == "side":
        return s0 + 3.5, s0 + 16.5
    return s0 + 8.0, s0 + 34.0


def _draw_crowd_person(
    role: str,
    *,
    s0: float,
    v_ego: float,
    gait_lat: float,
    lim: float,
    rng: random.Random,
    duration_s: float,
    slot: int = 0,
) -> CrowdPersonSpec:
    v_lo, v_hi = 0.85, 1.55
    v_ego = max(0.0, float(v_ego))
    gait = float(gait_lat)
    facade = math.copysign(1.0, gait if abs(gait) > 1e-6 else 1.0)
    k = max(0, int(slot))
    if role == "parallel_facade":
        d0 = rng.uniform(11.0, 17.5) + k * 2.60
        ds = min(v_hi, max(v_lo, v_ego + rng.uniform(0.00, 0.12)))
        lat = gait + facade * rng.uniform(0.92, 1.38)
        return CrowdPersonSpec(
            s=s0 + d0, lat=float(lat), ds=float(ds), lat_speed=0.0,
            lat_target=None, role=role, heading_sign=1.0,
        )
    if role == "same_path":
        d0 = rng.uniform(11.0, 16.0) + k * 2.40
        ds = min(v_hi, max(v_lo, v_ego + rng.uniform(0.00, 0.10)))
        return CrowdPersonSpec(
            s=s0 + d0, lat=gait + facade * rng.uniform(0.95, 1.25),
            ds=float(ds), lat_speed=0.0,
            lat_target=None, role=role, heading_sign=1.0,
        )
    if role == "intercept":
        ds = -rng.uniform(1.05, 1.55)
        tau = rng.uniform(4.0, 7.2) + k * 0.35
        s = s0 + (v_ego + abs(ds)) * tau
        lat = gait + facade * rng.uniform(-0.35, 0.35)
        if abs(lat) < 0.55:
            lat = facade * rng.uniform(0.65, 1.10)
        return CrowdPersonSpec(
            s=float(s), lat=float(lat), ds=float(ds), lat_speed=0.0,
            lat_target=None, role=role, heading_sign=-1.0,
        )
    if role == "junction":
        off = min(max(2.0, abs(float(lim)) * 0.82), 4.20)
        from_left = rng.random() < 0.5
        lat0 = -off if from_left else off
        s = s0 + rng.uniform(4.2, 9.5) + k * 1.60
        lat_t = float(gait) + facade * rng.uniform(-0.25, 0.45)
        speed = rng.uniform(0.88, 1.18)
        ds = rng.uniform(0.65, 1.05)
        return CrowdPersonSpec(
            s=float(s), lat=float(lat0), ds=float(ds), lat_speed=float(speed),
            lat_target=float(lat_t), role=role, heading_sign=1.0,
        )
    if role == "near_miss":
        s = s0 + rng.uniform(7.5, 13.5) + k * 2.20
        off = min(max(2.10, abs(float(lim)) * 0.88), 4.40)
        from_left = rng.random() < 0.5
        lat0 = -off if from_left else off
        speed = rng.uniform(0.62, 0.88)
        ds = rng.uniform(0.55, 0.90)
        return CrowdPersonSpec(
            s=float(s), lat=float(lat0), ds=float(ds), lat_speed=float(speed),
            lat_target=float(-lat0 * 0.85), role=role, heading_sign=1.0,
        )
    if role == "same_path_hold":
        d0 = rng.uniform(4.8, 7.2) + k * 2.40
        ds = min(v_hi, max(v_lo, v_ego + rng.uniform(0.02, 0.16)))
        lat = gait + facade * (rng.uniform(0.40, 0.52) + k * 0.22)
        return CrowdPersonSpec(
            s=s0 + d0, lat=float(lat), ds=float(ds), lat_speed=0.0,
            lat_target=None, role=role, heading_sign=1.0,
        )
    if role == "oncoming":
        ds = -rng.uniform(0.90, 1.40)
        tau = rng.uniform(6.5, 11.5)
        s = s0 + (v_ego + abs(ds)) * tau
        lat = -gait if abs(gait) > 0.40 else -facade * rng.uniform(1.6, 2.5)
        return CrowdPersonSpec(
            s=float(s), lat=float(lat), ds=float(ds), lat_speed=0.0,
            lat_target=None, role=role, heading_sign=-1.0,
        )
    if role == "side":
        sign = 1.0 if rng.random() < 0.55 else -1.0
        ds = sign * rng.uniform(0.90, 1.40)
        s = s0 + rng.uniform(4.0, 13.0) + k * 1.40
        lat = -gait if abs(gait) > 0.40 else -facade * rng.uniform(1.5, 2.4)
        return CrowdPersonSpec(
            s=float(s), lat=float(lat), ds=float(ds), lat_speed=0.0,
            lat_target=None, role=role, heading_sign=sign,
        )
    # Diagonal cut: walk +s while crossing so they are not a 15 s s-wall
    # that every same-path person would phase through.
    s = s0 + rng.uniform(6.5, 11.5) + k * 2.40
    off = min(max(1.70, abs(float(lim)) * 0.58), 3.50)
    from_left = rng.random() < 0.5
    lat0 = -off if from_left else off
    speed = rng.uniform(0.75, 1.10)
    ds = rng.uniform(0.70, 1.15)
    return CrowdPersonSpec(
        s=float(s), lat=float(lat0), ds=float(ds), lat_speed=float(speed),
        lat_target=float(-lat0), role="cross", heading_sign=1.0,
    )


def crowd_people_plan(
    v_ego: float,
    gait_lat: float,
    *,
    n: Optional[int] = None,
    mode: str = "mixed",
    rng: Optional[random.Random] = None,
    s0: float = 3.0,
    corridor_lim: float = 5.0,
    duration_s: float = 15.0,
    min_sep_m: float = CROWD_MIN_SEP_M,
) -> CrowdPeoplePlan:
    """2–5 people with Frenet occupancy + integrator separation.

    ``mode='mixed'`` (``crowd``): junction / cross / far head-on mix; no
    +s leader on ego gait near the camera. ``mode='path'`` (``crowd_path``):
    same variety with at most one far-ahead facade-parallel walker.
    Pairwise hypot(Δs, Δlat) stays ≥ ``min_sep_m`` (floor 0.45 m) over 15 s.
    """
    rng = rng or random.Random(0)
    key = "path" if str(mode).strip().lower() in ("path", "crowd_path") else "mixed"
    if n is None:
        n = rng.randint(CROWD_N_PEOPLE[0], CROWD_N_PEOPLE[1])
    n = max(2, min(5, int(n)))
    v_ego = max(0.0, float(v_ego))
    gait = float(gait_lat)
    s0 = float(s0)
    lim = max(1.80, float(corridor_lim))
    span = max(3.0, float(duration_s))
    need = max(CROWD_PAIRWISE_FLOOR_M, float(min_sep_m))
    hs, hl = crowd_person_extents()
    role_slot = {
        "same_path": 0, "same_path_hold": 0, "parallel_facade": 0,
        "oncoming": 0, "intercept": 0, "side": 0, "cross": 0,
        "junction": 0, "near_miss": 0,
    }

    last: Optional[CrowdPeoplePlan] = None
    for attempt in range(120):
        n_try = n if attempt < 100 else max(2, n - (attempt - 100) // 10)
        sess = ComposeSession(
            names=["crowd"],
            s_lo=s0 + 1.4,
            s_hi=s0 + 34.0,
            nudge_s=1.35,
            max_nudges=16,
            horizon=span,
            dt_sample=0.08,
            clearance_s=0.22,
            clearance_lat=0.20,
        )
        sess.begin("crowd")
        sess.capsules.append(
            FrenetCapsule(
                s=s0,
                lat=gait,
                half_s=0.45,
                half_lat=0.32,
                ds=v_ego,
                class_name="person",
                behavior="ego",
                tag="ego",
            )
        )
        people: list[CrowdPersonSpec] = []
        ok = True
        slots = dict(role_slot)
        for role in _crowd_roles(n_try, key, rng):
            slot = int(slots.get(role, 0))
            slots[role] = slot + 1
            spec = _draw_crowd_person(
                role, s0=s0, v_ego=v_ego, gait_lat=gait, lim=lim,
                rng=rng, duration_s=span, slot=slot,
            )
            lo, hi = _crowd_s_band(role, s0)
            placed = sess.try_reserve(
                s=spec.s,
                lat=spec.lat,
                half_s=hs,
                half_lat=hl,
                ds=spec.ds,
                lat_speed=spec.lat_speed,
                lat_target=spec.lat_target,
                class_name="person",
                behavior=role,
                allow_flip_lat=spec.lat_target is not None,
                s_lo=lo,
                s_hi=hi,
            )
            if placed is None:
                ok = False
                break
            person = CrowdPersonSpec(
                s=float(placed.s),
                lat=float(placed.lat),
                ds=float(spec.ds),
                lat_speed=float(spec.lat_speed),
                lat_target=placed.lat_target,
                role=spec.role,
                heading_sign=float(spec.heading_sign),
            )
            if crowd_motion_m_s(person) < _CROWD_WALK_LO_M_S - 1e-6:
                ok = False
                break
            people.append(person)
        if not ok or len(people) < 2:
            continue
        crowd_caps = [c for c in sess.capsules if c.tag not in ("ego", "background")]
        dmin = crowd_min_pairwise_m(crowd_caps, horizon=span, dt=0.05)
        deuler = crowd_euler_min_pairwise_m(crowd_caps, horizon=span, dt=0.05)
        if min(dmin, deuler) + 1e-6 < need:
            continue
        plan = CrowdPeoplePlan(
            n=len(people),
            mode=key,
            duration_s=span,
            v_ego=v_ego,
            gait_lat=gait,
            people=tuple(people),
            min_pairwise_m=float(min(dmin, deuler)),
        )
        last = plan
        if _crowd_has_gait_leader(plan.people, s0=s0, gait_lat=gait):
            continue
        if key == "mixed" and plan.n_on_path() > 0:
            continue
        if key == "mixed" and plan.n_off_axis() < 1:
            continue
        if n_try >= 3 and plan.n_off_axis() < 1:
            continue
        if key == "mixed" and n_try >= 3 and len(plan.heading_buckets()) < 2:
            continue
        if key == "mixed" and n_try >= 2 and "oncoming" not in plan.heading_buckets():
            if "intercept" not in plan.heading_buckets():
                continue
        if key == "path" and plan.n_same_way() > 1:
            continue
        if key == "path" and n_try >= 3 and plan.n_off_axis() < n_try - 1:
            continue
        return plan
    if last is not None:
        return last
    # Deterministic fallback: far head-on + junction cross (no gait leader).
    ahead = CrowdPersonSpec(
        s=s0 + 14.0, lat=-gait if abs(gait) > 0.4 else -2.0,
        ds=-1.15, lat_speed=0.0, lat_target=None, role="oncoming",
        heading_sign=-1.0,
    )
    off = min(max(2.0, abs(lim) * 0.85), 4.0)
    other = CrowdPersonSpec(
        s=s0 + 7.0, lat=-off, ds=0.80, lat_speed=0.95,
        lat_target=float(off * 0.9), role="junction", heading_sign=1.0,
    )
    caps = [
        FrenetCapsule(
            s=p.s, lat=p.lat, half_s=0.55, half_lat=0.38, ds=p.ds,
            class_name="person", behavior=p.role, tag="crowd",
        )
        for p in (ahead, other)
    ]
    return CrowdPeoplePlan(
        n=2, mode=key, duration_s=span, v_ego=v_ego, gait_lat=gait,
        people=(ahead, other),
        min_pairwise_m=crowd_min_pairwise_m(caps, horizon=span, dt=0.05),
    )


def scenario_request_from_tokens(
    scenario_flags: Optional[Sequence[str]],
    scenarios_csv: Optional[str],
) -> str:
    """Join ``--scenario`` repeats and ``--scenarios`` into one request string."""
    chunks: list[str] = []
    if scenarios_csv:
        chunks.append(str(scenarios_csv))
    if scenario_flags:
        chunks.extend(str(x) for x in scenario_flags if str(x).strip())
    if not chunks:
        return "auto"
    return ",".join(chunks)


# ---------------------------------------------------------------------------
# Self-test (system Python; no Blender)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rng = random.Random(0)
    cfg = {
        "scenarios": {
            "ratios": {"safe": 0.40, "near_miss": 0.30, "critical": 0.30},
            "safe_pool": ("safe_walk", "car_approaching", "pothole_offset"),
            "near_miss_pool": ("jaywalker_from_left",),
            "critical_pool": ("jaywalker", "pothole_on_path", "sudden_stop"),
        }
    }
    assert split_scenario_tokens("jaywalker, car, pothole") == [
        "jaywalker",
        "car",
        "pothole",
    ]
    assert split_scenario_tokens("jaywalker+car_approaching+pothole_on_path")[0] == "jaywalker"
    names = pick_scenarios(rng, cfg, "jaywalker,car,pothole")
    assert names == ["jaywalker", "car_approaching", "pothole_on_path"], names
    assert compose_slug(names) == "jaywalker__car_approaching__pothole_on_path"
    assert pick_scenarios(rng, cfg, "auto")[0] in canonical_name_set(cfg)
    ordered = sort_for_inject(names)
    assert ordered[0] == "pothole_on_path"
    assert ordered[-1] == "jaywalker"

    sess = ComposeSession(names=names, s_lo=6.0, s_hi=28.0, dt_sample=0.10)
    a = sess.reserve(s=8.0, lat=0.0, half_s=0.9, half_lat=0.55, class_name="pothole")
    b = sess.reserve(s=8.0, lat=0.0, half_s=0.7, half_lat=0.42, class_name="person")
    assert abs(b.s - a.s) >= 1.5, (a, b)

    # Through-crosser vs oncoming car: same s, overlapping lat at t=0 must nudge.
    sess2 = ComposeSession(names=["jaywalker", "car_approaching"], s_lo=6.0, s_hi=32.0)
    car = sess2.reserve(
        s=22.0, lat=1.7, half_s=2.45, half_lat=1.05, ds=-6.0, class_name="vehicle",
    )
    ped = sess2.reserve(
        s=8.0, lat=-4.0, half_s=0.7, half_lat=0.42, ds=0.0,
        lat_speed=1.2, lat_target=4.0, class_name="person", allow_flip_lat=True,
    )
    # Predicted tubes must not overlap at the committed poses.
    cap_car = sess2.capsules[0]
    cap_ped = sess2.capsules[1]
    conflict = False
    for i in range(41):
        t = i * 0.12
        s1, l1 = cap_car.pose_at(t)
        s2, l2 = cap_ped.pose_at(t)
        if abs(s1 - s2) < cap_car.half_s + cap_ped.half_s + 0.2 and abs(l1 - l2) < cap_car.half_lat + cap_ped.half_lat + 0.18:
            conflict = True
            break
    assert not conflict, (car, ped, "predicted overlap after reserve")

    d0, side0 = sess.take_cross_layout(True)
    d1, side1 = sess.take_cross_layout(True)
    assert d0 == 0.0 and side0 is True
    assert d1 == sess.cross_stride and side1 is False

    from config import get_config

    live = get_config()
    catalog = all_scenario_names(live)
    assert "shape_head_on" in catalog
    assert "cube_on_path" in catalog
    assert "shapes_on_path" in catalog
    assert "crossing_street" in catalog
    assert "crossing_car_side" in catalog
    assert "group_crossing" in catalog
    assert "scooter_from_sidewalk" in catalog
    assert "parked_car_door" in catalog
    assert "cyclist_overtake" in catalog
    assert "same_dir_closing" in catalog
    assert "same_dir_hold" in catalog
    assert "same_dir_pulling" in catalog
    from outcome_groups import FORKABLE_SCENARIOS
    assert "same_dir_closing" not in FORKABLE_SCENARIOS
    assert "same_dir_hold" not in FORKABLE_SCENARIOS
    assert "same_dir_pulling" not in FORKABLE_SCENARIOS
    assert resolve_scenario_name("follow_closing", random.Random(1), live) == "same_dir_closing"
    assert resolve_scenario_name("follow_hold", random.Random(1), live) == "same_dir_hold"
    assert resolve_scenario_name("follow_pulling", random.Random(1), live) == "same_dir_pulling"
    times = (0.0, 5.0, 10.0, 15.0)
    print("same-dir follow d(t)  [along-path gap, metres]")
    for v_ego in (0.80, 1.35, 1.90):
        print(f"  v_ego={v_ego:.2f} m/s")
        for row in same_dir_follow_table(v_ego, times=times):
            d = row["d"]
            print(
                f"    {row['mode']:8s}  v_other={row['v_other']:.3f}  "
                f"d0={row['d0']:.2f}  d(0/5/10/15)={d[0]:.2f}/{d[1]:.2f}/{d[2]:.2f}/{d[3]:.2f}"
            )
            assert row["v_other"] > 0.0
            if row["mode"] == "closing":
                assert row["v_other"] < v_ego - 0.10
                assert d[0] > d[1] > d[2] > d[3]
                assert d[3] >= 1.15
                assert d[0] >= 4.50
            elif row["mode"] == "hold":
                assert abs(row["v_other"] - v_ego) <= 0.03
                assert all(abs(x - d[0]) <= 0.50 for x in d)
                assert 4.40 <= d[0] <= 6.20
            else:
                assert row["v_other"] > v_ego + 0.15
                assert d[0] < d[1] < d[2] < d[3]
                assert d[0] >= 3.00
                assert d[0] + (row["v_other"] - v_ego) * 2.0 <= 8.50
    # RNG jitter must keep the inequalities.
    jitter = same_dir_follow_table(1.35, rng=random.Random(7))
    jc, jh, jp = jitter
    assert jc["d"][0] > jc["d"][-1] >= 1.15
    assert all(abs(x - jh["d"][0]) <= 0.50 for x in jh["d"])
    assert jp["d"][0] < jp["d"][-1]
    assert "backing_vehicle" in catalog
    assert "crossing_head_on" in catalog
    assert "periph_empty" in catalog
    assert "periph_car_turn" in catalog
    assert "periph_ped_cut" in catalog
    assert "tree_on_path" in catalog
    assert "lamp_on_path" in catalog
    assert "hasty_look" in catalog
    assert "parallel_person" in catalog
    assert resolve_scenario_name("pair_person", random.Random(1), live) == "parallel_person"
    assert resolve_scenario_name("parallel_pair", random.Random(1), live) == "parallel_person"
    assert is_parallel_person_pair("parallel_person")
    assert not is_parallel_person_pair("jaywalker")
    assert pair_shares_rgb_prefix("parallel_person") is False
    assert pair_shares_rgb_prefix("jaywalker") is True
    assert set(CLEAR_CENTER_SCENARIOS).issubset(set(catalog))
    assert resolve_scenario_name("side_cut", random.Random(1), live) == "periph_ped_cut"
    assert resolve_scenario_name("side_turn", random.Random(1), live) == "periph_car_turn"
    assert resolve_scenario_name("cross", random.Random(1), live) == "crossing_street"
    assert resolve_scenario_name("scooter", random.Random(1), live) == "scooter_from_sidewalk"
    assert resolve_scenario_name("shapes", random.Random(1), live) == "shapes_on_path"
    assert resolve_scenario_name("sphere", random.Random(1), live) == "shape_head_on"
    assert extents_for("threat_lump") == extents_for("threat_cube")
    assert extents_for("threat_pyramid")[0] > 0.0

    assert "static_shape" in catalog
    assert resolve_scenario_name("shape_pair", random.Random(1), live) == "static_shape"
    assert "static_shape" not in FORKABLE_SCENARIOS
    auto_mix = set(live["scenarios"]["safe_pool"]) | set(
        live["scenarios"]["near_miss_pool"]
    ) | set(live["scenarios"]["critical_pool"])
    assert "static_shape" not in auto_mix
    assert "parallel_person" not in auto_mix
    assert "parallel_person" not in FORKABLE_SCENARIOS
    paths = static_shape_pair_relpaths("miss_left")
    assert paths["collision_files"] == "collision/bg_00"
    assert paths["miss_files"] == "miss_left/bg_00"
    assert "collision/bg_01" in paths["empty"]
    assert pair_shares_rgb_prefix("static_shape") is False

    print("static-shape pair  [t_hit, lead=v_ego*t_hit, sideline CPA]")
    for v_ego in (0.80, 1.35, 1.90):
        for side in ("miss_left", "miss_right"):
            for seed in (0, 11, 29):
                plan = static_shape_pair_plan(
                    v_ego, -2.10, miss_side=side, rng=random.Random(seed),
                    corridor_lim=5.0,
                )
                assert plan.t_hit >= 12.2, plan
                assert plan.t_hit <= 14.5, plan
                assert abs(plan.lead_m - plan.v_ego * plan.t_hit) < 1e-6, plan
                assert plan.kind in SHAPE_PAIR_KINDS
                assert plan.size >= 0.55, plan.size
                assert plan.cpa_m("collision") < 0.20
                cpa_miss = plan.cpa_m(side)
                ego_r, obj_r = 0.30, 0.5 * plan.size
                assert cpa_miss >= 1.20, (cpa_miss, plan)
                assert cpa_miss > ego_r + obj_r + 0.15, (cpa_miss, plan.size)
                same = static_shape_pair_plan(
                    v_ego, -2.10, miss_side=side, rng=random.Random(seed),
                    corridor_lim=5.0,
                )
                assert same.kind == plan.kind and abs(same.size - plan.size) < 1e-9
                assert abs(same.lead_m - plan.lead_m) < 1e-9
                assert abs(same.lat_for("collision") - plan.lat_for(side)) > 1.0
    from spatial_threat import object_threat_score
    cam = (0.0, 0.0, 1.6)
    heading = (0.0, 1.0, 0.0)
    walk = 1.35
    s_on = object_threat_score(
        cam, (0.0, 2.2, 0.40), (0.0, 0.0, 0.0), heading, walk, 0.36, 0.36,
        path_s=2.2, path_lat=0.0, class_name="threat_cube",
    )
    s_near = object_threat_score(
        cam, (0.0, 1.4, 0.40), (0.0, 0.0, 0.0), heading, walk, 0.36, 0.36,
        path_s=1.4, path_lat=0.0, class_name="threat_cube",
    )
    s_side = object_threat_score(
        cam, (1.85, 2.2, 0.40), (0.0, 0.0, 0.0), heading, walk, 0.36, 0.36,
        path_s=2.2, path_lat=1.85, class_name="threat_cube",
    )
    s_behind = object_threat_score(
        cam, (0.0, 2.2, 0.40), (0.0, 0.0, 0.0), heading, walk, 0.36, 0.36,
        path_s=2.2, path_lat=0.0, class_name="threat_cube", behind_camera=True,
    )
    assert s_on > 0.20, s_on
    assert s_near > s_on, (s_near, s_on)
    assert s_side < 0.12, s_side
    assert s_behind < 0.02, s_behind

    _self_test_crowd(live, auto_mix)

    print("scenario_compose self-test ok")


def _self_test_crowd(live: dict, auto_mix: set[str]) -> None:
    """Count 2–5, no phase-through, mixed headings n≥3. No Blender."""
    from outcome_groups import FORKABLE_SCENARIOS

    catalog = all_scenario_names(live)
    assert "crowd" in catalog and "crowd_path" in catalog
    assert resolve_scenario_name("crowds", random.Random(1), live) == "crowd"
    assert resolve_scenario_name("crowd_path", random.Random(1), live) == "crowd_path"
    assert "crowd" not in FORKABLE_SCENARIOS
    assert "crowd_path" not in FORKABLE_SCENARIOS
    assert "crowd" not in auto_mix and "crowd_path" not in auto_mix
    assert pair_clip_relpath("crowd", "collision", "bg_00") == ""
    assert pair_clip_relpath("crowd_path", "miss_left", "bg_01") == ""
    assert crowd_pack_relpath() == ""
    assert pair_shares_rgb_prefix("crowd") is False
    clip = pairs_family_crowd(random.Random(3), index=4)
    assert clip.family == PAIRS_FAMILY_CROWD
    assert clip.scenario in CROWD_PACK_SCENARIOS
    assert clip.relpath == ""
    assert clip.outcomes == ()
    assert clip.bg_ids == ()
    assert clip.layout == "flat"
    assert clip.frames == 450 and clip.fps == 30
    assert clip.duration_s == 15.0
    assert clip.injector == "_inject_crowd"
    assert clip.pack_relpath(7) == f"crowd/sample_0007_{clip.scenario}"
    path_clip = pairs_family_crowd(prefer="crowd_path")
    assert path_clip.scenario == "crowd_path"
    alias = pairs_crowd_clip(random.Random(3), index=4)
    assert alias.scenario == clip.scenario

    # Occupancy AABB is conservative; hypot is the mesh-phase check.
    clash = [
        FrenetCapsule(s=10.0, lat=-2.0, half_s=0.55, half_lat=0.38, ds=1.2, tag="crowd"),
        FrenetCapsule(s=10.0, lat=-2.0, half_s=0.55, half_lat=0.38, ds=1.1, tag="crowd"),
    ]
    assert crowd_min_pairwise_m(clash, horizon=15.0, dt=0.05) < 0.10
    assert crowd_euler_min_pairwise_m(clash, horizon=15.0, dt=0.05) < 0.10

    print("crowd occupancy  [n, headings, min hypot over 15 s]")
    floor = CROWD_PAIRWISE_FLOOR_M
    for n in (2, 3, 4, 5):
        for mode in ("mixed", "path"):
            for seed in (0, 8, 19, 41, 77):
                plan = crowd_people_plan(
                    1.35, -2.10, n=n, mode=mode, rng=random.Random(seed),
                    s0=3.0, corridor_lim=5.0, duration_s=15.0,
                    min_sep_m=CROWD_MIN_SEP_M,
                )
                assert 2 <= plan.n <= 5, plan
                assert plan.n == n, (mode, n, seed, plan.n)
                assert not _crowd_has_gait_leader(
                    plan.people, s0=3.0, gait_lat=plan.gait_lat,
                ), (mode, n, seed, plan)
                caps = plan.capsules()
                assert len(caps) == plan.n
                d_closed = crowd_min_pairwise_m(caps, horizon=15.0, dt=0.05)
                d_euler = crowd_euler_min_pairwise_m(caps, horizon=15.0, dt=0.05)
                assert d_closed + 1e-6 >= floor, (mode, n, seed, d_closed)
                assert d_euler + 1e-6 >= floor, (mode, n, seed, d_euler)
                assert abs(d_closed - d_euler) < 0.12, (d_closed, d_euler)
                assert plan.min_pairwise_m + 1e-6 >= floor
                for person in plan.people:
                    assert crowd_motion_m_s(person) >= _CROWD_WALK_LO_M_S - 1e-6, person
                    assert person.role != "statue"
                buckets = plan.heading_buckets()
                if mode == "mixed":
                    assert plan.n_on_path() == 0, (n, seed, plan)
                    assert plan.n_off_axis() >= 1, (n, seed, plan)
                    if n >= 3:
                        assert len(buckets) >= 2, (n, seed, buckets)
                    assert buckets & {"oncoming", "intercept", "cross", "junction"}
                if mode == "path":
                    assert plan.n_on_path() == 0, (n, seed, plan)
                    assert plan.n_same_way() <= 1, (n, seed, plan)
                    if n >= 3:
                        assert plan.n_off_axis() >= 2, (n, seed, plan)
                print(
                    f"  mode={mode:5s} n={plan.n} seed={seed:2d}  "
                    f"on={plan.n_on_path()} off={plan.n_off_axis()}  "
                    f"heads={sorted(buckets)}  "
                    f"dmin={min(d_closed, d_euler):.2f} m"
                )

    audit = crowd_people_plan(1.35, -2.10, n=4, mode="mixed", rng=random.Random(69))
    assert audit.n == 4 and audit.n_on_path() == 0
    assert not _crowd_has_gait_leader(audit.people, s0=3.0, gait_lat=audit.gait_lat)
    assert len(audit.heading_buckets()) >= 2

    from spatial_threat import empty_grid, object_threat_score, splat_object
    cam = (0.0, 0.0, 1.6)
    heading = (0.0, 1.0, 0.0)
    walk = 1.20
    s_behind = object_threat_score(
        cam, (0.25, -1.1, 1.6), (0.0, 1.0, 0.0), heading, walk, 0.28, 0.38,
        class_name="person", behind_camera=True,
    )
    s_passed = object_threat_score(
        cam, (0.20, -0.90, 1.6), (0.0, 1.0, 0.0), heading, walk, 0.28, 0.38,
        class_name="person",
    )
    s_close = object_threat_score(
        cam, (0.20, 3.2, 1.6), (0.0, 0.75, 0.0), heading, walk, 0.28, 0.38,
        class_name="person",
    )
    s_hold = object_threat_score(
        cam, (0.30, 4.0, 1.6), (0.0, walk, 0.0), heading, walk, 0.28, 0.38,
        class_name="person",
    )
    assert s_behind < 0.02, s_behind
    assert s_passed < 0.02, s_passed
    assert s_hold < 0.05, s_hold
    assert s_close > s_hold + 0.15, (s_close, s_hold)
    g_off = empty_grid(3)
    splat_object(
        g_off,
        p_cam=cam, p_obj=(0.25, 3.0, 1.6), v_obj=(0.0, 0.75, 0.0),
        heading=heading, walk_speed=walk, half_s=0.28, half_lat=0.38,
        bbox=(0.0, 0.0, 800.0, 1080.0), res_x=1920, res_y=1080,
        boxes=[], class_name="person",
    )
    assert max(max(row) for row in g_off) < 0.02, g_off
    print("crowd self-test ok")


if __name__ == "__main__":
    _self_test()
