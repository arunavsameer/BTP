"""Compound scenario composition: CLI tokens, aliases, Frenet occupancy.

Injector bodies stay in ``world_generator.py``. This module is pure Python
(no ``bpy``) so the occupancy math can be unit-tested with system Python.

Spatial model
-------------
Actors live in the road Frenet frame ``(s, lateral)``. A *capsule* is an
axis-aligned rectangle in that plane (half-extents along-track and across)
that moves with the same piecewise-linear ``(ds/dt, d(lateral)/dt)`` the
integrator uses. Two capsules *conflict* if those rectangles overlap at any
sampled time in the first few seconds — the window that is actually on camera.

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
    }
)

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
}

_NOOP = frozenset({"safe_walk", "empty_street", "periph_empty"})
_STATIC_CLASSES = frozenset({
    "pothole", "crater", "broken_slab", "debris", "tree",
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
    "parked_car_opposite": 1,
    "empty_street": 2,
    "safe_walk": 2,
    "oncoming_pedestrian": 10,
    "parallel_pedestrian": 10,
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
}

_TOKEN_SPLIT = re.compile(r"[,+\s]+")

_EXTENT_S = {
    "vehicle": 2.45,
    "bicycle": 1.15,
    "pothole": 0.90,
    "threat_cube": 0.48,
    "person": 0.70,
    "tree": 0.55,
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
    max_nudges: int = 14
    horizon: float = 5.0
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
            max_nudges=int(block.get("max_nudges", 14)),
            horizon=float(block.get("horizon_s", 5.0)),
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
    assert "backing_vehicle" in catalog
    assert "crossing_head_on" in catalog
    assert "periph_empty" in catalog
    assert "periph_car_turn" in catalog
    assert "periph_ped_cut" in catalog
    assert set(CLEAR_CENTER_SCENARIOS).issubset(set(catalog))
    assert resolve_scenario_name("side_cut", random.Random(1), live) == "periph_ped_cut"
    assert resolve_scenario_name("side_turn", random.Random(1), live) == "periph_car_turn"
    assert resolve_scenario_name("cross", random.Random(1), live) == "crossing_street"
    assert resolve_scenario_name("scooter", random.Random(1), live) == "scooter_from_sidewalk"
    assert resolve_scenario_name("shapes", random.Random(1), live) == "shapes_on_path"
    assert resolve_scenario_name("sphere", random.Random(1), live) == "shape_head_on"
    assert extents_for("threat_lump") == extents_for("threat_cube")
    assert extents_for("threat_pyramid")[0] > 0.0

    print("scenario_compose self-test ok")


if __name__ == "__main__":
    _self_test()
