"""Pair pack: rotating parallel-person / static-shape / crowd layout.

A *sample index* ``i`` emits three family folders in that order so a cancel
leaves roughly equal counts:

    parallel_person/sample_XXXX_parallel_person/{collision,miss_*}/bg_00/  # files
                                                    .../bg_01/            # empty mkdir
    static_shape/sample_XXXX_static_shape/{collision,miss_*}/bg_00|bg_01
    crowd/sample_XXXX_crowd/   # files here; no bg wrap, no collision/miss

``--n`` is per family. ``--bg-variants`` is fake: only ``bg_00`` is rendered;
``bg_01`` is always created empty. One miss side per pair (50/50 left/right).

Parallel-person clips share world, ego, heading, and ``v_other``. Only
lateral changes. Pixels differ from t=0 — do **not** concat-copy a groups
prefix. Static-shape / crowd injectors belong to the sibling workstream.

Pure Python, no ``bpy``. ``python pair_pack.py`` runs the self-test.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from outcome_groups import (
    MISS_SIDES,
    bg_id_for,
    pick_miss_side,
    sample_dir_name,
)
from scenario_compose import (
    CROWD_PACK_SCENARIOS,
    PAIR_EMPTY_BG_ID,
    PAIR_RENDER_BG_ID,
    STATIC_SHAPE_PAIR_SCENARIOS,
    is_crowd_pack_scenario,
    pair_clip_relpath,
    pairs_family_crowd,
    static_shape_pair_relpaths,
)


PAIR_FAMILIES: tuple[str, ...] = ("parallel_person", "static_shape", "crowd")
FAMILY_LETTER: dict[str, str] = {
    "parallel_person": "P",
    "static_shape": "S",
    "crowd": "C",
}
FAMILY_SCENARIO: dict[str, str] = {
    "parallel_person": "parallel_person",
    "static_shape": "static_shape",
    "crowd": "crowd",
}

# Walking-speed band. Never a statue, never a sprint. Ego is an input.
_V_MIN = 0.55
_V_MAX = 2.45
_DV = (0.22, 0.58)
_HIT_S = (12.2, 14.5)
_CPA = (0.10, 0.20)
_MISS = (1.20, 2.00)
_SPAWN = (3.50, 32.0)
_T_CLIP = 15.0
_HEADINGS = ("same", "oncoming")
_SPEED_RELS = ("slower", "faster")


def family_letters(families: Sequence[str]) -> str:
    return ",".join(FAMILY_LETTER.get(str(f), "?") for f in families)


def pair_empty_bg_relpaths(miss_side: str) -> tuple[str, ...]:
    """Empty ``bg_01`` dirs for a collision/miss pair sample."""
    empty = static_shape_pair_relpaths(miss_side).get("empty") or ()
    return tuple(str(p) for p in empty)


def ensure_pair_empty_dirs(sample_dir: Path, miss_side: str) -> list[Path]:
    """mkdir the fake ``bg_01`` folders. No files."""
    made: list[Path] = []
    for rel in pair_empty_bg_relpaths(miss_side):
        path = Path(sample_dir) / str(rel)
        path.mkdir(parents=True, exist_ok=True)
        made.append(path)
    return made


def is_pair_family(name: str) -> bool:
    tok = str(name or "").strip().lower()
    return (
        tok in PAIR_FAMILIES
        or tok in STATIC_SHAPE_PAIR_SCENARIOS
        or tok in CROWD_PACK_SCENARIOS
    )


# ---------------------------------------------------------------------------
# Parallel-person pair kinematics (no bpy)
# ---------------------------------------------------------------------------
#
# Ego walks +s at v_ego. The other person keeps one heading and |v| for both
# clips. Signed speed is +v_other (same-direction) or −v_other (oncoming).
#
#     d(t) = s_other(t) − s_ego(t) = d0 + (v_signed − v_ego) t
#     d(t_hit) = 0  ⇒  d0 = (v_ego − v_signed) t_hit
#
# Oncoming: spawn ahead, gap (v_ego+v_other) t_hit.
# Same-dir + slower + ahead: catch-up, gap (v_ego−v_other) t_hit.
# Same-dir + faster: they pull away if ahead, so start behind (rear-end).
# A forward camera cannot see a rear-end approach — that combo is FOV-false
# and the drawer resamples rather than shipping an empty-road "collision".
#
# Do not speed the ego. Tune spawn s and v_other. t_hit ~ U(12.2, 14.5).


@dataclass(frozen=True)
class ParallelPersonPairPlan:
    heading: str
    speed_rel: str
    v_ego: float
    v_other: float
    signed_speed: float
    t_hit: float
    d0: float
    collision_dlat: float
    miss_dlat: float
    miss_side: str
    cpa_m: float
    miss_clearance_m: float
    duration_s: float = _T_CLIP
    fov_ok: bool = True

    @property
    def heading_sign(self) -> float:
        return -1.0 if self.heading == "oncoming" else 1.0

    def gap_s(self, t: float) -> float:
        """Along-track gap d(t) = s_other − s_ego."""
        return float(self.d0) + (float(self.signed_speed) - float(self.v_ego)) * float(t)

    def dlat_for(self, outcome: str) -> float:
        name = str(outcome or "collision").strip().lower()
        if name in ("collision", "hit", "collide", ""):
            return float(self.collision_dlat)
        return float(self.miss_dlat)

    def world_lat(self, gait_lat: float, outcome: str, *, corridor_lim: float = 8.0) -> float:
        return parallel_person_pair_world_lat(
            gait_lat, self.dlat_for(outcome), corridor_lim=corridor_lim,
        )

    def dist_xy(self, t: float, outcome: str) -> float:
        return math.hypot(self.gap_s(t), self.dlat_for(outcome))

    def as_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "heading_sign": round(float(self.heading_sign), 4),
            "speed_rel": self.speed_rel,
            "v_ego": round(float(self.v_ego), 4),
            "v_other": round(float(self.v_other), 4),
            "signed_speed": round(float(self.signed_speed), 4),
            "t_hit": round(float(self.t_hit), 4),
            "d0": round(float(self.d0), 4),
            "collision_dlat": round(float(self.collision_dlat), 4),
            "miss_dlat": round(float(self.miss_dlat), 4),
            "miss_side": self.miss_side,
            "cpa_m": round(float(self.cpa_m), 4),
            "miss_clearance_m": round(float(self.miss_clearance_m), 4),
            "fov_ok": bool(self.fov_ok),
        }


def parallel_person_signed_speed(heading: str, v_other: float) -> float:
    v = abs(float(v_other))
    if str(heading).strip().lower() in ("oncoming", "toward", "head_on", "headon"):
        return -v
    return v


def parallel_person_spawn_gap(
    v_ego: float,
    heading: str,
    v_other: float,
    t_hit: float,
) -> float:
    """d0 = s_other(0) − s_ego(0) so along-track CPA is at ``t_hit``."""
    v_signed = parallel_person_signed_speed(heading, v_other)
    return (float(v_ego) - float(v_signed)) * float(t_hit)


def parallel_person_pair_world_lat(
    gait_lat: float,
    dlat: float,
    *,
    corridor_lim: float,
) -> float:
    """World Frenet lateral. Collision keeps the CPA; miss keeps |Δ|≥1.2 if possible.

    Negative ``corridor_lim`` is treated as a half-width. A 1.2–2.0 m sideline
    toward the facade may sit on the planting strip — that is still a miss.
    """
    gait = float(gait_lat)
    dlat = float(dlat)
    lim = max(1.60, abs(float(corridor_lim)))
    target = gait + dlat
    clipped = max(-lim, min(lim, target))
    if abs(dlat) < 0.50:
        return float(clipped)
    if abs(clipped - gait) >= 1.15:
        return float(clipped)
    sign = 1.0 if dlat >= 0.0 else -1.0
    room = (lim - gait) if sign >= 0.0 else (gait + lim)
    use = abs(dlat)
    if room > 1.20:
        use = max(1.20, min(use, room * 0.92))
    else:
        use = max(1.20, room)
    return max(-lim, min(lim, gait + sign * use))


def parallel_person_in_forward_fov(
    d0: float,
    heading: str,
    v_other: float,
    v_ego: float,
    t_hit: float,
    *,
    min_ahead: float = _SPAWN[0],
    max_ahead: float = _SPAWN[1],
) -> bool:
    """True if the person is ahead of a +s camera for the first seconds.

    Same-dir + faster starts behind (d0 < 0) and stays behind until impact:
    the collision clip would be empty road. Reject rather than fake it.
    Catch-up may start only a few metres ahead — still on camera. Do not
    require a 0.8 m gap at t_hit−0.6; that is the hit.
    """
    v_signed = parallel_person_signed_speed(heading, v_other)
    d0 = float(d0)
    t_hit = float(t_hit)
    ahead_lo = min(0.80, max(0.40, float(min_ahead)))
    if d0 > float(max_ahead) or d0 < ahead_lo:
        return False
    for t in (0.0, min(2.5, max(0.0, t_hit * 0.35))):
        gap = d0 + (v_signed - float(v_ego)) * float(t)
        if gap < 0.80:
            return False
    return True


def parallel_person_kinematics(
    v_ego: float,
    heading: str,
    speed_rel: str,
    *,
    t_hit: float,
    v_other: float,
    miss_side: str,
    cpa_m: float,
    miss_clearance_m: float,
    collision_sign: float = 1.0,
    duration_s: float = _T_CLIP,
) -> ParallelPersonPairPlan:
    """Closed-form pair for a frozen heading × speed draw. No resampling."""
    heading_k = "oncoming" if str(heading).strip().lower() in (
        "oncoming", "toward", "head_on", "headon",
    ) else "same"
    rel = "faster" if str(speed_rel).strip().lower() in ("faster", "fast") else "slower"
    side = "miss_left" if "left" in str(miss_side or "") else "miss_right"
    v_ego = max(0.0, float(v_ego))
    v_other = max(_V_MIN, min(_V_MAX, abs(float(v_other))))
    t_hit = float(t_hit)
    v_signed = parallel_person_signed_speed(heading_k, v_other)
    d0 = parallel_person_spawn_gap(v_ego, heading_k, v_other, t_hit)
    cpa = min(max(float(cpa_m), _CPA[0]), _CPA[1])
    miss_c = min(max(float(miss_clearance_m), _MISS[0]), _MISS[1])
    csign = 1.0 if float(collision_sign) >= 0.0 else -1.0
    msign = -1.0 if side == "miss_left" else 1.0
    fov = parallel_person_in_forward_fov(d0, heading_k, v_other, v_ego, t_hit)
    return ParallelPersonPairPlan(
        heading=heading_k,
        speed_rel=rel,
        v_ego=float(v_ego),
        v_other=float(v_other),
        signed_speed=float(v_signed),
        t_hit=float(t_hit),
        d0=float(d0),
        collision_dlat=float(csign * cpa),
        miss_dlat=float(msign * miss_c),
        miss_side=side,
        cpa_m=float(cpa),
        miss_clearance_m=float(miss_c),
        duration_s=max(3.0, float(duration_s)),
        fov_ok=bool(fov),
    )


def _draw_v_other(v_ego: float, speed_rel: str, rng: random.Random) -> Optional[float]:
    v_ego = max(0.0, float(v_ego))
    dv = float(rng.uniform(*_DV))
    if speed_rel == "slower":
        v = min(v_ego - 0.18, v_ego - dv)
        v = max(_V_MIN, v)
        if v >= v_ego - 0.12:
            return None
        return float(v)
    v = max(v_ego + 0.18, v_ego + dv)
    v = min(_V_MAX, v)
    if v <= v_ego + 0.12:
        return None
    return float(v)


def _draw_t_hit(rng: random.Random, span: float) -> float:
    lo, hi = _HIT_S
    t = float(rng.uniform(lo, hi))
    return min(max(t, lo), min(hi, max(12.05, float(span) - 0.40)))


def draw_parallel_person_pair_plan(
    v_ego: float,
    *,
    miss_side: str,
    rng: Optional[random.Random] = None,
    duration_s: float = _T_CLIP,
    heading: Optional[str] = None,
    speed_rel: Optional[str] = None,
) -> ParallelPersonPairPlan:
    """Random heading × relative speed, resampled until FOV is real.

    Same-dir + faster is kinematically a rear-end from behind a forward
    camera: it is drawn, then rejected. Oncoming + faster is kept only when
    the spawn gap still fits the look-ahead (typically a stroll ego).
    """
    rng = rng or random.Random(0)
    v_ego = max(0.0, float(v_ego))
    span = max(3.0, float(duration_s))
    side = "miss_left" if "left" in str(miss_side or "") else (
        "miss_right" if "right" in str(miss_side or "") else pick_miss_side(rng)
    )
    forced = heading is not None or speed_rel is not None
    if forced:
        h0 = "oncoming" if str(heading or "same").lower() in (
            "oncoming", "toward", "head_on", "headon",
        ) else "same"
        r0 = "faster" if str(speed_rel or "slower").lower() in ("faster", "fast") else "slower"
        combos = [(h0, r0)]
    else:
        combos = [(h, r) for h in _HEADINGS for r in _SPEED_RELS]
        rng.shuffle(combos)
        combos = combos * 8

    last: Optional[ParallelPersonPairPlan] = None
    for h, r in combos:
        v_other = _draw_v_other(v_ego, r, rng)
        if v_other is None:
            continue
        t_hit = _draw_t_hit(rng, span)
        cpa = float(rng.uniform(*_CPA))
        miss_c = float(rng.uniform(*_MISS))
        csign = 1.0 if rng.random() < 0.5 else -1.0
        for _try in range(6):
            plan = parallel_person_kinematics(
                v_ego, h, r, t_hit=t_hit, v_other=v_other, miss_side=side,
                cpa_m=cpa, miss_clearance_m=miss_c, collision_sign=csign,
                duration_s=span,
            )
            last = plan
            if plan.fov_ok:
                return plan
            if abs(plan.d0) > _SPAWN[1] and t_hit > _HIT_S[0] + 0.05:
                t_hit = max(_HIT_S[0], t_hit - 0.55)
                continue
            if h == "oncoming" and r == "faster" and v_other > v_ego + 0.20:
                v_other = max(v_ego + 0.18, v_other - 0.12)
                continue
            break
        if forced:
            break

    t_hit = float(_HIT_S[0])
    v_other = min(1.05, max(_V_MIN, min(v_ego - 0.18, v_ego - 0.28)))
    v_other = max(_V_MIN, v_other)
    cap = float(_SPAWN[1])
    if (v_ego + v_other) * t_hit > cap:
        v_other = max(_V_MIN, cap / max(t_hit, 12.0) - v_ego - 0.05)
        v_other = max(_V_MIN, v_other)
    plan = parallel_person_kinematics(
        v_ego, "oncoming", "slower", t_hit=t_hit, v_other=v_other,
        miss_side=side, cpa_m=0.14, miss_clearance_m=1.60,
        collision_sign=1.0, duration_s=span,
    )
    if plan.fov_ok:
        return plan
    return last or plan


def parallel_person_pair_table(
    v_ego: float,
    *,
    t_hit: float = 13.0,
    miss_side: str = "miss_left",
    times: Sequence[float] = (0.0, 6.0, 12.0, 13.0, 15.0),
) -> list[dict[str, Any]]:
    """Heading × speed rows for the unit test / dry print."""
    rows: list[dict[str, Any]] = []
    v_slower = max(_V_MIN, min(v_ego - 0.32, v_ego - 0.28))
    gap_need = float(_SPAWN[0]) / max(float(t_hit), 1.0)
    if v_ego - v_slower < gap_need:
        v_slower = max(_V_MIN, float(v_ego) - gap_need - 0.01)
    v_faster = min(_V_MAX, max(v_ego + 0.32, v_ego + 0.28))
    for heading in _HEADINGS:
        for rel, v_o in (("slower", v_slower), ("faster", v_faster)):
            plan = parallel_person_kinematics(
                v_ego, heading, rel, t_hit=t_hit, v_other=v_o,
                miss_side=miss_side, cpa_m=0.15, miss_clearance_m=1.60,
                collision_sign=1.0,
            )
            rows.append(
                {
                    "heading": plan.heading,
                    "speed_rel": plan.speed_rel,
                    "v_ego": round(plan.v_ego, 4),
                    "v_other": round(plan.v_other, 4),
                    "d0": round(plan.d0, 4),
                    "t_hit": round(plan.t_hit, 4),
                    "fov_ok": plan.fov_ok,
                    "times": [float(t) for t in times],
                    "d_s": [round(plan.gap_s(t), 4) for t in times],
                    "xy_hit_collision": round(plan.dist_xy(plan.t_hit, "collision"), 4),
                    "xy_hit_miss": round(plan.dist_xy(plan.t_hit, miss_side), 4),
                    "min_xy_miss": round(
                        min(plan.dist_xy(t, miss_side) for t in times), 4,
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Plan / clip iterator
# ---------------------------------------------------------------------------


def _pair_backgrounds() -> list[dict[str, Any]]:
    """Fake K=2: bg_00 is the only rendered world; bg_01 is an empty dir."""
    return [
        {
            "id": PAIR_RENDER_BG_ID,
            "seed": 0,
            "style": "full",
            "lighting": "",
            "real": True,
        },
        {
            "id": PAIR_EMPTY_BG_ID,
            "seed": 0,
            "style": "full",
            "lighting": "",
            "real": False,
        },
    ]


def build_pair_plan(
    n: int,
    seed: int,
    cfg: dict | None = None,
    *,
    families: Sequence[str] | None = None,
) -> dict:
    """Deterministic rotating recipe. Same ``(n, seed)`` ⇒ same samples.

    ``n`` is the count **per family**. Emission order is
    parallel, static, crowd, parallel, … so a cancel leaves ~equal counts.
    """
    from config import get_config

    cfg = cfg or get_config()
    n = max(1, int(n))
    fams = [str(f).strip().lower() for f in (families or PAIR_FAMILIES)]
    fams = [f for f in fams if f in FAMILY_SCENARIO] or list(PAIR_FAMILIES)
    rng = random.Random(int(seed))
    fps = max(1, int(cfg["render"]["fps"]))
    frames = int(cfg["render"]["frames_per_episode"])
    samples: list[dict[str, Any]] = []
    rotation: list[str] = []
    for i in range(n):
        for family in fams:
            scenario = FAMILY_SCENARIO[family]
            scene_seed = rng.randrange(1, 2**31)
            sample_rel = f"{family}/{sample_dir_name(i, scenario)}"
            flat = family == "crowd" or is_crowd_pack_scenario(scenario)
            if family == "crowd":
                crowd_clip = pairs_family_crowd(rng, index=i)
                scenario = str(crowd_clip.scenario)
                sample_rel = crowd_clip.pack_relpath(i)
                flat = True
            miss_side = pick_miss_side(rng) if not flat else ""
            if not flat:
                rng.uniform(*_MISS)
                rng.uniform(*_CPA)
            samples.append(
                {
                    "id": i,
                    "family": family,
                    "scenario": scenario,
                    "dir": sample_rel,
                    "scene_seed": scene_seed,
                    "files_layout": "flat" if flat else "outcome_bg",
                    "outcomes": ["clip"] if flat else ["collision", miss_side],
                    "miss_side": miss_side,
                    "backgrounds": [] if flat else _pair_backgrounds(),
                    "bg_real": None if flat else PAIR_RENDER_BG_ID,
                }
            )
            rotation.append(family)
    return {
        "layout": "pairs",
        "seed": int(seed),
        "n": n,
        "n_bg": 2,
        "bg_real": PAIR_RENDER_BG_ID,
        "bg_empty": PAIR_EMPTY_BG_ID,
        "frames": frames,
        "fps": fps,
        "families": list(fams),
        "rotation": family_letters(rotation),
        "samples": samples,
        "prefix_cache": False,
    }


def iter_pair_clips(plan: dict) -> Iterable[dict]:
    """Flatten the rotating sample list into renderable clips.

    Empty ``bg_01`` is **not** yielded (mkdir only). Crowd is one clip at
    the sample root. Pair families: collision/bg_00 then the one miss/bg_00.
    """
    for sample in plan.get("samples") or []:
        family = str(sample.get("family") or sample.get("scenario") or "")
        scenario = str(sample.get("scenario") or family)
        layout = str(sample.get("files_layout") or "outcome_bg")
        if layout == "flat" or is_crowd_pack_scenario(scenario):
            yield {
                "sample_id": int(sample["id"]),
                "family": family,
                "sample_dir": str(sample["dir"]),
                "scenario": scenario,
                "scene_seed": int(sample["scene_seed"]),
                "files_layout": "flat",
                "outcome": "clip",
                "bg_id": None,
                "bg_real": False,
                "miss_side": "",
                "relpath": "",
                "prefix_relpath": None,
            }
            continue
        outcomes = [o for o in (sample.get("outcomes") or []) if o != "clip"]
        if not outcomes:
            outcomes = ["collision", str(sample.get("miss_side") or "miss_left")]
        ordered = ["collision"] + [o for o in outcomes if o != "collision"]
        for outcome in ordered:
            rel = pair_clip_relpath(scenario, str(outcome), PAIR_RENDER_BG_ID)
            yield {
                "sample_id": int(sample["id"]),
                "family": family,
                "sample_dir": str(sample["dir"]),
                "scenario": scenario,
                "scene_seed": int(sample["scene_seed"]),
                "files_layout": "outcome_bg",
                "outcome": str(outcome),
                "bg_id": PAIR_RENDER_BG_ID,
                "bg_real": True,
                "miss_side": str(sample.get("miss_side") or ""),
                "relpath": rel,
                "prefix_relpath": None,
            }


def _self_test() -> None:
    from config import get_config

    _ = bg_id_for  # re-export used by the runner docs
    cfg = get_config()
    a = build_pair_plan(2, 21, cfg)
    b = build_pair_plan(2, 21, cfg)
    assert a["samples"] == b["samples"]
    assert a["layout"] == "pairs"
    assert a["n"] == 2
    assert a["rotation"] == "P,S,C,P,S,C", a["rotation"]
    assert a["prefix_cache"] is False
    fams = [s["family"] for s in a["samples"]]
    assert fams == [
        "parallel_person", "static_shape", "crowd",
        "parallel_person", "static_shape", "crowd",
    ], fams
    assert [s["id"] for s in a["samples"]] == [0, 0, 0, 1, 1, 1]
    assert family_letters(fams) == "P,S,C,P,S,C"

    for sample in a["samples"]:
        if sample["family"] == "crowd":
            assert sample["files_layout"] == "flat"
            assert sample["dir"].startswith("crowd/sample_")
            assert sample["scenario"] in ("crowd", "crowd_path")
            assert sample["outcomes"] == ["clip"]
            assert sample["backgrounds"] == []
            continue
        outs = list(sample["outcomes"])
        assert outs[0] == "collision" and len(outs) == 2, outs
        assert outs[1] in MISS_SIDES
        bgs = sample["backgrounds"]
        assert [x["id"] for x in bgs] == [PAIR_RENDER_BG_ID, PAIR_EMPTY_BG_ID]
        assert bgs[0]["real"] is True and bgs[1]["real"] is False
        assert sample["dir"].startswith(sample["family"] + "/sample_")

    clips = list(iter_pair_clips(a))
    assert len(clips) == 10, len(clips)
    assert all(cl.get("prefix_relpath") is None for cl in clips)
    seq = [(cl["family"], cl["outcome"], cl.get("bg_id")) for cl in clips]
    assert seq[0][0] == "parallel_person" and seq[0][1] == "collision"
    assert seq[1][0] == "parallel_person" and seq[1][1] in MISS_SIDES
    assert seq[2][0] == "static_shape"
    assert seq[4][0] == "crowd" and seq[4][1] == "clip" and seq[4][2] is None
    fam_stream = [cl["family"] for cl in clips]
    compact: list[str] = []
    for f in fam_stream:
        if not compact or compact[-1] != f:
            compact.append(f)
    assert compact == [
        "parallel_person", "static_shape", "crowd",
        "parallel_person", "static_shape", "crowd",
    ], compact

    big = build_pair_plan(16, 7, cfg)
    pair_miss = [
        s["outcomes"][1] for s in big["samples"] if s["family"] == "parallel_person"
    ]
    assert set(pair_miss) == set(MISS_SIDES), pair_miss
    empty = pair_empty_bg_relpaths("miss_left")
    assert f"collision/{PAIR_EMPTY_BG_ID}" in empty
    assert f"miss_left/{PAIR_EMPTY_BG_ID}" in empty

    print("parallel-person pair d(t)  [along-track gap, metres]")
    times = (0.0, 6.0, 12.0, 13.0, 15.0)
    for v_ego in (0.80, 1.35, 1.90):
        print(f"  v_ego={v_ego:.2f} m/s")
        for row in parallel_person_pair_table(v_ego, t_hit=13.0, times=times):
            d = row["d_s"]
            print(
                f"    {row['heading']:8s} {row['speed_rel']:7s}  "
                f"v_other={row['v_other']:.3f}  d0={row['d0']:.2f}  "
                f"fov={int(row['fov_ok'])}  "
                f"d(0/6/12/13)={d[0]:.2f}/{d[1]:.2f}/{d[2]:.2f}/{d[3]:.2f}  "
                f"xy_hit_c={row['xy_hit_collision']:.2f} xy_hit_m={row['xy_hit_miss']:.2f}"
            )
            plan = parallel_person_kinematics(
                v_ego, row["heading"], row["speed_rel"],
                t_hit=13.0, v_other=row["v_other"], miss_side="miss_left",
                cpa_m=0.15, miss_clearance_m=1.60, collision_sign=1.0,
            )
            assert plan.heading == row["heading"]
            assert abs(plan.v_other - row["v_other"]) < 1e-9
            assert plan.t_hit >= 12.0
            assert abs(plan.gap_s(plan.t_hit)) < 1e-6, (plan, plan.gap_s(plan.t_hit))
            assert 0.09 <= plan.dist_xy(plan.t_hit, "collision") <= 0.22
            miss_xy = [plan.dist_xy(t, "miss_left") for t in times]
            assert min(miss_xy) >= 1.15, miss_xy
            assert plan.dist_xy(plan.t_hit, "miss_left") >= 1.15
            assert plan.dlat_for("collision") != plan.dlat_for("miss_left")
            assert abs(plan.dlat_for("miss_left")) >= 1.20
            if plan.heading == "same" and plan.speed_rel == "faster":
                assert plan.d0 < 0.0
                assert plan.fov_ok is False
            if plan.heading == "same" and plan.speed_rel == "slower":
                assert plan.d0 > 0.0
                assert plan.v_other < v_ego
                assert plan.fov_ok is True
            if plan.heading == "oncoming":
                assert plan.signed_speed < 0.0
                assert plan.d0 > 0.0

    shipped = [
        draw_parallel_person_pair_plan(1.35, miss_side="miss_right", rng=random.Random(s))
        for s in range(40, 80)
    ]
    assert all(p.fov_ok for p in shipped), [p.as_dict() for p in shipped if not p.fov_ok]
    assert not any(p.heading == "same" and p.speed_rel == "faster" for p in shipped)
    assert any(p.heading == "same" for p in shipped)
    assert any(p.heading == "oncoming" for p in shipped)
    assert all(p.v_other >= _V_MIN for p in shipped)
    assert all(p.t_hit >= 12.0 for p in shipped)
    slow = draw_parallel_person_pair_plan(0.80, miss_side="miss_left", rng=random.Random(11))
    assert slow.fov_ok and slow.t_hit >= 12.0
    hurry = draw_parallel_person_pair_plan(2.05, miss_side="miss_right", rng=random.Random(19))
    assert hurry.fov_ok and hurry.t_hit >= 12.0
    left = draw_parallel_person_pair_plan(1.1, miss_side="miss_left", rng=random.Random(3))
    assert left.miss_side == "miss_left" and left.miss_dlat < 0.0
    right = draw_parallel_person_pair_plan(1.1, miss_side="miss_right", rng=random.Random(3))
    assert right.miss_side == "miss_right" and right.miss_dlat > 0.0
    kin = parallel_person_kinematics(
        1.2, "oncoming", "slower", t_hit=12.8, v_other=0.85,
        miss_side="miss_left", cpa_m=0.12, miss_clearance_m=1.8,
    )
    kin_r = parallel_person_kinematics(
        1.2, "oncoming", "slower", t_hit=12.8, v_other=0.85,
        miss_side="miss_right", cpa_m=0.12, miss_clearance_m=1.8,
    )
    assert kin.v_other == kin_r.v_other and kin.heading == kin_r.heading
    assert kin.d0 == kin_r.d0 and kin.signed_speed == kin_r.signed_speed
    assert kin.miss_dlat == -kin_r.miss_dlat
    gait = 4.41
    lim = 5.68 + 1.20
    assert abs(parallel_person_pair_world_lat(gait, 0.15, corridor_lim=lim) - (gait + 0.15)) < 1e-6
    assert abs(parallel_person_pair_world_lat(gait, 1.60, corridor_lim=lim) - gait) >= 1.15
    assert abs(parallel_person_pair_world_lat(gait, -1.60, corridor_lim=lim) - gait) >= 1.15
    assert abs(kin.world_lat(gait, "miss_left", corridor_lim=lim) - gait) >= 1.15

    print("pair_pack self-test: OK")


if __name__ == "__main__":
    _self_test()
