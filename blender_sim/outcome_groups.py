"""Outcome groups: shared-prefix collision vs miss-left vs miss-right.

A *sample* is one threat scenario (the object that would go red on the
heat overlay). It expands to:

    sample_XXXX_<scenario>/
      collision/bg_00/   collision/bg_01/  …
      miss_left/bg_00/   …
      miss_right/bg_00/  …

Until ``diverge_s`` every clip of a (sample, background) triple has the
same ego, the same threat kinematics, and the same RGB. After that the
threat either continues into the walker or peels to the walker's left or
right. Background variants rebuild trees / buildings / furniture **and**
pin a distinct time-of-day lighting slot (noon / night / dusk / …).

Pure Python, no ``bpy``. ``python outcome_groups.py`` runs the self-test.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence


OUTCOMES: tuple[str, str, str] = ("collision", "miss_left", "miss_right")

# Styles for non-threat scatter. The threat injector is unchanged.
BG_STYLES: tuple[str, str, str] = ("full", "sparse", "bare")

# Time-of-day slots cycled across bg_YY of a sample. Adjacent entries are
# contrasty (noon vs night, dusk vs overcast) so two variants never share
# illumination. Weather stays mild so the threat silhouette stays readable.
LIGHTING_SLOTS: tuple[str, ...] = (
    "noon", "night", "dusk", "overcast", "harsh_glare", "dawn",
)
WEATHER_FOR_LIGHTING: dict[str, tuple[str, ...]] = {
    "noon": ("clear",),
    "night": ("clear",),
    "dusk": ("clear", "light_fog"),
    "overcast": ("light_fog", "clear"),
    "harsh_glare": ("clear",),
    "dawn": ("clear", "light_fog"),
}


def weather_choices_for(lighting: str) -> tuple[str, ...]:
    name = str(lighting or "").strip().lower()
    return WEATHER_FOR_LIGHTING.get(name, ("clear",))

# Moving (or roll-aside) intercepts that can fork left / right in the last
# seconds. Static holes and lamp-on-path are not in this set.
FORKABLE_SCENARIOS: tuple[str, ...] = (
    "jaywalker",
    "jaywalker_from_left",
    "jaywalker_from_right",
    "jaywalker_turn_toward",
    "swerve_vehicle",
    "cube_head_on",
    "cube_from_left",
    "cube_from_right",
    "cube_on_path",
    "shape_head_on",
    "shape_from_left",
    "shape_from_right",
    "car_cut_in",
    "cyclist_head_on",
    "car_cross_critical",
    "car_runs_off_road",
    "child_darting",
    "crossing_car_side",
    "periph_car_turn",
    "periph_ped_cut",
    "periph_child_cut",
)


def sample_dir_name(sample_id: int, scenario: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(scenario).strip())
    slug = slug or "scenario"
    return f"sample_{int(sample_id):04d}_{slug}"


def clip_relpath(outcome: str, bg_id: str) -> str:
    return f"{outcome}/{bg_id}"


def bg_id_for(index: int) -> str:
    return f"bg_{int(index):02d}"


def link_or_copy_prefix(
    src: Path, dst: Path, n: int, *, suffix: str = ".png",
) -> int:
    """Hard-link (or copy) frames ``000000`` .. ``n-1`` for a miss clip.

    Same filesystem → ``os.link`` so miss RGB / heat prefix is not rewritten.
    Falls back to ``copy2`` across devices. Stops at the first missing source.
    """
    dst.mkdir(parents=True, exist_ok=True)
    suf = str(suffix)
    if not suf.startswith("."):
        suf = "." + suf
    copied = 0
    for i in range(max(0, int(n))):
        a = src / f"{i:06d}{suf}"
        if not a.is_file():
            break
        b = dst / f"{i:06d}{suf}"
        if b.is_file():
            copied += 1
            continue
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)
        copied += 1
    return copied


def diverge_and_hit_s(cfg: dict, episode_seconds: float) -> tuple[float, float]:
    """(diverge_s, hit_s) inside the clip, with room to see the ending.

    Config values are written for a 15 s episode. Shorter clips keep a 2 s
    fork window ending 0.4 s before the last frame rather than scaling the
    whole intercept (that is what scrambled far cars when we only stretched
    τ).
    """
    sc = dict(cfg.get("scenarios") or {})
    span = max(3.0, float(episode_seconds))
    hit = float(sc.get("outcome_hit_s", 13.0))
    div = float(sc.get("outcome_diverge_s", 11.0))
    if hit >= span - 0.35 or div >= hit:
        hit = max(span - 0.40, 2.6)
        div = max(hit - 2.00, 0.80)
    return float(div), float(hit)


def miss_target_lat(
    gait_lat: float,
    outcome: str,
    *,
    clearance_m: float,
    corridor_lim: float,
    road_lane_m: float = 1.75,
    min_clearance_m: float | None = None,
) -> float | None:
    """Frenet lateral the threat steers to after diverge. None = collision.

    Both misses keep at least ``min_clearance_m`` from the gait after
    clipping. If that side is a wall (typical sidewalk + car), fall back
    to the far driving lane rather than sitting on the walker.
    """
    name = str(outcome or "collision").strip().lower()
    if name in ("collision", "hit", "collide"):
        return None
    sign = -1.0 if "left" in name else 1.0
    lim = max(0.80, float(corridor_lim))
    gait = float(gait_lat)
    pad = max(0.90, float(clearance_m))
    need = float(min_clearance_m) if min_clearance_m is not None else min(pad, 1.70)
    need = max(1.20, need)
    target = gait + sign * pad
    clipped = max(-lim, min(lim, target))
    if abs(clipped - gait) + 1e-6 >= need:
        return clipped
    alt = -math.copysign(max(float(road_lane_m), pad * 0.70), gait if abs(gait) > 1e-6 else -sign)
    return max(-lim, min(lim, alt))


def apply_bg_style(cfg: dict, style: str) -> dict:
    """Scale or drop non-threat scatter. Mutates ``cfg['world']`` in place."""
    wcfg = cfg.setdefault("world", {})
    name = str(style or "full").strip().lower()
    if name not in BG_STYLES:
        name = "full"
    poisson = wcfg.setdefault("poisson", {})

    def _scale_pair(key: str, k: float) -> None:
        pair = wcfg.get(key)
        if not pair:
            return
        lo, hi = int(pair[0]), int(pair[1])
        wcfg[key] = (max(0, int(round(lo * k))), max(0, int(round(hi * k))))

    if name == "bare":
        for key in (
            "n_trees", "n_grass_clumps", "n_street_trees",
            "n_median_trees", "n_path_trees",
        ):
            wcfg[key] = (0, 0)
        poisson["n_ground_static"] = (0, 1)
        wcfg["n_streetlamps"] = max(2, int(wcfg.get("n_streetlamps", 10) // 2))
        # Fewer lots, bigger gaps — not a demolished street (that would
        # change lighting of the threat).
        wcfg["building_gap"] = (4.0, 9.0)
        wcfg["building_width"] = (5.0, 9.5)
    elif name == "sparse":
        for key in (
            "n_trees", "n_grass_clumps", "n_street_trees",
            "n_median_trees", "n_path_trees",
        ):
            _scale_pair(key, 0.45)
        pair = poisson.get("n_ground_static")
        if pair:
            lo, hi = int(pair[0]), int(pair[1])
            poisson["n_ground_static"] = (max(0, lo // 2), max(1, (hi + 1) // 2))
        wcfg["n_streetlamps"] = max(3, int(round(int(wcfg.get("n_streetlamps", 10)) * 0.6)))
        wcfg["building_gap"] = (1.6, 4.8)
    return cfg


def build_group_plan(
    n: int,
    seed: int,
    cfg: dict | None = None,
    *,
    n_bg: int = 3,
    scenarios: Sequence[str] | None = None,
) -> dict:
    """Deterministic group recipe. Same ``(n, seed, n_bg)`` ⇒ same samples."""
    from config import get_config

    cfg = cfg or get_config()
    n = max(1, int(n))
    n_bg = max(1, min(8, int(n_bg)))
    pool = list(scenarios) if scenarios else list(FORKABLE_SCENARIOS)
    pool = [s for s in pool if s in FORKABLE_SCENARIOS] or list(FORKABLE_SCENARIOS)
    rng = random.Random(int(seed))
    fps = max(1, int(cfg["render"]["fps"]))
    frames = int(cfg["render"]["frames_per_episode"])
    span = frames / float(fps)
    diverge_s, hit_s = diverge_and_hit_s(cfg, span)

    samples: list[dict] = []
    for i in range(n):
        scenario = pool[i % len(pool)]
        scene_seed = rng.randrange(1, 2**31)
        backgrounds: list[dict] = []
        styles = list(BG_STYLES)
        rng.shuffle(styles)
        light_start = rng.randrange(len(LIGHTING_SLOTS))
        for b in range(n_bg):
            backgrounds.append(
                {
                    "id": bg_id_for(b),
                    "seed": rng.randrange(1, 2**31),
                    "style": styles[b % len(styles)],
                    "lighting": LIGHTING_SLOTS[(light_start + b) % len(LIGHTING_SLOTS)],
                }
            )
        samples.append(
            {
                "id": i,
                "dir": sample_dir_name(i, scenario),
                "scenario": scenario,
                "scene_seed": scene_seed,
                "diverge_s": round(diverge_s, 4),
                "hit_s": round(hit_s, 4),
                "diverge_frame": int(math.floor(float(diverge_s) * fps + 1e-9)) + 1,
                "backgrounds": backgrounds,
                "outcomes": list(OUTCOMES),
            }
        )
    return {
        "layout": "groups",
        "seed": int(seed),
        "n": n,
        "n_bg": n_bg,
        "frames": frames,
        "fps": fps,
        "diverge_s": round(diverge_s, 4),
        "hit_s": round(hit_s, 4),
        "outcomes": list(OUTCOMES),
        "samples": samples,
    }


def iter_clips(plan: dict) -> Iterable[dict]:
    """Flatten samples × backgrounds × outcomes for the Blender loop.

    Collision is always first inside a background so miss clips can copy
    its prefix RGB, heat PPMs, and JSON.
    """
    for sample in plan.get("samples") or []:
        bgs = list(sample.get("backgrounds") or [])
        outcomes = list(sample.get("outcomes") or OUTCOMES)
        ordered = [o for o in OUTCOMES if o in outcomes]
        ordered.extend(o for o in outcomes if o not in ordered)
        for bg in bgs:
            collision_rel = clip_relpath("collision", str(bg["id"]))
            for outcome in ordered:
                yield {
                    "sample_id": int(sample["id"]),
                    "sample_dir": str(sample["dir"]),
                    "scenario": str(sample["scenario"]),
                    "scene_seed": int(sample["scene_seed"]),
                    "diverge_s": float(sample["diverge_s"]),
                    "hit_s": float(sample["hit_s"]),
                    "diverge_frame": int(sample["diverge_frame"]),
                    "bg_id": str(bg["id"]),
                    "bg_seed": int(bg["seed"]),
                    "bg_style": str(bg.get("style") or "full"),
                    "bg_lighting": str(bg.get("lighting") or ""),
                    "outcome": str(outcome),
                    "relpath": clip_relpath(str(outcome), str(bg["id"])),
                    "prefix_relpath": collision_rel if outcome != "collision" else None,
                }


def _grid_max(grid: Any) -> float:
    if not grid:
        return 0.0
    return max(max(float(v) for v in row) for row in grid)


def _grids_close(a: Any, b: Any, atol: float) -> bool:
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for xa, xb in zip(ra, rb):
            if abs(float(xa) - float(xb)) > atol:
                return False
    return True


def verify_group_pack(root: str, *, heat_atol: float = 1e-4) -> list[str]:
    """Check shared prefixes and forked endings in a rendered/sim pack.

    Returns a list of problem strings (empty = ok). Reads spatial JSON only.
    Prefix is frames ``[0, diverge_frame)``; the fork applies at ``diverge_s``.
    """
    root_p = Path(root)
    problems: list[str] = []
    samples = sorted(p for p in root_p.glob("sample_*") if p.is_dir())
    for sample in samples:
        group_path = sample / "group.json"
        diverge = 330
        bgs: list[str] = []
        if group_path.is_file():
            g = json.loads(group_path.read_text(encoding="utf-8"))
            diverge = int(g.get("diverge_frame") or diverge)
            bgs = [str(b["id"]) for b in g.get("backgrounds") or []]
        if not bgs and (sample / "collision").is_dir():
            bgs = sorted(p.name for p in (sample / "collision").glob("bg_*") if p.is_dir())
        for bg in bgs:
            grids: dict[str, list] = {}
            for outcome in OUTCOMES:
                path = (
                    sample / outcome / bg / "spatial_annotations" / "spatial_annotations.json"
                )
                if not path.is_file():
                    problems.append(f"missing {path.relative_to(root_p)}")
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                frames = list(payload.get("frames") or [])
                grids[outcome] = [fr.get("matrix") for fr in frames]
            if len(grids) < 3:
                continue
            n = min(len(g) for g in grids.values())
            if n < diverge + 8:
                problems.append(f"{sample.name}/{bg}: only {n} frames (diverge={diverge})")
                continue
            for i in range(min(diverge, n)):
                a = grids["collision"][i]
                for outcome in ("miss_left", "miss_right"):
                    if not _grids_close(a, grids[outcome][i], heat_atol):
                        problems.append(
                            f"{sample.name}/{bg}: prefix diverge at frame {i} ({outcome})"
                        )
                        break
                else:
                    continue
                break
            # Probe near hit, not the last frame: after the walker passes the
            # meeting cell every ending can be empty.
            hit_i = min(n - 1, diverge + int(0.70 * 30))
            hit_c = _grid_max(grids["collision"][hit_i])
            hit_l = _grid_max(grids["miss_left"][hit_i])
            hit_r = _grid_max(grids["miss_right"][hit_i])
            if _grids_close(grids["collision"][hit_i], grids["miss_left"][hit_i], heat_atol) and _grids_close(
                grids["collision"][hit_i], grids["miss_right"][hit_i], heat_atol
            ):
                problems.append(f"{sample.name}/{bg}: three endings identical at frame {hit_i}")
            elif hit_c + 0.08 < max(hit_l, hit_r):
                problems.append(
                    f"{sample.name}/{bg}: collision heat {hit_c:.3f} < miss "
                    f"{max(hit_l, hit_r):.3f} at frame {hit_i}"
                )
    return problems


def _self_test() -> None:
    from config import get_config

    cfg = get_config()
    a = build_group_plan(4, 7, cfg, n_bg=3)
    b = build_group_plan(4, 7, cfg, n_bg=3)
    assert a["samples"] == b["samples"]
    c = build_group_plan(4, 8, cfg, n_bg=3)
    assert a["samples"][0]["scene_seed"] != c["samples"][0]["scene_seed"]
    assert a["layout"] == "groups"
    assert a["samples"][0]["dir"].startswith("sample_0000_")
    names = {s["scenario"] for s in a["samples"]}
    assert names.issubset(set(FORKABLE_SCENARIOS))
    clips = list(iter_clips(a))
    assert len(clips) == 4 * 3 * 3
    # Collision is first per background so prefix copies have a source.
    by_bg = [cl for cl in clips if cl["sample_id"] == 0 and cl["bg_id"] == "bg_00"]
    assert [cl["outcome"] for cl in by_bg] == list(OUTCOMES)
    assert by_bg[0]["prefix_relpath"] is None
    assert by_bg[1]["prefix_relpath"] == "collision/bg_00"
    lights = [b["lighting"] for b in a["samples"][0]["backgrounds"]]
    assert len(set(lights)) == len(lights), lights
    assert all(x in LIGHTING_SLOTS for x in lights)
    assert by_bg[0]["bg_lighting"] == a["samples"][0]["backgrounds"][0]["lighting"]
    # Same sample, different bg ⇒ different lighting labels.
    clip_bgs = [cl for cl in clips if cl["sample_id"] == 0 and cl["outcome"] == "collision"]
    assert len({cl["bg_lighting"] for cl in clip_bgs}) == len(clip_bgs)

    div, hit = diverge_and_hit_s(cfg, 15.0)
    assert 11.0 <= div < hit <= 14.8, (div, hit)
    short_div, short_hit = diverge_and_hit_s(cfg, 6.0)
    assert short_div < short_hit < 6.0

    gait = -2.1
    assert miss_target_lat(gait, "collision", clearance_m=2.4, corridor_lim=5.0) is None
    left = miss_target_lat(gait, "miss_left", clearance_m=2.4, corridor_lim=5.0)
    right = miss_target_lat(gait, "miss_right", clearance_m=2.4, corridor_lim=5.0)
    assert left is not None and right is not None
    assert left < gait < right
    # Sidewalk against a wall: do not clip a car miss back onto the walker.
    tight = miss_target_lat(
        4.2, "miss_right", clearance_m=3.4, corridor_lim=5.0, road_lane_m=1.75,
        min_clearance_m=2.50,
    )
    assert tight is not None and abs(tight - 4.2) >= 2.50, tight
    assert tight < 0.0, tight

    import tempfile

    td = Path(tempfile.mkdtemp())
    src, dst = td / "src", td / "dst"
    src.mkdir()
    for i in range(4):
        (src / f"{i:06d}.ppm").write_bytes(b"P6 dummy")
    nlink = link_or_copy_prefix(src, dst, 3, suffix=".ppm")
    assert nlink == 3, nlink
    assert (dst / "000002.ppm").is_file()
    assert not (dst / "000003.ppm").is_file()
    assert (dst / "000000.ppm").read_bytes() == b"P6 dummy"

    bare = apply_bg_style(get_config(), "bare")
    assert bare["world"]["n_trees"] == (0, 0)
    assert bare["world"]["building_gap"][0] >= 3.0
    print("outcome_groups self-test: OK")


if __name__ == "__main__":
    _self_test()
