#!/usr/bin/env python3
"""Build a mixed synthetic pack: N episodes, one new folder under datasets/.

Each episode already randomizes lighting, weather, path shape, walk speed,
FOV, body type, colours, and clutter (that is the per-episode RNG in
``WorldGenerator`` / ``CameraRig``). This script only chooses *which
scenario(s)* go in each episode so the pack is balanced — singles from
every threat bucket plus compounds (jaywalk + car + pothole, …).

Usage (system Python; launches Blender via ``run.sh``)::

    ./gen_dataset.py --n 24 --seed 7
    python gen_dataset.py --n 8 --seed 1 --dry-run
    python gen_dataset.py --n 16 --seed 3 --spatial-overlay --media both
    python gen_dataset.py --n 40 --theme peripheral --name side40 --dry-run

``python gen_dataset.py`` with no args that need Blender still runs the
planner self-test when you pass ``--self-test``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_config
from scenario_compose import CLEAR_CENTER_SCENARIOS, pick_scenarios, peripheral_pools


# Families: at most one member per compound so occupancy stays sane.
_FAMILIES: dict[str, tuple[str, ...]] = {
    "jaywalk": (
        "jaywalker",
        "jaywalker_offset",
        "jaywalker_from_left",
        "jaywalker_from_right",
        "jaywalker_turn_toward",
        "jaywalker_turn_away",
        "distant_jaywalk",
    ),
    "pothole": ("pothole_on_path", "pothole_near", "pothole_offset"),
    "tree": ("tree_on_path", "tree_near"),
    "lamp": ("lamp_on_path", "lamp_near"),
    "car": (
        "car_approaching",
        "car_pass_far",
        "parked_car_opposite",
        "near_miss_pass",
        "car_near_miss_lane",
        "car_cross_front",
        "car_cut_in",
        "swerve_vehicle",
        "car_cross_critical",
    ),
    "cyclist": (
        "cyclist_same_way", "cyclist_near_miss", "cyclist_head_on", "cyclist_weaving",
    ),
    "cube": (
        "cube_near_miss", "cube_head_on", "cube_from_left", "cube_from_right",
        "cube_on_path",
    ),
    "shape": (
        "shape_near_miss", "shape_head_on", "shape_from_left", "shape_from_right",
        "shapes_on_path",
    ),
    "ped": ("oncoming_pedestrian", "parallel_pedestrian", "sudden_stop", "child_darting"),
    "erratic_car": ("car_erratic_swerve", "car_runs_off_road"),
    "crossing": (
        "crossing_street",
        "crossing_car_side",
        "group_crossing",
        "crossing_head_on",
    ),
    "sidewalk_dyn": (
        "scooter_from_sidewalk",
        "cyclist_overtake",
        "parked_car_door",
        "backing_vehicle",
    ),
}

# Pack mix. The leftover after compounds is split 40 / 30 / 30 like ``--scenario auto``.
_COMPOUND_FRAC = 0.22
_SAFE_FRAC = 0.40
_NEAR_FRAC = 0.30


def _pools(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    sc = cfg["scenarios"]
    return (
        list(sc["safe_pool"]),
        list(sc["near_miss_pool"]),
        list(sc["critical_pool"]),
    )


def _quotas(n: int) -> dict[str, int]:
    """How many singles (safe / near / critical) and compounds for a pack of n."""
    n = max(0, int(n))
    if n == 0:
        return {"safe": 0, "near_miss": 0, "critical": 0, "compound": 0}
    if n == 1:
        return {"safe": 0, "near_miss": 0, "critical": 1, "compound": 0}
    n_comp = 0 if n < 5 else max(1, int(round(n * _COMPOUND_FRAC)))
    n_comp = min(n_comp, n - 1)
    rest = n - n_comp
    n_safe = int(round(rest * _SAFE_FRAC))
    n_near = int(round(rest * _NEAR_FRAC))
    n_crit = rest - n_safe - n_near
    if n_crit < 0:
        n_near += n_crit
        n_crit = 0
    return {
        "safe": max(0, n_safe),
        "near_miss": max(0, n_near),
        "critical": max(0, n_crit),
        "compound": n_comp,
    }


def _cycle_draw(rng: random.Random, pool: list[str] | tuple[str, ...], count: int) -> list[str]:
    """Shuffle-copy the pool; refill only after every name was used once."""
    out: list[str] = []
    bag: list[str] = []
    names = list(pool)
    if not names or count <= 0:
        return out
    while len(out) < count:
        if not bag:
            bag = names[:]
            rng.shuffle(bag)
        out.append(bag.pop())
    return out


def _family_of(name: str) -> str:
    for fam, members in _FAMILIES.items():
        if name in members:
            return fam
    return name


def draw_compound(rng: random.Random, cfg: dict) -> str:
    """2- or 3-injector street. One name per family (jaywalk / hole / car / …)."""
    safe, near, crit = _pools(cfg)
    known = set(safe + near + crit)
    families = {k: [n for n in v if n in known] for k, v in _FAMILIES.items()}
    families = {k: v for k, v in families.items() if v}

    k = 3 if rng.random() < 0.28 else 2
    prefer: list[str] = []
    if rng.random() < 0.78:
        prefer.append("jaywalk")
    if rng.random() < 0.58:
        prefer.append("pothole")
    if rng.random() < 0.62:
        prefer.append("car")
    if rng.random() < 0.22:
        prefer.append("cyclist")
    if rng.random() < 0.18:
        prefer.append("cube")
    if rng.random() < 0.28:
        prefer.append("crossing")
    if rng.random() < 0.16:
        prefer.append("sidewalk_dyn")

    picked: list[str] = []
    used: set[str] = set()
    for fam in prefer:
        if fam in used or fam not in families:
            continue
        picked.append(rng.choice(families[fam]))
        used.add(fam)
        if len(picked) >= k:
            break
    rest = [f for f in families if f not in used]
    rng.shuffle(rest)
    for fam in rest:
        if len(picked) >= k:
            break
        picked.append(rng.choice(families[fam]))
        used.add(fam)
    if not picked:
        picked.append(rng.choice(crit or safe))
    # Canonical CSV that ``pick_scenarios`` already understands.
    return ",".join(dict.fromkeys(picked))


_THEME_ALIASES = {
    "mixed": "mixed",
    "peripheral": "peripheral",
    "periph": "peripheral",
    "side": "peripheral",
    "clear_center": "peripheral",
    "clear-center": "peripheral",
}


def normalize_theme(raw: str) -> str:
    key = str(raw or "mixed").strip().lower().replace("-", "_")
    if key not in _THEME_ALIASES:
        known = ", ".join(sorted(set(_THEME_ALIASES.values())))
        raise ValueError(f"unknown --theme {raw!r}. Known: {known}, side, clear_center")
    return _THEME_ALIASES[key]


def _periph_quotas(n: int) -> dict[str, int]:
    """Empty sidewalk / side-stay / side-cut mix. No compounds."""
    n = max(0, int(n))
    if n == 0:
        return {"empty": 0, "side": 0, "issue": 0, "compound": 0}
    if n == 1:
        return {"empty": 0, "side": 0, "issue": 1, "compound": 0}
    if n == 2:
        return {"empty": 1, "side": 0, "issue": 1, "compound": 0}
    n_empty = max(1, int(round(n * 0.18)))
    n_issue = max(1, int(round(n * 0.45)))
    n_side = n - n_empty - n_issue
    if n_side < 0:
        n_issue += n_side
        n_side = 0
    return {
        "empty": n_empty,
        "side": max(0, n_side),
        "issue": max(0, n_issue),
        "compound": 0,
    }


def build_plan_peripheral(n: int, seed: int, cfg: dict | None = None) -> dict:
    """Empty-center pack: nothing on the gait, events enter from the side."""
    cfg = cfg or get_config()
    rng = random.Random(int(seed))
    safe, _near, crit = peripheral_pools(cfg)
    empty = [x for x in safe if x == "periph_empty"] or ["periph_empty"]
    side = [x for x in safe if x != "periph_empty"]
    issue = list(crit)
    q = _periph_quotas(n)
    recipes: list[str] = []
    recipes.extend(_cycle_draw(rng, empty, q["empty"]))
    recipes.extend(_cycle_draw(rng, side or empty, q["side"]))
    recipes.extend(_cycle_draw(rng, issue or empty, q["issue"]))
    while len(recipes) < n:
        recipes.append(rng.choice(issue or empty))
    recipes = recipes[:n]
    rng.shuffle(recipes)
    return _episodes_from_recipes(recipes, seed, n, q, cfg, rng)


def _episodes_from_recipes(
    recipes: list[str],
    seed: int,
    n: int,
    q: dict[str, int],
    cfg: dict,
    rng: random.Random,
) -> dict:
    episodes = []
    for i, raw in enumerate(recipes):
        names = pick_scenarios(rng, cfg, raw)
        episodes.append(
            {
                "id": i,
                "scenario": ",".join(names),
                "kind": "compound" if len(names) > 1 else "single",
                "bucket": _bucket_of(names[0], cfg) if len(names) == 1 else "compound",
            }
        )
    return {
        "seed": int(seed),
        "n": int(n),
        "quotas": q,
        "episodes": episodes,
    }


def build_plan(
    n: int,
    seed: int,
    cfg: dict | None = None,
    theme: str = "mixed",
) -> dict:
    """Deterministic pack recipe. Same ``(n, seed, theme)`` ⇒ same scenario list."""
    cfg = cfg or get_config()
    theme = normalize_theme(theme)
    if theme == "peripheral":
        return build_plan_peripheral(n, seed, cfg)
    rng = random.Random(int(seed))
    safe, near, crit = _pools(cfg)
    q = _quotas(n)
    recipes: list[str] = []
    recipes.extend(_cycle_draw(rng, safe, q["safe"]))
    recipes.extend(_cycle_draw(rng, near, q["near_miss"]))
    recipes.extend(_cycle_draw(rng, crit, q["critical"]))
    recipes.extend(draw_compound(rng, cfg) for _ in range(q["compound"]))
    # One leftover slot (rounding) — draw from the critical pool.
    while len(recipes) < n:
        recipes.append(rng.choice(crit or safe))
    recipes = recipes[:n]
    rng.shuffle(recipes)
    return _episodes_from_recipes(recipes, seed, n, q, cfg, rng)


def _bucket_of(name: str, cfg: dict) -> str:
    sc = cfg["scenarios"]
    if name in sc["safe_pool"] or name in (sc.get("peripheral_safe") or ()):
        return "safe"
    if name in sc["near_miss_pool"] or name in (sc.get("peripheral_near") or ()):
        return "near_miss"
    if name in sc["critical_pool"] or name in (sc.get("peripheral_critical") or ()):
        return "critical"
    return "other"


def pack_dirname(*, seed: int, n: int, name: str = "", when: datetime | None = None) -> str:
    when = when or datetime.now()
    ts = when.strftime("%Y%m%d_%H%M%S")
    if name:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_-") or "pack"
        return f"{slug}_s{seed}_n{n}"
    return f"pack_{ts}_s{seed}_n{n}"


def unique_pack_dir(root: Path, dirname: str) -> Path:
    path = root / dirname
    if not path.exists():
        return path
    for i in range(2, 1000):
        alt = root / f"{dirname}_{i}"
        if not alt.exists():
            return alt
    raise RuntimeError(f"could not allocate a free pack folder under {root}")


def write_pack(pack_dir: Path, plan: dict, extra: dict) -> Path:
    pack_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(pack_dir),
        **extra,
        **plan,
    }
    path = pack_dir / "pack.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    plan_path = pack_dir / "plan.json"
    plan_path.write_text(
        json.dumps({"seed": plan["seed"], "episodes": plan["episodes"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return plan_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a mixed episode pack under datasets/. "
            "Each episode is a different scenario (balanced singles + compounds); "
            "lighting, path, speed, FOV, people, and colours are randomized per episode."
        ),
    )
    p.add_argument("-n", "--n", "--samples", type=int, default=8, dest="n", metavar="N",
                   help="Number of episodes / videos. Default: 8.")
    p.add_argument("--seed", type=int, default=42, help="Pack + world RNG seed. Default: 42.")
    p.add_argument("--name", type=str, default="", help="Optional pack name (folder slug).")
    p.add_argument("--datasets", type=str, default="", dest="datasets",
                   help="Parent folder. Default: <repo>/datasets.")
    p.add_argument("--media", choices=("frames", "video", "both"), default="both",
                   help="Visual product per episode. Default: both.")
    p.add_argument(
        "--no-rgb",
        action="store_true",
        help="Delete each episode's rgb/ folder after the video is muxed.",
    )
    p.add_argument("--frames", type=int, default=0, help="Override frames_per_episode (0 = config).")
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Episode seconds; frames = round(duration * fps). --frames wins if both set.",
    )
    p.add_argument("--threat-grid", type=int, default=3, dest="threat_grid", metavar="K")
    p.add_argument("--spatial-overlay", action="store_true",
                   help="Also write spatial_overlay.mp4 on each episode.")
    p.add_argument("--biome", type=str, default="auto",
                   help="street | avenue | park | plaza | alley | residential | market | auto.")
    p.add_argument("--ego-mode", "--ego", type=str, default="auto", dest="ego_mode",
                   help="walk | diagonal_cross | crosswalk | erratic | hasty | seated | auto.")
    p.add_argument("--ego-height", type=str, default="auto", dest="ego_height",
                   help="short | typical | tall | auto | metres.")
    p.add_argument("--chaos", type=float, default=None,
                   help="Lock appearance chaos in [0,1] for every episode.")
    p.add_argument("--no-trees", action="store_true",
                   help="Disable procedural trees and grass.")
    p.add_argument("--wind", type=str, default="auto",
                   help="calm | breeze | windy | auto. Default: auto.")
    p.add_argument("--no-render", action="store_true", help="JSON only (no EEVEE / video).")
    p.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip object-box annotations/annotations.json. Spatial matrices are still written.",
    )
    p.add_argument(
        "--theme",
        type=str,
        default="mixed",
        help=(
            "mixed = balanced catalog (default). "
            "peripheral | side | clear_center = empty sidewalk ahead, "
            "threats enter from the side only."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Write pack.json + plan.json and print the mix; do not launch Blender.")
    p.add_argument("--self-test", action="store_true", help="Run the planner unit test and exit.")
    return p.parse_args(argv)


def _print_mix(plan: dict) -> None:
    q = plan["quotas"]
    kinds = Counter(ep["kind"] for ep in plan["episodes"])
    buckets = Counter(ep["bucket"] for ep in plan["episodes"])
    if "empty" in q and "side" in q:
        print(
            f"  quotas  empty={q['empty']}  side={q['side']}  "
            f"issue={q['issue']}  compound={q.get('compound', 0)}"
        )
    else:
        print(
            f"  quotas  safe={q.get('safe', 0)}  near_miss={q.get('near_miss', 0)}  "
            f"critical={q.get('critical', 0)}  compound={q.get('compound', 0)}"
        )
    print(f"  kinds   {dict(kinds)}   buckets {dict(buckets)}")
    for ep in plan["episodes"]:
        mark = "+" if ep["kind"] == "compound" else " "
        print(f"    {ep['id']:04d}{mark}  {ep['scenario']}")


def launch_blender(plan_path: Path, pack_dir: Path, args: argparse.Namespace) -> int:
    run = _ROOT / "run.sh"
    if not run.is_file():
        print(f"missing {run}", file=sys.stderr)
        return 127
    cmd = [
        str(run),
        "--plan", str(plan_path),
        "--output", str(pack_dir),
        "--seed", str(int(args.seed)),
        "--start-episode", "0",
        "--episodes", str(int(args.n)),
        "--media", str(args.media),
        "--threat-grid", str(int(args.threat_grid)),
    ]
    if args.frames > 0:
        cmd += ["--frames", str(int(args.frames))]
    elif float(getattr(args, "duration", 0.0) or 0.0) > 0.0:
        cmd += ["--duration", str(float(args.duration))]
    if args.spatial_overlay:
        cmd.append("--spatial-overlay")
    if args.no_render:
        cmd.append("--no-render")
    if args.no_annotations:
        cmd.append("--no-annotations")
    if args.no_rgb:
        cmd.append("--no-rgb")
    if str(getattr(args, "biome", "auto")) not in ("", "auto"):
        cmd += ["--biome", str(args.biome)]
    if str(getattr(args, "ego_mode", "auto")) not in ("", "auto"):
        cmd += ["--ego-mode", str(args.ego_mode)]
    if str(getattr(args, "ego_height", "auto")) not in ("", "auto"):
        cmd += ["--ego-height", str(args.ego_height)]
    if args.chaos is not None:
        cmd += ["--chaos", str(float(args.chaos))]
    if args.no_trees:
        cmd.append("--no-trees")
    if str(getattr(args, "wind", "auto")) not in ("", "auto"):
        cmd += ["--wind", str(args.wind)]
    print(f"[gen_dataset] exec: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    return int(subprocess.call(cmd, cwd=str(_ROOT), env=env))


def _self_test() -> None:
    cfg = get_config()
    a = build_plan(12, 1, cfg)
    b = build_plan(12, 1, cfg)
    assert [e["scenario"] for e in a["episodes"]] == [e["scenario"] for e in b["episodes"]]
    c = build_plan(12, 2, cfg)
    assert [e["scenario"] for e in a["episodes"]] != [e["scenario"] for e in c["episodes"]]

    q = _quotas(20)
    assert q["safe"] + q["near_miss"] + q["critical"] + q["compound"] == 20, q
    assert q["compound"] >= 1
    assert q["safe"] >= q["near_miss"]

    plan = build_plan(20, 99, cfg)
    assert len(plan["episodes"]) == 20
    n_comp = sum(1 for e in plan["episodes"] if e["kind"] == "compound")
    assert n_comp == q["compound"], (n_comp, q)
    for ep in plan["episodes"]:
        names = pick_scenarios(random.Random(0), cfg, ep["scenario"])
        assert names, ep
        fams = [_family_of(n) for n in names]
        if len(names) > 1:
            assert len(fams) == len(set(fams)), (names, fams)

    assert normalize_theme("side") == "peripheral"
    side = build_plan(16, 5, cfg, theme="peripheral")
    side2 = build_plan(16, 5, cfg, theme="side")
    assert [e["scenario"] for e in side["episodes"]] == [e["scenario"] for e in side2["episodes"]]
    assert len(side["episodes"]) == 16
    assert side["quotas"]["compound"] == 0
    for ep in side["episodes"]:
        names = ep["scenario"].split(",")
        assert names and set(names).issubset(CLEAR_CENTER_SCENARIOS), ep
        assert ep["kind"] == "single"
    mixed_names = {e["scenario"] for e in build_plan(16, 5, cfg)["episodes"]}
    assert not mixed_names.issubset(CLEAR_CENTER_SCENARIOS)
    fps = int(cfg["render"]["fps"])
    assert int(round(5.0 * fps)) == int(cfg["render"]["frames_per_episode"])
    safe, near, crit = _pools(cfg)
    catalog = set(safe + near + crit)
    assert "tree_on_path" in catalog
    assert "lamp_on_path" in catalog
    assert "hasty_look" in catalog
    print("gen_dataset self-test: OK")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if int(args.n) < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    if int(args.threat_grid) < 1:
        print("--threat-grid must be >= 1", file=sys.stderr)
        return 2

    cfg = get_config()
    try:
        theme = normalize_theme(args.theme)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    plan = build_plan(int(args.n), int(args.seed), cfg, theme=theme)
    parent = Path(args.datasets) if args.datasets else (_ROOT / "datasets")
    parent.mkdir(parents=True, exist_ok=True)
    dirname = pack_dirname(seed=int(args.seed), n=int(args.n), name=str(args.name or ""))
    pack_dir = unique_pack_dir(parent, dirname)
    extra = {
        "media": args.media,
        "frames": int(args.frames),
        "threat_grid": int(args.threat_grid),
        "spatial_overlay": bool(args.spatial_overlay),
        "no_render": bool(args.no_render),
        "no_annotations": bool(args.no_annotations),
        "no_rgb": bool(args.no_rgb),
        "biome": str(args.biome),
        "ego_mode": str(args.ego_mode),
        "ego_height": str(getattr(args, "ego_height", "auto")),
        "chaos": args.chaos,
        "no_trees": bool(args.no_trees),
        "theme": theme,
        "note": (
            "Scenario tokens are chosen here. Lighting, weather, path type, "
            "walk speed, FOV, body proportions, colours, and clutter are "
            "drawn per episode from the pipeline RNG (same --seed)."
            + (
                " Theme 'peripheral': empty sidewalk ahead; cars/people stay "
                "on the side until they cut in."
                if theme == "peripheral"
                else ""
            )
        ),
    }
    plan_path = write_pack(pack_dir, plan, extra)
    print(f"[gen_dataset] pack → {pack_dir}", flush=True)
    extra_flags = []
    if args.no_rgb:
        extra_flags.append("no-rgb")
    if args.spatial_overlay:
        extra_flags.append("spatial-overlay")
    if args.biome != "auto":
        extra_flags.append(f"biome={args.biome}")
    if args.ego_mode != "auto":
        extra_flags.append(f"ego={args.ego_mode}")
    flag_s = ("  " + " ".join(extra_flags)) if extra_flags else ""
    print(
        f"  seed={args.seed}  n={args.n}  theme={theme}  media={args.media}{flag_s}",
        flush=True,
    )
    _print_mix(plan)
    sys.stdout.flush()
    if args.dry_run:
        print("[gen_dataset] dry-run; Blender not launched")
        return 0
    rc = launch_blender(plan_path, pack_dir, args)
    if rc != 0:
        print(f"[gen_dataset] blender exited {rc}", file=sys.stderr)
    else:
        print(f"[gen_dataset] done → {pack_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
