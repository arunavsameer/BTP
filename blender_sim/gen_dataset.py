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
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "ped": (
        "oncoming_pedestrian",
        "parallel_pedestrian",
        "same_dir_closing",
        "same_dir_hold",
        "same_dir_pulling",
        "sudden_stop",
        "child_darting",
    ),
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
    dump = {k: plan[k] for k in plan}
    if "episodes" not in dump and "samples" not in dump:
        dump["episodes"] = []
    plan_path.write_text(json.dumps(dump, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
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
    p.add_argument(
        "--groups",
        action="store_true",
        help=(
            "Outcome-group pack: each sample is collision plus one miss side "
            "(miss_left or miss_right, 50/50 from --seed) with shared prefix "
            "frames, plus --bg-variants background-only worlds. Folder layout is "
            "sample_XXXX_<scenario>/{collision,miss_left|miss_right}/bg_YY/."
        ),
    )
    p.add_argument(
        "--pairs",
        action="store_true",
        help=(
            "Pair pack: --n of EACH family (parallel_person, static_shape, crowd) "
            "in rotating order P,S,C,P,S,C so a cancel leaves ~equal counts. "
            "Pair families write collision plus one miss side under bg_00/bg_01. "
            "Crowd writes the same clip under sample/bg_YY (no collision/miss). "
            "Not the old 3-outcome groups pack; pixels are not prefix-cached. "
            "Resume skips clips that already have preview+overlay."
        ),
    )
    p.add_argument(
        "--bg-variants",
        type=int,
        default=2,
        dest="bg_variants",
        metavar="K",
        help=(
            "Background variants per sample for --groups and --pairs. Default: 2. "
            "Each variant reseeds atmosphere/scatter/clothes; motion stays on scene_seed. "
            "Pairs also pin injector kinematics from existing bg_00 episode.json on resume."
        ),
    )
    p.add_argument(
        "--group-scenarios",
        type=str,
        default="",
        dest="group_scenarios",
        help=(
            "Comma-separated forkable scenario names for --groups "
            "(e.g. jaywalker,car_cut_in). Default: cycle the catalog."
        ),
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Blender CPU threads for each Blender worker (0 = AUTO). Sets BTP_THREADS.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        metavar="J",
        help=(
            "With --pipeline (default when J>1): --gpu-jobs EEVEE workers + "
            "(J - gpu-jobs) CPU sim workers. Default gpu-jobs=1 so J=2 is "
            "1 EEVEE + 1 CPU sim-ahead. --no-pipeline restores J full CPU+GPU "
            "processes (EEVEE may contend). Env: BTP_JOBS. Default 1."
        ),
    )
    p.add_argument(
        "--pipeline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "True clip pipeline: CPU sim workers bake upcoming clips while GPU "
            "workers EEVEE finished bakes. GPU concurrency is --gpu-jobs (default 1). "
            "Default: on when --jobs>1. --no-pipeline = old N concurrent full Blenders."
        ),
    )
    p.add_argument(
        "--gpu-jobs",
        type=int,
        default=None,
        metavar="G",
        help=(
            "Concurrent EEVEE workers in --pipeline mode. Default 1 (safe). "
            "Example: --jobs 4 --gpu-jobs 2 → 2 EEVEE + 2 CPU sim. "
            "Two EEVEEs on one 8 GB GPU can OOM or run slower. Env: BTP_GPU_JOBS."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Write pack.json + plan.json and print the mix; do not launch Blender.")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse datasets/<name>_s<seed>_n<N> if it already exists instead of "
            "allocating a new folder. Finished clips are skipped."
        ),
    )
    p.add_argument("--self-test", action="store_true", help="Run the planner unit test and exit.")
    return p.parse_args(argv)


def _print_mix(plan: dict) -> None:
    if str(plan.get("layout") or "") == "pairs":
        from pair_pack import FAMILY_LETTER, iter_pair_clips

        clips = list(iter_pair_clips(plan))
        rotation = str(plan.get("rotation") or "")
        print(
            f"  layout=pairs  n={plan.get('n')}/family  "
            f"samples={len(plan.get('samples') or [])}  "
            f"clips={len(clips)}  prefix_cache={plan.get('prefix_cache')}"
        )
        print(f"  rotation: {rotation}")
        for sample in plan.get("samples") or []:
            letter = FAMILY_LETTER.get(str(sample.get("family") or ""), "?")
            outs = ",".join(sample.get("outcomes") or [])
            layout = sample.get("files_layout") or "outcome_bg"
            if layout in ("flat", "bg_only"):
                print(
                    f"    {letter}  {sample['dir']}  crowd  "
                    f"scenario={sample['scenario']}  "
                    f"bg={len(sample.get('backgrounds') or [])}"
                )
            else:
                miss = sample.get("miss_side") or "?"
                print(
                    f"    {letter}  {sample['dir']}  outcomes=[{outs}]  "
                    f"bg={len(sample.get('backgrounds') or [])}  miss={miss}"
                )
        return
    if str(plan.get("layout") or "") == "groups":
        n_clips = sum(
            len(s.get("outcomes") or []) * len(s.get("backgrounds") or [])
            for s in plan.get("samples") or []
        )
        print(
            f"  layout=groups  samples={plan.get('n')}  "
            f"bg={plan.get('n_bg')}  clips={n_clips}  "
            f"diverge={plan.get('diverge_s')}s  hit={plan.get('hit_s')}s"
        )
        for sample in plan.get("samples") or []:
            styles = ",".join(b.get("style", "?") for b in sample.get("backgrounds") or [])
            lights = ",".join(b.get("lighting", "?") for b in sample.get("backgrounds") or [])
            outs = ",".join(sample.get("outcomes") or [])
            print(
                f"    {sample['dir']}  outcomes=[{outs}]  "
                f"bg=[{styles}]  light=[{lights}]"
            )
        return
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


def resolve_jobs(args: argparse.Namespace) -> int:
    raw = getattr(args, "jobs", None)
    if raw is None:
        env = os.environ.get("BTP_JOBS", "").strip()
        raw = int(env) if env else 1
    return max(1, int(raw))


def resolve_pipeline(args: argparse.Namespace, jobs: int) -> bool:
    flag = getattr(args, "pipeline", None)
    if flag is None:
        return jobs > 1
    return bool(flag) and jobs > 1


def resolve_gpu_jobs(args: argparse.Namespace, jobs: int) -> int:
    raw = getattr(args, "gpu_jobs", None)
    if raw is None:
        env = os.environ.get("BTP_GPU_JOBS", "").strip()
        raw = int(env) if env else 1
    n_gpu = max(1, int(raw))
    # Keep at least one sim worker when more than one job is requested.
    if jobs > 1:
        n_gpu = min(n_gpu, jobs - 1)
    return max(1, n_gpu)


def _blender_child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if os.environ.get("OMP_NUM_THREADS", "").strip() == "1":
        env.pop("OMP_NUM_THREADS", None)
    if getattr(args, "threads", None) is not None:
        n = int(args.threads)
        env["BTP_THREADS"] = str(n)
        if n > 0:
            env["BLENDER_SYSTEM_THREADS"] = str(n)
    return env


def build_blender_cmd(
    plan_path: Path,
    pack_dir: Path,
    args: argparse.Namespace,
    *,
    plan_sample_idx: str = "",
    plan_episode_idx: str = "",
    plan_clip_idx: str = "",
    pipeline_stage: str = "",
) -> list[str]:
    run = _ROOT / "run.sh"
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
    if plan_sample_idx:
        cmd += ["--plan-sample-idx", plan_sample_idx]
    if plan_episode_idx:
        cmd += ["--plan-episode-idx", plan_episode_idx]
    if plan_clip_idx:
        cmd += ["--plan-clip-idx", str(plan_clip_idx)]
    if pipeline_stage:
        cmd += ["--pipeline-stage", str(pipeline_stage)]
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
    if getattr(args, "threads", None) is not None:
        cmd += ["--threads", str(int(args.threads))]
    return cmd


def plan_job_shards(plan: dict) -> list[tuple[str, str]]:
    """Return (plan_sample_idx, plan_episode_idx) shards safe for parallel Blender."""
    layout = str(plan.get("layout") or "")
    if layout in ("pairs", "groups"):
        n = len(plan.get("samples") or [])
        return [(str(i), "") for i in range(n)]
    episodes = plan.get("episodes") or []
    if episodes:
        return [("", str(i)) for i in range(len(episodes))]
    return [("", "")]


def _clip_done(
    pack_dir: Path,
    clip: dict,
    *,
    layout: str,
    want_overlay: bool,
    video_name: str = "preview.mp4",
) -> bool:
    if layout == "pairs":
        sample_dir = pack_dir / str(clip["sample_dir"])
        rel = str(clip.get("relpath") or "")
        ep_dir = (sample_dir / rel) if rel else sample_dir
    else:
        ep_dir = pack_dir / str(clip["sample_dir"]) / str(clip["relpath"])
    video_path = ep_dir / video_name
    if not video_path.is_file():
        return False
    if want_overlay:
        return (ep_dir / "spatial_overlay.mp4").is_file()
    return True


def count_pack_clips_done(
    pack_dir: Path, plan: dict, args: argparse.Namespace
) -> tuple[int, int]:
    """How many planned clips have required deliverables on disk."""
    layout = str(plan.get("layout") or "")
    want_overlay = bool(args.spatial_overlay) and not bool(args.no_render)
    if layout == "pairs":
        from pair_pack import iter_pair_clips

        clips = list(iter_pair_clips(plan))
        n_ok = sum(
            1
            for c in clips
            if _clip_done(pack_dir, c, layout=layout, want_overlay=want_overlay)
        )
        return n_ok, len(clips)
    if layout == "groups":
        from outcome_groups import iter_clips

        clips = list(iter_clips(plan))
        n_ok = sum(
            1
            for c in clips
            if _clip_done(pack_dir, c, layout=layout, want_overlay=want_overlay)
        )
        return n_ok, len(clips)
    return 0, 0


def verify_pack_complete(
    pack_dir: Path, plan: dict, args: argparse.Namespace
) -> int:
    n_ok, n_clips = count_pack_clips_done(pack_dir, plan, args)
    if n_clips <= 0:
        return 0
    if n_ok >= n_clips:
        return 0
    layout = str(plan.get("layout") or "pack")
    print(
        f"[gen_dataset] pack incomplete: {n_ok}/{n_clips} clip(s) on disk ({layout})",
        file=sys.stderr,
    )
    want_overlay = bool(args.spatial_overlay) and not bool(args.no_render)
    from pair_pack import iter_pair_clips
    from outcome_groups import iter_clips

    if layout == "pairs":
        todo = list(iter_pair_clips(plan))
    elif layout == "groups":
        todo = list(iter_clips(plan))
    else:
        todo = []
    for clip in todo:
        if not _clip_done(pack_dir, clip, layout=layout, want_overlay=want_overlay):
            rel = clip.get("relpath") or clip.get("outcome") or "?"
            print(
                f"  missing: {clip.get('sample_dir', '?')}/{rel}",
                file=sys.stderr,
            )
    return 2


def finalize_pack_index(pack_dir: Path, plan: dict, args: argparse.Namespace) -> None:
    layout = str(plan.get("layout") or "")
    want_overlay = bool(args.spatial_overlay) and not bool(args.no_render)
    if layout == "pairs":
        from pair_pack import iter_pair_clips

        clips = list(iter_pair_clips(plan))
        index_rows = []
        for clip in clips:
            if not _clip_done(pack_dir, clip, layout=layout, want_overlay=want_overlay):
                continue
            outcome = str(clip["outcome"])
            y = None if outcome == "clip" else (1 if outcome == "collision" else 0)
            sample_dir = pack_dir / str(clip["sample_dir"])
            rel = str(clip.get("relpath") or "")
            ep_dir = (sample_dir / rel) if rel else sample_dir
            index_rows.append(
                {
                    "sample_id": clip["sample_id"],
                    "family": clip["family"],
                    "scenario": clip["scenario"],
                    "outcome": outcome,
                    "bg_id": clip.get("bg_id"),
                    "path": str(ep_dir.relative_to(pack_dir)),
                    "video": "preview.mp4",
                    "spatial_overlay": "spatial_overlay.mp4" if want_overlay else None,
                    "y": y,
                }
            )
        payload = {
            "layout": "pairs",
            "n_ok": len(index_rows),
            "n_clips": len(clips),
            "rotation": plan.get("rotation"),
            "prefix_cache": False,
            "train_glob": (
                "{parallel_person,static_shape}/sample_*/"
                "{collision,miss_left,miss_right}/bg_*/preview.mp4"
            ),
            "crowd_glob": "crowd/sample_*/bg_*/preview.mp4",
            "clips": index_rows,
        }
        (pack_dir / "pairs_index.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return
    if layout == "groups":
        from outcome_groups import iter_clips

        clips = list(iter_clips(plan))
        index_rows = []
        for clip in clips:
            if not _clip_done(pack_dir, clip, layout=layout, want_overlay=want_overlay):
                continue
            sample_dir = pack_dir / str(clip["sample_dir"])
            ep_dir = sample_dir / str(clip["relpath"])
            outcome = str(clip["outcome"])
            index_rows.append(
                {
                    "sample_id": clip["sample_id"],
                    "scenario": clip["scenario"],
                    "outcome": outcome,
                    "bg_id": clip["bg_id"],
                    "bg_style": clip["bg_style"],
                    "bg_lighting": clip.get("bg_lighting"),
                    "path": str(ep_dir.relative_to(pack_dir)),
                    "video": "preview.mp4",
                    "spatial_overlay": "spatial_overlay.mp4" if want_overlay else None,
                    "diverge_frame": clip["diverge_frame"],
                    "y": 1 if outcome == "collision" else 0,
                }
            )
        payload = {
            "layout": "groups",
            "n_ok": len(index_rows),
            "n_clips": len(clips),
            "train_glob": "sample_*/{collision,miss_left,miss_right}/bg_*/preview.mp4",
            "clips": index_rows,
        }
        (pack_dir / "groups_index.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def launch_blender(plan_path: Path, pack_dir: Path, args: argparse.Namespace) -> int:
    run = _ROOT / "run.sh"
    if not run.is_file():
        print(f"missing {run}", file=sys.stderr)
        return 127
    jobs = resolve_jobs(args)
    env = _blender_child_env(args)
    plan_doc: dict = {}
    if plan_path.is_file():
        try:
            plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            plan_doc = {}
    shards = plan_job_shards(plan_doc) if plan_doc else [("", "")]
    if jobs <= 1 or len(shards) <= 1:
        cmd = build_blender_cmd(plan_path, pack_dir, args)
        print(f"[gen_dataset] exec: {' '.join(cmd)}", flush=True)
        rc = int(subprocess.call(cmd, cwd=str(_ROOT), env=env))
        if plan_doc:
            finalize_pack_index(pack_dir, plan_doc, args)
            pack_rc = verify_pack_complete(pack_dir, plan_doc, args)
            if pack_rc != 0 and rc == 0:
                rc = pack_rc
        return rc

    use_pipe = resolve_pipeline(args, jobs)
    if use_pipe:
        n_gpu = resolve_gpu_jobs(args, jobs)
        n_sim = max(1, jobs - n_gpu)
        print(
            f"[gen_dataset] pipeline jobs={jobs}  "
            f"({n_gpu} GPU EEVEE + {n_sim} CPU sim worker(s); "
            f"sim stays ahead of render)",
            flush=True,
        )
        lock_path = pack_dir / ".btp_gpu.lock"
        env = dict(env)
        env["BTP_GPU_LOCK"] = str(lock_path)
        env["BTP_GPU_SLOTS"] = str(n_gpu)
        rc = _launch_pipeline(shards, jobs, plan_path, pack_dir, args, env)
    else:
        print(
            f"[gen_dataset] parallel jobs={jobs}  shards={len(shards)}  "
            f"(one Blender per sample/episode; collision+miss stay serial per shard; "
            f"EEVEE may contend)",
            flush=True,
        )

        def _run_shard(shard: tuple[str, str]) -> int:
            sample_idx, ep_idx = shard
            cmd = build_blender_cmd(
                plan_path,
                pack_dir,
                args,
                plan_sample_idx=sample_idx,
                plan_episode_idx=ep_idx,
            )
            tag = sample_idx or ep_idx or "all"
            print(f"[gen_dataset] shard {tag}: {' '.join(cmd)}", flush=True)
            return int(subprocess.call(cmd, cwd=str(_ROOT), env=env))

        rc = 0
        workers = min(jobs, len(shards))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_shard, shard): shard for shard in shards}
            for fut in as_completed(futures):
                sample_idx, ep_idx = futures[fut]
                shard_rc = int(fut.result())
                tag = sample_idx or ep_idx or "all"
                print(f"[gen_dataset] shard {tag} exited {shard_rc}", flush=True)
                if shard_rc != 0:
                    rc = shard_rc
    if plan_doc:
        finalize_pack_index(pack_dir, plan_doc, args)
        pack_rc = verify_pack_complete(pack_dir, plan_doc, args)
        if pack_rc != 0:
            rc = pack_rc if rc == 0 else rc
    return rc


def _pipeline_clip_dir(pack_dir: Path, plan: dict, clip: dict) -> Path:
    layout = str(plan.get("layout") or "")
    sample_dir = pack_dir / str(clip["sample_dir"])
    rel = str(clip.get("relpath") or "")
    if layout == "pairs":
        return (sample_dir / rel) if rel else sample_dir
    return sample_dir / rel


def pipeline_work_units(
    plan: dict, pack_dir: Path, args: argparse.Namespace
) -> list[dict]:
    """Clip-level sim→GPU units. Finished videos are omitted (--resume)."""
    layout = str(plan.get("layout") or "")
    want_overlay = bool(args.spatial_overlay) and not bool(args.no_render)
    units: list[dict] = []
    if layout == "pairs":
        from pair_pack import iter_pair_clips

        clips = list(iter_pair_clips(plan))
        for i, clip in enumerate(clips):
            ep_dir = _pipeline_clip_dir(pack_dir, plan, clip)
            done = _clip_done(pack_dir, clip, layout=layout, want_overlay=want_overlay)
            bake = (
                (ep_dir / "_baked.blend").is_file()
                and (ep_dir / "_pose_cache.json").is_file()
                and (ep_dir / "_bake_meta.json").is_file()
            )
            units.append(
                {
                    "idx": i,
                    "kind": "clip",
                    "tag": f"{clip.get('sample_dir')}/{clip.get('relpath') or clip.get('outcome')}",
                    "ep_dir": ep_dir,
                    "done": done,
                    "bake": bake,
                    "sample_idx": "",
                    "episode_idx": "",
                }
            )
        return units
    if layout == "groups":
        from outcome_groups import iter_clips

        clips = list(iter_clips(plan))
        for i, clip in enumerate(clips):
            ep_dir = _pipeline_clip_dir(pack_dir, plan, clip)
            done = _clip_done(pack_dir, clip, layout=layout, want_overlay=want_overlay)
            bake = (
                (ep_dir / "_baked.blend").is_file()
                and (ep_dir / "_pose_cache.json").is_file()
                and (ep_dir / "_bake_meta.json").is_file()
            )
            units.append(
                {
                    "idx": i,
                    "kind": "clip",
                    "tag": f"{clip.get('sample_dir')}/{clip.get('relpath')}",
                    "ep_dir": ep_dir,
                    "done": done,
                    "bake": bake,
                    "sample_idx": "",
                    "episode_idx": "",
                }
            )
        return units
    episodes = list(plan.get("episodes") or [])
    for i, ep in enumerate(episodes):
        marker = pack_dir / f"_pipe_{i}.json"
        ep_dir = None
        if marker.is_file():
            try:
                info = json.loads(marker.read_text(encoding="utf-8"))
                rel = str(info.get("dir") or "")
                if rel:
                    ep_dir = pack_dir / rel
            except json.JSONDecodeError:
                ep_dir = None
        bake = bool(
            ep_dir is not None
            and (ep_dir / "_baked.blend").is_file()
            and (ep_dir / "_pose_cache.json").is_file()
        )
        video_ok = bool(ep_dir is not None and (ep_dir / "preview.mp4").is_file())
        if want_overlay:
            video_ok = video_ok and bool(ep_dir is not None and (ep_dir / "spatial_overlay.mp4").is_file())
        units.append(
            {
                "idx": i,
                "kind": "episode",
                "tag": f"episode[{i}]/{ep.get('scenario')}",
                "ep_dir": ep_dir,
                "done": video_ok,
                "bake": bake,
                "sample_idx": "",
                "episode_idx": str(i),
            }
        )
    return units


def _launch_pipeline(
    shards: list[tuple[str, str]],
    jobs: int,
    plan_path: Path,
    pack_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> int:
    """CPU sim workers stay ahead; one GPU process EEVEEs finished bakes."""
    plan_doc: dict = {}
    if plan_path.is_file():
        try:
            plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            plan_doc = {}
    units = pipeline_work_units(plan_doc, pack_dir, args)
    todo = [u for u in units if not u["done"]]
    if not todo:
        print("[gen_dataset] pipeline: all clips already on disk", flush=True)
        return 0

    n_gpu = resolve_gpu_jobs(args, jobs)
    n_sim = max(1, int(jobs) - n_gpu)
    cond = threading.Condition()
    sim_pending = [u for u in todo if not u["bake"]]
    gpu_ready: list[dict] = [u for u in todo if u["bake"]]
    gpu_pending_ids = {u["idx"] for u in todo}
    inflight_sim: dict[threading.Thread, dict] = {}
    inflight_gpu: dict[threading.Thread, dict] = {}
    results: dict[tuple[str, int], int] = {}

    def _sim_env() -> dict[str, str]:
        e = dict(env)
        e.pop("BTP_GPU_LOCK", None)
        e["BTP_GPU"] = os.environ.get("BTP_SIM_GPU", "amd")
        e["CUDA_VISIBLE_DEVICES"] = ""
        e.pop("BTP_BLEND_FILE", None)
        return e

    def _pump(kind: str, unit: dict, cmd: list[str], child_env: dict[str, str]) -> None:
        tag = unit["tag"]
        print(f"[gen_dataset] {kind} {tag}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"[{kind} {unit['idx']}] {line}")
            sys.stdout.flush()
        rc = int(proc.wait())
        with cond:
            results[(kind, int(unit["idx"]))] = rc
            cond.notify_all()

    def _start_sim(unit: dict) -> None:
        kwargs = {
            "plan_clip_idx": str(unit["idx"]),
            "pipeline_stage": "sim",
        }
        if unit["kind"] == "episode":
            kwargs["plan_episode_idx"] = str(unit["episode_idx"])
        cmd = build_blender_cmd(plan_path, pack_dir, args, **kwargs)
        thread = threading.Thread(
            target=_pump, args=("sim", unit, cmd, _sim_env()), daemon=True,
        )
        inflight_sim[thread] = unit
        thread.start()

    def _start_gpu(unit: dict) -> None:
        child = dict(env)
        ep_dir = unit.get("ep_dir")
        if ep_dir is None:
            marker = pack_dir / f"_pipe_{int(unit['idx'])}.json"
            if marker.is_file():
                try:
                    info = json.loads(marker.read_text(encoding="utf-8"))
                    rel = str(info.get("dir") or "")
                    if rel:
                        ep_dir = pack_dir / rel
                        unit["ep_dir"] = ep_dir
                except json.JSONDecodeError:
                    ep_dir = None
        if ep_dir is not None:
            child["BTP_BLEND_FILE"] = str(Path(ep_dir) / "_baked.blend")
        kwargs = {
            "plan_clip_idx": str(unit["idx"]),
            "pipeline_stage": "render",
        }
        if unit["kind"] == "episode":
            kwargs["plan_episode_idx"] = str(unit["episode_idx"])
        cmd = build_blender_cmd(plan_path, pack_dir, args, **kwargs)
        thread = threading.Thread(
            target=_pump, args=("gpu", unit, cmd, child), daemon=True,
        )
        inflight_gpu[thread] = unit
        thread.start()

    with cond:
        while sim_pending or gpu_ready or inflight_sim or inflight_gpu or gpu_pending_ids:
            while sim_pending and len(inflight_sim) < n_sim:
                _start_sim(sim_pending.pop(0))
            while gpu_ready and len(inflight_gpu) < n_gpu:
                _start_gpu(gpu_ready.pop(0))

            done_sim = [t for t in list(inflight_sim) if not t.is_alive()]
            for thread in done_sim:
                unit = inflight_sim.pop(thread)
                thread.join()
                rc = int(results.get(("sim", int(unit["idx"])), 1))
                print(f"[gen_dataset] sim {unit['tag']} exited {rc}", flush=True)
                if rc == 0 and int(unit["idx"]) in gpu_pending_ids:
                    gpu_ready.append(unit)
                elif rc != 0:
                    gpu_pending_ids.discard(int(unit["idx"]))

            done_gpu = [t for t in list(inflight_gpu) if not t.is_alive()]
            for thread in done_gpu:
                unit = inflight_gpu.pop(thread)
                thread.join()
                rc = int(results.get(("gpu", int(unit["idx"])), 1))
                print(f"[gen_dataset] gpu {unit['tag']} exited {rc}", flush=True)
                gpu_pending_ids.discard(int(unit["idx"]))

            while sim_pending and len(inflight_sim) < n_sim:
                _start_sim(sim_pending.pop(0))
            while gpu_ready and len(inflight_gpu) < n_gpu:
                _start_gpu(gpu_ready.pop(0))

            if inflight_sim or inflight_gpu:
                cond.wait(timeout=1.0)
            elif sim_pending or gpu_ready:
                continue
            elif gpu_pending_ids:
                print(
                    "[gen_dataset] pipeline stalled: clips waiting without bake",
                    file=sys.stderr,
                )
                break
            else:
                break

    rc = 0
    for (kind, _idx), shard_rc in results.items():
        if int(shard_rc) != 0:
            rc = int(shard_rc)
            print(f"[gen_dataset] pipeline {kind} failed rc={shard_rc}", file=sys.stderr)
            break
    try:
        (pack_dir / ".btp_gpu.lock").unlink(missing_ok=True)
    except OSError:
        pass
    return rc


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
    assert int(round(15.0 * fps)) == int(cfg["render"]["frames_per_episode"])
    from outcome_groups import FORKABLE_SCENARIOS, MISS_SIDES, build_group_plan, iter_clips
    g = build_group_plan(3, 11, cfg, n_bg=2)
    g2 = build_group_plan(3, 11, cfg, n_bg=2)
    assert g["samples"] == g2["samples"]
    assert g["layout"] == "groups"
    g_clips = list(iter_clips(g))
    assert len(g_clips) == 3 * 2 * 2
    for sample in g["samples"]:
        lights = [b["lighting"] for b in sample["backgrounds"]]
        assert len(set(lights)) == len(lights), lights
        outs = list(sample["outcomes"])
        assert outs[0] == "collision" and len(outs) == 2, outs
        assert outs[1] in MISS_SIDES, outs
        miss = {
            cl["outcome"]
            for cl in g_clips
            if cl["sample_id"] == sample["id"] and cl["outcome"] != "collision"
        }
        assert miss == {outs[1]}, (sample["id"], miss, outs)
    safe, near, crit = _pools(cfg)
    catalog = set(safe + near + crit)
    assert "tree_on_path" in catalog
    assert "lamp_on_path" in catalog
    assert "hasty_look" in catalog
    assert "same_dir_closing" in catalog
    assert "same_dir_hold" in catalog
    assert "same_dir_pulling" in catalog
    assert not {"same_dir_closing", "same_dir_hold", "same_dir_pulling"} & set(FORKABLE_SCENARIOS)
    from scenario_compose import all_scenario_names
    named = set(all_scenario_names(cfg))
    assert "static_shape" in named
    assert "static_shape" not in catalog
    assert "static_shape" not in FORKABLE_SCENARIOS
    assert "parallel_person" in named
    assert "parallel_person" not in catalog
    assert "parallel_person" not in FORKABLE_SCENARIOS

    from pair_pack import build_pair_plan, iter_pair_clips as iter_pair
    pp = build_pair_plan(2, 21, cfg)
    assert pp["layout"] == "pairs"
    assert pp["rotation"] == "P,S,C,P,S,C"
    assert len(list(iter_pair(pp))) == 20
    assert pp["prefix_cache"] is False
    pipe_args = argparse.Namespace(spatial_overlay=True, no_render=False)
    pipe_root = Path(tempfile.mkdtemp(prefix="btp_pipe_"))
    units = pipeline_work_units(pp, pipe_root, pipe_args)
    assert len(units) == len(list(iter_pair(pp)))
    assert units[0]["idx"] == 0 and units[0]["done"] is False
    assert units[1]["idx"] == 1
    assert "collision" in str(units[0]["tag"]) or "clip" in str(units[0]["tag"])
    assert resolve_gpu_jobs(argparse.Namespace(gpu_jobs=None), 2) == 1
    assert resolve_gpu_jobs(argparse.Namespace(gpu_jobs=2), 4) == 2
    assert resolve_gpu_jobs(argparse.Namespace(gpu_jobs=2), 2) == 1

    parent = Path(tempfile.mkdtemp(prefix="btp_ds_"))
    existing = parent / pack_dirname(seed=1, n=2, name="groups15")
    existing.mkdir()
    (existing / "keep.txt").write_text("ok", encoding="utf-8")
    rc = main([
        "--dry-run", "--groups", "-n", "2", "--seed", "1",
        "--datasets", str(parent), "--name", "groups15",
    ])
    assert rc == 0
    kids = sorted(p.name for p in parent.iterdir())
    assert kids == [existing.name], kids
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "ok"

    import io
    from contextlib import redirect_stdout

    pair_parent = Path(tempfile.mkdtemp(prefix="btp_pairs_"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_p = main([
            "--dry-run", "--pairs", "-n", "2", "--seed", "21",
            "--datasets", str(pair_parent), "--name", "pairs15_side_static_crowd",
        ])
    assert rc_p == 0
    printed = buf.getvalue()
    assert "P,S,C,P,S,C" in printed, printed
    assert "rotation:" in printed
    assert list(pair_parent.iterdir()) == []
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
    if bool(args.groups) and bool(getattr(args, "pairs", False)):
        print("--groups and --pairs are mutually exclusive", file=sys.stderr)
        return 2
    if bool(args.groups):
        from outcome_groups import build_group_plan
        n_bg = max(1, min(8, int(args.bg_variants)))
        names = [
            p.strip() for p in str(getattr(args, "group_scenarios", "") or "").split(",")
            if p.strip()
        ]
        plan = build_group_plan(
            int(args.n), int(args.seed), cfg, n_bg=n_bg,
            scenarios=names or None,
        )
        if str(getattr(args, "ego_mode", "auto")) in ("", "auto"):
            args.ego_mode = "walk"
    elif bool(getattr(args, "pairs", False)):
        from pair_pack import build_pair_plan
        plan = build_pair_plan(
            int(args.n), int(args.seed), cfg,
            n_bg=max(1, min(8, int(args.bg_variants))),
        )
        if str(getattr(args, "ego_mode", "auto")) in ("", "auto"):
            args.ego_mode = "walk"
    else:
        plan = build_plan(int(args.n), int(args.seed), cfg, theme=theme)
    parent = Path(args.datasets) if args.datasets else (_ROOT / "datasets")
    parent.mkdir(parents=True, exist_ok=True)
    pack_name = str(args.name or "")
    if bool(args.groups) and not pack_name:
        pack_name = "groups15"
    if bool(getattr(args, "pairs", False)) and not pack_name:
        pack_name = "pairs15_side_static_crowd"
    dirname = pack_dirname(seed=int(args.seed), n=int(args.n), name=pack_name)
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
        "theme": (
            "pairs" if bool(getattr(args, "pairs", False))
            else ("groups" if bool(args.groups) else theme)
        ),
        "note": (
            "Pair pack: --n of each family in rotating order "
            "parallel, static_shape, crowd. Collision plus one miss side "
            "(miss_left or miss_right, 50/50 from --seed). K background "
            "variants (bg_00, bg_01, …): same path/ego/injector kinematics, "
            "new atmosphere/scatter/clothes. Crowd clips under sample/bg_YY. "
            "World+ego cached per (sample, bg); miss clip only changes "
            "actor lateral. Do not concat-copy a groups prefix."
            if bool(getattr(args, "pairs", False))
            else (
                "Outcome groups: one threat scenario, collision plus one miss "
                "side (miss_left or miss_right, 50/50 per sample from --seed) "
                "that share frames until diverge_s, and K background variants "
                "(trees/buildings/furniture plus a distinct time-of-day lighting "
                "slot per bg_YY)."
                if bool(args.groups)
                else (
                    "Scenario tokens are chosen here. Lighting, weather, path type, "
                    "walk speed, FOV, body proportions, colours, and clutter are "
                    "drawn per episode from the pipeline RNG (same --seed)."
                    + (
                        " Theme 'peripheral': empty sidewalk ahead; cars/people stay "
                        "on the side until they cut in."
                        if theme == "peripheral"
                        else ""
                    )
                )
            )
        ),
    }
    if bool(getattr(args, "resume", False)):
        would_dir = parent / dirname
        existing_plan = would_dir / "plan.json"
        if existing_plan.is_file() and bool(getattr(args, "pairs", False)):
            try:
                disk = json.loads(existing_plan.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                disk = {}
            if str(disk.get("layout") or "") == "pairs" and disk.get("samples"):
                from pair_pack import upgrade_pair_plan

                upgrade_pair_plan(disk, n_bg=max(1, min(8, int(args.bg_variants))))
                plan = disk
    else:
        would_dir = unique_pack_dir(parent, dirname)
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
        f"  seed={args.seed}  n={args.n}  "
        f"theme={'pairs' if getattr(args, 'pairs', False) else ('groups' if args.groups else theme)}  "
        f"media={args.media}{flag_s}",
        flush=True,
    )
    _print_mix(plan)
    sys.stdout.flush()
    if args.dry_run:
        # Do not mkdir under datasets/: a named pack that already exists
        # used to leave groups15_*_2 leftovers with only plan.json.
        tmp = Path(tempfile.mkdtemp(prefix="btp_dryrun_"))
        plan_path = write_pack(tmp, plan, extra)
        print(f"[gen_dataset] dry-run; would write {would_dir}")
        print(f"  plan → {plan_path}")
        print("[gen_dataset] dry-run; Blender not launched")
        return 0
    pack_dir = would_dir
    if bool(getattr(args, "resume", False)) and bool(getattr(args, "pairs", False)):
        from pair_pack import migrate_pair_pack_crowd

        n_moved = migrate_pair_pack_crowd(pack_dir)
        if n_moved:
            print(f"[gen_dataset] moved {n_moved} crowd sample(s) into bg_00/")
    plan_path = write_pack(pack_dir, plan, extra)
    print(f"[gen_dataset] pack → {pack_dir}", flush=True)
    rc = launch_blender(plan_path, pack_dir, args)
    if rc != 0:
        print(f"[gen_dataset] blender exited {rc}", file=sys.stderr)
    else:
        print(f"[gen_dataset] done → {pack_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
