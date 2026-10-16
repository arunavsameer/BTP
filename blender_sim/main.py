"""Headless orchestrator: build episode → step kinematics → render → write JSON.

Execution (Blender 4.x on Arch Linux)::

    blender --background --python /storage/BTP/blender_sim/main.py -- \\
        --episodes 10 --scenario auto --media both --output /storage/BTP/blender_sim/output

Compound events: ``--scenario jaywalker,car,pothole`` (see IMPLEMENTATION.md §6.5.1).

Everything after ``--`` is argparse. ``--media`` selects PNG frames, an H.264
video, or both. Per-frame JSON is always written. ``--no-render`` skips EEVEE
entirely (annotations only; useful without an EGL context).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# Blender's embedded interpreter does not put the script directory on sys.path.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bpy
from mathutils import Vector

from camera_kinematics import (
    EGO_MODES,
    CameraRig,
    CameraState,
    apply_ego_height,
    choose_ego_profile,
)
from config import apply_camera_fov, get_config
from projection import (
    INNER_BOX_SCALE,
    bound_cache_for,
    camera_projection_matrix,
    camera_view_matrix,
    clear_bound_caches,
    footprint_mouth_corners,
    inset_world_corners,
    project_inner_boxes,
    project_object,
    threat_point_from_corners,
)
from spatial_overlay import overlay_filter_complex, write_heat_sequence
from spatial_threat import (
    empty_grid,
    episode_spatial_payload,
    frame_spatial_entry,
    horizontal_extents,
    splat_object,
)
from threat_math import (
    blender_zup_to_yup,
    classify_threat,
    relative_kinematics,
    vec_to_list,
)
from scenario_compose import (
    CROSS_EGO_SCENARIOS,
    SCENARIO_ALIASES,
    all_scenario_names,
    compose_display,
    pick_scenarios,
    scenario_request_from_tokens,
)
from outcome_groups import iter_clips, link_or_copy_prefix
from world_generator import (
    SUNKEN_CLASSES,
    WIND_LABELS,
    WorldGenerator,
    prepare_still_render,
    release_episode,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _argv_after_double_dash(argv: list[str]) -> list[str]:
    """Blender forwards operator args after a bare ``--``."""
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    # Direct `python main.py ...` (threat-math CI, or bpy-as-module).
    return [a for a in argv[1:] if not a.endswith("main.py")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Egocentric pedestrian synthetic dataset (Blender 4.x / EEVEE-Next).",
    )
    p.add_argument("--episodes", type=int, default=4, help="Number of episodes to generate.")
    p.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help=(
            "First episode index. Default: one past the highest episode_* folder "
            "already in --output, so a new run never overwrites. Pass 0 to start "
            "from episode_0000 (sharding / reproducible packs)."
        ),
    )
    p.add_argument(
        "--scenario",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Injector name, alias, or comma/plus-separated compound. Repeat the "
            "flag to compose: --scenario jaywalker --scenario car --scenario pothole. "
            "Also: --scenario jaywalker,car,pothole. Default: auto."
        ),
    )
    p.add_argument(
        "--scenarios",
        default=None,
        metavar="LIST",
        help=(
            "Same as --scenario; comma or plus separated. "
            "Example: jaywalker,car,pothole. Overrides nothing — joined with "
            "--scenario if both are set."
        ),
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print every named scenario, short aliases, and compound syntax, then exit.",
    )
    p.add_argument("--output", type=str, default="", help="Output root (default: <repo>/output).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=0, help="Override frames_per_episode (0 = config).")
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Episode length in seconds. frames = round(duration * fps). Default 15 s. --frames wins if both set.",
    )
    p.add_argument(
        "--media",
        choices=("frames", "video", "both"),
        default=None,
        help="Visual output: PNG frames, H.264 .mp4, or both. Default: config (both).",
    )
    p.add_argument(
        "--no-rgb",
        action="store_true",
        dest="no_rgb",
        help=(
            "Delete the rgb/ PNG folder after preview.mp4 (and spatial_overlay.mp4) "
            "are written. Use this when you only need the video. "
            "PNGs are kept if video mux fails."
        ),
    )
    p.add_argument("--no-render", action="store_true", help="Skip EEVEE, write JSON only.")
    p.add_argument(
        "--no-annotations",
        action="store_true",
        dest="no_annotations",
        help=(
            "Do not write object-box annotations (annotations/annotations.json). "
            "Spatial threat matrices are still written."
        ),
    )
    p.add_argument("--no-occlusion", action="store_true", help="Disable raycast occlusion flag.")
    p.add_argument(
        "--biome",
        type=str,
        default="auto",
        help=(
            "World type: street | avenue | park | plaza | alley | "
            "residential | market | auto. Default: auto (weighted draw)."
        ),
    )
    p.add_argument(
        "--ego-mode",
        "--ego",
        type=str,
        default="auto",
        dest="ego_mode",
        help=(
            "Ego trajectory: walk | diagonal_cross | crosswalk | erratic | "
            "hasty | seated | auto. 'seated' pins walk_speed to 0. 'hasty' is "
            "high-frequency look jitter. Crossing scenarios force crosswalk "
            "unless this flag locks another mode. Default: auto."
        ),
    )
    p.add_argument(
        "--ego-height",
        type=str,
        default="auto",
        dest="ego_height",
        help=(
            "Standing eye height: short (1.35–1.50) | typical | tall (1.75–1.90) "
            "| auto | metres (e.g. 1.45). Seated still uses the bench range."
        ),
    )
    p.add_argument(
        "--chaos",
        type=float,
        default=None,
        help=(
            "Appearance-randomization dial in [0,1], locked for every episode. "
            "0 reproduces the tame palette; 1 is maximum material anarchy. "
            "Default: per-episode draw from config material_chaos."
        ),
    )
    p.add_argument(
        "--no-trees",
        action="store_true",
        dest="no_trees",
        help="Disable procedural trees and grass (faster, less cluttered).",
    )
    p.add_argument(
        "--wind",
        type=str,
        default="auto",
        help="Leaf wind: calm | breeze | windy | auto. Default: auto.",
    )
    p.add_argument(
        "--hfov",
        type=float,
        default=None,
        help="Horizontal FOV in degrees (locks all episodes). Example: 70.",
    )
    p.add_argument(
        "--lens-mm",
        type=float,
        default=None,
        dest="lens_mm",
        help="Focal length in millimetres (locks all episodes). Ignored if --hfov is set.",
    )
    p.add_argument(
        "--no-random-fov",
        action="store_true",
        help="Do not randomize FOV per episode; use config lens_mm (or --hfov / --lens-mm).",
    )
    p.add_argument(
        "--threat-grid",
        type=int,
        default=3,
        metavar="K",
        dest="threat_grid",
        help=(
            "k×k spatial-annotation resolution (always written). "
            "Default: 3. Output: spatial_annotations/spatial_annotations.json."
        ),
    )
    p.add_argument(
        "--spatial-overlay",
        action="store_true",
        dest="spatial_overlay",
        help=(
            "Write spatial_overlay.mp4: hazy RGB with the k×k threat grid on top "
            "(blue=0, bright red=1). Requires a render (not --no-render)."
        ),
    )
    p.add_argument(
        "--plan",
        type=str,
        default="",
        help="JSON plan from gen_dataset.py (per-episode scenario list). Overrides --scenario.",
    )
    return p.parse_args(argv if argv is not None else _argv_after_double_dash(sys.argv))


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _round_pyr(state: CameraState) -> list[float]:
    return [round(state.pitch, 4), round(state.yaw, 4), round(state.roll, 4)]


def build_frame_record(
    *,
    frame_index: int,
    t: float,
    cam: CameraState,
    cam_obj: Any,
    world: WorldGenerator,
    scene: Any,
    cfg: dict,
    do_occlusion: bool,
    threat_k: int = 3,
) -> tuple[dict, list[list[float]]]:
    """Project every annotatable actor and attach a threat label.

    Also builds the k×k spatial threat matrix (second return value).
    """
    depsgraph = bpy.context.view_layer.depsgraph
    res_x = int(cfg["render"]["resolution_x"])
    res_y = int(cfg["render"]["resolution_y"])
    max_dist = float(cfg["annotation"]["max_distance"])
    occ_eps = float(cfg["annotation"]["occlusion_epsilon"])
    p_cam = (cam.position.x, cam.position.y, cam.position.z)
    v_cam = (cam.velocity.x, cam.velocity.y, cam.velocity.z)

    objects: list[dict] = []
    k = max(1, int(threat_k))
    threat_grid = empty_grid(k)
    view = camera_view_matrix(cam_obj)
    proj = camera_projection_matrix(cam_obj, depsgraph, res_x, res_y)
    cam_x, cam_y, cam_z = p_cam
    tx, ty, tz = cam.tangent.x, cam.tangent.y, cam.tangent.z
    max_dist_pad = max_dist + 5.0
    max_dist_pad2 = max_dist_pad * max_dist_pad
    fps = max(int(cfg["render"]["fps"]), 1)

    for actor in world.annotatable():
        origin = actor.obj.matrix_world.translation
        dx = origin.x - cam_x
        dy = origin.y - cam_y
        dz = origin.z - cam_z
        if dx * dx + dy * dy + dz * dz > max_dist_pad2:
            continue
        if dx * tx + dy * ty + dz * tz < -6.0:
            continue

        if actor.class_name in SUNKEN_CLASSES:
            # The render mesh drops below the pavement so it reads as a hole;
            # the box must come from the mouth slab, not the buried volume.
            corners = getattr(actor, "_world_corners", None)
            if corners is None:
                corners = footprint_mouth_corners(actor.obj)
                actor._world_corners = corners
        elif actor.category == "static":
            corners = getattr(actor, "_world_corners", None)
            if corners is None:
                corners = bound_cache_for(actor.obj).world_corners()
                actor._world_corners = corners
        else:
            # Hold the cache on the actor: `bound_cache_for` is a dict lookup
            # on `as_pointer()`, and this runs for every dynamic actor on
            # every frame of every episode.
            bounds = actor._bounds
            if bounds is None:
                bounds = actor._bounds = bound_cache_for(actor.obj)
            corners = bounds.world_corners()
        if not corners:
            continue

        # A tree's silhouette is its canopy but its hazard is its trunk, so
        # the threat point and the Frenet extents come from `threat_obj`
        # while the 2-D box keeps using the full hierarchy. Streetlamps use
        # the pole mesh only — the cobra arm must not inflate half_lat.
        if actor.class_name in ("tree", "streetlamp"):
            mesh = actor.threat_obj if actor.threat_obj is not None else actor.obj
            threat_corners = bound_cache_for(mesh).part_world_corners(
                only_obj=mesh, scale=1.0,
            ) or corners
        elif actor.threat_obj is not None and actor.threat_obj is not actor.obj:
            threat_corners = bound_cache_for(actor.threat_obj).world_corners() or corners
        else:
            threat_corners = corners

        p_obj_vec = threat_point_from_corners(threat_corners, cam.position.z, actor.threat_mode)
        v_obj_vec = actor.velocity.copy()
        if actor.velocity.length <= 1e-6 and actor.category not in ("static",) and not actor.stopped:
            v_obj_vec = actor.finite_velocity(1.0 / fps)

        p_obj = (p_obj_vec.x, p_obj_vec.y, p_obj_vec.z)
        v_obj = (v_obj_vec.x, v_obj_vec.y, v_obj_vec.z)
        # Parks / plazas have no kerb step: a 1.6 m eye-height offset is not
        # clearance from a bollard. threat_point already clamps Z, and this
        # planar solve is the belt-and-braces so a seated walker vs a trunk
        # cannot be labelled SAFE on a vertical residual.
        planar = str(getattr(world, "biome", "street")) in ("park", "plaza")
        kin = relative_kinematics(
            p_cam,
            v_cam,
            p_obj,
            v_obj,
            rel_speed_eps=float(cfg["threat"]["rel_speed_eps"]),
            converging_eps=float(cfg["threat"]["converging_eps"]),
            planar=planar,
        )
        if kin.distance > max_dist:
            continue

        bbox = project_object(
            actor.obj,
            cam_obj,
            scene,
            depsgraph,
            res_x,
            res_y,
            occlusion_epsilon=occ_eps,
            do_occlusion=do_occlusion,
            view=view,
            proj=proj,
            corners=corners,
        )
        if bbox is None:
            continue

        # Heat paints inscribed mesh-part boxes (cabin, wheel, limb, trunk),
        # not the outer 2-D AABB. JSON `bounding_box_2d` stays the outer box.
        splat_root = actor.obj
        only_part = None
        if actor.class_name in ("tree", "streetlamp") and actor.threat_obj is not None:
            splat_root = actor.threat_obj
            only_part = actor.threat_obj
        if actor.class_name in SUNKEN_CLASSES:
            inner_sets = [inset_world_corners(corners, INNER_BOX_SCALE)]
        else:
            splat_bounds = bound_cache_for(splat_root)
            inner_sets = splat_bounds.inner_world_corner_sets(
                INNER_BOX_SCALE, only_obj=only_part,
            )
        inner_boxes = project_inner_boxes(inner_sets, view, proj, res_x, res_y)
        cam_space = view @ Vector((p_obj[0], p_obj[1], p_obj[2], 1.0))
        look_dot = (
            (p_obj[0] - cam_x) * tx
            + (p_obj[1] - cam_y) * ty
            + (p_obj[2] - cam_z) * tz
        )
        behind_camera = bool(cam_space.z >= -1e-3) or look_dot < -0.05

        label = classify_threat(kin, v_obj_vec.length, cfg["threat"])
        half_s, half_lat = horizontal_extents(threat_corners, (tx, ty, tz))
        path_s = path_lat = None
        if actor.follow_spline is not None or actor.category == "static":
            path_s = float(actor.s) - float(cam.arc_length)
            path_lat = float(actor.lateral) - float(cam.lateral)
        splat_object(
            threat_grid,
            p_cam=p_cam,
            p_obj=p_obj,
            v_obj=v_obj,
            heading=(tx, ty, tz),
            walk_speed=float(cam.walk_speed),
            half_s=half_s,
            half_lat=half_lat,
            bbox=(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax),
            res_x=res_x,
            res_y=res_y,
            path_s=path_s,
            path_lat=path_lat,
            boxes=inner_boxes,
            class_name=actor.class_name,
            occluded=bool(bbox.occluded),
            behind_camera=behind_camera,
            z_cam=float(cam_space.z),
        )
        objects.append(
            {
                "instance_id": actor.instance_id,
                "class_name": actor.class_name,
                "threat_label": label,
                "kinematics": {
                    "world_position": vec_to_list(blender_zup_to_yup(origin)),
                    "velocity": vec_to_list(blender_zup_to_yup(v_obj)),
                    "relative_velocity": vec_to_list(blender_zup_to_yup(kin.v_rel)),
                    "distance": round(kin.distance, 4),
                    "ttc": round(kin.ttc, 4) if math.isfinite(kin.ttc) else 9999.0,
                    "cpa": round(kin.cpa, 4) if math.isfinite(kin.cpa) else round(kin.distance, 4),
                },
                "bounding_box_2d": bbox.as_dict(),
                "flags": {
                    "truncated": bool(bbox.truncated),
                    "occluded": bool(bbox.occluded),
                },
            }
        )

    env = world.state.environment if world.state else {"lighting": "unknown", "weather": "unknown"}
    cam_cfg = cfg.get("camera", {})
    lens_mm = float(cam_cfg.get("lens_mm", 24.0))
    sensor_w = float(cam_cfg.get("sensor_width_mm", 36.0))
    hfov = cam_cfg.get("hfov_deg")
    if hfov is None:
        hfov = math.degrees(2.0 * math.atan((sensor_w * 0.5) / max(lens_mm, 1e-3)))
    record = {
        "frame_id": f"{frame_index:06d}",
        "timestamp": round(t, 4),
        "camera_data": {
            "world_position": vec_to_list(blender_zup_to_yup(cam.position)),
            "velocity": vec_to_list(blender_zup_to_yup(cam.velocity)),
            "pitch_yaw_roll": _round_pyr(cam),
            "lens_mm": round(lens_mm, 3),
            "hfov_deg": round(float(hfov), 2),
            "sensor_width_mm": round(sensor_w, 3),
            # Instantaneous ground speed: 0 while seated or hesitating, so a
            # consumer can tell "no optical flow" from "sensor dropout".
            "ego_mode": cam.mode,
            "ego_speed": round(float(cam.walk_speed), 4),
        },
        "environment": {
            "lighting": env.get("lighting", "unknown"),
            "weather": env.get("weather", "unknown"),
            "biome": env.get("biome", "street"),
            "dappled": bool(env.get("dappled", False)),
            "chaos": round(float(env.get("chaos", getattr(world, "chaos", 0.0))), 4),
            "wind": env.get("wind", "breeze"),
            "wind_strength": round(float(env.get("wind_strength", 0.4)), 3),
        },
        "objects": objects,
    }
    return record, threat_grid


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


# episode_0007 / episode_0007_jaywalker
_EPISODE_DIR_RE = re.compile(r"^episode_(\d{4,})(?:_|$)")


def _episode_dir_pattern():
    return _EPISODE_DIR_RE


def existing_episode_ids(out_root: Path) -> list[int]:
    """Numeric ids already used by episode_* folders in `out_root`."""
    pat = _episode_dir_pattern()
    ids: list[int] = []
    if not out_root.is_dir():
        return ids
    for child in out_root.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def next_free_episode_id(out_root: Path) -> int:
    ids = existing_episode_ids(out_root)
    return (max(ids) + 1) if ids else 0


def episode_dir_name(episode_id: int, scenario: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(scenario).strip()) or "episode"
    return f"episode_{episode_id:04d}_{slug}"


def allocate_episode_dir(out_root: Path, episode_id: int, scenario: str) -> tuple[int, Path]:
    """Pick `episode_XXXX_<scenario>` that does not already exist.

    If that name is taken (re-run of the same id), bump the numeric id so we
    never clobber a finished episode.
    """
    while True:
        path = out_root / episode_dir_name(episode_id, scenario)
        if not path.exists():
            return episode_id, path
        episode_id += 1


def rebuild_dataset_summary(out_root: Path, cfg: dict, args: argparse.Namespace, media: str) -> dict:
    """Scan every episode_*/episode.json so a new run does not erase old rows."""
    manifest_name = str(cfg["output"]["manifest_name"])
    rows: list[dict] = []
    hist: Counter[str] = Counter()
    if out_root.is_dir():
        for child in sorted(out_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or not _episode_dir_pattern().match(child.name):
                continue
            man_path = child / manifest_name
            if not man_path.is_file():
                continue
            try:
                man = json.loads(man_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            labels = man.get("label_histogram") or {}
            hist.update(labels)
            rows.append(
                {
                    "episode_id": man.get("episode_id"),
                    "dir": child.name,
                    "scenario": man.get("scenario"),
                    "scenarios": man.get("scenarios") or None,
                    "labels": labels,
                }
            )
    return {
        "episodes_on_disk": len(rows),
        "episodes_this_run": int(args.episodes),
        "scenario": getattr(args, "scenario_request", None) or args.scenario,
        "seed": int(args.seed),
        "output": str(out_root),
        "media": "annotations" if args.no_render else media,
        "label_histogram": dict(hist),
        "episodes": rows,
    }


# ---------------------------------------------------------------------------
# Video mux (PNG sequence → H.264)
# ---------------------------------------------------------------------------

def _ffmpeg_has_encoder(exe: str, name: str) -> bool:
    proc = subprocess.run(
        [exe, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and name in (proc.stdout or "")


_FFMPEG_NVENC: bool | None = None


def _nvenc_usable(exe: str) -> bool:
    """True when ffmpeg lists h264_nvenc. The real encode is the probe.

    A lavfi dummy probe often fails under PRIME/EGL even when encoding a PNG
    sequence with NVENC works, so we only check the encoder is compiled in.
    """
    global _FFMPEG_NVENC
    if _FFMPEG_NVENC is None:
        _FFMPEG_NVENC = _ffmpeg_has_encoder(exe, "h264_nvenc")
    return _FFMPEG_NVENC


def _encode_with_ffmpeg(
    png_dir: Path,
    n_frames: int,
    fps: int,
    out_path: Path,
    crf: int,
    ffmpeg_bin: str,
    encoder: str = "auto",
    x264_preset: str = "veryfast",
    nvenc_preset: str = "p4",
    extra_inputs: list[str] | None = None,
    filter_complex: str | None = None,
    filter_map: str = "[out]",
    start_number: int = 0,
) -> tuple[bool, str]:
    """Mux `%06d.png` with system ffmpeg. Returns (ok, message)."""
    exe = shutil.which(ffmpeg_bin)
    if exe is None:
        return False, f"{ffmpeg_bin} not found on PATH"
    start_number = max(0, int(start_number))
    first = png_dir / f"{start_number:06d}.png"
    if not first.is_file():
        return False, f"missing first frame: {first}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    want = str(encoder or "auto").lower()
    use_nvenc = want == "nvenc" or (want == "auto" and _nvenc_usable(exe))
    if want == "nvenc" and not _nvenc_usable(exe):
        use_nvenc = False

    def _seq_in(path_glob: str) -> list[str]:
        return [
            "-framerate", str(int(fps)),
            "-start_number", str(start_number),
            "-i", path_glob,
        ]

    common_in = [exe, "-y"] + _seq_in(str(png_dir / "%06d.png"))
    for extra in extra_inputs or []:
        common_in += _seq_in(extra)
    common_in += ["-frames:v", str(int(n_frames))]
    filter_args: list[str] = []
    if filter_complex:
        filter_args = ["-filter_complex", filter_complex, "-map", filter_map]
    common_out = filter_args + [
        "-r", str(int(fps)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]

    if use_nvenc:
        cmd = common_in + [
            "-c:v", "h264_nvenc",
            "-preset", str(nvenc_preset),
            "-tune", "hq",
            "-rc", "constqp",
            "-qp", str(int(crf)),
        ] + common_out
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return True, f"{exe} h264_nvenc"
        err = (proc.stderr or proc.stdout or "").strip()
        if want == "nvenc":
            return False, err[-800:] or f"nvenc exited {proc.returncode}"
        # Fall through to libx264.

    cmd = common_in + [
        "-c:v", "libx264",
        "-preset", str(x264_preset),
        "-crf", str(int(crf)),
        "-threads", "0",
    ] + common_out
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err[-800:] or f"ffmpeg exited {proc.returncode}"
    return True, f"{exe} libx264 {x264_preset}"


def _count_mp4_frames(path: Path) -> int | None:
    """Best-effort packet count so concat-copy cannot silently drop frames."""
    exe = shutil.which("ffprobe")
    if exe is None or not path.is_file():
        return None
    proc = subprocess.run(
        [
            exe, "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets,nb_frames",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    nums: list[int] = []
    for part in (proc.stdout or "").replace("\n", ",").split(","):
        part = part.strip()
        if part.isdigit():
            nums.append(int(part))
    return max(nums) if nums else None


def _concat_copy_mp4(parts: list[Path], out_path: Path, ffmpeg_bin: str) -> tuple[bool, str]:
    """Concat demuxer, bitstream copy. Parts must share codec / timebase / size."""
    exe = shutil.which(ffmpeg_bin)
    if exe is None:
        return False, f"{ffmpeg_bin} not found on PATH"
    if any(not p.is_file() for p in parts):
        missing = [str(p) for p in parts if not p.is_file()]
        return False, f"concat part missing: {missing}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.parent / f".{out_path.stem}.concat.txt"
    try:
        lines = []
        for p in parts:
            s = p.resolve().as_posix().replace("'", r"'\''")
            lines.append(f"file '{s}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [
                exe, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
                "-c", "copy", "-movflags", "+faststart", str(out_path),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            return False, err[-800:] or f"concat exited {proc.returncode}"
        return True, "concat-copy"
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass


def _encode_full_ffmpeg(
    *,
    png_dir: Path,
    n_frames: int,
    cfg: dict,
    out_path: Path,
    extra_inputs: list[str] | None = None,
    filter_complex: str | None = None,
    start_number: int = 0,
) -> tuple[bool, str]:
    fps = int(cfg["render"]["fps"])
    return _encode_with_ffmpeg(
        png_dir, int(n_frames), fps, out_path,
        int(cfg["output"].get("video_crf", 18)),
        str(cfg["output"].get("ffmpeg_bin", "ffmpeg")),
        encoder=str(cfg["output"].get("video_encoder", "auto")),
        x264_preset=str(cfg["output"].get("x264_preset", "veryfast")),
        nvenc_preset=str(cfg["output"].get("nvenc_preset", "p4")),
        extra_inputs=extra_inputs,
        filter_complex=filter_complex,
        start_number=start_number,
    )


def _encode_segmented(
    *,
    png_dir: Path,
    n_frames: int,
    prefix_len: int,
    cfg: dict,
    out_path: Path,
    extra_inputs: list[str] | None = None,
    filter_complex: str | None = None,
    prefix_mp4: Path | None = None,
    write_prefix_to: Path | None = None,
) -> tuple[bool, str, Path | None]:
    """Encode prefix once, encode the tail, concat-copy. Fallback: full PNG mux."""
    n_frames = int(n_frames)
    prefix_len = max(0, int(prefix_len))
    ffmpeg_bin = str(cfg["output"].get("ffmpeg_bin", "ffmpeg"))

    def _full(reason: str) -> tuple[bool, str, Path | None]:
        ok, msg = _encode_full_ffmpeg(
            png_dir=png_dir, n_frames=n_frames, cfg=cfg, out_path=out_path,
            extra_inputs=extra_inputs, filter_complex=filter_complex,
        )
        cached = write_prefix_to if write_prefix_to is not None and write_prefix_to.is_file() else None
        if prefix_mp4 is not None and prefix_mp4.is_file():
            cached = prefix_mp4
        return ok, f"{reason}; full ({msg})" if ok else f"{reason}; full failed ({msg})", cached

    if prefix_len <= 0 or prefix_len >= n_frames:
        ok, msg, cached = _full("no split")
        return ok, msg.split("; ", 1)[-1], cached

    dest_prefix = prefix_mp4 if prefix_mp4 is not None and prefix_mp4.is_file() else None
    if dest_prefix is None:
        dest_prefix = write_prefix_to or out_path.with_name(f".{out_path.stem}_prefix.mp4")
        ok, msg = _encode_full_ffmpeg(
            png_dir=png_dir, n_frames=prefix_len, cfg=cfg, out_path=dest_prefix,
            extra_inputs=extra_inputs, filter_complex=filter_complex, start_number=0,
        )
        if not ok:
            return _full(f"prefix encode failed ({msg})")

    tail_n = n_frames - prefix_len
    tail_mp4 = out_path.with_name(f".{out_path.stem}_tail.mp4")
    try:
        ok, msg = _encode_full_ffmpeg(
            png_dir=png_dir, n_frames=tail_n, cfg=cfg, out_path=tail_mp4,
            extra_inputs=extra_inputs, filter_complex=filter_complex,
            start_number=prefix_len,
        )
        if not ok:
            return _full(f"tail encode failed ({msg})")
        ok, msg = _concat_copy_mp4([dest_prefix, tail_mp4], out_path, ffmpeg_bin)
        if not ok:
            return _full(f"concat failed ({msg})")
        got = _count_mp4_frames(out_path)
        if got is not None and got != n_frames:
            return _full(f"concat dropped frames ({got} != {n_frames})")
        via = f"concat-copy prefix={prefix_len} tail={tail_n}"
        return True, via, dest_prefix
    finally:
        try:
            tail_mp4.unlink()
        except OSError:
            pass


def _sequence_collection(sed: Any) -> Any:
    """Blender 4.x uses `sequences`; 5.x renamed the collection to `strips`."""
    return getattr(sed, "sequences", None) or getattr(sed, "strips", None)


def _encode_with_blender_vse(
    scene: Any,
    png_dir: Path,
    n_frames: int,
    fps: int,
    out_path: Path,
    res_x: int,
    res_y: int,
) -> tuple[bool, str]:
    """Fallback: image-sequence strip → Blender's bundled FFmpeg encoder."""
    first = png_dir / "000000.png"
    if not first.is_file():
        return False, f"missing first frame: {first}"

    if scene.sequence_editor is None:
        scene.sequence_editor_create()
    sed = scene.sequence_editor
    seqs = _sequence_collection(sed)
    if seqs is None:
        return False, "Sequence Editor API not available"

    old = {
        "filepath": scene.render.filepath,
        "file_format": scene.render.image_settings.file_format,
        "use_sequencer": scene.render.use_sequencer,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
    }
    strip = None
    try:
        for existing in list(seqs):
            if getattr(existing, "name", "").startswith("synth_rgb"):
                seqs.remove(existing)

        strip = seqs.new_image("synth_rgb", str(first), 1, 1)
        for i in range(1, n_frames):
            strip.elements.append(f"{i:06d}.png")
        if hasattr(strip, "frame_final_duration"):
            strip.frame_final_duration = n_frames

        scene.frame_start = 1
        scene.frame_end = n_frames
        scene.render.fps = int(fps)
        scene.render.resolution_x = int(res_x)
        scene.render.resolution_y = int(res_y)
        scene.render.use_sequencer = True
        scene.render.image_settings.file_format = "FFMPEG"
        # Trailing separator-less path: Blender writes a single container file.
        scene.render.filepath = str(out_path)

        ff = scene.render.ffmpeg
        try:
            ff.format = "MPEG4"
        except Exception:
            pass
        try:
            ff.codec = "H264"
        except Exception:
            pass
        for attr, value in (
            ("constant_rate_factor", "MEDIUM"),
            ("ffmpeg_preset", "GOOD"),
            ("audio_codec", "NONE"),
        ):
            if hasattr(ff, attr):
                try:
                    setattr(ff, attr, value)
                except Exception:
                    pass

        bpy.ops.render.render(animation=True)
        if not out_path.is_file():
            # Some builds append an extension / frame token; accept a sibling .mp4.
            siblings = list(out_path.parent.glob(out_path.stem + "*.mp4"))
            if siblings:
                if siblings[0] != out_path:
                    siblings[0].replace(out_path)
            else:
                return False, f"Blender FFmpeg produced no file at {out_path}"
        return True, "blender-vse"
    except Exception as exc:
        return False, f"blender VSE encode failed: {exc}"
    finally:
        if strip is not None:
            try:
                seqs.remove(strip)
            except Exception:
                pass
        scene.render.filepath = old["filepath"]
        scene.render.image_settings.file_format = old["file_format"]
        scene.render.use_sequencer = old["use_sequencer"]
        scene.frame_start = old["frame_start"]
        scene.frame_end = old["frame_end"]


def encode_episode_video(
    *,
    scene: Any,
    png_dir: Path,
    n_frames: int,
    cfg: dict,
    out_path: Path,
    prefix_len: int = 0,
    prefix_mp4: Path | None = None,
    write_prefix_to: Path | None = None,
) -> tuple[bool, str]:
    """Prefer system ffmpeg; fall back to Blender's bundled encoder."""
    fps = int(cfg["render"]["fps"])
    if 0 < int(prefix_len) < int(n_frames):
        ok, msg, _cached = _encode_segmented(
            png_dir=png_dir,
            n_frames=n_frames,
            prefix_len=int(prefix_len),
            cfg=cfg,
            out_path=out_path,
            prefix_mp4=prefix_mp4,
            write_prefix_to=write_prefix_to,
        )
        if ok:
            return True, f"ffmpeg ({msg})"
        print(f"  [video] segmented mux failed ({msg})")
    ok, msg = _encode_full_ffmpeg(
        png_dir=png_dir, n_frames=n_frames, cfg=cfg, out_path=out_path,
    )
    if ok:
        return True, f"ffmpeg ({msg})"
    print(f"  [video] ffmpeg unavailable or failed ({msg}); trying Blender VSE")
    return _encode_with_blender_vse(
        scene,
        png_dir,
        n_frames,
        fps,
        out_path,
        int(cfg["render"]["resolution_x"]),
        int(cfg["render"]["resolution_y"]),
    )


def encode_spatial_overlay_video(
    *,
    rgb_dir: Path,
    spatial_payload: dict,
    n_frames: int,
    cfg: dict,
    out_path: Path,
    heat_dir: Path,
    write_heat: bool = True,
    prefix_len: int = 0,
    prefix_mp4: Path | None = None,
    write_prefix_to: Path | None = None,
) -> tuple[bool, str]:
    """Hazy RGB + k×k threat heat → ``spatial_overlay.mp4``."""
    frames = list(spatial_payload.get("frames") or [])
    k = int(spatial_payload.get("k") or 3)
    if len(frames) < n_frames:
        return False, f"spatial JSON has {len(frames)} frames, expected {n_frames}"
    if write_heat:
        try:
            written = write_heat_sequence(frames[:n_frames], heat_dir, k=k)
        except Exception as exc:
            return False, f"heat sequence failed: {exc}"
        if written < n_frames:
            return False, f"wrote {written} heat frames, expected {n_frames}"
    first_heat = heat_dir / "000000.ppm"
    if not first_heat.is_file():
        return False, f"missing heat frame: {first_heat}"
    last_heat = heat_dir / f"{int(n_frames) - 1:06d}.ppm"
    if not last_heat.is_file():
        return False, f"missing heat frame: {last_heat}"
    extra = [str(heat_dir / "%06d.ppm")]
    filt = overlay_filter_complex(
        k,
        int(cfg["render"]["resolution_x"]),
        int(cfg["render"]["resolution_y"]),
    )

    def _try(filt_graph: str) -> tuple[bool, str]:
        if 0 < int(prefix_len) < int(n_frames):
            ok, msg, _cached = _encode_segmented(
                png_dir=rgb_dir,
                n_frames=n_frames,
                prefix_len=int(prefix_len),
                cfg=cfg,
                out_path=out_path,
                extra_inputs=extra,
                filter_complex=filt_graph,
                prefix_mp4=prefix_mp4,
                write_prefix_to=write_prefix_to,
            )
            if ok:
                return True, msg
        ok, msg = _encode_full_ffmpeg(
            png_dir=rgb_dir, n_frames=n_frames, cfg=cfg, out_path=out_path,
            extra_inputs=extra, filter_complex=filt_graph,
        )
        return ok, msg

    ok, msg = _try(filt)
    if ok:
        return True, f"ffmpeg ({msg})"
    # Older ffmpeg builds may lack drawgrid; retry the blend-only graph.
    head, sep, _ = filt.partition(";[mix]drawgrid")
    filt_plain = f"{head};[mix]copy[out]" if sep else ""
    if filt_plain and "drawgrid" in filt:
        ok, msg = _try(filt_plain)
        if ok:
            return True, f"ffmpeg ({msg}, no drawgrid)"
    return False, msg


# ---------------------------------------------------------------------------
# Pose cache for animation replay
# ---------------------------------------------------------------------------

def _collect_pose_objects(cam_obj: Any, actors: list) -> list:
    """Camera + every actor root and descendant (gait bones, wheels)."""
    out: list = []
    seen: set[int] = set()

    def add(obj: Any) -> None:
        if obj is None:
            return
        pid = obj.as_pointer()
        if pid in seen:
            return
        seen.add(pid)
        out.append(obj)
        for ch in obj.children:
            add(ch)

    add(cam_obj)
    for actor in actors:
        add(actor.obj)
    return out


def _snapshot_poses(objects: list) -> list[tuple]:
    snaps: list[tuple] = []
    for obj in objects:
        mode = obj.rotation_mode
        loc = obj.location.copy()
        if mode == "QUATERNION":
            rot = obj.rotation_quaternion.copy()
        else:
            rot = obj.rotation_euler.copy()
        snaps.append((mode, loc, rot))
    return snaps


def _restore_poses(objects: list, snaps: list[tuple]) -> None:
    for obj, (mode, loc, rot) in zip(objects, snaps):
        if obj.rotation_mode != mode:
            obj.rotation_mode = mode
        obj.location = loc
        if mode == "QUATERNION":
            obj.rotation_quaternion = rot
        else:
            obj.rotation_euler = rot


def _render_animation_sequence(
    scene: Any,
    rgb_dir: Path,
    n_frames: int,
    pose_objects: list,
    frame_poses: list[list[tuple]],
    png_compression: int,
    start_frame: int = 0,
) -> bool:
    """Replay cached poses inside one EEVEE animation render (GPU stays warm)."""

    def on_frame(scene_arg: Any, _depsgraph: Any = None) -> None:
        i = int(scene_arg.frame_current)
        if 0 <= i < len(frame_poses):
            snap = frame_poses[i]
            if snap is not None:
                _restore_poses(pose_objects, snap)

    handlers = bpy.app.handlers.frame_change_pre
    handlers.append(on_frame)
    try:
        prepare_still_render(scene, png_compression=png_compression)
        scene.frame_start = max(0, int(start_frame))
        scene.frame_end = n_frames - 1
        scene.render.filepath = str(rgb_dir / "######")
        bpy.ops.render.render(animation=True)
    except Exception:
        traceback.print_exc()
        return False
    finally:
        while on_frame in handlers:
            handlers.remove(on_frame)

    return _ensure_six_digit_pngs(rgb_dir, n_frames)


def _ensure_six_digit_pngs(rgb_dir: Path, n_frames: int) -> bool:
    """Accept 4-digit Blender names and rename to `%06d.png`."""
    if (rgb_dir / "000000.png").is_file() and (rgb_dir / f"{n_frames - 1:06d}.png").is_file():
        return True
    if (rgb_dir / "0000.png").is_file():
        for i in range(n_frames):
            src = rgb_dir / f"{i:04d}.png"
            dst = rgb_dir / f"{i:06d}.png"
            if src.is_file() and src != dst:
                src.replace(dst)
        return (rgb_dir / "000000.png").is_file()
    pngs = sorted(rgb_dir.glob("*.png"))
    if len(pngs) >= n_frames:
        for i, src in enumerate(pngs[:n_frames]):
            dst = rgb_dir / f"{i:06d}.png"
            if src != dst:
                src.replace(dst)
        return (rgb_dir / "000000.png").is_file()
    return False


def _render_stills_fallback(
    scene: Any,
    rgb_dir: Path,
    n_frames: int,
    pose_objects: list,
    frame_poses: list[list[tuple]],
    png_compression: int,
    start_frame: int = 0,
) -> None:
    prepare_still_render(scene, png_compression=png_compression)
    for i in range(max(0, int(start_frame)), n_frames):
        snap = frame_poses[i]
        if snap is None:
            continue
        _restore_poses(pose_objects, snap)
        bpy.context.view_layer.update()
        scene.render.filepath = str(rgb_dir / f"{i:06d}.png")
        bpy.ops.render.render(write_still=True)


def copy_prefix_pngs(src: Path, dst: Path, n: int) -> int:
    """Hard-link or copy the shared prefix so miss tails do not re-render it."""
    return link_or_copy_prefix(src, dst, n, suffix=".png")

def run_episode(
    *,
    episode_id: int,
    scenario_request: str,
    cfg: dict,
    rng: random.Random,
    out_root: Path,
    no_render: bool,
    no_occlusion: bool,
    frames_override: int,
    media: str,
    hfov_deg: float | None = None,
    lens_mm: float | None = None,
    lock_fov: bool = False,
    threat_k: int = 3,
    spatial_overlay: bool = False,
    write_annotations: bool = True,
    drop_rgb: bool = False,
    biome: str = "auto",
    ego_mode: str = "auto",
    ego_height: str = "auto",
    chaos: float | None = None,
    no_trees: bool = False,
    wind: str = "auto",
) -> dict:
    cfg = copy.deepcopy(cfg)
    names = pick_scenarios(rng, cfg, scenario_request)
    # Dedicated episode RNG so two episodes never share Perlin / Poisson streams.
    ep_rng = random.Random(rng.randrange(1, 2**31))
    apply_camera_fov(
        cfg,
        hfov_deg=hfov_deg,
        lens_mm=lens_mm,
        rng=ep_rng,
        lock=lock_fov,
    )
    lens = float(cfg["camera"]["lens_mm"])
    hfov = float(cfg["camera"]["hfov_deg"])
    print(
        f"[episode {episode_id:04d}] scenario={compose_display(names)}  "
        f"lens={lens:.1f}mm  hfov={hfov:.1f}°"
    )

    if chaos is not None:
        # A locked dial overrides the per-episode draw at both ends of the
        # range, so --chaos 0 really is the pre-chaos pipeline.
        c = max(0.0, min(1.0, float(chaos)))
        cfg["domain_randomization"]["material_chaos"] = (c, c)
    if no_trees:
        for key in ("n_trees", "n_grass_clumps", "n_street_trees", "n_median_trees", "n_path_trees"):
            cfg["world"][key] = (0, 0)
        for prof in (cfg["world"].get("biomes") or {}).values():
            for key in ("n_trees", "n_grass_clumps", "n_street_trees", "n_median_trees", "n_path_trees"):
                prof.pop(key, None)
    requested_wind = str(wind or "auto").strip().lower()
    if requested_wind in WIND_LABELS:
        cfg["domain_randomization"]["wind_weights"] = {requested_wind: 1.0}

    apply_ego_height(cfg, ego_height)
    gen = WorldGenerator(cfg, ep_rng)
    # Biome first: a sparse scenario must be able to override the biome's
    # traffic counts, not the other way round.
    chosen_biome = gen.prepare_biome(biome, names=names)
    gen.prepare_scenario(names)
    clear_bound_caches()
    state = gen.build()
    cam_obj = gen.create_camera()

    fps = int(cfg["render"]["fps"])
    n_frames = int(frames_override) if frames_override > 0 else int(cfg["render"]["frames_per_episode"])
    # The ego profile needs the episode length up front so a hesitation
    # window lands inside the episode rather than after the last frame.
    requested_mode = str(ego_mode or "auto").strip().lower()
    if requested_mode in ("", "auto", "random") and getattr(gen, "force_ego_mode", ""):
        requested_mode = str(gen.force_ego_mode)
    if requested_mode not in ("", "auto", "random"):
        if requested_mode not in EGO_MODES:
            raise ValueError(
                f"unknown --ego-mode {requested_mode!r}. Known: {', '.join(EGO_MODES)}"
            )
        profile = choose_ego_profile(
            {**cfg, "ego": {**cfg["ego"], "mode_weights": {requested_mode: 1.0}}},
            ep_rng,
            episode_seconds=n_frames / float(fps),
        )
    else:
        profile = choose_ego_profile(cfg, ep_rng, episode_seconds=n_frames / float(fps))

    rig = CameraRig(
        cam_obj=cam_obj,
        spline=state.road,
        cfg=cfg,
        rng=ep_rng,
        walk_speed=0.0 if profile.stationary else state.walk_speed,
        lateral=state.sidewalk_lateral,
        profile=profile,
        # Sidesteps and diagonal crossings must stay on the built ribbon.
        lateral_limit=state.corridor.lateral_limit(0.30, True),
    )
    injected = gen.inject_scenarios(names, rig)
    gen.place_ego_props(rig)
    # No more actors will be created; partition them for the per-frame loop.
    gen.freeze()
    lib_stats = gen.lib.stats()
    episode_id, ep_dir = allocate_episode_dir(out_root, episode_id, injected)
    print(
        f"  dir={ep_dir.name}\n"
        f"  biome={chosen_biome}  ego={profile.mode}  "
        f"eye={profile.eye_height:.2f}m({getattr(profile, 'stature', '?')})  "
        f"chaos={gen.chaos:.2f}  wind={state.environment.get('wind', 'breeze')}"
        f"({float(state.environment.get('wind_strength', 0)):.2f})  "
        f"light={state.environment.get('lighting')}"
        f"{'+dappled' if state.environment.get('dappled') else ''}"
    )

    scene = bpy.context.scene
    # Do not walk off the spline.
    n_frames = min(n_frames, max(2, int(rig.max_time() * fps)))
    dt = 1.0 / float(fps)
    scene.frame_start = 0
    scene.frame_end = n_frames - 1
    scene.render.fps = fps

    rgb_dir = ep_dir / str(cfg["output"]["rgb_dirname"])
    ann_dir = ep_dir / str(cfg["output"]["ann_dirname"])
    ann_path = ann_dir / "annotations.json"
    spatial_dir = ep_dir / "spatial_annotations"
    spatial_path = spatial_dir / "spatial_annotations.json"
    video_path = ep_dir / str(cfg["output"].get("video_name", "preview.mp4"))
    overlay_path = ep_dir / "spatial_overlay.mp4"
    if write_annotations:
        ann_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)
    grid_k = max(1, int(threat_k))

    do_render = not no_render
    want_video = do_render and media in ("video", "both")
    keep_frames = do_render and media in ("frames", "both")
    if drop_rgb:
        if want_video:
            keep_frames = False
        elif do_render:
            print("  [--no-rgb] ignored: no video is being written (use --media video or both)")
    want_overlay = bool(spatial_overlay) and do_render
    write_png = do_render  # video is muxed from the PNG sequence
    if spatial_overlay and not do_render:
        print("  [spatial-overlay] skipped (--no-render has no RGB to tint)")
    if write_png:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    png_compression = int(cfg["render"].get("png_compression", 1))
    prepare_still_render(scene, png_compression=png_compression)

    label_hist: Counter[str] = Counter()
    object_frames: list[dict] = []
    spatial_frames: list[dict] = []
    pose_objects = _collect_pose_objects(cam_obj, state.actors) if write_png else []
    frame_poses: list[list[tuple]] = []

    t_sim0 = time.perf_counter()
    for i in range(n_frames):
        t = i * dt
        cam_state = rig.update(t, dt)
        gen.update(t, dt)
        # Actors write location / Euler, not matrix_world. Without this the
        # depsgraph keeps the previous pose and every AABB projects empty.
        bpy.context.view_layer.update()

        if write_png:
            frame_poses.append(_snapshot_poses(pose_objects))

        record, threat_grid = build_frame_record(
            frame_index=i,
            t=t,
            cam=cam_state,
            cam_obj=cam_obj,
            world=gen,
            scene=scene,
            cfg=cfg,
            do_occlusion=not no_occlusion,
            threat_k=grid_k,
        )
        for obj in record["objects"]:
            label_hist[obj["threat_label"]] += 1
        if write_annotations:
            object_frames.append(record)
        spatial_frames.append(frame_spatial_entry(record["frame_id"], threat_grid))

        if (i + 1) % 30 == 0 or i == 0 or i == n_frames - 1:
            print(f"  sim {i:04d}/{n_frames - 1:04d}  t={t:5.2f}s  objects={len(record['objects'])}")
    t_sim = time.perf_counter() - t_sim0
    if write_annotations:
        _write_json(ann_path, {"frames": object_frames})
    _write_json(spatial_path, episode_spatial_payload(grid_k, spatial_frames))

    t_rnd = 0.0
    if write_png:
        t_r0 = time.perf_counter()
        ok_anim = _render_animation_sequence(
            scene, rgb_dir, n_frames, pose_objects, frame_poses, png_compression,
        )
        if not ok_anim:
            print("  [render] animation batch missed files; falling back to stills")
            _render_stills_fallback(
                scene, rgb_dir, n_frames, pose_objects, frame_poses, png_compression,
            )
        t_rnd = time.perf_counter() - t_r0
        print(f"  timing  sim={t_sim:.2f}s  render={t_rnd:.2f}s  ({n_frames} frames)")
    else:
        print(f"  timing  sim={t_sim:.2f}s  render=skipped")

    video_ok = False
    video_via = None
    if want_video:
        video_ok, video_via = encode_episode_video(
            scene=scene,
            png_dir=rgb_dir,
            n_frames=n_frames,
            cfg=cfg,
            out_path=video_path,
        )
        if video_ok:
            print(f"  video → {video_path}  ({video_via})")
        else:
            print(f"  [video] FAILED: {video_via}", file=sys.stderr)
            # Keep PNGs so the episode is still inspectable.
            keep_frames = True

    overlay_ok = False
    overlay_via = None
    if want_overlay:
        heat_dir = spatial_dir / "_heat"
        overlay_ok, overlay_via = encode_spatial_overlay_video(
            rgb_dir=rgb_dir,
            spatial_payload=episode_spatial_payload(grid_k, spatial_frames),
            n_frames=n_frames,
            cfg=cfg,
            out_path=overlay_path,
            heat_dir=heat_dir,
        )
        if heat_dir.is_dir():
            shutil.rmtree(heat_dir, ignore_errors=True)
        if overlay_ok:
            print(f"  spatial overlay → {overlay_path}  ({overlay_via})")
        else:
            print(f"  [spatial-overlay] FAILED: {overlay_via}", file=sys.stderr)
            keep_frames = True

    if write_png and not keep_frames and rgb_dir.is_dir():
        shutil.rmtree(rgb_dir)
        print(f"  removed {rgb_dir.name}/")

    manifest = {
        "episode_id": episode_id,
        "dir": ep_dir.name,
        "scenario": injected,
        "scenarios": list(names),
        "scenario_requested": scenario_request,
        "frames": n_frames,
        "fps": fps,
        "walk_speed": round(0.0 if profile.stationary else state.walk_speed, 4),
        "sidewalk_lateral": round(state.sidewalk_lateral, 4),
        "biome": chosen_biome,
        "ego": {
            "mode": profile.mode,
            "eye_height_m": round(float(profile.eye_height), 3),
            "stature": str(getattr(profile, "stature", "typical")),
            "stationary": bool(profile.stationary),
            "sidestep_amp_m": round(float(profile.sidestep_amp), 3),
            "halt_s": (
                [round(float(profile.halt_t0), 3), round(float(profile.halt_t1), 3)]
                if profile.halt_t0 is not None and profile.halt_t1 is not None
                else None
            ),
        },
        "camera": {
            "lens_mm": round(float(cfg["camera"]["lens_mm"]), 3),
            "hfov_deg": round(float(cfg["camera"]["hfov_deg"]), 2),
            "sensor_width_mm": round(float(cfg["camera"]["sensor_width_mm"]), 3),
        },
        "environment": state.environment,
        "label_histogram": dict(label_hist),
        "render": do_render,
        "media": media if do_render else "annotations",
        "rgb_dir": str(cfg["output"]["rgb_dirname"]) if keep_frames else None,
        "video": video_path.name if video_ok else None,
        "video_encoder": video_via if video_ok else None,
        "annotations": "annotations/annotations.json" if write_annotations else None,
        "spatial_annotations_k": grid_k,
        "spatial_annotations": "spatial_annotations/spatial_annotations.json",
        "spatial_overlay": overlay_path.name if overlay_ok else None,
        "spatial_overlay_encoder": overlay_via if overlay_ok else None,
    }
    _write_json(ep_dir / str(cfg["output"]["manifest_name"]), manifest)
    print(f"  labels={dict(label_hist)}")

    # Free everything this episode allocated. Without this, a 2000-episode
    # pack accumulates every mesh and material it ever built, because the
    # next build() only unlinks objects — it cannot reach node groups and
    # other second-order dependents.
    freed = release_episode(state)
    clear_bound_caches()
    print(f"  cleanup: purged {freed} datablock(s)  [{lib_stats}]")
    return manifest


def _mux_clip_videos(
    *,
    scene: Any,
    rgb_dir: Path,
    n_frames: int,
    cfg: dict,
    video_path: Path,
    overlay_path: Path,
    spatial_dir: Path,
    spatial_frames: list,
    grid_k: int,
    want_video: bool,
    want_overlay: bool,
    prefix_len: int = 0,
    prefix_video: Path | None = None,
    prefix_overlay: Path | None = None,
    write_prefix_video: Path | None = None,
    write_prefix_overlay: Path | None = None,
    heat_dir: Path | None = None,
    write_heat: bool = True,
    keep_heat: bool = False,
) -> tuple[bool, str | None, bool, str | None, bool]:
    """Returns video_ok, video_via, overlay_ok, overlay_via, keep_frames."""
    keep_frames = False
    video_ok = False
    video_via = None
    split = 0 < int(prefix_len) < int(n_frames)
    if want_video:
        video_ok, video_via = encode_episode_video(
            scene=scene, png_dir=rgb_dir, n_frames=n_frames, cfg=cfg, out_path=video_path,
            prefix_len=int(prefix_len) if split else 0,
            prefix_mp4=prefix_video if split else None,
            write_prefix_to=write_prefix_video if split else None,
        )
        if video_ok:
            print(f"  video → {video_path}  ({video_via})")
        else:
            print(f"  [video] FAILED: {video_via}", file=sys.stderr)
            keep_frames = True
    overlay_ok = False
    overlay_via = None
    if want_overlay:
        heat_dir = heat_dir or (spatial_dir / "_heat")
        overlay_ok, overlay_via = encode_spatial_overlay_video(
            rgb_dir=rgb_dir,
            spatial_payload=episode_spatial_payload(grid_k, spatial_frames),
            n_frames=n_frames,
            cfg=cfg,
            out_path=overlay_path,
            heat_dir=heat_dir,
            write_heat=write_heat,
            prefix_len=int(prefix_len) if split else 0,
            prefix_mp4=prefix_overlay if split else None,
            write_prefix_to=write_prefix_overlay if split else None,
        )
        if heat_dir.is_dir() and not keep_heat:
            shutil.rmtree(heat_dir, ignore_errors=True)
        if overlay_ok:
            print(f"  spatial overlay → {overlay_path}  ({overlay_via})")
        else:
            print(f"  [spatial-overlay] FAILED: {overlay_via}", file=sys.stderr)
            keep_frames = True
    return video_ok, video_via, overlay_ok, overlay_via, keep_frames


def run_group_plan(
    *,
    plan: dict,
    cfg: dict,
    out_root: Path,
    args: argparse.Namespace,
    frames_override: int,
    media: str,
    threat_k: int,
) -> int:
    """Render sample × background × {collision, one miss side}.

    One world build per background. Collision is rendered in full; the
    single miss clip copies its prefix RGB / heat / JSON and only EEVEE
    the tail.
    """
    clips = list(iter_clips(plan))
    if not clips:
        print("[groups] plan has no clips", file=sys.stderr)
        return 2
    fps = int(cfg["render"]["fps"])
    n_frames = int(frames_override) if frames_override > 0 else int(cfg["render"]["frames_per_episode"])
    grid_k = max(1, int(threat_k))
    do_render = not bool(args.no_render)
    want_video = do_render and media in ("video", "both")
    want_overlay = bool(args.spatial_overlay) and do_render
    drop_rgb = bool(args.no_rgb)
    write_annotations = not bool(args.no_annotations)
    png_compression = int(cfg["render"].get("png_compression", 1))

    n_ok = 0
    index_rows: list[dict] = []
    current_key = None
    gen = None
    state = None
    cam_obj = None
    profile = None
    scene_cfg = None
    prefix_rgb: Path | None = None
    prefix_heat: Path | None = None
    prefix_video: Path | None = None
    prefix_overlay: Path | None = None
    prefix_spatial: list | None = None
    prefix_records: list | None = None
    rgb_dirs_to_drop: list[Path] = []

    def _drop_prefix_caches() -> None:
        nonlocal prefix_rgb, prefix_heat, prefix_video, prefix_overlay
        nonlocal prefix_spatial, prefix_records
        for cache in (prefix_video, prefix_overlay):
            if cache is not None and cache.is_file():
                try:
                    cache.unlink()
                except OSError:
                    pass
        if prefix_heat is not None and prefix_heat.is_dir():
            shutil.rmtree(prefix_heat, ignore_errors=True)
        prefix_rgb = None
        prefix_heat = None
        prefix_video = None
        prefix_overlay = None
        prefix_spatial = None
        prefix_records = None

    def _close_world() -> None:
        nonlocal gen, state, cam_obj
        if state is not None:
            freed = release_episode(state)
            clear_bound_caches()
            print(f"  cleanup: purged {freed} datablock(s)")
        gen = None
        state = None
        cam_obj = None
        _drop_prefix_caches()

    for clip in clips:
        key = (clip["sample_id"], clip["bg_id"])
        sample_dir = out_root / str(clip["sample_dir"])
        ep_dir = sample_dir / str(clip["relpath"])
        outcome = str(clip["outcome"])
        diverge_frame = max(1, min(n_frames - 2, int(clip["diverge_frame"])))
        diverge_s = float(clip["diverge_s"])
        video_name = str(cfg["output"].get("video_name", "preview.mp4"))
        video_path = ep_dir / video_name
        overlay_path = ep_dir / "spatial_overlay.mp4"
        # Resume: skip clips that already have their final video(s) on disk.
        already_done = video_path.is_file() and (
            (not want_overlay) or overlay_path.is_file()
        )
        if already_done:
            print(
                f"[group {clip['sample_id']:04d}/{clip['scenario']}] "
                f"{outcome}/{clip['bg_id']} skip (exists)"
            )
            index_rows.append(
                {
                    "sample_id": clip["sample_id"],
                    "scenario": clip["scenario"],
                    "outcome": outcome,
                    "bg_id": clip["bg_id"],
                    "bg_style": clip["bg_style"],
                    "bg_lighting": clip.get("bg_lighting"),
                    "path": str(ep_dir.relative_to(out_root)),
                    "video": video_name,
                    "spatial_overlay": overlay_path.name if want_overlay else None,
                    "diverge_frame": diverge_frame,
                    "y": 1 if outcome == "collision" else 0,
                    "resumed": True,
                }
            )
            n_ok += 1
            continue

        ep_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[group {clip['sample_id']:04d}/{clip['scenario']}] "
            f"{outcome}/{clip['bg_id']} style={clip['bg_style']} "
            f"light={clip.get('bg_lighting') or '?'} "
            f"diverge_f={diverge_frame} t={diverge_s:.2f}s"
        )

        if current_key != key:
            _close_world()
            for leftover in rgb_dirs_to_drop:
                if leftover.is_dir():
                    shutil.rmtree(leftover, ignore_errors=True)
            rgb_dirs_to_drop = []
            current_key = key
            scene_cfg = copy.deepcopy(cfg)
            scene_rng = random.Random(int(clip["scene_seed"]))
            bg_rng = random.Random(int(clip["bg_seed"]))
            apply_camera_fov(
                scene_cfg,
                hfov_deg=args.hfov,
                lens_mm=args.lens_mm,
                rng=scene_rng,
                lock=bool(args.no_random_fov),
            )
            if args.chaos is not None:
                c = max(0.0, min(1.0, float(args.chaos)))
                scene_cfg["domain_randomization"]["material_chaos"] = (c, c)
            if args.no_trees:
                for tree_key in ("n_trees", "n_grass_clumps", "n_street_trees", "n_median_trees", "n_path_trees"):
                    scene_cfg["world"][tree_key] = (0, 0)
            requested_wind = str(args.wind or "auto").strip().lower()
            if requested_wind in WIND_LABELS:
                scene_cfg["domain_randomization"]["wind_weights"] = {requested_wind: 1.0}
            apply_ego_height(scene_cfg, str(getattr(args, "ego_height", "auto")))
            names = pick_scenarios(random.Random(int(clip["scene_seed"]) + 3), scene_cfg, clip["scenario"])
            gen = WorldGenerator(scene_cfg, random.Random(int(clip["scene_seed"])), bg_rng=bg_rng)
            gen.group_mode = True
            gen.force_critical = True
            gen.bg_style = str(clip["bg_style"])
            gen.bg_lighting = str(clip.get("bg_lighting") or "")
            gen.prepare_biome(str(args.biome), names=names)
            gen.prepare_scenario(names)
            clear_bound_caches()
            state = gen.build()
            cam_obj = gen.create_camera()
            cam_rng = random.Random(int(clip["scene_seed"]) + 17)
            ego_rng = random.Random(int(clip["scene_seed"]) + 23)
            requested_mode = str(getattr(args, "ego_mode", "auto") or "auto").strip().lower()
            if requested_mode in ("", "auto", "random"):
                requested_mode = "walk"
            if any(n in CROSS_EGO_SCENARIOS for n in names):
                requested_mode = "crosswalk"
            if requested_mode == "seated":
                requested_mode = "walk"
            profile = choose_ego_profile(
                {**scene_cfg, "ego": {**scene_cfg["ego"], "mode_weights": {requested_mode: 1.0}}},
                ego_rng,
                episode_seconds=n_frames / float(fps),
            )
            rig = CameraRig(
                cam_obj=cam_obj,
                spline=state.road,
                cfg=scene_cfg,
                rng=cam_rng,
                walk_speed=0.0 if profile.stationary else state.walk_speed,
                lateral=state.sidewalk_lateral,
                profile=profile,
                lateral_limit=state.corridor.lateral_limit(0.30, True),
            )
            gen.inject_scenarios(names, rig)
            gen.place_ego_props(rig)
            gen.freeze()
            gen.snapshot_actors()
            print(
                f"  biome={gen.biome}  ego={profile.mode}  "
                f"eye={profile.eye_height:.2f}m  walk={state.walk_speed:.2f}m/s  "
                f"style={clip['bg_style']}  "
                f"light={state.environment.get('lighting')}/"
                f"{state.environment.get('weather')}"
            )

        assert gen is not None and state is not None and cam_obj is not None and profile is not None
        assert scene_cfg is not None
        gen.restore_actors()
        gen.apply_outcome_fork(outcome, diverge_s, float(state.sidewalk_lateral))
        cam_rng = random.Random(int(clip["scene_seed"]) + 17)
        rig = CameraRig(
            cam_obj=cam_obj,
            spline=state.road,
            cfg=scene_cfg,
            rng=cam_rng,
            walk_speed=0.0 if profile.stationary else state.walk_speed,
            lateral=state.sidewalk_lateral,
            profile=profile,
            lateral_limit=state.corridor.lateral_limit(0.30, True),
        )

        scene = bpy.context.scene
        clip_frames = min(n_frames, max(2, int(rig.max_time() * fps)))
        dt = 1.0 / float(fps)
        scene.frame_start = 0
        scene.frame_end = clip_frames - 1
        scene.render.fps = fps

        rgb_dir = ep_dir / str(cfg["output"]["rgb_dirname"])
        ann_dir = ep_dir / str(cfg["output"]["ann_dirname"])
        spatial_dir = ep_dir / "spatial_annotations"
        if write_annotations:
            ann_dir.mkdir(parents=True, exist_ok=True)
        spatial_dir.mkdir(parents=True, exist_ok=True)
        if do_render:
            rgb_dir.mkdir(parents=True, exist_ok=True)

        start_frame = 0
        if do_render and clip["prefix_relpath"] and prefix_rgb is not None:
            copied = copy_prefix_pngs(prefix_rgb, rgb_dir, diverge_frame)
            start_frame = min(diverge_frame, copied)
            print(f"  cache prefix {start_frame} pngs ← {prefix_rgb}")

        label_hist: Counter[str] = Counter()
        object_frames: list[dict] = []
        spatial_frames: list[dict] = []
        copied_prefix_json = False
        json_from = start_frame
        if (
            outcome != "collision"
            and prefix_spatial is not None
            and len(prefix_spatial) >= max(1, start_frame or diverge_frame)
        ):
            json_from = start_frame if start_frame > 0 else diverge_frame
            json_from = min(json_from, len(prefix_spatial))
            if json_from > 0:
                spatial_frames = list(prefix_spatial[:json_from])
                if prefix_records is not None and len(prefix_records) >= json_from:
                    object_frames = list(prefix_records[:json_from])
                    for rec in object_frames:
                        for obj in rec.get("objects") or []:
                            label_hist[obj["threat_label"]] += 1
                copied_prefix_json = True
                print(f"  cache prefix {json_from} json frames")

        pose_objects = _collect_pose_objects(cam_obj, state.actors) if do_render else []
        frame_poses: list = [None] * clip_frames if do_render else []
        t_sim0 = time.perf_counter()
        for i in range(clip_frames):
            t = i * dt
            cam_state = rig.update(t, dt)
            gen.update(t, dt)
            # Prefix kinematics still run so wheels / gait / CameraRig velocity
            # at diverge match collision. Skip depsgraph + labels when the
            # collision JSON for [0, diverge) is already in hand.
            need_labels = (not copied_prefix_json) or (i >= json_from)
            need_pose = do_render and (i >= start_frame)
            if need_labels or need_pose:
                bpy.context.view_layer.update()
            if need_pose:
                frame_poses[i] = _snapshot_poses(pose_objects)
            if need_labels:
                record, threat_grid = build_frame_record(
                    frame_index=i,
                    t=t,
                    cam=cam_state,
                    cam_obj=cam_obj,
                    world=gen,
                    scene=scene,
                    cfg=scene_cfg,
                    do_occlusion=not bool(args.no_occlusion),
                    threat_k=grid_k,
                )
                for obj in record["objects"]:
                    label_hist[obj["threat_label"]] += 1
                object_frames.append(record)
                spatial_frames.append(frame_spatial_entry(record["frame_id"], threat_grid))
            if (i + 1) % 30 == 0 or i == 0 or i == clip_frames - 1:
                print(f"  sim {i:04d}/{clip_frames - 1:04d}  t={t:5.2f}s")
        t_sim = time.perf_counter() - t_sim0
        if write_annotations:
            _write_json(ann_dir / "annotations.json", {"frames": object_frames})
        _write_json(spatial_dir / "spatial_annotations.json", episode_spatial_payload(grid_k, spatial_frames))

        t_rnd = 0.0
        if do_render:
            t_r0 = time.perf_counter()
            prepare_still_render(scene, png_compression=png_compression)
            ok_anim = _render_animation_sequence(
                scene, rgb_dir, clip_frames, pose_objects, frame_poses, png_compression,
                start_frame=start_frame,
            )
            if not ok_anim:
                print("  [render] animation batch missed files; falling back to stills")
                _render_stills_fallback(
                    scene, rgb_dir, clip_frames, pose_objects, frame_poses, png_compression,
                    start_frame=start_frame,
                )
            t_rnd = time.perf_counter() - t_r0
            if outcome == "collision":
                prefix_rgb = rgb_dir

        if outcome == "collision":
            prefix_spatial = list(spatial_frames)
            prefix_records = list(object_frames)

        t_heat = 0.0
        heat_dir = spatial_dir / "_heat"
        if want_overlay:
            t_h0 = time.perf_counter()
            if start_frame > 0 and prefix_heat is not None:
                copied_h = link_or_copy_prefix(prefix_heat, heat_dir, start_frame, suffix=".ppm")
                print(f"  cache prefix {copied_h} heat ← {prefix_heat}")
                write_heat_sequence(
                    spatial_frames[start_frame:clip_frames], heat_dir, k=grid_k,
                )
            else:
                write_heat_sequence(spatial_frames[:clip_frames], heat_dir, k=grid_k)
            t_heat = time.perf_counter() - t_h0

        split_len = 0
        if diverge_frame > 1 and diverge_frame < clip_frames:
            if outcome == "collision":
                split_len = diverge_frame
            elif start_frame == diverge_frame:
                video_ready = (not want_video) or (
                    prefix_video is not None and prefix_video.is_file()
                )
                overlay_ready = (not want_overlay) or (
                    prefix_overlay is not None and prefix_overlay.is_file()
                )
                if video_ready and overlay_ready and (want_video or want_overlay):
                    split_len = diverge_frame
        cache_preview = ep_dir / "_cache_prefix_preview.mp4" if outcome == "collision" else None
        cache_overlay = ep_dir / "_cache_prefix_overlay.mp4" if outcome == "collision" else None

        video_path = ep_dir / str(cfg["output"].get("video_name", "preview.mp4"))
        overlay_path = ep_dir / "spatial_overlay.mp4"
        t_m0 = time.perf_counter()
        video_ok, video_via, overlay_ok, overlay_via, keep_fail = _mux_clip_videos(
            scene=scene,
            rgb_dir=rgb_dir,
            n_frames=clip_frames,
            cfg=cfg,
            video_path=video_path,
            overlay_path=overlay_path,
            spatial_dir=spatial_dir,
            spatial_frames=spatial_frames,
            grid_k=grid_k,
            want_video=want_video,
            want_overlay=want_overlay,
            prefix_len=split_len,
            prefix_video=prefix_video,
            prefix_overlay=prefix_overlay,
            write_prefix_video=cache_preview,
            write_prefix_overlay=cache_overlay,
            heat_dir=heat_dir if want_overlay else None,
            write_heat=False,
            keep_heat=bool(want_overlay and outcome == "collision"),
        )
        t_mux = time.perf_counter() - t_m0
        print(
            f"  timing  sim={t_sim:.2f}s  render={t_rnd:.2f}s  "
            f"heat={t_heat:.2f}s  mux={t_mux:.2f}s  start={start_frame}"
        )
        if outcome == "collision":
            if cache_preview is not None and cache_preview.is_file():
                prefix_video = cache_preview
            if cache_overlay is not None and cache_overlay.is_file():
                prefix_overlay = cache_overlay
            if want_overlay and heat_dir.is_dir():
                prefix_heat = heat_dir
        keep_frames = (media in ("frames", "both") and do_render) or keep_fail
        if drop_rgb and want_video and not keep_fail:
            keep_frames = False
        # Collision PNGs are the prefix cache for miss clips of this
        # background. Drop them only when the next (sample, bg) starts.
        if do_render and not keep_frames:
            rgb_dirs_to_drop.append(rgb_dir)

        manifest = {
            "layout": "groups",
            "sample_id": clip["sample_id"],
            "sample_dir": clip["sample_dir"],
            "outcome": outcome,
            "bg_id": clip["bg_id"],
            "bg_style": clip["bg_style"],
            "bg_lighting": clip.get("bg_lighting") or state.environment.get("lighting"),
            "diverge_s": diverge_s,
            "diverge_frame": diverge_frame,
            "hit_s": clip["hit_s"],
            "scenario": clip["scenario"],
            "dir": str(ep_dir.relative_to(out_root)),
            "frames": clip_frames,
            "fps": fps,
            "walk_speed": round(0.0 if profile.stationary else state.walk_speed, 4),
            "sidewalk_lateral": round(state.sidewalk_lateral, 4),
            "biome": gen.biome,
            "ego": {
                "mode": profile.mode,
                "eye_height_m": round(float(profile.eye_height), 3),
                "stature": str(getattr(profile, "stature", "typical")),
                "stationary": bool(profile.stationary),
            },
            "camera": {
                "lens_mm": round(float(scene_cfg["camera"]["lens_mm"]), 3),
                "hfov_deg": round(float(scene_cfg["camera"]["hfov_deg"]), 2),
            },
            "environment": state.environment,
            "label_histogram": dict(label_hist),
            "render": do_render,
            "media": media if do_render else "annotations",
            "rgb_dir": str(cfg["output"]["rgb_dirname"]) if keep_frames else None,
            "video": video_path.name if video_ok else None,
            "video_encoder": video_via if video_ok else None,
            "annotations": "annotations/annotations.json" if write_annotations else None,
            "spatial_annotations_k": grid_k,
            "spatial_annotations": "spatial_annotations/spatial_annotations.json",
            "spatial_overlay": overlay_path.name if overlay_ok else None,
            "prefix_cached": bool(start_frame > 0),
            "prefix_json_cached": bool(copied_prefix_json),
            "prefix_mux": "concat-copy" if split_len else None,
        }
        _write_json(ep_dir / str(cfg["output"]["manifest_name"]), manifest)
        index_rows.append(
            {
                "sample_id": clip["sample_id"],
                "scenario": clip["scenario"],
                "outcome": outcome,
                "bg_id": clip["bg_id"],
                "bg_style": clip["bg_style"],
                "bg_lighting": clip.get("bg_lighting")
                or (state.environment.get("lighting") if state else None),
                "path": str(ep_dir.relative_to(out_root)),
                "video": "preview.mp4" if video_ok else None,
                "spatial_overlay": "spatial_overlay.mp4" if overlay_ok else None,
                "diverge_frame": diverge_frame,
                "y": 1 if outcome == "collision" else 0,
            }
        )
        n_ok += 1

    _close_world()
    for leftover in rgb_dirs_to_drop:
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)

    for sample in plan.get("samples") or []:
        sdir = out_root / str(sample["dir"])
        sample_outcomes = list(sample.get("outcomes") or ["collision"])
        index = {
            outcome: {bg["id"]: f"{outcome}/{bg['id']}" for bg in sample["backgrounds"]}
            for outcome in sample_outcomes
        }
        _write_json(
            sdir / "group.json",
            {
                "sample_id": sample["id"],
                "scenario": sample["scenario"],
                "diverge_s": sample["diverge_s"],
                "diverge_frame": sample["diverge_frame"],
                "hit_s": sample["hit_s"],
                "scene_seed": sample["scene_seed"],
                "backgrounds": sample["backgrounds"],
                "outcomes": sample_outcomes,
                "index": index,
            },
        )

    _write_json(
        out_root / "groups_index.json",
        {
            "layout": "groups",
            "n_ok": n_ok,
            "n_clips": len(clips),
            "train_glob": "sample_*/{collision,miss_left,miss_right}/bg_*/preview.mp4",
            "clips": index_rows,
        },
    )
    print(f"[done] groups wrote {n_ok}/{len(clips)} clip(s) → {out_root}")
    return 0 if n_ok == len(clips) else 2


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = get_config()
    if args.list_scenarios:
        sc = cfg["scenarios"]
        print("safe_pool:")
        for name in sc["safe_pool"]:
            print(f"  {name}")
        print("near_miss_pool:")
        for name in sc["near_miss_pool"]:
            print(f"  {name}")
        print("critical_pool:")
        for name in sc["critical_pool"]:
            print(f"  {name}")
        print("peripheral_safe (empty-center pack; not in --scenario auto):")
        for name in sc.get("peripheral_safe") or ():
            print(f"  {name}")
        print("peripheral_critical:")
        for name in sc.get("peripheral_critical") or ():
            print(f"  {name}")
        print(f"total={len(all_scenario_names(cfg))}")
        print()
        print("short aliases (for compounds):")
        for src, dst in sorted(SCENARIO_ALIASES.items()):
            print(f"  {src} → {dst}")
        print()
        print("compound examples:")
        print("  --scenario jaywalker,car,pothole")
        print("  --scenarios jaywalker+car_approaching+pothole_on_path")
        print("  --scenario jaywalker --scenario car --scenario pothole")
        print("  --scenario empty_street,jaywalker")
        print("  --scenario periph_ped_cut")
        print("  gen_dataset.py --theme peripheral --n 40")
        return 0
    request = scenario_request_from_tokens(args.scenario, args.scenarios)
    args.scenario_request = request
    threat_k = int(args.threat_grid)
    if threat_k < 1:
        print(f"invalid --threat-grid {threat_k}; expected an integer >= 1", file=sys.stderr)
        return 2
    media = str(args.media or cfg["output"].get("media", "both")).lower()
    if media not in ("frames", "video", "both"):
        print(f"invalid --media {media!r}; expected frames|video|both", file=sys.stderr)
        return 2
    frames_override = int(args.frames)
    if frames_override <= 0:
        dur = float(getattr(args, "duration", 0.0) or 0.0)
        if dur > 0.0:
            fps = max(1, int(cfg["render"]["fps"]))
            frames_override = max(1, int(round(dur * fps)))
    plan_eps: list[dict] | None = None
    plan_doc: dict | None = None
    if args.plan:
        plan_path = Path(args.plan)
        if not plan_path.is_file():
            print(f"--plan not found: {plan_path}", file=sys.stderr)
            return 2
        try:
            plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"invalid --plan JSON: {exc}", file=sys.stderr)
            return 2
        if str(plan_doc.get("layout") or "") == "groups":
            out_root = Path(args.output) if args.output else (_ROOT / str(cfg["output"]["root"]))
            out_root.mkdir(parents=True, exist_ok=True)
            print(f"[run] group plan={plan_path}  samples={len(plan_doc.get('samples') or [])}")
            return run_group_plan(
                plan=plan_doc,
                cfg=cfg,
                out_root=out_root,
                args=args,
                frames_override=int(frames_override),
                media=media,
                threat_k=threat_k,
            )
        plan_eps = list(plan_doc.get("episodes") or [])
        if not plan_eps:
            print("--plan has no episodes[]", file=sys.stderr)
            return 2
        print(f"[run] plan={plan_path}  {len(plan_eps)} episode(s)")
    n_ep = len(plan_eps) if plan_eps is not None else int(args.episodes)
    args.episodes = n_ep
    out_root = Path(args.output) if args.output else (_ROOT / str(cfg["output"]["root"]))
    out_root.mkdir(parents=True, exist_ok=True)

    if args.start_episode is None:
        start_id = 0 if plan_eps is not None else next_free_episode_id(out_root)
        if plan_eps is None:
            print(f"[run] --start-episode omitted; continuing at {start_id:04d} under {out_root}")
    else:
        start_id = int(args.start_episode)

    master = random.Random(int(args.seed))
    # Advance the master RNG so --start-episode N is a stable shard and so
    # an auto-continued run with the same --seed does not clone episode 0.
    for _ in range(max(0, start_id)):
        master.randrange(1, 2**31)

    n_ok = 0
    for k in range(n_ep):
        episode_id = start_id + k
        if plan_eps is not None:
            request_k = str(plan_eps[k].get("scenario") or "auto")
        else:
            request_k = request
        try:
            run_episode(
                episode_id=episode_id,
                scenario_request=request_k,
                cfg=cfg,
                rng=master,
                out_root=out_root,
                no_render=bool(args.no_render),
                no_occlusion=bool(args.no_occlusion),
                frames_override=int(frames_override),
                media=media,
                hfov_deg=args.hfov,
                lens_mm=args.lens_mm,
                lock_fov=bool(args.no_random_fov),
                threat_k=threat_k,
                spatial_overlay=bool(args.spatial_overlay),
                write_annotations=not bool(args.no_annotations),
                drop_rgb=bool(args.no_rgb),
                biome=str(args.biome),
                ego_mode=str(args.ego_mode),
                ego_height=str(getattr(args, "ego_height", "auto")),
                chaos=args.chaos,
                no_trees=bool(args.no_trees),
                wind=str(args.wind),
            )
            n_ok += 1
        except Exception:
            print(f"[episode {episode_id:04d}] FAILED", file=sys.stderr)
            traceback.print_exc()

    summary = rebuild_dataset_summary(out_root, cfg, args, media)
    _write_json(out_root / "dataset_summary.json", summary)
    print(f"[done] wrote {n_ok}/{n_ep} new episode(s) → {out_root}")
    print(f"       {summary['episodes_on_disk']} episode(s) on disk  labels={summary['label_histogram']}")
    return 0 if n_ok == n_ep else 2


if __name__ == "__main__":
    sys.exit(main())
