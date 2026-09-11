#!/usr/bin/env python3
"""Central orchestrator for the egocentric accessibility synthetic generator.

Usage (from this directory or via python /storage/BTP/synth_sim/main.py):

    python main.py --episodes 8 --seconds 20 --output-mode both --output ./dataset_output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import glm
import numpy as np

from actors import quat_to_xyzw, vec3_to_list
from biomechanics import BiomechanicalCameraRig, CameraPose, SidewalkPath
from collision_engine import (
    CollisionDirector,
    EpisodeWorld,
    advance_actor,
    episode_realized_class,
    kinematics_vs_camera,
)
from config import SimulationConfig
from core_gl import OffscreenRenderer
from dataset_writer import DatasetWriter, build_annotation
from projection import compute_occlusion_and_truncation, project_obb_to_2d


SKIP_ANNOTATE = frozenset({"ground", "road", "sidewalk", "curb", "building"})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless egocentric accessibility synthetic data generator")
    p.add_argument("--episodes", type=int, default=4, help="Number of episodes to render")
    p.add_argument("--output", type=Path, default=_ROOT / "dataset_output")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--frames", type=int, default=None, help="Override frames per episode (default 20s at --fps)")
    p.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Episode length in seconds (default 20, minimum 15 unless --frames is set)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--output-mode",
        choices=("frames", "video", "both"),
        default="both",
        help="Write PNG/NPY/JSON frames, an MP4 per episode, or both",
    )
    return p.parse_args(argv)


def make_config(args: argparse.Namespace) -> SimulationConfig:
    cfg = SimulationConfig(seed=int(args.seed), writer_workers=int(args.workers))
    cfg.camera = type(cfg.camera)(width=int(args.width), height=int(args.height))
    cfg.output_mode = str(args.output_mode)
    if args.frames is not None:
        cfg.frames_per_episode = int(args.frames)
    elif args.seconds is not None:
        secs = max(float(cfg.min_episode_seconds), float(args.seconds))
        cfg.frames_per_episode = int(round(secs * cfg.fps))
    else:
        cfg.frames_per_episode = int(round(cfg.default_episode_seconds * cfg.fps))
    return cfg


def step_actors(world: EpisodeWorld, frame: int, dt: float) -> None:
    for act in world.actors:
        advance_actor(act, world, dt, frame)


def sky_tuple(v: glm.vec3) -> Tuple[float, float, float]:
    return (float(v.x), float(v.y), float(v.z))


def annotate_frame_full(
    cfg: SimulationConfig,
    world: EpisodeWorld,
    pose: CameraPose,
    proj: glm.mat4,
    depth: np.ndarray,
    inst: np.ndarray,
    frame: int,
    episode: int,
) -> Tuple[Dict, str]:
    env = world.environment
    objects: List[Dict] = []
    labels: List[str] = []
    w, h = cfg.camera.width, cfg.camera.height
    for act in world.actors:
        if act.class_name in SKIP_ANNOTATE:
            continue
        kin = kinematics_vs_camera(act, pose.position, pose.velocity, cfg.threat)
        box = project_obb_to_2d(act, pose.view, proj, (w, h), near=cfg.camera.near)
        occ, is_occ, trunc, is_trunc, clamped = compute_occlusion_and_truncation(
            act, depth, pose.view, proj, cfg, box=box, instance_mask=inst
        )
        if box.behind and kin["euclidean_distance"] > 12.0:
            continue
        if (not box.visible) and kin["euclidean_distance"] > 18.0:
            continue
        labels.append(str(kin["threat_classification"]))
        ttc = kin["time_to_collision_sec"]
        xmin, ymin, xmax, ymax = clamped.as_int()
        objects.append(
            {
                "instance_id": int(act.instance_id),
                "class_name": act.class_name,
                "threat_classification": kin["threat_classification"],
                "kinematics": {
                    "world_position": [float(x) for x in kin["world_position"]],
                    "world_velocity": [float(x) for x in kin["world_velocity"]],
                    "relative_position": [float(x) for x in kin["relative_position"]],
                    "relative_velocity": [float(x) for x in kin["relative_velocity"]],
                    "euclidean_distance": float(kin["euclidean_distance"]),
                    "time_to_collision_sec": None if ttc is None else float(ttc),
                    "closest_point_of_approach_dist": float(kin["closest_point_of_approach_dist"]),
                },
                "bounding_box_3d": {
                    "center": vec3_to_list(act.position),
                    "extents": vec3_to_list(act.extents),
                    "orientation_quaternion": quat_to_xyzw(act.orientation),
                },
                "bounding_box_2d": {
                    "xmin": int(xmin),
                    "ymin": int(ymin),
                    "xmax": int(xmax),
                    "ymax": int(ymax),
                },
                "visibility": {
                    "occlusion_ratio": float(occ),
                    "truncation_ratio": float(trunc),
                    "is_occluded": bool(is_occ),
                    "is_truncated": bool(is_trunc),
                },
            }
        )
    realized = episode_realized_class(labels)
    ann = build_annotation(
        cfg,
        episode,
        frame,
        timestamp=float(frame * cfg.dt),
        cam_pos=vec3_to_list(pose.position),
        cam_vel=vec3_to_list(pose.velocity),
        euler_deg={
            "pitch": float(pose.euler_deg.x),
            "yaw": float(pose.euler_deg.y),
            "roll": float(pose.euler_deg.z),
        },
        environment={
            "sun_azimuth_deg": float(env.sun_azimuth_deg),
            "sun_elevation_deg": float(env.sun_elevation_deg),
            "road_condition": env.road_condition,
            "ambient_light_intensity": float(env.ambient),
        },
        objects=objects,
    )
    ann["episode"] = {
        "scenario": world.scenario_name,
        "target_class": world.target_class,
        "realized_class": realized,
        "k_star": int(world.k_star),
    }
    return ann, realized


def render_episode(
    cfg: SimulationConfig,
    renderer: OffscreenRenderer,
    writer: DatasetWriter,
    episode: int,
) -> str:
    director = CollisionDirector(cfg)
    rng = np.random.default_rng(cfg.seed + 104729 * episode)
    world = director.build_episode(cfg.seed, episode)

    path = SidewalkPath(
        world.spline.positions,
        world.spline.tangents,
        world.spline.normals,
        world.spline.ups,
        world.spline.arclength,
        lateral_offset=world.sidewalk_lateral,
    )
    rig = BiomechanicalCameraRig(cfg, path, seed=cfg.seed + 13 * episode, s0=8.0)
    poses = list(rig.precompute(cfg.frames_per_episode, cfg.dt))
    director.finalize_with_camera(world, poses, rng)

    env = world.environment
    sun_dir = env.sun_direction()
    sun_col = env.sun_color()
    sky = env.sky_color()
    proj = renderer.projection

    realized_worst = "SAFE"
    t0 = time.time()
    sink = writer.begin_episode(episode)
    for frame in range(cfg.frames_per_episode):
        if frame > 0:
            step_actors(world, frame, cfg.dt)
        pose = poses[frame]
        focus = pose.position + (pose.target - pose.position) * 16.0
        renderer.compute_light_vp(sun_dir, focus)
        renderer.begin_shadow()
        for prop in world.static_props:
            if prop.class_name == "ground":
                continue
            renderer.draw_shadow(prop.mesh, prop.model)
        for act in world.actors:
            if act.mesh is None:
                continue
            renderer.draw_shadow(act.mesh, act.model_matrix())

        renderer.begin_frame(sky_tuple(sky))
        renderer.draw_sky(pose.view, sun_dir, sun_col, env.sun_elevation_deg)
        renderer.set_frame_uniforms(
            view=pose.view,
            sun_dir=sun_dir,
            sun_color=sun_col,
            ambient=env.ambient,
            sky=sky,
            fog=env.fog_density,
            cam_pos=pose.position,
            wetness=env.wetness,
            sun_elevation=env.sun_elevation_deg,
        )
        for prop in world.static_props:
            renderer.draw(
                prop.mesh,
                prop.model,
                prop.albedo,
                prop.instance_id,
                specularity=prop.specularity,
                shininess=prop.shininess,
                style=prop.style,
                wetness=env.wetness * prop.wetness_scale,
            )
        for act in world.actors:
            if act.mesh is None:
                continue
            wet = env.wetness if act.class_name in ("vehicle", "scooter") else 0.0
            renderer.draw(
                act.mesh,
                act.model_matrix(),
                act.albedo,
                act.instance_id,
                specularity=act.specularity,
                shininess=act.shininess,
                style=act.style,
                wetness=wet,
            )
        rgb, depth, inst = renderer.read_targets()
        ann, realized = annotate_frame_full(cfg, world, pose, proj, depth, inst, frame, episode)
        rank = {"SAFE": 0, "NEAR_MISS": 1, "CRITICAL_THREAT": 2}
        if rank.get(realized, 0) > rank.get(realized_worst, 0):
            realized_worst = realized
        sink.submit(frame, rgb, depth, inst, ann)

    sink.close()

    dt = time.time() - t0
    fps = cfg.frames_per_episode / max(dt, 1e-6)
    print(
        f"[episode {episode:05d}] scenario={world.scenario_name:20s} "
        f"target={world.target_class:16s} worst={realized_worst:16s} "
        f"{fps:.1f} frames/s"
    )
    return realized_worst


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"EgocentricAccessibilitySim-Scratch {cfg.generator_version}  "
        f"{cfg.camera.width}x{cfg.camera.height} @ {cfg.fps} Hz  "
        f"frames/ep={cfg.frames_per_episode} ({cfg.episode_duration:.1f}s)  output={cfg.output_mode}"
    )
    renderer = OffscreenRenderer(cfg)
    print(f"GL context: {renderer.ctx_info}  id_dtype={renderer.id_dtype}  vendor-ok")
    counts = {"SAFE": 0, "NEAR_MISS": 0, "CRITICAL_THREAT": 0}
    with DatasetWriter(args.output, cfg, output_mode=args.output_mode, max_workers=args.workers) as writer:
        for i in range(int(args.episodes)):
            ep = int(args.start_episode) + i
            realized = render_episode(cfg, renderer, writer, ep)
            counts[realized] = counts.get(realized, 0) + 1
    renderer.release()
    print("realized class counts (worst-in-episode):", counts)
    print(f"wrote dataset to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
