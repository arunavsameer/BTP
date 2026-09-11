"""Trajectory inversion, TTC / CPA metrics, threat labels, and scenario directors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import glm
import numpy as np

from actors import (
    Actor,
    EnvironmentState,
    make_actor,
    np_to_glm,
    pick_albedo,
    reset_instance_ids,
    sample_environment,
    yaw_to_quat,
)
from biomechanics import CameraPose
from config import SimulationConfig, ThreatThresholds
from geometry import (
    Mesh,
    StreetSpline,
    generate_bench_mesh,
    generate_bollard_mesh,
    generate_building_mesh,
    generate_control_points,
    generate_cyclist_mesh,
    generate_dustbin_mesh,
    generate_ground_plane,
    generate_head_hazard_mesh,
    generate_hydrant_mesh,
    generate_pedestrian_mesh,
    generate_planter_mesh,
    generate_pothole_mesh,
    generate_road_markings,
    generate_scooter_mesh,
    generate_street_lamp_mesh,
    generate_street_mesh,
    generate_tree_mesh,
    generate_vehicle_mesh,
    scatter_on_sidewalk,
    sidewalk_center_lateral,
)


INF = 1.0e6

SAFE_SCENARIOS = ("sidewalk_stroll", "opposite_flow", "far_crosswalk", "parked_street")
NEAR_SCENARIOS = (
    "near_miss_ped",
    "near_miss_cyclist",
    "jaywalker",
    "close_overtake",
    "approaching_group",
    "curb_stepout",
)
CRITICAL_SCENARIOS = (
    "critical_inversion",
    "sidewalk_incursion",
    "jaywalker",
    "sudden_brake",
    "head_overhang",
    "pothole_stepin",
    "crosswalk_conflict",
)


def glm_to_np(v: glm.vec3) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def ttc_cpa(p_rel: np.ndarray, v_rel: np.ndarray) -> Tuple[float, float, bool]:
    """Constant-velocity TTC and closest-point-of-approach distance.

    TTC = - (P_rel · V_rel) / ||V_rel||^2   if converging, else +inf.
    t_cpa = max(0, - (P_rel · V_rel) / ||V_rel||^2)
    D_cpa = || P_rel + V_rel t_cpa ||
    """
    closing = float(np.dot(p_rel, v_rel))
    v2 = float(np.dot(v_rel, v_rel))
    converging = closing < 0.0 and v2 > 1e-10
    if v2 < 1e-10:
        t_cpa = 0.0
        ttc = INF
    elif converging:
        t_cpa = -closing / v2
        ttc = t_cpa
    else:
        t_cpa = 0.0
        ttc = INF
    d_cpa = float(np.linalg.norm(p_rel + v_rel * t_cpa))
    return float(ttc), d_cpa, bool(converging)


def classify_threat(
    speed: float,
    ttc: float,
    d_cpa: float,
    converging: bool,
    thr: ThreatThresholds,
) -> str:
    """Priority: CRITICAL_THREAT > NEAR_MISS > SAFE_DYNAMIC / SAFE_STATIC."""
    if converging and ttc < thr.critical_ttc and d_cpa < thr.critical_cpa:
        return "CRITICAL_THREAT"
    if (
        converging
        and thr.near_miss_ttc_lo <= ttc <= thr.near_miss_ttc_hi
        and thr.near_miss_cpa_lo <= d_cpa <= thr.near_miss_cpa_hi
    ):
        return "NEAR_MISS"
    # Static objects sitting in the stride envelope still count as critical / near-miss.
    if speed < thr.static_speed:
        if converging and d_cpa < thr.critical_cpa and ttc < 3.5:
            return "CRITICAL_THREAT"
        if converging and d_cpa <= thr.near_miss_cpa_hi and ttc <= 4.0:
            return "NEAR_MISS"
        return "SAFE_STATIC"
    if (not converging) or (ttc > thr.safe_dynamic_ttc and d_cpa > thr.safe_dynamic_cpa):
        return "SAFE_DYNAMIC"
    if converging and ttc < 3.5 and d_cpa < 1.5:
        return "NEAR_MISS"
    return "SAFE_DYNAMIC"


def invert_constant_velocity(
    p_spawn: np.ndarray,
    p_target: np.ndarray,
    duration: float,
    speed_lo: float,
    speed_hi: float,
) -> np.ndarray:
    """V_required = (P_target - P_spawn) / (t* - t_spawn), clamped into a kinematic envelope."""
    dur = max(float(duration), 1e-3)
    delta = p_target - p_spawn
    vel = delta / dur
    spd = float(np.linalg.norm(vel))
    if spd < 1e-8:
        return np.zeros(3)
    if spd < speed_lo:
        vel = vel * (speed_lo / spd)
    elif spd > speed_hi:
        vel = vel * (speed_hi / spd)
    return vel


@dataclass
class StaticProp:
    mesh: Mesh
    model: glm.mat4
    albedo: Tuple[float, float, float]
    instance_id: int
    class_name: str
    specularity: float = 0.08
    shininess: float = 16.0
    style: int = 0
    wetness_scale: float = 0.0


@dataclass
class EpisodeWorld:
    spline: StreetSpline
    road_width: float
    curb_width: float
    curb_height: float
    sidewalk_width: float
    camera_side: int
    sidewalk_lateral: float
    static_props: List[StaticProp]
    actors: List[Actor]
    environment: EnvironmentState
    target_class: str
    scenario_name: str
    k_star: int
    street_meshes: Dict[str, Mesh]


def arclength_near(spline: StreetSpline, pos: np.ndarray | glm.vec3) -> float:
    p = glm_to_np(pos) if isinstance(pos, glm.vec3) else np.asarray(pos, dtype=np.float64).reshape(3)
    d = np.linalg.norm(spline.positions - p[None, :], axis=1)
    return float(spline.arclength[int(np.argmin(d))])


def bind_spline_motion(act: Actor, s: float, lat: float, ds: float, y: float) -> Actor:
    """Lock an actor to the Frenet frame so they stay on the street for a long clip."""
    act._follow_spline = True  # type: ignore[attr-defined]
    act._path_s = float(s)  # type: ignore[attr-defined]
    act._path_lat = float(lat)  # type: ignore[attr-defined]
    act._path_ds = float(ds)  # type: ignore[attr-defined]
    act._path_y = float(y)  # type: ignore[attr-defined]
    act.face_velocity = False
    return act


def bind_crossing(
    act: Actor,
    s: float,
    lat0: float,
    lat1: float,
    duration: float,
    t0: float = 0.0,
    ds_after: float = 1.2,
) -> Actor:
    act._crossing = True  # type: ignore[attr-defined]
    act._cross_s = float(s)  # type: ignore[attr-defined]
    act._cross_lat0 = float(lat0)  # type: ignore[attr-defined]
    act._cross_lat1 = float(lat1)  # type: ignore[attr-defined]
    act._cross_dur = float(max(duration, 0.4))  # type: ignore[attr-defined]
    act._cross_t = float(t0)  # type: ignore[attr-defined]
    act._path_ds = float(ds_after)  # type: ignore[attr-defined]
    act.face_velocity = False
    return act


def sync_path_pose(act: Actor, spline: StreetSpline) -> None:
    s = float(np.clip(act._path_s, 0.4, spline.length - 0.4))  # type: ignore[attr-defined]
    p, t, n, _ = spline.frame_at(s)
    pos = p + n * float(act._path_lat)  # type: ignore[attr-defined]
    y = float(act._path_y)  # type: ignore[attr-defined]
    ds = float(act._path_ds)  # type: ignore[attr-defined]
    vel = t * ds
    act.position = glm.vec3(float(pos[0]), y, float(pos[2]))
    act.velocity = glm.vec3(float(vel[0]), 0.0, float(vel[2]))
    if abs(ds) > 0.05:
        act.orientation = yaw_to_quat(math.atan2(float(vel[0]), float(vel[2])))


def advance_actor(act: Actor, world: EpisodeWorld, dt: float, frame: int) -> None:
    """One integration step: cross the road, follow the spline, or free Euler."""
    spline = world.spline
    if getattr(act, "_crossing", False):
        act._cross_t = float(act._cross_t) + dt  # type: ignore[attr-defined]
        dur = max(float(act._cross_dur), 1e-3)  # type: ignore[attr-defined]
        u = min(1.0, float(act._cross_t) / dur)  # type: ignore[attr-defined]
        lat = (1.0 - u) * float(act._cross_lat0) + u * float(act._cross_lat1)  # type: ignore[attr-defined]
        p, t, n, _ = spline.frame_at(float(act._cross_s))  # type: ignore[attr-defined]
        on_road = abs(lat) < 0.5 * world.road_width + world.curb_width
        y = 0.86 + (0.0 if on_road else world.curb_height)
        pos = p + n * lat
        dlat = (float(act._cross_lat1) - float(act._cross_lat0)) / dur  # type: ignore[attr-defined]
        vel = n * dlat
        act.position = glm.vec3(float(pos[0]), y, float(pos[2]))
        act.velocity = glm.vec3(float(vel[0]), 0.0, float(vel[2]))
        if abs(dlat) > 1e-4:
            act.orientation = yaw_to_quat(math.atan2(float(vel[0]), float(vel[2])))
        if u >= 1.0:
            act._crossing = False  # type: ignore[attr-defined]
            bind_spline_motion(
                act,
                float(act._cross_s),  # type: ignore[attr-defined]
                float(act._cross_lat1),  # type: ignore[attr-defined]
                float(getattr(act, "_path_ds", 1.2)),
                0.86 + world.curb_height,
            )
            sync_path_pose(act, spline)
        return

    if getattr(act, "_follow_spline", False):
        brake_f = getattr(act, "_brake_frame", None)
        ds = float(act._path_ds)  # type: ignore[attr-defined]
        if brake_f is not None and frame >= int(brake_f):
            acc = float(getattr(act, "_brake_accel", -3.0))
            nds = ds + acc * dt * (1.0 if ds >= 0.0 else -1.0)
            if ds == 0.0 or nds * ds <= 0.0:
                nds = 0.0
            ds = nds
            act._path_ds = ds  # type: ignore[attr-defined]
        s = float(act._path_s) + ds * dt  # type: ignore[attr-defined]
        lo, hi = 0.7, spline.length - 0.7
        if s < lo or s > hi:
            s = float(np.clip(s, lo, hi))
            if act.class_name in ("pedestrian", "cyclist", "scooter") and abs(ds) > 0.05:
                act._path_ds = -ds  # type: ignore[attr-defined]
                ds = -ds
            else:
                act._path_ds = 0.0  # type: ignore[attr-defined]
                ds = 0.0
        act._path_s = s  # type: ignore[attr-defined]
        sync_path_pose(act, spline)
        return

    brake_f = getattr(act, "_brake_frame", None)
    if brake_f is not None and frame >= int(brake_f):
        spd = glm.length(act.velocity)
        if spd > 0.05:
            act.acceleration = (act.velocity / spd) * float(getattr(act, "_brake_accel", -3.0))
        else:
            act.velocity = glm.vec3(0.0)
            act.acceleration = glm.vec3(0.0)
    act.step(dt)
    if act.class_name != "pothole":
        act.position.y = max(act.position.y, 0.0)


class CollisionDirector:
    """Chooses a target threat class and spawns actors that realize it at frame k*."""

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg

    def pick_target_class(self, rng: np.random.Generator) -> str:
        names = list(self.cfg.class_ratios.keys())
        probs = np.array([self.cfg.class_ratios[n] for n in names], dtype=np.float64)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))

    def build_episode(self, seed: int, episode_index: int) -> EpisodeWorld:
        cfg = self.cfg
        rng = np.random.default_rng(int(seed) + 7919 * int(episode_index))
        reset_instance_ids(100 + episode_index * 1000)
        target = self.pick_target_class(rng)
        env = sample_environment(cfg, rng)
        street = cfg.street
        road_w = float(rng.uniform(street.road_width_min, street.road_width_max))
        walk_w = float(rng.uniform(street.sidewalk_width_min, street.sidewalk_width_max))
        controls = generate_control_points(street, rng)
        packed = generate_street_mesh(
            controls,
            width=road_w,
            length=street.street_length,
            resolution=street.spline_resolution_m,
            curb_width=street.curb_width,
            curb_height=street.curb_height,
            sidewalk_width=walk_w,
        )
        spline: StreetSpline = packed.pop("spline")  # type: ignore[assignment]
        street_meshes = {k: v for k, v in packed.items() if isinstance(v, Mesh)}
        camera_side = -1  # walk on the right sidewalk (N is left)
        lat = sidewalk_center_lateral(road_w, street.curb_width, walk_w, camera_side)

        static_props: List[StaticProp] = []
        iid_ground = 1
        static_props.append(
            StaticProp(
                mesh=generate_ground_plane(220.0, y=-0.04),
                model=glm.mat4(1.0),
                albedo=(0.20, 0.28, 0.12),
                instance_id=iid_ground,
                class_name="ground",
                style=11,
            )
        )
        static_props.append(
            StaticProp(
                mesh=street_meshes["road"],
                model=glm.mat4(1.0),
                albedo=(0.11, 0.11, 0.12),
                instance_id=2,
                class_name="road",
                specularity=0.15 + 0.7 * env.wetness,
                shininess=8.0 + 56.0 * env.wetness,
                style=3,
                wetness_scale=1.0,
            )
        )
        static_props.append(
            StaticProp(
                mesh=street_meshes["curb"],
                model=glm.mat4(1.0),
                albedo=(0.55, 0.54, 0.50),
                instance_id=3,
                class_name="curb",
                specularity=0.08,
                style=13,
            )
        )
        if "verge" in street_meshes:
            static_props.append(
                StaticProp(
                    mesh=street_meshes["verge"],
                    model=glm.mat4(1.0),
                    albedo=(0.22, 0.32, 0.12),
                    instance_id=5,
                    class_name="ground",
                    specularity=0.04,
                    style=11,
                )
            )
        static_props.append(
            StaticProp(
                mesh=street_meshes["sidewalk"],
                model=glm.mat4(1.0),
                albedo=(0.52, 0.50, 0.46),
                instance_id=4,
                class_name="sidewalk",
                specularity=0.06 + 0.3 * env.wetness,
                style=2,
                wetness_scale=0.5,
            )
        )

        self._scatter_buildings(spline, road_w, street.curb_width, walk_w, rng, static_props)
        self._scatter_street_infra(spline, road_w, street.curb_width, walk_w, rng, static_props, env)

        actors: List[Actor] = []
        k_star = int(rng.integers(int(0.40 * cfg.frames_per_episode), int(0.80 * cfg.frames_per_episode)))

        if target == "SAFE":
            scenario = str(rng.choice(SAFE_SCENARIOS))
            avoid = True
        elif target == "NEAR_MISS":
            scenario = str(rng.choice(NEAR_SCENARIOS))
            avoid = True
        else:
            scenario = str(rng.choice(CRITICAL_SCENARIOS))
            avoid = False
        self._spawn_street_furniture(
            spline, road_w, street.curb_width, walk_w, camera_side, rng, actors, avoid_corridor=avoid
        )

        world = EpisodeWorld(
            spline=spline,
            road_width=road_w,
            curb_width=street.curb_width,
            curb_height=street.curb_height,
            sidewalk_width=walk_w,
            camera_side=camera_side,
            sidewalk_lateral=lat,
            static_props=static_props,
            actors=actors,
            environment=env,
            target_class=target,
            scenario_name=scenario,
            k_star=k_star,
            street_meshes=street_meshes,
        )
        return world

    def finalize_with_camera(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        """Populate the street, then place the scripted beat using the camera path."""
        name = world.scenario_name
        self._spawn_street_life(world, poses, rng)
        if name == "far_crosswalk":
            self._spawn_cross_group(world, poses, rng, ahead=16.0, n=int(rng.integers(3, 6)), aimed=False)
        elif name == "approaching_group":
            self._spawn_approaching_group(world, poses, rng)
        elif name == "curb_stepout":
            self._spawn_curb_stepout(world, poses, rng)
        elif name == "crosswalk_conflict":
            self._spawn_cross_group(world, poses, rng, ahead=None, n=int(rng.integers(3, 5)), aimed=False)
            self._spawn_jaywalker(world, poses, rng, critical=True)
        elif name == "near_miss_ped":
            self._spawn_inverted(world, poses, rng, kind="pedestrian", near_miss=True)
        elif name == "near_miss_cyclist":
            self._spawn_inverted(world, poses, rng, kind="cyclist", near_miss=True)
        elif name == "close_overtake":
            self._spawn_overtake(world, poses, rng)
        elif name == "critical_inversion":
            self._spawn_inverted(world, poses, rng, kind="pedestrian", near_miss=False)
        elif name == "sidewalk_incursion":
            self._spawn_sidewalk_incursion(world, poses, rng)
        elif name == "jaywalker":
            self._spawn_jaywalker(world, poses, rng, critical=world.target_class == "CRITICAL_THREAT")
        elif name == "sudden_brake":
            self._spawn_sudden_brake(world, poses, rng)
        elif name == "head_overhang":
            self._spawn_head_overhang(world, poses, rng)
        elif name == "pothole_stepin":
            self._spawn_pothole(world, poses, rng)
        elif world.target_class != "SAFE":
            self._spawn_inverted(world, poses, rng, kind="pedestrian", near_miss=world.target_class == "NEAR_MISS")

    # ------------------------------------------------------------------
    # Ambient population
    # ------------------------------------------------------------------

    def _scatter_buildings(
        self,
        spline: StreetSpline,
        road_w: float,
        curb_w: float,
        walk_w: float,
        rng: np.random.Generator,
        props: List[StaticProp],
    ) -> None:
        s = 6.0
        while s < spline.length - 6.0:
            width = float(rng.uniform(6.0, 14.0))
            depth = float(rng.uniform(6.0, 11.0))
            height = float(rng.uniform(7.0, 22.0))
            gap = float(rng.uniform(0.6, 3.5))
            mesh = generate_building_mesh(width, depth, height, rng=rng)
            style = int(rng.choice([1, 7, 10], p=[0.35, 0.40, 0.25]))
            for side in (+1, -1):
                lat = side * (0.5 * road_w + curb_w + walk_w + 0.5 * depth + 0.15)
                p, t, n, _ = spline.frame_at(s + 0.5 * width)
                yaw = math.atan2(t[0], t[2])
                pos = p + n * lat + np.array([0.0, 0.5 * height, 0.0])
                model = glm.translate(glm.mat4(1.0), np_to_glm(pos)) * glm.rotate(
                    glm.mat4(1.0), yaw, glm.vec3(0.0, 1.0, 0.0)
                )
                from actors import next_instance_id

                props.append(
                    StaticProp(
                        mesh=mesh,
                        model=model,
                        albedo=pick_albedo("building", rng),
                        instance_id=next_instance_id(),
                        class_name="building",
                        specularity=0.04 if style != 10 else 0.12,
                        style=style,
                    )
                )
            s += width + gap

    def _scatter_street_infra(
        self,
        spline: StreetSpline,
        road_w: float,
        curb_w: float,
        walk_w: float,
        rng: np.random.Generator,
        props: List[StaticProp],
        env,
    ) -> None:
        from actors import next_instance_id

        marks = generate_road_markings(spline, road_w)
        props.append(
            StaticProp(
                mesh=marks,
                model=glm.mat4(1.0),
                albedo=(0.92, 0.90, 0.78),
                instance_id=next_instance_id(),
                class_name="road",
                specularity=0.15,
                style=12,
                wetness_scale=0.4,
            )
        )
        lamp = generate_street_lamp_mesh()
        s = 8.0
        while s < spline.length - 6.0:
            for side in (+1, -1):
                lat = side * (0.5 * road_w + curb_w + 0.35)
                p, t, n, _ = spline.frame_at(s)
                yaw = math.atan2(t[0], t[2])
                # Arm should hang toward the roadway: rotate 180 on the left side.
                extra = 0.0 if side < 0 else math.pi
                pos = p + n * lat
                model = glm.translate(glm.mat4(1.0), glm.vec3(float(pos[0]), 0.0, float(pos[2]))) * glm.rotate(
                    glm.mat4(1.0), yaw + extra, glm.vec3(0.0, 1.0, 0.0)
                )
                props.append(
                    StaticProp(
                        mesh=lamp,
                        model=model,
                        albedo=(0.18, 0.18, 0.20),
                        instance_id=next_instance_id(),
                        class_name="sign",
                        specularity=0.4,
                        style=9,
                    )
                )
            s += float(rng.uniform(11.0, 16.0))
        # Occasional benches on the outer sidewalk.
        bench = generate_bench_mesh()
        s = 14.0
        while s < spline.length - 10.0:
            side = int(rng.choice([-1, 1]))
            lat = side * (0.5 * road_w + curb_w + walk_w - 0.45)
            p, t, n, _ = spline.frame_at(s)
            yaw = math.atan2(t[0], t[2])
            pos = p + n * lat
            model = glm.translate(glm.mat4(1.0), glm.vec3(float(pos[0]), self.cfg.street.curb_height, float(pos[2]))) * glm.rotate(
                glm.mat4(1.0), yaw, glm.vec3(0.0, 1.0, 0.0)
            )
            props.append(
                StaticProp(
                    mesh=bench,
                    model=model,
                    albedo=(0.28, 0.18, 0.10),
                    instance_id=next_instance_id(),
                    class_name="dustbin",
                    specularity=0.08,
                    style=8,
                )
            )
            s += float(rng.uniform(18.0, 28.0))
        hydrant = generate_hydrant_mesh()
        s = 11.0
        while s < spline.length - 8.0:
            side = int(rng.choice([-1, 1]))
            # Sit on the gutter, not in the walking lane.
            lat = side * (0.5 * road_w + 0.18)
            p, t, n, _ = spline.frame_at(s)
            yaw = math.atan2(t[0], t[2])
            pos = p + n * lat
            model = glm.translate(glm.mat4(1.0), glm.vec3(float(pos[0]), 0.40, float(pos[2]))) * glm.rotate(
                glm.mat4(1.0), yaw, glm.vec3(0.0, 1.0, 0.0)
            )
            props.append(
                StaticProp(
                    mesh=hydrant,
                    model=model,
                    albedo=(0.72, 0.12, 0.10),
                    instance_id=next_instance_id(),
                    class_name="bollard",
                    specularity=0.35,
                    style=9,
                )
            )
            s += float(rng.uniform(16.0, 26.0))
        planter = generate_planter_mesh(rng)
        s = 20.0
        while s < spline.length - 10.0:
            side = int(rng.choice([-1, 1]))
            lat = side * (0.5 * road_w + curb_w + walk_w - 0.55)
            p, t, n, _ = spline.frame_at(s)
            yaw = math.atan2(t[0], t[2])
            pos = p + n * lat
            model = glm.translate(
                glm.mat4(1.0), glm.vec3(float(pos[0]), self.cfg.street.curb_height + 0.32, float(pos[2]))
            ) * glm.rotate(glm.mat4(1.0), yaw, glm.vec3(0.0, 1.0, 0.0))
            props.append(
                StaticProp(
                    mesh=planter,
                    model=model,
                    albedo=(0.16, 0.38, 0.12),
                    instance_id=next_instance_id(),
                    class_name="dustbin",
                    specularity=0.06,
                    style=4,
                )
            )
            s += float(rng.uniform(22.0, 34.0))

    def _spawn_street_furniture(
        self,
        spline: StreetSpline,
        road_w: float,
        curb_w: float,
        walk_w: float,
        camera_side: int,
        rng: np.random.Generator,
        actors: List[Actor],
        avoid_corridor: bool,
    ) -> None:
        other_side = -camera_side
        for side, rmin, factory, cls, n_max in (
            (other_side, self.cfg.poisson.dustbin, generate_dustbin_mesh, "dustbin", 6),
            (other_side, self.cfg.poisson.bollard, generate_bollard_mesh, "bollard", 8),
            (other_side, self.cfg.poisson.scooter, generate_scooter_mesh, "scooter", 3),
            (camera_side, self.cfg.poisson.dustbin, generate_dustbin_mesh, "dustbin", 3 if avoid_corridor else 5),
            (camera_side, self.cfg.poisson.tree, lambda: generate_tree_mesh(rng), "tree", 3),
        ):
            pts = scatter_on_sidewalk(spline, road_w, curb_w, walk_w, rmin, rng, side=side)
            rng.shuffle(pts)
            placed = 0
            corridor_lat = sidewalk_center_lateral(road_w, curb_w, walk_w, camera_side)
            for sp in pts:
                if placed >= n_max:
                    break
                if avoid_corridor and side == camera_side:
                    if abs(sp.lateral - corridor_lat) < 1.25:
                        continue
                y = 0.0
                mesh = factory()
                ext = {
                    "dustbin": glm.vec3(0.48, 1.05, 0.48),
                    "bollard": glm.vec3(0.20, 0.95, 0.20),
                    "scooter": glm.vec3(0.40, 1.20, 1.10),
                    "tree": glm.vec3(1.6, 3.2, 1.6),
                }[cls]
                pos = glm.vec3(float(sp.world[0]), float(ext.y * 0.5 if cls != "tree" else 0.0), float(sp.world[2]))
                if cls == "tree":
                    pos = glm.vec3(float(sp.world[0]), 0.0, float(sp.world[2]))
                    mesh = generate_tree_mesh(rng)
                    act = make_actor("tree", pos, mesh=mesh, rng=rng, extents=glm.vec3(2.2, 6.0, 2.2))
                    act.style = 4
                    act.velocity = glm.vec3(0.0)
                    act.face_velocity = False
                    actors.append(act)
                    placed += 1
                    continue
                act = make_actor(cls if cls != "tree" else "head_obstacle", pos, mesh=mesh, rng=rng, extents=ext)
                if cls in ("dustbin", "bollard", "scooter"):
                    act.position.y = float(ext.y) * 0.5 + self.cfg.street.curb_height
                act.velocity = glm.vec3(0.0)
                act.face_velocity = False
                actors.append(act)
                placed += 1

    def _walk_lat(self, world: EpisodeWorld, side: int) -> float:
        return sidewalk_center_lateral(world.road_width, world.curb_width, world.sidewalk_width, side)

    def _building_lat(self, world: EpisodeWorld, side: int) -> float:
        return side * (0.5 * world.road_width + world.curb_width + world.sidewalk_width - 0.40)

    def _ped_y(self, world: EpisodeWorld) -> float:
        return 0.86 + world.curb_height

    def _make_ped(
        self,
        world: EpisodeWorld,
        rng: np.random.Generator,
        s: float,
        lat: float,
        ds: float,
    ) -> Actor:
        y = self._ped_y(world)
        p, t, n, _ = world.spline.frame_at(s)
        pos = p + n * lat
        heading = math.atan2(t[0], t[2])
        if ds < 0.0:
            heading += math.pi
        act = make_actor(
            "pedestrian",
            glm.vec3(float(pos[0]), y, float(pos[2])),
            mesh=generate_pedestrian_mesh(rng),
            velocity=glm.vec3(math.sin(heading) * abs(ds), 0.0, math.cos(heading) * abs(ds)),
            rng=rng,
            yaw=heading,
        )
        bind_spline_motion(act, s, lat, ds, y)
        sync_path_pose(act, world.spline)
        return act

    def _spawn_street_life(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        pop = self.cfg.population
        cam_s0 = arclength_near(world.spline, poses[0].position)
        cam_s1 = arclength_near(world.spline, poses[-1].position)
        clear = world.target_class == "SAFE"
        self._spawn_sidewalk_crowd(world, rng, cam_s0, cam_s1, clear)
        self._spawn_ambient_crossers(world, rng, cam_s0, cam_s1, clear)
        self._spawn_traffic(world, rng, cam_s0, cam_s1)
        n_cyc = int(rng.integers(pop.cyclist_min, pop.cyclist_max + 1))
        self._spawn_lane_cyclists(world, rng, cam_s0, n_cyc)
        for _ in range(int(pop.scooter_moving)):
            self._spawn_moving_scooter(world, rng, cam_s0)

    def _spawn_sidewalk_crowd(
        self,
        world: EpisodeWorld,
        rng: np.random.Generator,
        cam_s0: float,
        cam_s1: float,
        clear: bool,
    ) -> None:
        pop = self.cfg.population
        n_walk = int(rng.integers(pop.ped_walk_min, pop.ped_walk_max + 1))
        for _ in range(n_walk):
            # Prefer the opposite sidewalk so the wearer's corridor stays walkable.
            if clear:
                side = int(-world.camera_side if rng.random() < 0.78 else world.camera_side)
                ds_sign = 1.0 if side == world.camera_side else float(rng.choice([-1.0, 1.0]))
            else:
                side = int(rng.choice([-1, 1]))
                ds_sign = 1.0 if rng.random() < 0.55 else -1.0
            spd = float(rng.uniform(0.85, 1.65))
            s = float(rng.uniform(4.0, world.spline.length - 6.0))
            if side == world.camera_side and clear:
                # Same-direction, well ahead, hugging the building line — never oncoming.
                lat = self._building_lat(world, side) + float(rng.uniform(-0.08, 0.08))
                s = max(s, cam_s0 + float(rng.uniform(12.0, 28.0)))
                ds_sign = 1.0
            else:
                lat = self._walk_lat(world, side) + float(rng.uniform(-0.35, 0.35))
            world.actors.append(self._make_ped(world, rng, s, lat, ds_sign * spd))

        n_stand = int(rng.integers(pop.ped_stand_min, pop.ped_stand_max + 1))
        for _ in range(n_stand):
            side = int(rng.choice([-1, 1]))
            s = float(rng.uniform(8.0, world.spline.length - 8.0))
            if clear and side == world.camera_side and cam_s0 - 2.0 < s < cam_s1 + 3.0:
                continue
            lat = self._building_lat(world, side)
            world.actors.append(self._make_ped(world, rng, s, lat, 0.0))

        n_pairs = int(rng.integers(pop.ped_pair_min, pop.ped_pair_max + 1))
        for _ in range(n_pairs):
            side = int(-world.camera_side if clear else rng.choice([-1, 1]))
            s = float(rng.uniform(10.0, world.spline.length - 10.0))
            if side == world.camera_side and clear:
                s = max(s, cam_s0 + 10.0)
            mid = self._walk_lat(world, side) if side != world.camera_side else self._building_lat(world, side)
            ds = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.95, 1.45))
            world.actors.append(self._make_ped(world, rng, s, mid - 0.32, ds))
            world.actors.append(self._make_ped(world, rng, s + 0.15, mid + 0.32, ds))

    def _spawn_ambient_crossers(
        self,
        world: EpisodeWorld,
        rng: np.random.Generator,
        cam_s0: float,
        cam_s1: float,
        clear: bool,
    ) -> None:
        pop = self.cfg.population
        n = int(rng.integers(pop.ped_cross_min, pop.ped_cross_max + 1))
        lo, hi = world.spline.length * 0.08, world.spline.length - 8.0
        for _ in range(n):
            if clear:
                if rng.random() < 0.65:
                    s = float(rng.uniform(min(hi, cam_s1 + 5.0), min(hi, cam_s1 + 32.0)))
                else:
                    s = float(rng.uniform(lo, max(lo + 1.0, cam_s0 - 3.0)))
            else:
                s = float(rng.uniform(lo, hi))
            start_side = int(rng.choice([-1, 1]))
            dest_side = -start_side
            lat0 = self._walk_lat(world, start_side)
            if dest_side == world.camera_side and clear:
                lat1 = self._building_lat(world, dest_side)
            else:
                lat1 = self._walk_lat(world, dest_side)
            dur = float(rng.uniform(5.2, 8.0))
            t0 = float(rng.uniform(0.0, 0.45 * dur))
            y = self._ped_y(world)
            p, t, nrm, _ = world.spline.frame_at(s)
            pos = p + nrm * lat0
            act = make_actor(
                "pedestrian",
                glm.vec3(float(pos[0]), y, float(pos[2])),
                mesh=generate_pedestrian_mesh(rng),
                rng=rng,
            )
            bind_crossing(act, s, lat0, lat1, dur, t0=t0, ds_after=float(rng.choice([-1.2, 1.2])))
            advance_actor(act, world, 0.0, 0)
            world.actors.append(act)

    def _lane_free(self, occupied: List[Tuple[float, float]], s: float, lane: float, gap: float) -> bool:
        for os, ol in occupied:
            if abs(ol - lane) < 0.15 and abs(os - s) < gap:
                return False
        return True

    def _spawn_traffic(
        self,
        world: EpisodeWorld,
        rng: np.random.Generator,
        cam_s0: float,
        cam_s1: float,
    ) -> None:
        pop = self.cfg.population
        occupied: List[Tuple[float, float]] = []
        n_move = int(rng.integers(pop.veh_moving_min, pop.veh_moving_max + 1))
        for i in range(n_move):
            with_traffic = rng.random() < 0.5
            lane = (-1.0 if with_traffic else 1.0) * (0.28 * world.road_width)
            ds = float(rng.uniform(7.5, 13.5)) * (1.0 if with_traffic else -1.0)
            if with_traffic:
                s = float(rng.uniform(2.0, max(6.0, cam_s0 + 8.0))) if rng.random() < 0.45 else float(
                    rng.uniform(cam_s0 + 10.0, min(world.spline.length - 10.0, cam_s1 + 40.0))
                )
            else:
                s = float(rng.uniform(min(world.spline.length - 12.0, cam_s0 + 25.0), world.spline.length - 6.0))
            if not self._lane_free(occupied, s, lane, 14.0):
                continue
            occupied.append((s, lane))
            self._spawn_path_vehicle(world, rng, s, lane, ds)

        n_park = int(rng.integers(pop.veh_parked_min, pop.veh_parked_max + 1))
        extra_park = 2 if world.scenario_name == "parked_street" else 0
        for _ in range(n_park + extra_park):
            side = int(rng.choice([-1, 1]))
            lat = side * (0.5 * world.road_width - 1.15)
            s = float(rng.uniform(8.0, world.spline.length - 8.0))
            if not self._lane_free(occupied, s, lat, 9.0):
                continue
            occupied.append((s, lat))
            self._spawn_path_vehicle(world, rng, s, lat, 0.0)

    def _spawn_path_vehicle(
        self,
        world: EpisodeWorld,
        rng: np.random.Generator,
        s: float,
        lat: float,
        ds: float,
    ) -> Actor:
        kind = str(rng.choice(["sedan", "van", "truck"], p=[0.58, 0.27, 0.15]))
        ext = glm.vec3(2.0, 1.5, 4.5) if kind == "sedan" else glm.vec3(2.0, 1.8, 5.5)
        y = float(ext.y) * 0.5
        p, t, n, _ = world.spline.frame_at(s)
        pos = p + n * lat
        heading = math.atan2(t[0], t[2])
        if ds < 0.0:
            heading += math.pi
        act = make_actor(
            "vehicle",
            glm.vec3(float(pos[0]), y, float(pos[2])),
            mesh=generate_vehicle_mesh(kind),
            rng=rng,
            yaw=heading,
            extents=ext,
        )
        bind_spline_motion(act, s, lat, ds, y)
        sync_path_pose(act, world.spline)
        world.actors.append(act)
        return act

    def _spawn_lane_cyclists(
        self,
        world: EpisodeWorld,
        rng: np.random.Generator,
        cam_s0: float,
        n: int,
    ) -> None:
        for i in range(n):
            with_traffic = i % 2 == 0
            lat = (-1.0 if with_traffic else 1.0) * (0.38 * world.road_width)
            ds = float(rng.uniform(4.0, 7.0)) * (1.0 if with_traffic else -1.0)
            s = float(rng.uniform(6.0, world.spline.length - 10.0))
            y = 0.85
            p, t, nrm, _ = world.spline.frame_at(s)
            pos = p + nrm * lat
            heading = math.atan2(t[0], t[2]) + (0.0 if with_traffic else math.pi)
            act = make_actor(
                "cyclist",
                glm.vec3(float(pos[0]), y, float(pos[2])),
                mesh=generate_cyclist_mesh(rng),
                rng=rng,
                yaw=heading,
                extents=glm.vec3(0.60, 1.70, 1.70),
            )
            bind_spline_motion(act, s, lat, ds, y)
            sync_path_pose(act, world.spline)
            world.actors.append(act)

    def _spawn_moving_scooter(
        self,
        world: EpisodeWorld,
        rng: np.random.Generator,
        cam_s0: float,
    ) -> None:
        side = int(-world.camera_side)
        lat = self._walk_lat(world, side) + float(rng.uniform(-0.2, 0.2))
        s = float(rng.uniform(8.0, world.spline.length - 8.0))
        ds = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(2.8, 5.0))
        y = 0.60 + world.curb_height
        p, t, n, _ = world.spline.frame_at(s)
        pos = p + n * lat
        act = make_actor(
            "scooter",
            glm.vec3(float(pos[0]), y, float(pos[2])),
            mesh=generate_scooter_mesh(),
            rng=rng,
            extents=glm.vec3(0.40, 1.20, 1.10),
        )
        bind_spline_motion(act, s, lat, ds, y)
        sync_path_pose(act, world.spline)
        world.actors.append(act)

    def _spawn_cross_group(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
        ahead: Optional[float],
        n: int,
        aimed: bool,
    ) -> None:
        cam_s0 = arclength_near(world.spline, poses[0].position)
        if ahead is None:
            pose_star = self._camera_at(poses, world.k_star)
            s = arclength_near(world.spline, pose_star.position)
        else:
            s = min(world.spline.length - 8.0, cam_s0 + float(ahead))
        start_side = int(-world.camera_side)
        dest_side = -start_side
        lat0 = self._walk_lat(world, start_side)
        if aimed:
            lat1 = world.sidewalk_lateral
        elif dest_side == world.camera_side and world.target_class == "SAFE":
            lat1 = self._building_lat(world, dest_side)
        else:
            lat1 = self._walk_lat(world, dest_side)
        for i in range(n):
            dur = float(rng.uniform(5.5, 7.5))
            t0 = float(i) * (0.35 * dur / max(n, 1)) + float(rng.uniform(0.0, 0.25))
            si = s + float(rng.uniform(-0.8, 0.8))
            p, t, nrm, _ = world.spline.frame_at(si)
            pos = p + nrm * lat0
            act = make_actor(
                "pedestrian",
                glm.vec3(float(pos[0]), self._ped_y(world), float(pos[2])),
                mesh=generate_pedestrian_mesh(rng),
                rng=rng,
            )
            bind_crossing(act, si, lat0, lat1, dur, t0=t0, ds_after=float(rng.choice([-1.15, 1.15])))
            advance_actor(act, world, 0.0, 0)
            world.actors.append(act)

    def _spawn_approaching_group(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        """Two or three oncoming sidewalk walkers that part around the wearer (near-miss CPA)."""
        k_star = world.k_star
        pose_star = self._camera_at(poses, k_star)
        p_cam = glm_to_np(pose_star.position)
        s_star = arclength_near(world.spline, p_cam)
        n_people = int(rng.integers(2, 4))
        duration = max(k_star * self.cfg.dt, 2.0)
        y = self._ped_y(world)
        for i in range(n_people):
            sign = -1.0 if i % 2 == 0 else 1.0
            mag = float(rng.uniform(*self.cfg.threat.near_miss_offset))
            lat = world.sidewalk_lateral + sign * mag
            speed = float(rng.uniform(1.15, 1.55))
            s_spawn = min(world.spline.length - 2.0, s_star + speed * duration)
            world.actors.append(self._make_ped(world, rng, s_spawn, lat, -speed))

    def _spawn_curb_stepout(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        """Person on the camera-side curb walks laterally into the stride envelope."""
        pose_star = self._camera_at(poses, world.k_star)
        s = arclength_near(world.spline, pose_star.position)
        lat0 = self._building_lat(world, world.camera_side)
        lat1 = world.sidewalk_lateral + world.camera_side * (-0.15)
        dur = max(2.2, world.k_star * self.cfg.dt * 0.35)
        p, t, n, _ = world.spline.frame_at(s)
        pos = p + n * lat0
        act = make_actor(
            "pedestrian",
            glm.vec3(float(pos[0]), self._ped_y(world), float(pos[2])),
            mesh=generate_pedestrian_mesh(rng),
            rng=rng,
        )
        bind_crossing(act, s, lat0, lat1, dur, t0=0.0, ds_after=0.0)
        world.actors.append(act)

    # ------------------------------------------------------------------
    # Scripted threats
    # ------------------------------------------------------------------

    def _camera_at(self, poses: Sequence[CameraPose], k: int) -> CameraPose:
        k = int(np.clip(k, 0, len(poses) - 1))
        return poses[k]

    def _offset_vector(self, rng: np.random.Generator, near_miss: bool, tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
        thr = self.cfg.threat
        mag = float(rng.uniform(*(thr.near_miss_offset if near_miss else thr.critical_offset)))
        # Prefer a lateral miss for near-miss; a mixed offset for critical hits.
        if near_miss:
            sign = -1.0 if rng.random() < 0.5 else 1.0
            return normal * (sign * mag)
        mix = rng.normal(0.0, 0.35, size=3)
        mix[1] = abs(mix[1]) * 0.2
        nrm = np.linalg.norm(mix)
        if nrm < 1e-6:
            mix = normal * mag
        else:
            mix = mix / nrm * mag
        return mix

    def _spawn_inverted(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
        kind: str,
        near_miss: bool,
    ) -> None:
        k_star = world.k_star
        pose_star = self._camera_at(poses, k_star)
        p_cam = glm_to_np(pose_star.position)
        d = np.linalg.norm(world.spline.positions - p_cam[None, :], axis=1)
        idx = int(np.argmin(d))
        t = world.spline.tangents[idx]
        n = world.spline.normals[idx]
        duration = max(k_star * self.cfg.dt, 1.6)
        offset = self._offset_vector(rng, near_miss, t, n)
        if kind == "pedestrian":
            speed_lo, speed_hi = self.cfg.envelopes.pedestrian
            mesh = generate_pedestrian_mesh(rng)
            ext = glm.vec3(0.50, 1.72, 0.40)
            y = 0.86 + world.curb_height
        else:
            speed_lo, speed_hi = self.cfg.envelopes.cyclist
            mesh = generate_cyclist_mesh()
            ext = glm.vec3(0.60, 1.70, 1.70)
            kind = "cyclist"
            y = 0.85
        speed = float(rng.uniform(speed_lo, min(speed_hi, speed_lo + 1.2)))
        # Oncoming along the corridor: they occupy the same s(t*) with a lateral offset
        # so they cannot clip the camera on earlier frames.
        p_star = np.array([p_cam[0], y, p_cam[2]], dtype=np.float64)
        p_target = p_star + offset
        p_target[1] = y
        p_spawn = p_target + t * (speed * duration)
        p_spawn[1] = y
        vel = -t * speed
        act = make_actor(
            kind,
            np_to_glm(p_spawn),
            mesh=mesh,
            velocity=np_to_glm(vel),
            rng=rng,
            extents=ext,
        )
        act.scripted = True
        world.actors.append(act)

    def _spawn_sidewalk_incursion(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        """Vehicle in the travel lane steers onto the sidewalk through P_cam(t*)."""
        k_star = world.k_star
        pose_star = self._camera_at(poses, k_star)
        p_cam = glm_to_np(pose_star.position)
        d = np.linalg.norm(world.spline.positions - p_cam[None, :], axis=1)
        idx = int(np.argmin(d))
        t = world.spline.tangents[idx]
        n = world.spline.normals[idx]
        duration = max(k_star * self.cfg.dt, 1.2)
        # Aim slightly in front of the forehead so TTC ~ 1.2 s at first visible frames, 0 at impact.
        p_target = np.array([p_cam[0], 0.75, p_cam[2]]) + t * 0.2
        # Spawn in the near lane, behind and to the road side of the camera.
        lane_lat = world.camera_side * (-0.30 * world.road_width)  # toward road from sidewalk
        # camera_side = -1 (right). Road is toward +N (left of travel? N is left).
        # Right sidewalk is -N. Road center is 0. A lane toward the camera from center is -N * small.
        p_spawn = p_target - t * (duration * 10.0) + n * (world.camera_side * (-0.28 * world.road_width) - world.sidewalk_lateral)
        # Simpler: spawn on the road at earlier arclength.
        s_cam = world.spline.arclength[idx]
        s_spawn = max(1.0, s_cam - duration * float(rng.uniform(8.0, 14.0)))
        p_s, t_s, n_s, _ = world.spline.frame_at(s_spawn)
        spawn = p_s + n_s * (world.camera_side * 0.22 * world.road_width)
        spawn[1] = 0.75
        vel = invert_constant_velocity(spawn, p_target, duration, *self.cfg.envelopes.vehicle)
        spd = float(np.linalg.norm(vel))
        if spd > 1e-6:
            spawn = p_target - vel * duration
            spawn[1] = 0.75
        act = make_actor(
            "vehicle",
            np_to_glm(spawn),
            mesh=generate_vehicle_mesh("sedan"),
            velocity=np_to_glm(vel),
            rng=rng,
        )
        act.scripted = True
        world.actors.append(act)

    def _spawn_jaywalker(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
        critical: bool,
    ) -> None:
        k_star = world.k_star
        pose_star = self._camera_at(poses, k_star)
        p_cam = glm_to_np(pose_star.position)
        d = np.linalg.norm(world.spline.positions - p_cam[None, :], axis=1)
        idx = int(np.argmin(d))
        t = world.spline.tangents[idx]
        n = world.spline.normals[idx]
        mag = float(rng.uniform(*(self.cfg.threat.critical_offset if critical else self.cfg.threat.near_miss_offset)))
        p_target = np.array([p_cam[0], 0.86 + world.curb_height, p_cam[2]]) + n * (mag if not critical else 0.05)
        p_target[1] = 0.86 + world.curb_height
        duration = max(k_star * self.cfg.dt, 1.4)
        # Emerge from behind a parked van on the road edge.
        van_s = world.spline.arclength[idx] - 3.5
        p_v, t_v, n_v, _ = world.spline.frame_at(max(1.0, van_s))
        van_pos = p_v + n_v * (world.camera_side * (0.5 * world.road_width - 1.1))
        van_pos[1] = 0.9
        van = make_actor(
            "vehicle",
            np_to_glm(van_pos),
            mesh=generate_vehicle_mesh("van"),
            velocity=glm.vec3(0.0),
            rng=rng,
            extents=glm.vec3(1.9, 1.8, 5.2),
        )
        van.face_velocity = False
        world.actors.append(van)
        spawn = van_pos + n_v * (world.camera_side * 1.6) - t_v * 0.4
        spawn[1] = p_target[1]
        vel = invert_constant_velocity(spawn, p_target, duration, *self.cfg.envelopes.pedestrian)
        spawn = p_target - vel * duration
        spawn[1] = p_target[1]
        ped = make_actor(
            "pedestrian",
            np_to_glm(spawn),
            mesh=generate_pedestrian_mesh(rng),
            velocity=np_to_glm(vel),
            rng=rng,
        )
        ped.scripted = True
        world.actors.append(ped)

    def _spawn_sudden_brake(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        """Lead pedestrian matches gait then brakes at -3 m/s^2 so TTC collapses < 1 s."""
        k_star = world.k_star
        pose0 = poses[0]
        p0 = glm_to_np(pose0.position)
        d = np.linalg.norm(world.spline.positions - p0[None, :], axis=1)
        idx = int(np.argmin(d))
        t = world.spline.tangents[idx]
        n = world.spline.normals[idx]
        lat = world.sidewalk_lateral
        p_s, t_s, n_s, _ = world.spline.frame_at(world.spline.arclength[idx] + 2.4)
        spawn = p_s + n_s * lat
        spawn[1] = 0.86 + world.curb_height
        cam_v = glm_to_np(pose0.velocity)
        cam_v[1] = 0.0
        spd = float(np.clip(np.linalg.norm(cam_v), 1.1, 1.5))
        s_spawn = arclength_near(world.spline, spawn)
        act = self._make_ped(world, rng, s_spawn, lat, spd)
        brake_frame = max(1, k_star - int(0.7 * self.cfg.fps))
        act._brake_frame = brake_frame  # type: ignore[attr-defined]
        act._brake_accel = -3.0  # type: ignore[attr-defined]
        act.scripted = True
        world.actors.append(act)

    def _spawn_head_overhang(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        pose_star = self._camera_at(poses, world.k_star)
        p_cam = glm_to_np(pose_star.position)
        d = np.linalg.norm(world.spline.positions - p_cam[None, :], axis=1)
        idx = int(np.argmin(d))
        p, t, n, _ = (
            world.spline.positions[idx],
            world.spline.tangents[idx],
            world.spline.normals[idx],
            world.spline.ups[idx],
        )
        # Place a bar at Y=1.62 m on the camera corridor.
        pos = p + n * world.sidewalk_lateral + t * 0.0
        pos = np.array([p_cam[0], 0.0, p_cam[2]])
        mesh = generate_head_hazard_mesh("pipe", glm.vec3(float(pos[0]), 0.0, float(pos[2])), rng)
        act = make_actor(
            "head_obstacle",
            glm.vec3(float(pos[0]), 1.62, float(pos[2])),
            mesh=mesh,
            rng=rng,
            extents=glm.vec3(1.8, 0.20, 0.20),
        )
        act.face_velocity = False
        act.scripted = True
        world.actors.append(act)

    def _spawn_pothole(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        pose_star = self._camera_at(poses, world.k_star)
        p_cam = glm_to_np(pose_star.position)
        radius = 0.40
        depth = float(rng.uniform(0.08, 0.25))
        mesh = generate_pothole_mesh(
            (0.0, 0.0, 0.0),
            radius=radius,
            depth=depth,
            seed=int(rng.integers(0, 10_000)),
            surface_y=world.curb_height,
        )
        act = make_actor(
            "pothole",
            glm.vec3(float(p_cam[0]), 0.0, float(p_cam[2])),
            mesh=mesh,
            rng=rng,
            extents=glm.vec3(2.0 * radius, depth, 2.0 * radius),
        )
        act.face_velocity = False
        act.scripted = True
        world.actors.append(act)

    def _spawn_overtake(
        self,
        world: EpisodeWorld,
        poses: Sequence[CameraPose],
        rng: np.random.Generator,
    ) -> None:
        pose0 = poses[0]
        p0 = glm_to_np(pose0.position)
        d = np.linalg.norm(world.spline.positions - p0[None, :], axis=1)
        idx = int(np.argmin(d))
        t = world.spline.tangents[idx]
        n = world.spline.normals[idx]
        lat = world.sidewalk_lateral + (-world.camera_side) * float(rng.uniform(0.45, 0.85))
        s0 = arclength_near(world.spline, p0) - 2.4
        spd = float(rng.uniform(3.2, 5.5))
        y = 0.85
        p, t, n, _ = world.spline.frame_at(max(0.8, s0))
        pos = p + n * lat
        act = make_actor(
            "cyclist",
            glm.vec3(float(pos[0]), y, float(pos[2])),
            mesh=generate_cyclist_mesh(rng),
            rng=rng,
            extents=glm.vec3(0.60, 1.70, 1.70),
        )
        bind_spline_motion(act, max(0.8, s0), lat, spd, y)
        sync_path_pose(act, world.spline)
        act.scripted = True
        world.actors.append(act)


def kinematics_vs_camera(actor: Actor, cam_pos: glm.vec3, cam_vel: glm.vec3, thr: ThreatThresholds) -> Dict:
    p_i = glm_to_np(actor.position)
    v_i = glm_to_np(actor.velocity)
    p_c = glm_to_np(cam_pos)
    v_c = glm_to_np(cam_vel)
    p_rel = p_i - p_c
    v_rel = v_i - v_c
    # Locomotion threats are decided in the walking plane; head-level obstacles
    # keep the full 3-D hit vector so a branch at Y=1.62 m still collapses TTC.
    if actor.class_name in ("head_obstacle", "sign", "awning"):
        ttc, d_cpa, conv = ttc_cpa(p_rel, v_rel)
    else:
        p_h = np.array([p_rel[0], 0.0, p_rel[2]])
        v_h = np.array([v_rel[0], 0.0, v_rel[2]])
        ttc, d_cpa, conv = ttc_cpa(p_h, v_h)
    speed = float(np.linalg.norm(v_i))
    label = classify_threat(speed, ttc, d_cpa, conv, thr)
    dist = float(np.linalg.norm(p_rel))
    return {
        "world_position": p_i.tolist(),
        "world_velocity": v_i.tolist(),
        "relative_position": p_rel.tolist(),
        "relative_velocity": v_rel.tolist(),
        "euclidean_distance": dist,
        "time_to_collision_sec": None if ttc >= INF * 0.5 else float(ttc),
        "closest_point_of_approach_dist": float(d_cpa),
        "converging": conv,
        "threat_classification": label,
    }


def episode_realized_class(labels: Sequence[str]) -> str:
    if "CRITICAL_THREAT" in labels:
        return "CRITICAL_THREAT"
    if "NEAR_MISS" in labels:
        return "NEAR_MISS"
    if any(x.startswith("SAFE") for x in labels) or not labels:
        return "SAFE"
    return "SAFE"
