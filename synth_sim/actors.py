"""Kinematic actors: vehicles, pedestrians, cyclists, and static hazards."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import glm
import numpy as np

from geometry import Mesh, generate_cuboid_mesh
from config import SimulationConfig


def quat_to_xyzw(q: glm.quat) -> List[float]:
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def yaw_to_quat(yaw: float) -> glm.quat:
    return glm.quat(glm.vec3(0.0, yaw, 0.0))


def vec3_to_list(v: glm.vec3) -> List[float]:
    return [float(v.x), float(v.y), float(v.z)]


def np_to_glm(v: np.ndarray | Sequence[float]) -> glm.vec3:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return glm.vec3(float(a[0]), float(a[1]), float(a[2]))


@dataclass
class Actor:
    instance_id: int
    class_name: str
    extents: glm.vec3
    position: glm.vec3
    velocity: glm.vec3 = field(default_factory=lambda: glm.vec3(0.0))
    acceleration: glm.vec3 = field(default_factory=lambda: glm.vec3(0.0))
    orientation: glm.quat = field(default_factory=lambda: glm.quat(1.0, 0.0, 0.0, 0.0))
    mesh: Optional[Mesh] = None
    albedo: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    specularity: float = 0.12
    shininess: float = 24.0
    style: int = 0
    face_velocity: bool = True
    max_speed: float = 20.0
    alive: bool = True
    semantic_static: bool = False
    scripted: bool = False

    def model_matrix(self) -> glm.mat4:
        return glm.translate(glm.mat4(1.0), self.position) * glm.mat4_cast(self.orientation)

    def step(self, dt: float) -> None:
        self.velocity = self.velocity + self.acceleration * dt
        sp = glm.length(self.velocity)
        if sp > self.max_speed and sp > 1e-8:
            self.velocity = self.velocity * (self.max_speed / sp)
        self.position = self.position + self.velocity * dt
        if self.face_velocity:
            hx = float(self.velocity.x)
            hz = float(self.velocity.z)
            if hx * hx + hz * hz > 0.04:
                yaw = math.atan2(hx, hz)
                self.orientation = yaw_to_quat(yaw)

    def apply_social_force(
        self,
        others: Sequence["Actor"],
        desired: glm.vec3,
        dt: float,
        tau: float = 0.45,
        a_rep: float = 1.8,
        b_rep: float = 0.45,
    ) -> None:
        """Helbing-style relaxation toward a desired velocity plus exponential repulsion."""
        force = (desired - self.velocity) / max(tau, 1e-3)
        for o in others:
            if o is self:
                continue
            d = self.position - o.position
            dist = glm.length(d)
            radii = 0.5 * (float(self.extents.x + self.extents.z) * 0.5 + float(o.extents.x + o.extents.z) * 0.5)
            if dist < 1e-4:
                n = glm.vec3(1.0, 0.0, 0.0)
                dist = 1e-4
            else:
                n = d / dist
            force = force + n * (a_rep * math.exp((radii - dist) / b_rep))
            force.y = 0.0
        self.acceleration = force
        self.acceleration.y = 0.0


@dataclass
class EnvironmentState:
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_kelvin: float
    ambient: float
    wetness: float
    fog_density: float
    road_condition: str

    def sun_direction(self) -> glm.vec3:
        el = math.radians(self.sun_elevation_deg)
        az = math.radians(self.sun_azimuth_deg)
        x = math.cos(el) * math.sin(az)
        y = math.sin(el)
        z = math.cos(el) * math.cos(az)
        return glm.normalize(glm.vec3(x, y, z))

    def sun_color(self) -> glm.vec3:
        return np_to_glm(kelvin_to_rgb(self.sun_kelvin))

    def sky_color(self) -> glm.vec3:
        el = np.clip(self.sun_elevation_deg / 90.0, 0.0, 1.0)
        horizon = np.array([0.78, 0.62, 0.48])
        zenith = np.array([0.42, 0.62, 0.88])
        c = (1.0 - el) * horizon + el * zenith
        c = c * (0.35 + 0.65 * self.ambient / 0.6)
        return np_to_glm(c)


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """Tanner Helland approximation, returned in linear-ish 0..1 RGB."""
    t = float(np.clip(kelvin, 1000.0, 40000.0)) / 100.0
    if t <= 66.0:
        r = 255.0
        g = np.clip(99.4708025861 * math.log(t) - 161.1195681661, 0.0, 255.0)
    else:
        r = np.clip(329.698727446 * ((t - 60.0) ** -0.1332047592), 0.0, 255.0)
        g = np.clip(288.1221695283 * ((t - 60.0) ** -0.0755148492), 0.0, 255.0)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = np.clip(138.5177312231 * math.log(t - 10.0) - 305.0447927307, 0.0, 255.0)
    rgb = np.array([r, g, b], dtype=np.float64) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def sample_environment(cfg: SimulationConfig, rng: np.random.Generator) -> EnvironmentState:
    d = cfg.domain
    wet = float(rng.uniform(*d.wetness))
    cond = "wet_specular" if wet > 0.45 else "dry"
    return EnvironmentState(
        sun_azimuth_deg=float(rng.uniform(*d.sun_azimuth_deg)),
        sun_elevation_deg=float(rng.uniform(*d.sun_elevation_deg)),
        sun_kelvin=float(rng.uniform(*d.sun_kelvin)),
        ambient=float(rng.uniform(*d.ambient_sky)),
        wetness=wet,
        fog_density=float(rng.uniform(*d.fog_density)),
        road_condition=cond,
    )


def default_extents(class_name: str) -> glm.vec3:
    table = {
        "vehicle": glm.vec3(2.0, 1.5, 4.5),
        "pedestrian": glm.vec3(0.50, 1.72, 0.40),
        "cyclist": glm.vec3(0.60, 1.70, 1.70),
        "scooter": glm.vec3(0.40, 1.20, 1.10),
        "dustbin": glm.vec3(0.48, 1.05, 0.48),
        "bollard": glm.vec3(0.20, 0.95, 0.20),
        "pothole": glm.vec3(0.90, 0.25, 0.90),
        "head_obstacle": glm.vec3(1.80, 0.35, 0.35),
        "sign": glm.vec3(1.20, 0.80, 0.20),
        "awning": glm.vec3(1.80, 0.12, 3.20),
        "tree": glm.vec3(1.60, 3.20, 1.60),
        "building": glm.vec3(8.0, 12.0, 10.0),
    }
    return table.get(class_name, glm.vec3(1.0, 1.0, 1.0))


PALETTES = {
    "vehicle": [(0.15, 0.18, 0.22), (0.62, 0.12, 0.12), (0.12, 0.28, 0.55), (0.85, 0.85, 0.82), (0.08, 0.08, 0.09)],
    "pedestrian": [(0.18, 0.22, 0.45), (0.45, 0.18, 0.16), (0.20, 0.42, 0.28), (0.55, 0.45, 0.22), (0.12, 0.12, 0.12)],
    "cyclist": [(0.10, 0.10, 0.12), (0.70, 0.15, 0.10), (0.15, 0.45, 0.25)],
    "scooter": [(0.12, 0.12, 0.14), (0.85, 0.55, 0.10), (0.20, 0.55, 0.85)],
    "dustbin": [(0.18, 0.42, 0.28), (0.25, 0.25, 0.26), (0.55, 0.45, 0.15)],
    "bollard": [(0.75, 0.55, 0.12), (0.85, 0.85, 0.80)],
    "pothole": [(0.16, 0.14, 0.12)],
    "head_obstacle": [(0.28, 0.18, 0.10), (0.22, 0.38, 0.16)],
    "sign": [(0.75, 0.15, 0.12), (0.15, 0.35, 0.75)],
    "awning": [(0.45, 0.12, 0.12), (0.15, 0.18, 0.22), (0.55, 0.45, 0.22)],
    "tree": [(0.18, 0.38, 0.14), (0.12, 0.32, 0.10), (0.22, 0.42, 0.16)],
    "building": [(0.55, 0.42, 0.34), (0.62, 0.55, 0.45), (0.48, 0.50, 0.52), (0.72, 0.70, 0.62), (0.38, 0.36, 0.34)],
}


def pick_albedo(class_name: str, rng: np.random.Generator) -> Tuple[float, float, float]:
    opts = PALETTES.get(class_name, [(0.5, 0.5, 0.5)])
    c = opts[int(rng.integers(0, len(opts)))]
    jitter = rng.uniform(-0.04, 0.04, size=3)
    rgb = np.clip(np.array(c) + jitter, 0.02, 0.95)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


_NEXT_ID = 100


def next_instance_id() -> int:
    global _NEXT_ID
    i = _NEXT_ID
    _NEXT_ID += 1
    return i


def reset_instance_ids(start: int = 100) -> None:
    global _NEXT_ID
    _NEXT_ID = start


def make_actor(
    class_name: str,
    position: glm.vec3,
    mesh: Optional[Mesh] = None,
    velocity: Optional[glm.vec3] = None,
    rng: Optional[np.random.Generator] = None,
    yaw: float = 0.0,
    extents: Optional[glm.vec3] = None,
) -> Actor:
    rng = rng or np.random.default_rng()
    ext = extents if extents is not None else default_extents(class_name)
    act = Actor(
        instance_id=next_instance_id(),
        class_name=class_name,
        extents=ext,
        position=position,
        velocity=velocity if velocity is not None else glm.vec3(0.0),
        orientation=yaw_to_quat(yaw),
        mesh=mesh if mesh is not None else generate_cuboid_mesh(ext),
        albedo=pick_albedo(class_name, rng),
        face_velocity=class_name in ("vehicle", "pedestrian", "cyclist", "scooter"),
        semantic_static=class_name not in ("vehicle", "pedestrian", "cyclist", "scooter"),
        max_speed={
            "pedestrian": 3.2,
            "vehicle": 22.0,
            "cyclist": 10.0,
            "scooter": 10.0,
        }.get(class_name, 20.0),
    )
    if class_name in ("vehicle",):
        act.specularity = 0.45
        act.shininess = 48.0
        act.style = 5
    if class_name == "pedestrian":
        act.style = 6
    if class_name in ("cyclist", "scooter"):
        act.style = 5
    if class_name in ("tree", "head_obstacle"):
        act.style = 4
    if class_name in ("bollard", "sign"):
        act.style = 9
    if class_name == "dustbin":
        act.style = 9
    if class_name == "pothole":
        act.face_velocity = False
        act.specularity = 0.05
    return act
