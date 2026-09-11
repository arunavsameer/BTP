"""6-DOF biomechanical forehead camera rig.

Implements the inverted-pendulum gait of Section 3.1, with Ken Perlin
gradient noise (pure NumPy) driving exploratory head tremor as fBM.

World frame: +X right, +Y up, +Z forward.
The scalar "progress" coordinate is arclength along the sidewalk corridor,
which reduces to world Z on a straight street.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import glm
import numpy as np

from config import GaitConfig, SimulationConfig


def _fade(t: np.ndarray | float) -> np.ndarray | float:
    """Perlin improved fade: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class Perlin1D:
    """Deterministic 1-D gradient noise with a 256-entry permutation table."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        perm = rng.permutation(256).astype(np.int32)
        self._perm = np.concatenate([perm, perm])

    def noise(self, x: float) -> float:
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = float(_fade(xf))
        a = int(self._perm[xi])
        b = int(self._perm[xi + 1])
        ga = 1.0 if (a & 1) == 0 else -1.0
        gb = 1.0 if (b & 1) == 0 else -1.0
        va = ga * xf
        vb = gb * (xf - 1.0)
        return (1.0 - u) * va + u * vb

    def fbm(
        self,
        x: float,
        octaves: int = 5,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        frequency: float = 1.0,
    ) -> float:
        """Fractional Brownian motion: sum_i persistence^i * noise(freq * lac^i * x)."""
        amp = 1.0
        freq = frequency
        total = 0.0
        norm = 0.0
        for _ in range(octaves):
            total += amp * self.noise(x * freq)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / max(norm, 1e-8)


def view_matrix_z_forward(eye: glm.vec3, forward: glm.vec3, up: glm.vec3) -> glm.mat4:
    """World +Z forward, +X right, +Y up → OpenGL view (camera looks down -Z).

    glm.lookAt(eye, eye+Z, up) flips X because its z-axis is eye-center.
    We build R explicitly so sidewalk +X stays on the right of the image.
    """
    f = glm.normalize(forward)
    r = glm.normalize(glm.cross(up, f))  # right = up × forward
    u = glm.cross(f, r)
    m = glm.mat4(1.0)
    # Columns of the 3x3 rotation are world-axes expressed in camera space.
    m[0][0], m[0][1], m[0][2] = r.x, u.x, -f.x
    m[1][0], m[1][1], m[1][2] = r.y, u.y, -f.y
    m[2][0], m[2][1], m[2][2] = r.z, u.z, -f.z
    m[3][0] = -glm.dot(r, eye)
    m[3][1] = -glm.dot(u, eye)
    m[3][2] = glm.dot(f, eye)
    m[3][3] = 1.0
    return m


@dataclass
class CameraPose:
    position: glm.vec3
    target: glm.vec3
    up: glm.vec3
    velocity: glm.vec3
    euler_deg: glm.vec3  # pitch, yaw, roll in degrees
    t: float
    view: glm.mat4


class SidewalkPath:
    """Samples a sidewalk centerline given world samples and a lateral offset."""

    def __init__(
        self,
        positions: np.ndarray,
        tangents: np.ndarray,
        normals: np.ndarray,
        ups: np.ndarray,
        arclength: np.ndarray,
        lateral_offset: float,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        self.tangents = np.asarray(tangents, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        self.ups = np.asarray(ups, dtype=np.float64)
        self.arclength = np.asarray(arclength, dtype=np.float64)
        self.lateral_offset = float(lateral_offset)
        self.length = float(self.arclength[-1])

    def sample(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = float(np.clip(s, self.arclength[0], self.arclength[-1]))
        idx = int(np.searchsorted(self.arclength, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.arclength) - 2))
        s0 = self.arclength[idx]
        s1 = self.arclength[idx + 1]
        u = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        p = (1.0 - u) * self.positions[idx] + u * self.positions[idx + 1]
        t = (1.0 - u) * self.tangents[idx] + u * self.tangents[idx + 1]
        n = (1.0 - u) * self.normals[idx] + u * self.normals[idx + 1]
        up = (1.0 - u) * self.ups[idx] + u * self.ups[idx + 1]
        tn = np.linalg.norm(t)
        nn = np.linalg.norm(n)
        un = np.linalg.norm(up)
        t = t / tn if tn > 1e-8 else np.array([0.0, 0.0, 1.0])
        n = n / nn if nn > 1e-8 else np.array([1.0, 0.0, 0.0])
        up = up / un if un > 1e-8 else np.array([0.0, 1.0, 0.0])
        p = p + n * self.lateral_offset
        return p, t, n, up


class BiomechanicalCameraRig:
    """Forehead-mounted camera with heel-strike heave, lateral sway, and fBM tremor."""

    def __init__(
        self,
        cfg: SimulationConfig,
        path: SidewalkPath,
        seed: int,
        s0: float = 8.0,
        v0: Optional[float] = None,
        h_nominal: Optional[float] = None,
        phi_sway: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.gait: GaitConfig = cfg.gait
        self.path = path
        rng = np.random.default_rng(seed)
        self.v0 = float(v0 if v0 is not None else rng.uniform(self.gait.v0_min, self.gait.v0_max))
        self.h_nominal = float(
            h_nominal
            if h_nominal is not None
            else rng.normal(self.gait.h_nominal_mean, self.gait.h_nominal_std)
        )
        self.phi_sway = float(phi_sway if phi_sway is not None else rng.uniform(0.0, 2.0 * math.pi))
        self.s0 = float(s0)
        self.t = 0.0
        self._perlin_x = Perlin1D(seed + 17)
        self._perlin_y = Perlin1D(seed + 31)
        self._perlin_p = Perlin1D(seed + 47)
        self._perlin_yw = Perlin1D(seed + 61)
        self._perlin_r = Perlin1D(seed + 89)
        self._prev_pos: Optional[glm.vec3] = None
        self._velocity = glm.vec3(0.0, 0.0, self.v0)

    def arclength_at(self, t: float) -> float:
        """Closed-form integral of v0 (1 + alpha_v cos(4 pi f_step tau))."""
        g = self.gait
        omega = 4.0 * math.pi * g.f_step
        return self.s0 + self.v0 * (t + g.alpha_v * math.sin(omega * t) / omega)

    def pose_at(self, t: float, prev_position: Optional[glm.vec3] = None) -> CameraPose:
        g = self.gait
        s = self.arclength_at(t)
        base, tangent, normal, up_path = self.path.sample(s)

        eta_x = g.noise_sway * self._perlin_x.fbm(t * 3.4, octaves=4, frequency=1.1)
        eta_y = g.noise_heave * self._perlin_y.fbm(t * 5.1, octaves=4, frequency=1.3)
        sway = g.a_sway * math.sin(2.0 * math.pi * g.f_stride * t + self.phi_sway) + eta_x
        heave = g.a_heave * abs(math.sin(2.0 * math.pi * g.f_step * t)) + eta_y

        pos_np = base + normal * sway + up_path * (self.h_nominal + heave - up_path[1] * 0.0)
        # base already sits on the sidewalk surface; add head height along world up.
        pos_np = np.array(
            [base[0] + normal[0] * sway, self.h_nominal + heave, base[2] + normal[2] * sway],
            dtype=np.float64,
        )
        # Preserve path elevation if the spline ever leaves y=0.
        pos_np[1] += base[1]

        n_pitch = self._perlin_p.fbm(t * 0.85, octaves=5, frequency=0.7)
        n_yaw = self._perlin_yw.fbm(t * 0.55, octaves=5, frequency=0.45)
        n_roll = self._perlin_r.fbm(t * 0.95, octaves=4, frequency=0.8)

        pitch = math.radians(g.a_pitch_deg) * math.sin(4.0 * math.pi * g.f_step * t) + math.radians(
            g.beta_pitch
        ) * n_pitch
        yaw = math.radians(g.a_yaw_deg) * math.sin(2.0 * math.pi * g.f_stride * t) + math.radians(
            g.beta_yaw
        ) * n_yaw
        roll = math.radians(g.a_roll_deg) * math.cos(2.0 * math.pi * g.f_stride * t) + math.radians(
            g.beta_roll
        ) * n_roll

        pitch = float(np.clip(pitch, -math.radians(g.pitch_explore_deg), math.radians(g.pitch_explore_deg)))
        yaw = float(np.clip(yaw, -math.radians(g.yaw_explore_deg), math.radians(g.yaw_explore_deg)))
        roll = float(np.clip(roll, -math.radians(g.roll_explore_deg), math.radians(g.roll_explore_deg)))

        # Heading follows the path tangent in the XZ plane, then cervical Euler offsets.
        heading = math.atan2(tangent[0], tangent[2])
        total_yaw = heading + yaw

        cy, sy = math.cos(total_yaw), math.sin(total_yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # Intrinsic yaw (Y) → pitch (X) → roll (Z). Local +Z is optical axis.
        forward = np.array([sy * cp, -sp, cy * cp], dtype=np.float64)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / rn
        up_vec = np.cross(right, forward)
        up_vec = up_vec / max(np.linalg.norm(up_vec), 1e-8)
        # Apply roll about the optical axis.
        up_rolled = up_vec * cr + right * sr

        eye = glm.vec3(float(pos_np[0]), float(pos_np[1]), float(pos_np[2]))
        fwd = glm.vec3(float(forward[0]), float(forward[1]), float(forward[2]))
        upg = glm.vec3(float(up_rolled[0]), float(up_rolled[1]), float(up_rolled[2]))
        target = eye + fwd
        view = view_matrix_z_forward(eye, fwd, upg)

        if prev_position is None:
            vel = glm.vec3(tangent[0] * self.v0, 0.0, tangent[2] * self.v0)
        else:
            dt = self.cfg.dt
            vel = (eye - prev_position) / max(dt, 1e-8)

        return CameraPose(
            position=eye,
            target=target,
            up=upg,
            velocity=vel,
            euler_deg=glm.vec3(math.degrees(pitch), math.degrees(yaw), math.degrees(roll)),
            t=t,
            view=view,
        )

    def step(self, dt: float) -> Tuple[glm.vec3, glm.vec3]:
        self.t += float(dt)
        pose = self.pose_at(self.t, self._prev_pos)
        self._prev_pos = pose.position
        self._velocity = pose.velocity
        return pose.position, pose.target

    def view_matrix(self) -> glm.mat4:
        pose = self.pose_at(self.t, self._prev_pos)
        return pose.view

    def precompute(self, n_frames: int, dt: float) -> Sequence[CameraPose]:
        poses = []
        prev = None
        for i in range(n_frames):
            t = i * dt
            pose = self.pose_at(t, prev)
            poses.append(pose)
            prev = pose.position
        return poses
