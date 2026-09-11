"""Procedural mesh builders: Catmull-Rom streets, Poisson scattering, hazards.

Vertex layout is interleaved float32:
    position.xyz, normal.xyz, uv.st   (8 floats, tightly packed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import glm
import numpy as np

from config import SimulationConfig, StreetConfig


Vec3 = np.ndarray


def _as3(v: Sequence[float] | glm.vec3 | np.ndarray) -> np.ndarray:
    if isinstance(v, glm.vec3):
        return np.array([v.x, v.y, v.z], dtype=np.float64)
    return np.asarray(v, dtype=np.float64).reshape(3)


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 8) float32
    indices: np.ndarray  # (M,) uint32
    name: str = ""

    @property
    def vertex_bytes(self) -> bytes:
        return np.ascontiguousarray(self.vertices, dtype=np.float32).tobytes()

    @property
    def index_bytes(self) -> bytes:
        return np.ascontiguousarray(self.indices, dtype=np.uint32).tobytes()


class MeshBuilder:
    def __init__(self) -> None:
        self._p: List[List[float]] = []
        self._n: List[List[float]] = []
        self._uv: List[List[float]] = []
        self._i: List[int] = []

    def add_vertex(self, p: Sequence[float], n: Sequence[float], uv: Sequence[float]) -> int:
        idx = len(self._p)
        self._p.append([float(p[0]), float(p[1]), float(p[2])])
        self._n.append([float(n[0]), float(n[1]), float(n[2])])
        self._uv.append([float(uv[0]), float(uv[1])])
        return idx

    def add_tri(self, i0: int, i1: int, i2: int) -> None:
        self._i.extend([i0, i1, i2])

    def add_face(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
        p3: Optional[Sequence[float]] = None,
        n: Optional[Sequence[float]] = None,
        uv0: Sequence[float] = (0.0, 0.0),
        uv1: Sequence[float] = (1.0, 0.0),
        uv2: Sequence[float] = (1.0, 1.0),
        uv3: Sequence[float] = (0.0, 1.0),
    ) -> None:
        a = _as3(p0)
        b = _as3(p1)
        c = _as3(p2)
        if n is None:
            nrm = _norm(np.cross(b - a, c - a))
        else:
            nrm = _as3(n)
            nrm = _norm(nrm) if np.linalg.norm(nrm) > 1e-12 else _norm(np.cross(b - a, c - a))
        if p3 is None:
            i0 = self.add_vertex(a, nrm, uv0)
            i1 = self.add_vertex(b, nrm, uv1)
            i2 = self.add_vertex(c, nrm, uv2)
            self.add_tri(i0, i1, i2)
            return
        d = _as3(p3)
        i0 = self.add_vertex(a, nrm, uv0)
        i1 = self.add_vertex(b, nrm, uv1)
        i2 = self.add_vertex(c, nrm, uv2)
        i3 = self.add_vertex(d, nrm, uv3)
        self.add_tri(i0, i1, i2)
        self.add_tri(i0, i2, i3)

    def build(self, name: str = "") -> Mesh:
        if not self._p:
            v = np.zeros((0, 8), dtype=np.float32)
            i = np.zeros((0,), dtype=np.uint32)
            return Mesh(v, i, name)
        pos = np.asarray(self._p, dtype=np.float32)
        nrm = np.asarray(self._n, dtype=np.float32)
        uv = np.asarray(self._uv, dtype=np.float32)
        verts = np.concatenate([pos, nrm, uv], axis=1)
        idx = np.asarray(self._i, dtype=np.uint32)
        return Mesh(verts, idx, name)


# ---------------------------------------------------------------------------
# Cubic Hermite / Catmull-Rom spline
# ---------------------------------------------------------------------------

def catmull_rom_point(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float, tau: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a Catmull-Rom segment and its derivative at u in [0, 1].

    Hermite basis on endpoints P1, P2 with tangents
        T1 = tau * (P2 - P0),  T2 = tau * (P3 - P1).
    """
    u2 = u * u
    u3 = u2 * u
    t1 = tau * (p2 - p0)
    t2 = tau * (p3 - p1)
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    c = h00 * p1 + h10 * t1 + h01 * p2 + h11 * t2
    d00 = 6.0 * u2 - 6.0 * u
    d10 = 3.0 * u2 - 4.0 * u + 1.0
    d01 = -6.0 * u2 + 6.0 * u
    d11 = 3.0 * u2 - 2.0 * u
    cp = d00 * p1 + d10 * t1 + d01 * p2 + d11 * t2
    return c, cp


@dataclass
class StreetSpline:
    """Arclength-parameterized street centerline with a parallel-transport frame."""

    control_points: np.ndarray
    positions: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    ups: np.ndarray
    arclength: np.ndarray

    @property
    def length(self) -> float:
        return float(self.arclength[-1])

    def frame_at(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = float(np.clip(s, self.arclength[0], self.arclength[-1]))
        idx = int(np.searchsorted(self.arclength, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.arclength) - 2))
        s0, s1 = self.arclength[idx], self.arclength[idx + 1]
        u = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        p = (1.0 - u) * self.positions[idx] + u * self.positions[idx + 1]
        t = _norm((1.0 - u) * self.tangents[idx] + u * self.tangents[idx + 1])
        n = _norm((1.0 - u) * self.normals[idx] + u * self.normals[idx + 1])
        up = _norm((1.0 - u) * self.ups[idx] + u * self.ups[idx + 1])
        if np.linalg.norm(t) < 1e-8:
            t = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(n) < 1e-8:
            n = np.array([-1.0, 0.0, 0.0])
        if np.linalg.norm(up) < 1e-8:
            up = np.array([0.0, 1.0, 0.0])
        return p, t, n, up


def build_street_spline(
    control_points: Sequence[Sequence[float]],
    samples_per_segment: int = 24,
    tau: float = 0.5,
) -> StreetSpline:
    pts = [ _as3(p) for p in control_points ]
    if len(pts) < 2:
        raise ValueError("Need at least two control points")
    if len(pts) == 2:
        pts = [pts[0], pts[0], pts[1], pts[1]]
    padded = [pts[0]] + pts + [pts[-1]]
    positions: List[np.ndarray] = []
    tangents: List[np.ndarray] = []
    n_seg = len(padded) - 3
    for si in range(n_seg):
        p0, p1, p2, p3 = padded[si], padded[si + 1], padded[si + 2], padded[si + 3]
        n_samp = samples_per_segment if si < n_seg - 1 else samples_per_segment + 1
        for k in range(n_samp):
            u = k / float(samples_per_segment)
            c, cp = catmull_rom_point(p0, p1, p2, p3, u, tau)
            positions.append(c)
            tangents.append(_norm(cp) if np.linalg.norm(cp) > 1e-10 else np.array([0.0, 0.0, 1.0]))

    pos = np.stack(positions, axis=0)
    tan = np.stack(tangents, axis=0)
    diffs = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(diffs)])

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normals = np.zeros_like(pos)
    ups = np.zeros_like(pos)
    for i in range(len(pos)):
        t = tan[i]
        n = np.cross(t, world_up)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            n = np.array([-1.0, 0.0, 0.0])
        else:
            n = n / nn
        up = _norm(np.cross(n, t))
        normals[i] = n
        ups[i] = up
    return StreetSpline(np.stack(pts), pos, tan, normals, ups, s)


def generate_control_points(cfg: StreetConfig, rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(cfg.n_control_min, cfg.n_control_max + 1))
    zs = np.linspace(0.0, cfg.street_length, n)
    xs = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        xs[i] = xs[i - 1] + rng.uniform(-cfg.lateral_wander, cfg.lateral_wander) * 0.35
        xs[i] = float(np.clip(xs[i], -cfg.lateral_wander, cfg.lateral_wander))
    pts = np.stack([xs, np.zeros(n), zs], axis=1)
    return pts


def generate_street_mesh(
    spline_control_points: Sequence[Sequence[float]],
    width: float,
    length: float,
    resolution: float,
    curb_width: float = 0.25,
    curb_height: float = 0.15,
    sidewalk_width: float = 3.0,
) -> Dict[str, Mesh]:
    """Extrude asphalt, curb steps, and sidewalks along a Catmull-Rom centerline.

    Cross-section (left → right), looking along +T:
        left sidewalk outer, left sidewalk inner, left curb, road, right curb,
        right sidewalk inner, right sidewalk outer.
    Frenet normal N = T × world_up points to the left of travel.
    """
    samples = max(8, int(math.ceil(length / max(resolution, 0.1))))
    spline = build_street_spline(spline_control_points, samples_per_segment=max(8, samples // max(len(spline_control_points), 1)))
    n_sl = len(spline.positions)
    half_road = 0.5 * width

    def xsec_x(side_sign: float, dist: float) -> float:
        return side_sign * dist

    road_b = MeshBuilder()
    curb_b = MeshBuilder()
    walk_b = MeshBuilder()

    def strip(builder: MeshBuilder, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, uv_s0: float, uv_s1: float, uv_l0: float, uv_l1: float) -> None:
        builder.add_face(a, b, c, d, uv0=(uv_l0, uv_s0), uv1=(uv_l1, uv_s0), uv2=(uv_l1, uv_s1), uv3=(uv_l0, uv_s1))

    for i in range(n_sl - 1):
        p0, t0, n0, u0 = spline.positions[i], spline.tangents[i], spline.normals[i], spline.ups[i]
        p1, t1, n1, u1 = spline.positions[i + 1], spline.tangents[i + 1], spline.normals[i + 1], spline.ups[i + 1]
        s0 = spline.arclength[i]
        s1 = spline.arclength[i + 1]
        uv_s0 = s0 * 0.35
        uv_s1 = s1 * 0.35

        def P(p: np.ndarray, n: np.ndarray, u: np.ndarray, lat: float, y: float) -> np.ndarray:
            return p + n * lat + np.array([0.0, y, 0.0]) + u * 0.0

        # Cambered road: crown in the middle, gutters at the curb.
        crown = 0.042
        r0c = P(p0, n0, u0, 0.0, crown)
        r0l = P(p0, n0, u0, +half_road, 0.0)
        r0r = P(p0, n0, u0, -half_road, 0.0)
        r1c = P(p1, n1, u1, 0.0, crown)
        r1l = P(p1, n1, u1, +half_road, 0.0)
        r1r = P(p1, n1, u1, -half_road, 0.0)
        strip(road_b, r0l, r0c, r1c, r1l, uv_s0, uv_s1, 0.0, 0.5)
        strip(road_b, r0c, r0r, r1r, r1c, uv_s0, uv_s1, 0.5, 1.0)

        h = curb_height
        cw = curb_width
        sw = sidewalk_width
        bevel = min(0.06, 0.45 * cw)

        for sign, builder_curb, builder_walk in (
            (+1.0, curb_b, walk_b),
            (-1.0, curb_b, walk_b),
        ):
            # Slanted curb face (road gutter → tread) instead of a 90° wall.
            c_in0 = P(p0, n0, u0, sign * half_road, 0.0)
            c_in1 = P(p1, n1, u1, sign * half_road, 0.0)
            c_in0t = P(p0, n0, u0, sign * (half_road + bevel), h)
            c_in1t = P(p1, n1, u1, sign * (half_road + bevel), h)
            c_out0t = P(p0, n0, u0, sign * (half_road + cw), h)
            c_out1t = P(p1, n1, u1, sign * (half_road + cw), h)
            if sign > 0:
                strip(builder_curb, c_in0, c_in1, c_in1t, c_in0t, uv_s0, uv_s1, 0.0, h)
                strip(builder_curb, c_in0t, c_in1t, c_out1t, c_out0t, uv_s0, uv_s1, 0.0, cw)
            else:
                strip(builder_curb, c_in1, c_in0, c_in0t, c_in1t, uv_s0, uv_s1, 0.0, h)
                strip(builder_curb, c_out0t, c_out1t, c_in1t, c_in0t, uv_s0, uv_s1, 0.0, cw)

            w_in0 = c_out0t
            w_in1 = c_out1t
            w_out0 = P(p0, n0, u0, sign * (half_road + cw + sw), h)
            w_out1 = P(p1, n1, u1, sign * (half_road + cw + sw), h)
            if sign > 0:
                strip(builder_walk, w_in0, w_in1, w_out1, w_out0, uv_s0, uv_s1, 0.0, sw)
            else:
                strip(builder_walk, w_out0, w_out1, w_in1, w_in0, uv_s0, uv_s1, 0.0, sw)

            g_out0 = P(p0, n0, u0, sign * (half_road + cw + sw), 0.0)
            g_out1 = P(p1, n1, u1, sign * (half_road + cw + sw), 0.0)
            if sign > 0:
                strip(builder_walk, w_out0, w_out1, g_out1, g_out0, uv_s0, uv_s1, h, 0.0)
            else:
                strip(builder_walk, g_out0, g_out1, w_out1, w_out0, uv_s0, uv_s1, 0.0, h)

    verge_b = MeshBuilder()
    verge_w = 1.85
    for i in range(n_sl - 1):
        p0, n0 = spline.positions[i], spline.normals[i]
        p1, n1 = spline.positions[i + 1], spline.normals[i + 1]
        s0, s1 = spline.arclength[i], spline.arclength[i + 1]
        for sign in (+1.0, -1.0):
            lat0 = sign * (half_road + curb_width + sidewalk_width)
            lat1 = sign * (half_road + curb_width + sidewalk_width + verge_w)
            a = p0 + n0 * lat0 + np.array([0.0, 0.018, 0.0])
            bb = p0 + n0 * lat1 + np.array([0.0, 0.012, 0.0])
            c = p1 + n1 * lat1 + np.array([0.0, 0.012, 0.0])
            d = p1 + n1 * lat0 + np.array([0.0, 0.018, 0.0])
            if sign > 0:
                verge_b.add_face(a, d, c, bb, n=(0, 1, 0), uv0=(0, s0), uv1=(verge_w, s0), uv2=(verge_w, s1), uv3=(0, s1))
            else:
                verge_b.add_face(bb, c, d, a, n=(0, 1, 0), uv0=(0, s0), uv1=(verge_w, s0), uv2=(verge_w, s1), uv3=(0, s1))

    return {
        "road": road_b.build("road"),
        "curb": curb_b.build("curb"),
        "sidewalk": walk_b.build("sidewalk"),
        "verge": verge_b.build("verge"),
        "spline": spline,  # type: ignore[dict-item]
    }


def generate_ground_plane(extent: float = 120.0, y: float = -0.02, segments: int = 18) -> Mesh:
    """Subdivided dirt/grass plane with a gentle undulation so the horizon is not a slab."""
    b = MeshBuilder()
    e = extent
    xs = np.linspace(-e, e, segments + 1)
    zs = np.linspace(-e, e, segments + 1)

    def height(x: float, z: float) -> float:
        return y + 0.055 * math.sin(x * 0.055) * math.sin(z * 0.07) + 0.02 * math.sin(x * 0.19 + z * 0.13)

    for i in range(segments):
        for j in range(segments):
            x0, x1 = float(xs[i]), float(xs[i + 1])
            z0, z1 = float(zs[j]), float(zs[j + 1])
            p00 = (x0, height(x0, z0), z0)
            p10 = (x1, height(x1, z0), z0)
            p11 = (x1, height(x1, z1), z1)
            p01 = (x0, height(x0, z1), z1)
            b.add_face(
                p00, p10, p11, p01,
                uv0=(x0 * 0.25, z0 * 0.25),
                uv1=(x1 * 0.25, z0 * 0.25),
                uv2=(x1 * 0.25, z1 * 0.25),
                uv3=(x0 * 0.25, z1 * 0.25),
            )
    return b.build("ground")


def generate_cuboid_mesh(dimensions: glm.vec3 | Sequence[float], centered: bool = True) -> Mesh:
    d = _as3(dimensions)
    hx, hy, hz = 0.5 * d[0], 0.5 * d[1], 0.5 * d[2]
    if centered:
        x0, x1 = -hx, hx
        y0, y1 = -hy, hy
        z0, z1 = -hz, hz
    else:
        x0, x1 = 0.0, d[0]
        y0, y1 = 0.0, d[1]
        z0, z1 = 0.0, d[2]
    sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)
    b = MeshBuilder()
    # Metric UVs so facade shaders stay aligned after yaw.
    # +Z (front)
    b.add_face((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), n=(0, 0, 1),
               uv0=(0, 0), uv1=(sx, 0), uv2=(sx, sy), uv3=(0, sy))
    # -Z (back)
    b.add_face((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), n=(0, 0, -1),
               uv0=(0, 0), uv1=(sx, 0), uv2=(sx, sy), uv3=(0, sy))
    # +X
    b.add_face((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), n=(1, 0, 0),
               uv0=(0, 0), uv1=(sz, 0), uv2=(sz, sy), uv3=(0, sy))
    # -X
    b.add_face((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), n=(-1, 0, 0),
               uv0=(0, 0), uv1=(sz, 0), uv2=(sz, sy), uv3=(0, sy))
    # +Y
    b.add_face((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), n=(0, 1, 0),
               uv0=(0, 0), uv1=(sx, 0), uv2=(sx, sz), uv3=(0, sz))
    # -Y
    b.add_face((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), n=(0, -1, 0),
               uv0=(0, 0), uv1=(sx, 0), uv2=(sx, sz), uv3=(0, sz))
    return b.build("cuboid")


def generate_lofted_box(
    y0: float,
    y1: float,
    x0b: float,
    x1b: float,
    z0b: float,
    z1b: float,
    x0t: float,
    x1t: float,
    z0t: float,
    z1t: float,
    name: str = "loft",
) -> Mesh:
    """Convex prism with independent bottom/top rectangles (hoods, cabins, roofs)."""
    b = MeshBuilder()
    b.add_face((x0b, y0, z1b), (x1b, y0, z1b), (x1t, y1, z1t), (x0t, y1, z1t))
    b.add_face((x1b, y0, z0b), (x0b, y0, z0b), (x0t, y1, z0t), (x1t, y1, z0t))
    b.add_face((x1b, y0, z1b), (x1b, y0, z0b), (x1t, y1, z0t), (x1t, y1, z1t))
    b.add_face((x0b, y0, z0b), (x0b, y0, z1b), (x0t, y1, z1t), (x0t, y1, z0t))
    b.add_face((x0t, y1, z1t), (x1t, y1, z1t), (x1t, y1, z0t), (x0t, y1, z0t), n=(0, 1, 0))
    b.add_face((x0b, y0, z0b), (x1b, y0, z0b), (x1b, y0, z1b), (x0b, y0, z1b), n=(0, -1, 0))
    return b.build(name)


def generate_beveled_box(
    dimensions: Sequence[float],
    bevel: float = 0.06,
    centered: bool = True,
) -> Mesh:
    """Axis-aligned box with a small edge chamfer so silhouettes are not razor-sharp."""
    d = _as3(dimensions)
    b = min(float(bevel), 0.22 * float(min(d)))
    # Three overlapping boxes approximate a chamfer without 26 extra faces.
    parts = [
        generate_cuboid_mesh((d[0] - 2.0 * b, d[1], d[2]), centered=centered),
        generate_cuboid_mesh((d[0], d[1] - 2.0 * b, d[2]), centered=centered),
        generate_cuboid_mesh((d[0], d[1], d[2] - 2.0 * b), centered=centered),
    ]
    if not centered:
        parts = [
            _transformed_mesh(parts[0], (b, 0.0, 0.0)),
            _transformed_mesh(parts[1], (0.0, b, 0.0)),
            _transformed_mesh(parts[2], (0.0, 0.0, b)),
        ]
    return _merge_meshes(parts, "bevel_box")


def _center_y(mesh: Mesh) -> Mesh:
    """Shift so the Y AABB center is at 0 (Actor.position is the OBB center)."""
    if mesh.vertices.size == 0:
        return mesh
    ys = mesh.vertices[:, 1]
    mid = 0.5 * (float(ys.min()) + float(ys.max()))
    v = mesh.vertices.copy()
    v[:, 1] = ys - np.float32(mid)
    return Mesh(v, mesh.indices.copy(), mesh.name)


def _as_glass(mesh: Mesh) -> Mesh:
    """Sentinel UV so the shader treats this submesh as dark glass."""
    if mesh.vertices.size == 0:
        return mesh
    v = mesh.vertices.copy()
    v[:, 6] = np.float32(-100.0)
    v[:, 7] = np.float32(-100.0)
    return Mesh(v, mesh.indices.copy(), mesh.name)


def _displace_along_normal(mesh: Mesh, amp: float, scale: float) -> Mesh:
    """Cheap organic warp: sine-product displacement along the vertex normal."""
    if mesh.vertices.size == 0 or amp <= 0.0:
        return mesh
    v = mesh.vertices.copy()
    p = v[:, 0:3].astype(np.float64)
    n = v[:, 3:6].astype(np.float64)
    h = (
        np.sin(p[:, 0] * scale + 0.7)
        * np.sin(p[:, 1] * scale * 0.83 + 1.3)
        * np.sin(p[:, 2] * scale * 1.17 + 2.1)
    )
    v[:, 0:3] = (p + n * (amp * h)[:, None]).astype(np.float32)
    return Mesh(v, mesh.indices.copy(), mesh.name)


def generate_cylinder_mesh(
    radius: float,
    height: float,
    segments: int = 16,
    y0: float = 0.0,
) -> Mesh:
    b = MeshBuilder()
    y1 = y0 + height
    top_c = (0.0, y1, 0.0)
    bot_c = (0.0, y0, 0.0)
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        x0, z0 = radius * math.cos(a0), radius * math.sin(a0)
        x1, z1 = radius * math.cos(a1), radius * math.sin(a1)
        n0 = _norm(np.array([x0, 0.0, z0]))
        n1 = _norm(np.array([x1, 0.0, z1]))
        p00 = (x0, y0, z0)
        p10 = (x1, y0, z1)
        p11 = (x1, y1, z1)
        p01 = (x0, y1, z0)
        i00 = b.add_vertex(p00, n0, (i / segments, 0.0))
        i10 = b.add_vertex(p10, n1, ((i + 1) / segments, 0.0))
        i11 = b.add_vertex(p11, n1, ((i + 1) / segments, 1.0))
        i01 = b.add_vertex(p01, n0, (i / segments, 1.0))
        b.add_tri(i00, i10, i11)
        b.add_tri(i00, i11, i01)
        b.add_face(p01, p11, top_c, n=(0, 1, 0), uv0=(0, 0), uv1=(1, 0), uv2=(0.5, 1))
        b.add_face(p10, p00, bot_c, n=(0, -1, 0), uv0=(0, 0), uv1=(1, 0), uv2=(0.5, 1))
    return b.build("cylinder")


def generate_tapered_cylinder(
    radius_bottom: float,
    radius_top: float,
    height: float,
    segments: int = 12,
    y0: float = 0.0,
) -> Mesh:
    b = MeshBuilder()
    y1 = y0 + height
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        p00 = (radius_bottom * c0, y0, radius_bottom * s0)
        p10 = (radius_bottom * c1, y0, radius_bottom * s1)
        p11 = (radius_top * c1, y1, radius_top * s1)
        p01 = (radius_top * c0, y1, radius_top * s0)
        n0 = _norm(np.array([c0, (radius_bottom - radius_top) / max(height, 1e-4), s0]))
        n1 = _norm(np.array([c1, (radius_bottom - radius_top) / max(height, 1e-4), s1]))
        i00 = b.add_vertex(p00, n0, (i / segments, 0.0))
        i10 = b.add_vertex(p10, n1, ((i + 1) / segments, 0.0))
        i11 = b.add_vertex(p11, n1, ((i + 1) / segments, 1.0))
        i01 = b.add_vertex(p01, n0, (i / segments, 1.0))
        b.add_tri(i00, i10, i11)
        b.add_tri(i00, i11, i01)
    return b.build("taper")


def generate_uv_sphere(radius: float, stacks: int = 8, slices: int = 12, center: Sequence[float] = (0.0, 0.0, 0.0)) -> Mesh:
    b = MeshBuilder()
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    rings: List[List[int]] = []
    for st in range(stacks + 1):
        v = st / stacks
        phi = math.pi * v
        row: List[int] = []
        for sl in range(slices):
            u = sl / slices
            th = 2.0 * math.pi * u
            n = np.array([math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th)])
            p = (cx + radius * n[0], cy + radius * n[1], cz + radius * n[2])
            row.append(b.add_vertex(p, n, (u, v)))
        rings.append(row)
    for st in range(stacks):
        for sl in range(slices):
            sl2 = (sl + 1) % slices
            a, bb = rings[st][sl], rings[st][sl2]
            c, d = rings[st + 1][sl2], rings[st + 1][sl]
            if st != 0:
                b.add_tri(a, bb, c)
            if st != stacks - 1:
                b.add_tri(a, c, d)
    return b.build("sphere")


def generate_pothole_mesh(
    center: Sequence[float] | glm.vec3,
    radius: float,
    depth: float,
    rings: int = 8,
    slices: int = 24,
    rim_noise: float = 0.08,
    seed: int = 0,
    surface_y: float = 0.15,
) -> Mesh:
    """Depressed crater with a noisy rim, sitting in the sidewalk plane."""
    rng = np.random.default_rng(seed)
    c = _as3(center)
    b = MeshBuilder()
    radii = np.linspace(0.0, radius, rings + 1)
    verts_idx: List[List[int]] = []
    for ri, r in enumerate(radii):
        row: List[int] = []
        bowl = 0.5 * (1.0 + math.cos(math.pi * (r / max(radius, 1e-6))))  # 1 at center, 0 at rim
        for si in range(slices):
            ang = 2.0 * math.pi * si / slices
            jitter = 1.0 + rim_noise * (rng.uniform(-1.0, 1.0) if r > 0.2 * radius else 0.0)
            rr = r * jitter
            y_n = rng.uniform(-0.015, 0.015) if ri == rings else rng.uniform(-0.01, 0.01)
            y = surface_y - depth * bowl + y_n
            if ri == rings:
                y = surface_y + abs(y_n) * 0.5  # slightly proud lip
            p = (c[0] + rr * math.cos(ang), y, c[2] + rr * math.sin(ang))
            n = _norm(np.array([math.cos(ang) * bowl * 0.4, 1.0, math.sin(ang) * bowl * 0.4]))
            uv = (si / slices, ri / rings)
            row.append(b.add_vertex(p, n, uv))
        verts_idx.append(row)
    for ri in range(rings):
        for si in range(slices):
            sj = (si + 1) % slices
            a = verts_idx[ri][si]
            bb = verts_idx[ri][sj]
            c1 = verts_idx[ri + 1][sj]
            d = verts_idx[ri + 1][si]
            b.add_tri(a, bb, c1)
            b.add_tri(a, c1, d)
    return b.build("pothole")


def _merge_meshes(meshes: Sequence[Mesh], name: str) -> Mesh:
    if not meshes:
        return Mesh(np.zeros((0, 8), np.float32), np.zeros((0,), np.uint32), name)
    verts = []
    idxs = []
    offset = 0
    for m in meshes:
        verts.append(m.vertices)
        idxs.append(m.indices + offset)
        offset += len(m.vertices)
    return Mesh(np.concatenate(verts, axis=0), np.concatenate(idxs, axis=0), name)


def _transformed_mesh(
    mesh: Mesh,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
    rotation_y: float = 0.0,
    rotation_x: float = 0.0,
    rotation_z: float = 0.0,
) -> Mesh:
    r = np.eye(3, dtype=np.float64)
    if rotation_x:
        cx, sx = math.cos(rotation_x), math.sin(rotation_x)
        r = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64) @ r
    if rotation_y:
        cy, sy = math.cos(rotation_y), math.sin(rotation_y)
        r = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64) @ r
    if rotation_z:
        cz, sz = math.cos(rotation_z), math.sin(rotation_z)
        r = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64) @ r
    t = _as3(translation)
    v = mesh.vertices.copy()
    p = v[:, 0:3].astype(np.float64)
    n = v[:, 3:6].astype(np.float64)
    p = (p @ r.T) + t
    n = n @ r.T
    v[:, 0:3] = p.astype(np.float32)
    v[:, 3:6] = n.astype(np.float32)
    return Mesh(v, mesh.indices.copy(), mesh.name)


def generate_tree_mesh(rng: Optional[np.random.Generator] = None) -> Mesh:
    """Tapered trunk, forked limbs, and a lumpy multi-lobe canopy."""
    rng = rng or np.random.default_rng(0)
    h = float(rng.uniform(4.4, 7.8))
    rb = float(rng.uniform(0.17, 0.30))
    trunk = generate_tapered_cylinder(rb, rb * 0.32, h * 0.74, segments=12)
    parts = [trunk]
    n_branch = int(rng.integers(4, 8))
    for i in range(n_branch):
        yaw = (i / n_branch) * (2.0 * math.pi) + float(rng.uniform(-0.28, 0.28))
        elev = float(rng.uniform(0.40, 1.05))
        length = float(rng.uniform(0.85, 1.85))
        y = h * float(rng.uniform(0.36, 0.74))
        limb = generate_tapered_cylinder(0.065, 0.022, length, segments=8)
        limb = _transformed_mesh(limb, (0.0, 0.0, 0.0), rotation_z=elev)
        limb = _transformed_mesh(limb, (0.0, y, 0.0), rotation_y=yaw)
        parts.append(limb)
        if rng.random() < 0.55:
            twig = generate_tapered_cylinder(0.03, 0.012, length * 0.55, segments=6)
            twig = _transformed_mesh(twig, (0.0, 0.0, 0.0), rotation_z=elev + 0.45)
            twig = _transformed_mesh(twig, (0.0, y + 0.15, 0.0), rotation_y=yaw + 0.4)
            parts.append(twig)
    n_leaf = int(rng.integers(6, 11))
    for _ in range(n_leaf):
        rad = float(rng.uniform(0.50, 1.25))
        cy = h * float(rng.uniform(0.52, 0.98))
        cx = float(rng.uniform(-1.15, 1.15))
        cz = float(rng.uniform(-1.15, 1.15))
        lobe = generate_uv_sphere(rad, stacks=8, slices=10, center=(cx, cy, cz))
        lobe = _displace_along_normal(lobe, amp=rad * 0.22, scale=2.4 + float(rng.uniform(0.0, 1.2)))
        parts.append(lobe)
    if rng.random() < 0.7:
        hang = float(rng.choice([-1.0, 1.0]))
        bar = generate_tapered_cylinder(0.05, 0.028, 1.65, segments=8)
        bar = _transformed_mesh(bar, (0.0, 0.0, 0.0), rotation_z=1.22)
        bar = _transformed_mesh(bar, (hang * 0.16, float(rng.uniform(1.42, 1.82)), 0.0), rotation_y=hang * 0.25)
        parts.append(bar)
    return _merge_meshes(parts, "tree")


def generate_street_lamp_mesh() -> Mesh:
    pole = generate_tapered_cylinder(0.075, 0.042, 4.7, segments=10)
    base = generate_tapered_cylinder(0.16, 0.085, 0.32, segments=10)
    # Segmented arm so it reads as a curve rather than a stick.
    arm_a = _transformed_mesh(generate_cuboid_mesh((0.42, 0.055, 0.055)), (0.22, 4.52, 0.0))
    arm_b = _transformed_mesh(generate_cuboid_mesh((0.48, 0.05, 0.05)), (0.62, 4.42, 0.0), rotation_z=-0.35)
    fixture = generate_lofted_box(4.18, 4.34, 0.78, 1.18, -0.12, 0.12, 0.82, 1.14, -0.10, 0.10)
    glow = _transformed_mesh(generate_cuboid_mesh((0.32, 0.04, 0.18)), (0.98, 4.20, 0.0))
    return _merge_meshes([base, pole, arm_a, arm_b, fixture, glow], "lamp")


def generate_bench_mesh() -> Mesh:
    slats = []
    for i, y in enumerate((0.42, 0.50)):
        slat = generate_cuboid_mesh((0.44, 0.035, 1.48))
        slats.append(_transformed_mesh(slat, (0.0, y, 0.0)))
    back = []
    for z in np.linspace(-0.62, 0.62, 5):
        back.append(_transformed_mesh(generate_cuboid_mesh((0.04, 0.38, 0.07)), (-0.20, 0.70, float(z))))
    rail = _transformed_mesh(generate_cuboid_mesh((0.04, 0.04, 1.48)), (-0.20, 0.90, 0.0))
    legs = []
    for z in (-0.58, 0.58):
        legs.append(_transformed_mesh(generate_cuboid_mesh((0.38, 0.40, 0.06)), (0.0, 0.20, z)))
    return _merge_meshes([*slats, *back, rail, *legs], "bench")


def generate_hydrant_mesh() -> Mesh:
    barrel = generate_tapered_cylinder(0.13, 0.12, 0.55, segments=10, y0=0.08)
    dome = generate_uv_sphere(0.13, stacks=6, slices=10, center=(0.0, 0.68, 0.0))
    v = dome.vertices.copy()
    v[:, 1] = np.minimum(v[:, 1], 0.74)
    dome = Mesh(v, dome.indices, "dome")
    cap = generate_tapered_cylinder(0.05, 0.04, 0.10, segments=8, y0=0.70)
    side = generate_tapered_cylinder(0.04, 0.04, 0.16, segments=7)
    side = _transformed_mesh(side, (0.0, 0.0, 0.0), rotation_z=math.pi * 0.5)
    side = _transformed_mesh(side, (0.14, 0.42, 0.0))
    base = generate_tapered_cylinder(0.18, 0.16, 0.08, segments=10)
    return _center_y(_merge_meshes([base, barrel, dome, cap, side], "hydrant"))


def generate_planter_mesh(rng: Optional[np.random.Generator] = None) -> Mesh:
    rng = rng or np.random.default_rng(0)
    box = generate_beveled_box((0.62, 0.42, 0.62), bevel=0.04)
    box = _transformed_mesh(box, (0.0, 0.21, 0.0))
    dirt = _transformed_mesh(generate_cuboid_mesh((0.50, 0.06, 0.50)), (0.0, 0.40, 0.0))
    bush = generate_uv_sphere(0.28, stacks=6, slices=8, center=(0.0, 0.62, 0.0))
    bush = _displace_along_normal(bush, amp=0.07, scale=6.0)
    extra = []
    if rng.random() < 0.6:
        extra.append(generate_uv_sphere(0.16, stacks=5, slices=7, center=(0.12, 0.70, -0.08)))
    return _center_y(_merge_meshes([box, dirt, bush, *extra], "planter"))


def generate_head_hazard_mesh(
    type: str,
    position: glm.vec3 | Sequence[float],
    rng: Optional[np.random.Generator] = None,
) -> Mesh:
    """Cantilevered / branching obstacles in *local* space (origin at the mount)."""
    rng = rng or np.random.default_rng(0)
    p = _as3(position)
    kind = type.lower()
    y_bar = float(rng.uniform(1.40, 1.85))
    hang = -1.0 if p[0] >= 0.0 else 1.0
    if kind in ("branch", "tree", "tree_branch"):
        return generate_tree_mesh(rng)
    if kind in ("awning",):
        depth = float(rng.uniform(1.2, 2.0))
        width = float(rng.uniform(2.5, 4.5))
        box = generate_cuboid_mesh((depth, 0.08, width))
        ribs = [ _transformed_mesh(generate_cuboid_mesh((depth, 0.04, 0.05)), (0.0, -0.06, z))
                 for z in np.linspace(-width * 0.4, width * 0.4, 4) ]
        return _merge_meshes([_transformed_mesh(box, (hang * depth * 0.25, 0.0, 0.0)), *ribs], "awning")
    if kind in ("sign", "cantilever_sign"):
        pole = generate_tapered_cylinder(0.05, 0.04, 3.2, segments=8)
        panel = generate_cuboid_mesh((1.1, 0.7, 0.06))
        panel = _transformed_mesh(panel, (hang * 0.55, y_bar, 0.0))
        arm = generate_cuboid_mesh((1.15, 0.05, 0.05))
        arm = _transformed_mesh(arm, (hang * 0.55, y_bar + 0.4, 0.0))
        return _merge_meshes([pole, arm, panel], "sign")
    if kind in ("pipe", "tailgate"):
        length = float(rng.uniform(1.4, 2.4))
        bar = generate_tapered_cylinder(0.06, 0.06, length, segments=10)
        return _transformed_mesh(bar, (-0.5 * length, 0.0, 0.0), rotation_z=math.pi * 0.5)
    return generate_cuboid_mesh((1.6, 0.12, 0.12))


def _wheel(x: float, y: float, z: float, radius: float = 0.31, width: float = 0.18) -> Mesh:
    tire = generate_cylinder_mesh(radius, width, segments=14)
    tire = _transformed_mesh(tire, (0.0, 0.0, 0.0), rotation_z=math.pi * 0.5)
    hub = generate_cylinder_mesh(radius * 0.42, width * 1.15, segments=10)
    hub = _transformed_mesh(hub, (0.0, 0.0, 0.0), rotation_z=math.pi * 0.5)
    return _merge_meshes([
        _transformed_mesh(tire, (x, y, z)),
        _transformed_mesh(hub, (x, y, z)),
    ], "wheel")


def generate_vehicle_mesh(kind: str = "sedan") -> Mesh:
    """Lofted bodies with raked glass, wheel arches, and bumpers — not stacked boxes."""
    if kind == "van":
        rocker = generate_lofted_box(0.22, 0.78, -0.90, 0.90, -2.25, 2.30, -0.88, 0.88, -2.20, 2.22)
        cabin = generate_lofted_box(0.76, 1.95, -0.88, 0.88, -2.15, 2.05, -0.80, 0.80, -2.05, 1.55)
        hood = generate_lofted_box(0.78, 1.12, -0.86, 0.86, 1.40, 2.32, -0.78, 0.78, 1.25, 1.95)
        bumper = generate_lofted_box(0.28, 0.55, -0.92, 0.92, 2.22, 2.42, -0.90, 0.90, 2.20, 2.38)
        mirrors = [
            _transformed_mesh(generate_cuboid_mesh((0.10, 0.09, 0.16)), (0.95, 1.35, 1.15)),
            _transformed_mesh(generate_cuboid_mesh((0.10, 0.09, 0.16)), (-0.95, 1.35, 1.15)),
        ]
        wheels = [_wheel(x, 0.32, z, 0.33, 0.20) for x in (-0.82, 0.82) for z in (-1.45, 1.40)]
        return _center_y(_merge_meshes([rocker, cabin, hood, bumper, *mirrors, *wheels], "vehicle"))
    if kind == "truck":
        bed = generate_lofted_box(0.85, 1.85, -1.02, 1.02, -2.35, 0.85, -1.00, 1.00, -2.30, 0.80)
        cabin = generate_lofted_box(0.78, 2.15, -0.98, 0.98, 0.70, 2.45, -0.88, 0.88, 0.85, 2.05)
        hood = generate_lofted_box(0.78, 1.25, -0.96, 0.96, 2.00, 2.85, -0.88, 0.88, 1.90, 2.45)
        bumper = generate_lofted_box(0.30, 0.62, -1.05, 1.05, 2.75, 3.05, -1.02, 1.02, 2.72, 2.98)
        wheels = [_wheel(x, 0.38, z, 0.38, 0.22) for x in (-0.92, 0.92) for z in (-1.70, 0.15, 2.05)]
        return _center_y(_merge_meshes([bed, cabin, hood, bumper, *wheels], "vehicle"))
    rocker = generate_lofted_box(0.18, 0.70, -0.86, 0.86, -2.08, 2.10, -0.84, 0.84, -2.02, 2.02)
    belt = generate_lofted_box(0.68, 0.96, -0.84, 0.84, -2.00, 2.00, -0.80, 0.80, -1.88, 1.55)
    cabin = generate_lofted_box(0.94, 1.48, -0.80, 0.80, -1.20, 0.82, -0.70, 0.70, -1.08, 0.18)
    hood = generate_lofted_box(0.78, 0.98, -0.80, 0.80, 0.55, 2.02, -0.74, 0.74, 0.40, 1.42)
    trunk = generate_lofted_box(0.78, 1.02, -0.80, 0.80, -2.05, -1.05, -0.74, 0.74, -1.92, -1.15)
    bumper_f = generate_lofted_box(0.22, 0.48, -0.88, 0.88, 2.00, 2.20, -0.86, 0.86, 1.96, 2.16)
    bumper_r = generate_lofted_box(0.22, 0.48, -0.88, 0.88, -2.22, -2.02, -0.86, 0.86, -2.18, -1.98)
    lights = [
        _transformed_mesh(generate_cuboid_mesh((0.28, 0.10, 0.06)), (0.52, 0.52, 2.12)),
        _transformed_mesh(generate_cuboid_mesh((0.28, 0.10, 0.06)), (-0.52, 0.52, 2.12)),
    ]
    mirrors = [
        _transformed_mesh(generate_cuboid_mesh((0.09, 0.08, 0.14)), (0.88, 1.12, 0.35)),
        _transformed_mesh(generate_cuboid_mesh((0.09, 0.08, 0.14)), (-0.88, 1.12, 0.35)),
    ]
    wheels = [_wheel(x, 0.30, z) for x in (-0.78, 0.78) for z in (-1.28, 1.28)]
    return _center_y(_merge_meshes(
        [rocker, belt, cabin, hood, trunk, bumper_f, bumper_r, *lights, *mirrors, *wheels], "vehicle"
    ))


def generate_pedestrian_mesh(rng: Optional[np.random.Generator] = None, centered: bool = True) -> Mesh:
    """Capsule-like body with a stride pose so silhouettes are not door-shaped."""
    rng = rng or np.random.default_rng(0)
    stride = float(rng.uniform(-0.32, 0.32))
    head = generate_uv_sphere(0.115, stacks=7, slices=10, center=(0.0, 1.64, 0.0))
    neck = generate_tapered_cylinder(0.045, 0.05, 0.08, segments=7, y0=1.50)
    torso = generate_tapered_cylinder(0.155, 0.135, 0.50, segments=10, y0=0.96)
    hips = generate_uv_sphere(0.14, stacks=6, slices=8, center=(0.0, 0.92, 0.0))
    hv = hips.vertices.copy()
    hv[:, 1] = 0.92 + (hv[:, 1] - 0.92) * 0.55
    hv[:, 0] *= 1.15
    hips = Mesh(hv, hips.indices, "hips")
    parts = [head, neck, torso, hips]
    for side, x in ((-1.0, -0.12), (1.0, 0.12)):
        thigh = generate_tapered_cylinder(0.068, 0.052, 0.44, segments=8)
        thigh = _transformed_mesh(thigh, (0.0, 0.0, 0.0), rotation_x=side * stride)
        thigh = _transformed_mesh(thigh, (x, 0.46, 0.0))
        shin = generate_tapered_cylinder(0.052, 0.042, 0.40, segments=8)
        shin = _transformed_mesh(shin, (0.0, 0.0, 0.0), rotation_x=-side * stride * 0.35)
        shin = _transformed_mesh(shin, (x, 0.03, side * stride * 0.12))
        parts.extend([thigh, shin])
        upper = generate_tapered_cylinder(0.048, 0.040, 0.30, segments=7)
        upper = _transformed_mesh(upper, (0.0, 0.0, 0.0), rotation_z=side * 0.12, rotation_x=-side * stride * 0.5)
        upper = _transformed_mesh(upper, (x * 1.55, 1.18, 0.0))
        lower = generate_tapered_cylinder(0.040, 0.032, 0.28, segments=7)
        lower = _transformed_mesh(lower, (0.0, 0.0, 0.0), rotation_z=side * 0.08)
        lower = _transformed_mesh(lower, (x * 1.7, 0.88, 0.05))
        parts.extend([upper, lower])
        foot = generate_lofted_box(0.0, 0.055, x - 0.05, x + 0.05, -0.03 + side * stride * 0.08, 0.13 + side * stride * 0.08,
                                  x - 0.045, x + 0.045, -0.02 + side * stride * 0.08, 0.12 + side * stride * 0.08)
        parts.append(foot)
    mesh = _merge_meshes(parts, "pedestrian")
    return _center_y(mesh) if centered else mesh


def generate_cyclist_mesh(rng: Optional[np.random.Generator] = None) -> Mesh:
    rider = generate_pedestrian_mesh(rng=rng, centered=False)
    rider = _transformed_mesh(rider, (0.0, 0.12, -0.08), rotation_x=-0.38)
    wheel_f = generate_cylinder_mesh(0.32, 0.05, segments=10)
    wheel_f = _transformed_mesh(wheel_f, (0.0, 0.0, 0.0), rotation_z=math.pi * 0.5)
    wheel_f = _transformed_mesh(wheel_f, (0.0, 0.32, 0.55))
    wheel_r = _transformed_mesh(wheel_f, (0.0, 0.0, -1.10))
    frame = generate_cuboid_mesh((0.06, 0.06, 1.05))
    frame = _transformed_mesh(frame, (0.0, 0.55, 0.0))
    bar = generate_cuboid_mesh((0.52, 0.04, 0.04))
    bar = _transformed_mesh(bar, (0.0, 1.05, 0.42))
    return _center_y(_merge_meshes([rider, wheel_f, wheel_r, frame, bar], "cyclist"))


def generate_scooter_mesh() -> Mesh:
    deck = generate_cuboid_mesh((0.22, 0.08, 0.95))
    deck = _transformed_mesh(deck, (0.0, 0.14, 0.05))
    stem = generate_tapered_cylinder(0.03, 0.025, 1.0, segments=7)
    stem = _transformed_mesh(stem, (0.0, 0.16, 0.42), rotation_x=-0.15)
    bar = generate_tapered_cylinder(0.02, 0.02, 0.50, segments=6)
    bar = _transformed_mesh(bar, (0.0, 0.0, 0.0), rotation_z=math.pi * 0.5)
    bar = _transformed_mesh(bar, (0.0, 1.12, 0.48))
    wf = generate_cylinder_mesh(0.12, 0.05, segments=8)
    wf = _transformed_mesh(wf, (0.0, 0.0, 0.0), rotation_z=math.pi * 0.5)
    wr = _transformed_mesh(wf, (0.0, 0.12, -0.40))
    wf = _transformed_mesh(wf, (0.0, 0.12, 0.42))
    return _center_y(_merge_meshes([deck, stem, bar, wf, wr], "scooter"))


def generate_dustbin_mesh() -> Mesh:
    body = generate_tapered_cylinder(0.24, 0.22, 0.95, segments=12)
    lid = generate_uv_sphere(0.23, stacks=5, slices=10, center=(0.0, 0.98, 0.0))
    # Flatten lid by scaling Y in-place.
    v = lid.vertices.copy()
    v[:, 1] = 0.98 + (v[:, 1] - 0.98) * 0.35
    lid = Mesh(v, lid.indices, "lid")
    return _center_y(_merge_meshes([body, lid], "dustbin"))


def generate_bollard_mesh() -> Mesh:
    post = generate_tapered_cylinder(0.09, 0.08, 0.88, segments=10)
    cap = generate_uv_sphere(0.09, stacks=5, slices=8, center=(0.0, 0.90, 0.0))
    return _center_y(_merge_meshes([post, cap], "bollard"))


def generate_sky_dome(radius: float = 90.0, rings: int = 14, segments: int = 28) -> Mesh:
    """Inward-facing hemisphere (+Y) used as a sky background."""
    b = MeshBuilder()
    rings_idx: List[List[int]] = []
    for r in range(rings + 1):
        v = r / rings
        phi = (math.pi * 0.5) * v  # 0 at zenith-ish... actually 0 at horizon? 
        # v=0 equator (y=0), v=1 zenith
        phi = math.pi * 0.5 * (1.0 - v)
        row: List[int] = []
        for s in range(segments):
            u = s / segments
            th = 2.0 * math.pi * u
            x = radius * math.cos(phi) * math.cos(th)
            y = radius * math.sin(phi)
            z = radius * math.cos(phi) * math.sin(th)
            n = _norm(np.array([-x, -y, -z]))
            row.append(b.add_vertex((x, y, z), n, (u, v)))
        rings_idx.append(row)
    for r in range(rings):
        for s in range(segments):
            s2 = (s + 1) % segments
            a, bb = rings_idx[r][s], rings_idx[r][s2]
            c, d = rings_idx[r + 1][s2], rings_idx[r + 1][s]
            b.add_tri(a, d, c)
            b.add_tri(a, c, bb)
    return b.build("sky")


def generate_building_mesh(
    width: float,
    depth: float,
    height: float,
    rng: Optional[np.random.Generator] = None,
) -> Mesh:
    """Varied massing: setbacks, recessed windows, storefront, balconies, roof plant."""
    rng = rng or np.random.default_rng(0)
    parts: List[Mesh] = []
    setback = rng.random() < 0.42 and height > 10.0
    split_y = height * float(rng.uniform(0.42, 0.62)) if setback else height
    # Base mass (centered: y from -0.5H to +0.5H). Build relative to y=0 as building center.
    y_base0 = -0.5 * height
    y_base1 = y_base0 + split_y
    base_h = split_y
    parts.append(_transformed_mesh(generate_beveled_box((depth, base_h, width), bevel=0.07), (0.0, y_base0 + 0.5 * base_h, 0.0)))
    top_d, top_w, top_h = depth, width, height - split_y
    if setback:
        top_d = depth * float(rng.uniform(0.72, 0.90))
        top_w = width * float(rng.uniform(0.78, 0.94))
        parts.append(_transformed_mesh(
            generate_beveled_box((top_d, top_h, top_w), bevel=0.06),
            (float(rng.uniform(-0.15, 0.15)), y_base1 + 0.5 * top_h, 0.0),
        ))
    # Cornice + parapet
    parts.append(_transformed_mesh(generate_cuboid_mesh((depth + 0.30, 0.14, width + 0.30)), (0.0, y_base1 - 0.04, 0.0)))
    roof_y = 0.5 * height
    if setback:
        roof_y = y_base1 + top_h
    if rng.random() < 0.28:
        # Shallow hip roof
        parts.append(generate_lofted_box(
            roof_y - 0.02, roof_y + 1.15,
            -0.52 * top_d, 0.52 * top_d, -0.52 * top_w, 0.52 * top_w,
            -0.08, 0.08, -0.08, 0.08,
        ))
    else:
        parts.append(_transformed_mesh(generate_cuboid_mesh((top_d + 0.10, 0.42, top_w + 0.10)), (0.0, roof_y + 0.16, 0.0)))
    # Storefront ledge
    parts.append(_transformed_mesh(generate_cuboid_mesh((depth + 0.16, 0.10, width + 0.16)), (0.0, y_base0 + 3.05, 0.0)))
    if rng.random() < 0.75:
        hx = float(rng.uniform(-0.28 * top_d, 0.28 * top_d))
        hz = float(rng.uniform(-0.22 * top_w, 0.22 * top_w))
        parts.append(_transformed_mesh(
            generate_beveled_box(
                (float(rng.uniform(1.1, 2.1)), float(rng.uniform(0.7, 1.5)), float(rng.uniform(1.1, 2.3))),
                bevel=0.05,
            ),
            (hx, roof_y + 0.85, hz),
        ))
    # Recessed windows + frames on ±X (street faces).
    floors = max(2, int((height - 1.4) / 3.15))
    cols = max(2, int(width / 2.35))
    for side in (-1.0, 1.0):
        x_face = side * (0.5 * depth)
        door_z = float(rng.uniform(-0.35 * width, 0.35 * width))
        n_shop = max(2, min(4, cols))
        shop_w = min(width * 0.78, 2.2 * n_shop)
        for si in range(n_shop):
            gz = -0.5 * shop_w + (si + 0.5) * (shop_w / n_shop)
            pane = _as_glass(generate_cuboid_mesh((0.08, 1.85, 1.15)))
            parts.append(_transformed_mesh(pane, (side * (0.5 * depth - 0.10), y_base0 + 1.15, gz)))
            frame = generate_cuboid_mesh((0.05, 1.98, 1.28))
            parts.append(_transformed_mesh(frame, (x_face + side * 0.015, y_base0 + 1.15, gz)))
        door = _as_glass(generate_cuboid_mesh((0.07, 2.15, 0.92)))
        parts.append(_transformed_mesh(door, (side * (0.5 * depth - 0.06), y_base0 + 1.10, door_z)))
        for fi in range(1, floors):
            fy = y_base0 + 1.55 + fi * 3.15
            if fy > roof_y - 1.05:
                continue
            for ci in range(cols):
                if rng.random() < 0.08:
                    continue
                fz = -0.5 * width + (ci + 0.5) * (width / cols)
                inset = _as_glass(generate_cuboid_mesh((0.12, 1.32, 0.90)))
                parts.append(_transformed_mesh(inset, (side * (0.5 * depth - 0.10), fy, fz)))
                frame = generate_cuboid_mesh((0.05, 1.42, 1.02))
                parts.append(_transformed_mesh(frame, (x_face + side * 0.02, fy, fz)))
                if fi >= 2 and rng.random() < 0.18:
                    slab = generate_cuboid_mesh((0.55, 0.07, 1.15))
                    parts.append(_transformed_mesh(slab, (side * (0.5 * depth + 0.28), fy - 0.72, fz)))
                    rail = generate_cuboid_mesh((0.04, 0.32, 1.15))
                    parts.append(_transformed_mesh(rail, (side * (0.5 * depth + 0.52), fy - 0.52, fz)))
        # Downspout
        pipe = generate_tapered_cylinder(0.04, 0.035, height * 0.92, segments=6)
        parts.append(_transformed_mesh(pipe, (side * (0.5 * depth + 0.05), y_base0, side * 0.48 * width)))
    return _merge_meshes(parts, "building")


def generate_road_markings(
    spline: StreetSpline,
    road_width: float,
    dash: float = 2.4,
    gap: float = 3.2,
    stripe_w: float = 0.12,
) -> Mesh:
    """Centre dashed line and two solid edge lines following the spline."""
    b = MeshBuilder()
    half = 0.5 * road_width - 0.35
    s = 0.0
    paint = True
    while s < spline.length - 1.0:
        length = dash if paint else gap
        s1 = min(s + length, spline.length - 0.05)
        if paint:
            p0, t0, n0, _ = spline.frame_at(s)
            p1, t1, n1, _ = spline.frame_at(s1)
            for lat in (0.0,):
                a = p0 + n0 * (lat - stripe_w) + np.array([0.0, 0.012, 0.0])
                bb = p0 + n0 * (lat + stripe_w) + np.array([0.0, 0.012, 0.0])
                c = p1 + n1 * (lat + stripe_w) + np.array([0.0, 0.012, 0.0])
                d = p1 + n1 * (lat - stripe_w) + np.array([0.0, 0.012, 0.0])
                b.add_face(a, d, c, bb, n=(0, 1, 0))
        s = s1
        paint = not paint
    # Edge lines
    s = 0.0
    step = 2.0
    while s < spline.length - step:
        p0, t0, n0, _ = spline.frame_at(s)
        p1, t1, n1, _ = spline.frame_at(min(s + step, spline.length - 0.05))
        for sign in (-1.0, 1.0):
            lat0 = sign * half
            a = p0 + n0 * (lat0 - stripe_w * 0.6) + np.array([0.0, 0.012, 0.0])
            bb = p0 + n0 * (lat0 + stripe_w * 0.6) + np.array([0.0, 0.012, 0.0])
            c = p1 + n1 * (lat0 + stripe_w * 0.6) + np.array([0.0, 0.012, 0.0])
            d = p1 + n1 * (lat0 - stripe_w * 0.6) + np.array([0.0, 0.012, 0.0])
            b.add_face(a, d, c, bb, n=(0, 1, 0))
        s += step
    return b.build("markings")


# ---------------------------------------------------------------------------
# Bridson Poisson-disk sampling in 2-D
# ---------------------------------------------------------------------------

def poisson_disk_2d(
    width: float,
    height: float,
    r_min: float,
    rng: np.random.Generator,
    k: int = 28,
    existing: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    cell = r_min / math.sqrt(2.0)
    gw = max(1, int(math.ceil(width / cell)))
    gh = max(1, int(math.ceil(height / cell)))
    grid = -np.ones((gw, gh), dtype=np.int32)
    samples: List[Tuple[float, float]] = []
    active: List[int] = []

    def grid_coords(p: Tuple[float, float]) -> Tuple[int, int]:
        return int(p[0] / cell), int(p[1] / cell)

    def far_enough(p: Tuple[float, float]) -> bool:
        gx, gy = grid_coords(p)
        r2 = r_min * r_min
        for i in range(max(0, gx - 2), min(gw, gx + 3)):
            for j in range(max(0, gy - 2), min(gh, gy + 3)):
                sidx = int(grid[i, j])
                if sidx < 0:
                    continue
                q = samples[sidx]
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                if dx * dx + dy * dy < r2:
                    return False
        if existing:
            for q in existing:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                if dx * dx + dy * dy < r2:
                    return False
        return True

    def insert(p: Tuple[float, float]) -> None:
        samples.append(p)
        gx, gy = grid_coords(p)
        if 0 <= gx < gw and 0 <= gy < gh:
            grid[gx, gy] = len(samples) - 1
        active.append(len(samples) - 1)

    if existing:
        for q in existing:
            if 0 <= q[0] < width and 0 <= q[1] < height:
                gx, gy = grid_coords(q)
                if 0 <= gx < gw and 0 <= gy < gh and grid[gx, gy] < 0:
                    samples.append(q)
                    grid[gx, gy] = len(samples) - 1

    p0 = (float(rng.uniform(0.05 * width, 0.95 * width)), float(rng.uniform(0.05 * height, 0.95 * height)))
    if far_enough(p0):
        insert(p0)

    while active:
        ai = int(rng.integers(0, len(active)))
        src = samples[active[ai]]
        found = False
        for _ in range(k):
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            rad = float(rng.uniform(r_min, 2.0 * r_min))
            cand = (src[0] + rad * math.cos(ang), src[1] + rad * math.sin(ang))
            if cand[0] < 0.0 or cand[0] >= width or cand[1] < 0.0 or cand[1] >= height:
                continue
            if far_enough(cand):
                insert(cand)
                found = True
                break
        if not found:
            active.pop(ai)
    return samples


@dataclass
class ScatterPoint:
    world: np.ndarray
    s: float
    lateral: float


def scatter_on_sidewalk(
    spline: StreetSpline,
    road_width: float,
    curb_width: float,
    sidewalk_width: float,
    r_min: float,
    rng: np.random.Generator,
    side: int,
    margin: float = 0.35,
    s_start: float = 4.0,
    s_end: Optional[float] = None,
) -> List[ScatterPoint]:
    """Poisson-disk candidates in (s, lateral) then lifted through the Frenet frame.

    `side` = +1 is left sidewalk (along +N), -1 is right sidewalk (along -N).
    """
    s_end = spline.length - 4.0 if s_end is None else s_end
    length = max(1.0, s_end - s_start)
    width = max(0.4, sidewalk_width - 2.0 * margin)
    pts2 = poisson_disk_2d(width, length, r_min, rng)
    out: List[ScatterPoint] = []
    half = 0.5 * road_width
    base_lat = side * (half + curb_width + margin)
    for u, v in pts2:
        s = s_start + v
        lat_along = u  # 0 at inner walk edge
        lateral = base_lat + side * lat_along
        p, t, n, up = spline.frame_at(s)
        world = p + n * lateral + np.array([0.0, 0.0, 0.0])
        out.append(ScatterPoint(world=world, s=s, lateral=lateral))
    return out


def sidewalk_center_lateral(road_width: float, curb_width: float, sidewalk_width: float, side: int) -> float:
    """Signed offset from centerline to sidewalk midline. side=+1 left, -1 right."""
    return side * (0.5 * road_width + curb_width + 0.5 * sidewalk_width)
