"""Procedural street, Poisson-disk hazards, dynamic actors, and scenario injection.

The generator is deliberately asset-free: every mesh is a low-poly primitive
so the pipeline runs `blender --background --python main.py` on a stock
Blender 4.x install. Domain randomization of colour / roughness / lighting
is what prevents the downstream detector from overfitting to those primitives.

Coordinate frame inside this module is Blender Z-up. The camera and every
actor share the road spline's Frenet frame `(s, lateral)`; the sidewalk is
a lateral offset, not a second arc-length parameter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from scenario_compose import (
    CLEAR_CENTER_BIOMES,
    CLEAR_CENTER_SCENARIOS,
    CROSS_EGO_SCENARIOS,
    CROSS_GAP_SCENARIOS,
    SPARSE_SCENARIOS,
    ComposeSession,
    all_scenario_names,
    compose_slug,
    extents_for,
    is_noop,
    pick_scenario,
    pick_scenarios,
    resolve_scenario_name,
    sort_for_inject,
)

from camera_kinematics import Perlin1D

import bpy
from mathutils import Vector


# ===========================================================================
# Shared wind + foliage sway (visual only; trunk threat is unchanged)
# ===========================================================================
#
# Motion follows the GPU Gems 3 / Weber–Penn idea: a shared wind direction
# and gust envelope, then per-joint periodic modes. Branches take the slow
# cantilever bend; leaf sprigs add a 2–5 Hz flutter. The trunk never moves.

WIND_LABELS: tuple[str, ...] = ("calm", "breeze", "windy")


class WindField:
    """Episode-wide wind, evaluated once per sim time.

    ``strength`` is 0..1 (calm ≈ 0.05, breeze ≈ 0.4, windy ≈ 0.85).
    ``direction`` is a world-XY yaw: 0 blows toward +X.
    """

    __slots__ = ("noise", "strength", "direction", "label", "_t", "_env", "_gust")

    def __init__(
        self,
        seed: int,
        strength: float = 0.40,
        direction: float = 0.0,
        label: str = "breeze",
    ) -> None:
        self.noise = Perlin1D(int(seed))
        self.strength = max(0.0, float(strength))
        self.direction = float(direction)
        self.label = str(label)
        self._t: Optional[float] = None
        self._env = 0.0
        self._gust = 0.0

    def state(self, t: float) -> tuple[float, float, float, float]:
        """``(envelope, gust, dir_x, dir_y)``. Envelope already includes strength."""
        if self._t != t:
            self._t = t
            self._gust = self.noise.fbm(
                t * 0.17, octaves=3, persistence=0.55, lacunarity=2.1,
            )
            # Calm still has a breath of motion; windy never quite dies.
            self._env = self.strength * max(0.0, 0.52 + 0.48 * self._gust)
        return (
            self._env,
            self._gust,
            math.cos(self.direction),
            math.sin(self.direction),
        )


class FoliagePart:
    """One animated joint: a branch empty or a single triangle leaf."""

    __slots__ = ("obj", "rest", "phase", "flutter", "kind", "flex")

    def __init__(
        self,
        obj: Any,
        rest: tuple[float, float, float],
        phase: float,
        flutter: float,
        kind: str,
        flex: float = 1.0,
    ) -> None:
        self.obj = obj
        self.rest = rest
        self.phase = float(phase)
        self.flutter = float(flutter)
        self.kind = kind
        self.flex = float(flex)


def foliage_wind_euler(
    t: float,
    phase: float,
    flutter: float,
    kind: str,
    env: float,
    gust: float,
    dir_x: float,
    dir_y: float,
    flex: float = 1.0,
) -> tuple[float, float, float]:
    """Local Euler offset (radians) for one branch joint or leaf.

    GPU Gems 3 sum of sines, weighted by the shared envelope. Parent
    joints carry children, so a mid-limb bend moves the whole distal
    crown. ``flex`` is small near the bole and ~1 on twigs / leaves.
    """
    w = 2.0 * math.pi
    gain = max(0.0, float(flex))
    bend = (
        0.50 * math.sin(w * 0.31 * t + phase)
        + 0.28 * math.sin(w * 0.17 * t + phase * 0.71)
        + 0.22 * gust
    )
    if kind == "branch":
        amp = 0.14 * env * gain
        return (amp * bend * dir_y, amp * bend * dir_x, 0.0)
    rustle = (
        0.42 * math.sin(w * 2.05 * t + flutter)
        + 0.28 * math.sin(w * 3.35 * t + flutter * 1.37)
        + 0.16 * math.sin(w * 5.10 * t + flutter * 0.61)
    )
    amp = 0.48 * env * gain
    return (
        amp * (0.28 * bend * dir_y + 0.72 * rustle),
        amp * (0.28 * bend * dir_x + 0.58 * rustle),
        amp * 0.24 * rustle,
    )


# ===========================================================================
# Arc-length spline
# ===========================================================================

class PathSpline:
    """Piecewise-linear arc-length path with a well-defined Frenet frame.

    Control polygons are first densely sampled (Bézier or polyline), then
    resampled at a uniform ds so `evaluate(s)` / `tangent(s)` are O(log N)
    binary searches. The horizontal right vector is `tangent × world_up`.
    """

    def __init__(self, points: list[Vector], ds: float = 0.25) -> None:
        if len(points) < 2:
            raise ValueError("PathSpline requires at least two points")
        self.ds = float(ds)
        self._raw = [Vector(p) for p in points]
        self._build()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        raw = self._raw
        seglen = [0.0]
        acc = 0.0
        for a, b in zip(raw, raw[1:]):
            acc += (b - a).length
            seglen.append(acc)
        self.length = acc
        if acc < 1e-6:
            raise ValueError("PathSpline has zero length")

        n = max(2, int(math.ceil(acc / self.ds)) + 1)
        self.samples: list[Vector] = []
        self.tangents: list[Vector] = []
        for i in range(n):
            s = min(acc, i * acc / (n - 1))
            p, t = self._interp_raw(s, seglen)
            self.samples.append(p)
            self.tangents.append(t)
        self._s = [i * acc / (n - 1) for i in range(n)]

    def _interp_raw(self, s: float, seglen: list[float]) -> tuple[Vector, Vector]:
        s = min(max(0.0, s), seglen[-1])
        lo, hi = 0, len(seglen) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if seglen[mid] <= s:
                lo = mid
            else:
                hi = mid
        span = seglen[hi] - seglen[lo]
        u = 0.0 if span < 1e-9 else (s - seglen[lo]) / span
        a, b = self._raw[lo], self._raw[hi]
        p = a.lerp(b, u)
        t = b - a
        if t.length < 1e-8:
            t = Vector((0.0, 1.0, 0.0))
        else:
            t.normalize()
        return p, t

    def _index(self, s: float) -> tuple[int, float]:
        s = min(max(0.0, s), self.length)
        n = len(self._s)
        # Uniform parameterisation: direct index, then refine.
        i = int(s / self.length * (n - 1))
        i = min(max(0, i), n - 2)
        while i + 1 < n - 1 and self._s[i + 1] < s:
            i += 1
        while i > 0 and self._s[i] > s:
            i -= 1
        span = self._s[i + 1] - self._s[i]
        u = 0.0 if span < 1e-9 else (s - self._s[i]) / span
        return i, u

    def evaluate(self, s: float) -> Vector:
        i, u = self._index(s)
        return self.samples[i].lerp(self.samples[i + 1], u)

    def tangent(self, s: float) -> Vector:
        i, u = self._index(s)
        t = self.tangents[i].lerp(self.tangents[i + 1], u)
        if t.length < 1e-8:
            return Vector((0.0, 1.0, 0.0))
        t.normalize()
        return t

    def right(self, s: float) -> Vector:
        t = self.tangent(s)
        r = t.cross(Vector((0.0, 0.0, 1.0)))
        if r.length < 1e-6:
            return Vector((1.0, 0.0, 0.0))
        r.normalize()
        return r

    def frame(self, s: float) -> tuple[Vector, Vector, Vector]:
        return self.evaluate(s), self.tangent(s), self.right(s)

    def offset_point(self, s: float, lateral: float, z: float = 0.0) -> Vector:
        p, _t, r = self.frame(s)
        q = p + r * lateral
        q.z = z
        return q

    def offset_spline(self, lateral: float) -> "PathSpline":
        pts = [
            self.offset_point(s, lateral, z=0.0)
            for s in self._s
        ]
        return PathSpline(pts, ds=self.ds)

    def project(self, p: Vector) -> tuple[float, float]:
        """Nearest arc-length `s` and signed lateral of a world XY point.

        Closest point is the foot of the perpendicular on each polyline
        segment (not the nearest sample vertex). Lateral is the planar
        offset along `right(s)` (positive = road-right).
        """
        px, py = float(p.x), float(p.y)
        best_d = 1e18
        best_s = 0.0
        n = len(self.samples)
        for i in range(n - 1):
            ax, ay = self.samples[i].x, self.samples[i].y
            bx, by = self.samples[i + 1].x, self.samples[i + 1].y
            abx, aby = bx - ax, by - ay
            ab2 = abx * abx + aby * aby
            if ab2 < 1e-18:
                t = 0.0
            else:
                t = ((px - ax) * abx + (py - ay) * aby) / ab2
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            qx = ax + t * abx
            qy = ay + t * aby
            dx = px - qx
            dy = py - qy
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_s = self._s[i] + t * (self._s[i + 1] - self._s[i])
        s = min(max(0.05, best_s), max(0.10, self.length - 0.05))
        p0, _t, right = self.frame(s)
        lat = (px - p0.x) * right.x + (py - p0.y) * right.y
        return s, lat

    # -- factories ---------------------------------------------------------

    @staticmethod
    def _cubic_bezier(p0: Vector, p1: Vector, p2: Vector, p3: Vector, n: int) -> list[Vector]:
        pts: list[Vector] = []
        for i in range(n):
            u = i / (n - 1)
            omu = 1.0 - u
            p = (
                (omu ** 3) * p0
                + 3.0 * (omu ** 2) * u * p1
                + 3.0 * omu * (u ** 2) * p2
                + (u ** 3) * p3
            )
            pts.append(p)
        return pts

    @classmethod
    def generate(cls, path_type: str, length: float, rng: random.Random, ds: float) -> "PathSpline":
        """Randomised centreline: straight, gentle bend, S-curve, or 90° corner."""
        L = float(length)
        if path_type == "straight":
            pts = [Vector((0.0, 0.0, 0.0)), Vector((0.0, L, 0.0))]
            return cls(pts, ds=ds)

        if path_type == "gentle_curve":
            bend = rng.uniform(8.0, 20.0) * rng.choice((-1.0, 1.0))
            p0 = Vector((0.0, 0.0, 0.0))
            p1 = Vector((bend * 0.15, L * 0.30, 0.0))
            p2 = Vector((bend, L * 0.65, 0.0))
            p3 = Vector((bend * 0.55, L, 0.0))
            return cls(cls._cubic_bezier(p0, p1, p2, p3, n=80), ds=ds)

        if path_type == "s_curve":
            a = rng.uniform(10.0, 18.0) * rng.choice((-1.0, 1.0))
            p0 = Vector((0.0, 0.0, 0.0))
            p1 = Vector((a, L * 0.22, 0.0))
            p2 = Vector((-a, L * 0.62, 0.0))
            p3 = Vector((0.15 * a, L, 0.0))
            return cls(cls._cubic_bezier(p0, p1, p2, p3, n=100), ds=ds)

        # corner_90 — two straights joined by a circular fillet.
        sign = rng.choice((-1.0, 1.0))
        r = rng.uniform(5.5, 8.0)
        L1 = max(r + 4.0, L * 0.45)
        L2 = max(r + 4.0, L - L1)
        pts = [Vector((0.0, 0.0, 0.0)), Vector((0.0, L1 - r, 0.0))]
        # Fillet centre sits to the *inside* of the turn.
        cx, cy = sign * r, L1 - r
        n_arc = 16
        for i in range(1, n_arc + 1):
            # Incoming heading +Y; turn toward +X (sign=+1) or −X.
            # Angle at start (point (0, L1-r)): π if sign=+1, 0 if sign=−1.
            # Angle at end   (point (sign*r, L1)): π/2.
            if sign > 0.0:
                ang0, ang1 = math.pi, math.pi * 0.5
            else:
                ang0, ang1 = 0.0, math.pi * 0.5
            ang = ang0 + (ang1 - ang0) * (i / n_arc)
            pts.append(Vector((cx + r * math.cos(ang), cy + r * math.sin(ang), 0.0)))
        pts.append(Vector((sign * (r + L2), L1, 0.0)))
        return cls(pts, ds=ds)


# ===========================================================================
# Street ribbon (keeps actors on pavement, never in a facade)
# ===========================================================================

@dataclass
class StreetCorridor:
    """Frenet bounds of the drivable / walkable ribbon.

    Facades sit at `|lateral| = road_half + sidewalk_w + 1.6`. The outer
    pavement edge is `road_half + sidewalk_w`. Clamping to
    `max_abs_lateral = pavement_outer - margin` keeps every actor centre
    on asphalt or sidewalk, so a curve cannot chord them through a wall.
    """

    road: PathSpline
    road_half: float
    sidewalk_w: float
    max_abs_lateral: float
    curb_height: float = 0.12

    def clamp_s(self, s: float) -> float:
        return min(max(0.05, float(s)), max(0.10, self.road.length - 0.05))

    def lateral_limit(self, pad: float, allow_sidewalk: bool) -> float:
        raw = self.max_abs_lateral if allow_sidewalk else (self.road_half - 0.05)
        return max(0.20, float(raw) - max(0.0, float(pad)))

    def ground_z(self, lateral: float) -> float:
        """Road plane is z=0; sidewalks sit on the curb.

        The kerb is a raised-cosine over ~0.35 m of lateral so a through-
        crosser does not pop 12 cm when they leave the asphalt.
        """
        curb = float(self.curb_height)
        if curb <= 1e-6:
            return 0.0
        edge = float(self.road_half) - 0.08
        blend = 0.18
        a = abs(float(lateral))
        if a >= edge + blend:
            return curb
        if a <= edge - blend:
            return 0.0
        u = (a - (edge - blend)) / max(2.0 * blend, 1e-3)
        u = u * u * (3.0 - 2.0 * u)
        return curb * u

    def confine(
        self,
        s: float,
        lateral: float,
        pad: float = 0.35,
        allow_sidewalk: bool = True,
    ) -> tuple[float, float]:
        lim = self.lateral_limit(pad, allow_sidewalk)
        lat = max(-lim, min(lim, float(lateral)))
        return self.clamp_s(s), lat

    def world_to_sl(self, p: Vector) -> tuple[float, float]:
        return self.road.project(p)

    def world(self, s: float, lateral: float, z: float) -> Vector:
        s, lateral = self.confine(s, lateral, pad=0.0, allow_sidewalk=True)
        return self.road.offset_point(s, lateral, z)


# ===========================================================================
# Poisson-disk sampling (Bridson 2007) on a (s, lateral) strip
# ===========================================================================

def poisson_disk_strip(
    length: float,
    half_width: float,
    radius: float,
    rng: random.Random,
    n_max: int,
    k_candidates: int = 20,
    s_min: float = 6.0,
    s_max: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Uniform-ish samples in [s_min, s_max] × [−half_width, half_width].

    Distance is Euclidean in the (s, lateral) parameterisation, which is a
    close approximation to world distance on a gently curved sidewalk.
    """
    s0 = float(s_min)
    s1 = float(length if s_max is None else s_max)
    if s1 - s0 < radius or n_max <= 0:
        return []

    cell = radius / math.sqrt(2.0)
    width = 2.0 * half_width
    cols = max(1, int(math.ceil((s1 - s0) / cell)))
    rows = max(1, int(math.ceil(width / cell)))
    grid: list[list[Optional[int]]] = [[None] * rows for _ in range(cols)]
    points: list[tuple[float, float]] = []
    active: list[int] = []

    def grid_xy(s: float, lat: float) -> tuple[int, int]:
        gx = int((s - s0) / cell)
        gy = int((lat + half_width) / cell)
        return min(max(gx, 0), cols - 1), min(max(gy, 0), rows - 1)

    def far_enough(s: float, lat: float) -> bool:
        gx, gy = grid_xy(s, lat)
        r2 = radius * radius
        for ix in range(max(0, gx - 2), min(cols, gx + 3)):
            for iy in range(max(0, gy - 2), min(rows, gy + 3)):
                j = grid[ix][iy]
                if j is None:
                    continue
                ps, plat = points[j]
                if (ps - s) ** 2 + (plat - lat) ** 2 < r2:
                    return False
        return True

    def insert(s: float, lat: float) -> int:
        idx = len(points)
        points.append((s, lat))
        gx, gy = grid_xy(s, lat)
        grid[gx][gy] = idx
        active.append(idx)
        return idx

    insert(rng.uniform(s0, s1), rng.uniform(-half_width, half_width))
    while active and len(points) < n_max:
        ai = rng.randrange(len(active))
        i = active[ai]
        ps, plat = points[i]
        found = False
        for _ in range(k_candidates):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            rad = rng.uniform(radius, 2.0 * radius)
            ns = ps + rad * math.cos(ang)
            nlat = plat + rad * math.sin(ang)
            if ns < s0 or ns > s1 or nlat < -half_width or nlat > half_width:
                continue
            if far_enough(ns, nlat):
                insert(ns, nlat)
                found = True
                break
        if not found:
            active.pop(ai)
    return points[:n_max]


# ===========================================================================
# Blender mesh / material primitives
# ===========================================================================

def _link(obj: bpy.types.Object, collection: bpy.types.Collection) -> bpy.types.Object:
    collection.objects.link(obj)
    return obj


def ensure_collection(name: str, scene: bpy.types.Scene) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in scene.collection.children:
        # May already be nested; only link if orphaned.
        orphan = True
        for child in scene.collection.children:
            if child == col:
                orphan = False
                break
        if orphan:
            try:
                scene.collection.children.link(col)
            except RuntimeError:
                pass
    return col


def make_principled(name: str, color: tuple[float, float, float], roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    col4 = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = col4
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    # Ground ribbons are single-sided; never cull the camera-facing side.
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    return mat


def assign_mat(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.data is None:
        return
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def create_mesh(
    name: str,
    verts: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    collection: bpy.types.Collection,
    location: Vector,
    mat: Optional[bpy.types.Material] = None,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    _link(obj, collection)
    if mat is not None:
        assign_mat(obj, mat)
    return obj


def create_box(
    name: str,
    size: tuple[float, float, float],
    location: Vector,
    collection: bpy.types.Collection,
    mat: Optional[bpy.types.Material] = None,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    verts = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5),
        (0, 4, 5, 1), (1, 5, 6, 2),
        (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    obj = create_mesh(name, verts, faces, collection, location, mat)
    if rotation_z:
        obj.rotation_euler[2] = rotation_z
    _box_project_uv(obj)
    return obj


def create_z_cylinder(
    name: str,
    radius: float,
    z_top: float,
    z_bottom: float,
    collection: bpy.types.Collection,
    location: Vector,
    mat: Optional[bpy.types.Material] = None,
    segments: int = 16,
    cap_top: bool = True,
    cap_bottom: bool = True,
) -> bpy.types.Object:
    """Cylinder along local +Z. ``z_top`` / ``z_bottom`` are local metres."""
    n = max(8, int(segments))
    r = max(0.05, float(radius))
    zt = float(z_top)
    zb = float(z_bottom)
    if zt < zb:
        zt, zb = zb, zt
    verts: list[tuple[float, float, float]] = []
    for z in (zb, zt):
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((r * math.cos(a), r * math.sin(a), z))
    faces: list[tuple[int, ...]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    if cap_bottom:
        faces.append(tuple(range(n - 1, -1, -1)))
    if cap_top:
        faces.append(tuple(range(n, 2 * n)))
    return create_mesh(name, verts, faces, collection, location, mat)


def _box_project_uv(obj: bpy.types.Object) -> None:
    """Per-face planar UV in metres (dominant-axis). Materials prefer Object
    coords; this is the fallback so a stray UV-mapped graph still tiles."""
    mesh = obj.data
    if mesh is None or mesh.uv_layers:
        return
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        n = poly.normal
        ax = 0 if abs(n.x) >= abs(n.y) and abs(n.x) >= abs(n.z) else (
            1 if abs(n.y) >= abs(n.z) else 2
        )
        for li in poly.loop_indices:
            v = mesh.vertices[mesh.loops[li].vertex_index].co
            if ax == 0:
                uv_layer.data[li].uv = (v.y, v.z)
            elif ax == 1:
                uv_layer.data[li].uv = (v.x, v.z)
            else:
                uv_layer.data[li].uv = (v.x, v.y)


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "polygons"):
        return obj
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


# ===========================================================================
# Linked-data instancing
# ===========================================================================

class MeshLibrary:
    """Cache of mesh and material datablocks shared by many objects.

    Scattering trees, grass, and debris with :func:`create_mesh` would call
    ``bpy.data.meshes.new`` once per object. A thousand grass clumps then
    means a thousand unique vertex buffers, which is exactly how a scatter
    pass exhausts GPU memory even though the visible geometry is trivial.

    Here the *datablock* is keyed by shape, and every object is a linked
    duplicate of it (``bpy.data.objects.new(name, shared_mesh)``). Visual
    variety comes from a small number of shape keys multiplied by per-object
    scale and yaw, which cost nothing: the vertex buffer is uploaded once.

    Materials are cached the same way. A shared mesh carries its material on
    the *mesh* rather than the object, so all instances of a key render with
    the same slot without a per-object copy.
    """

    __slots__ = ("_meshes", "_mats", "built", "reused")

    def __init__(self) -> None:
        self._meshes: dict[str, Any] = {}
        self._mats: dict[str, Any] = {}
        self.built = 0
        self.reused = 0

    def mesh(self, key: str, builder: Any, smooth: bool = False) -> Any:
        """Get-or-build the mesh datablock for `key`. `builder() -> (verts, faces)`."""
        cached = self._meshes.get(key)
        if cached is not None:
            self.reused += 1
            return cached
        verts, faces = builder()
        mesh = bpy.data.meshes.new(f"lib_{key}")
        mesh.from_pydata(verts, [], faces)
        mesh.validate()
        mesh.update()
        if smooth:
            for poly in mesh.polygons:
                poly.use_smooth = True
        self._meshes[key] = mesh
        self.built += 1
        return mesh

    def material(self, key: str, builder: Any) -> Any:
        cached = self._mats.get(key)
        if cached is None:
            cached = builder()
            self._mats[key] = cached
        return cached

    def instance(
        self,
        name: str,
        key: str,
        builder: Any,
        collection: bpy.types.Collection,
        location: Vector,
        mat: Optional[bpy.types.Material] = None,
        rotation_z: float = 0.0,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        smooth: bool = False,
    ) -> bpy.types.Object:
        mesh = self.mesh(key, builder, smooth=smooth)
        if mat is not None and not mesh.materials:
            mesh.materials.append(mat)
        obj = bpy.data.objects.new(name, mesh)
        obj.location = location
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (0.0, 0.0, float(rotation_z))
        obj.scale = (float(scale[0]), float(scale[1]), float(scale[2]))
        _link(obj, collection)
        return obj

    def stats(self) -> str:
        return f"{self.built} unique mesh datablocks, {self.reused} linked instances"


def purge_orphans(max_passes: int = 8) -> int:
    """Delete every zero-user datablock, following dependency chains.

    Purging is iterative because freeing a mesh can orphan its material,
    which can orphan its node group. ``do_recursive`` handles most of that
    in one pass, but the loop guarantees convergence across the Blender
    versions where the flag is missing or partial. Without this, thousands
    of episodes in one process leak every material and mesh ever built.
    """
    total = 0
    for _ in range(max(1, int(max_passes))):
        removed = 0
        try:
            removed = int(
                bpy.data.orphans_purge(
                    do_local_ids=True, do_linked_ids=True, do_recursive=True
                )
            )
        except TypeError:
            try:
                removed = int(bpy.data.orphans_purge())
            except Exception:
                removed = 0
        except Exception:
            removed = 0
        total += removed
        if removed <= 0:
            break
    return total


def yaw_from_xy(direction: Vector) -> float:
    """Rotation about world +Z that maps local +Y onto `direction` in XY.

    Blender Euler XYZ, Z-only: :math:`R_z(\\theta)\\,(0,1,0) = (-\\sin\\theta,\\,\\cos\\theta)`.
    Matching a unit vector :math:`(d_x, d_y)` gives :math:`\\theta = \\mathrm{atan2}(-d_x, d_y)`.

    The old ``atan2(d_x, d_y)`` is correct only when the path is exactly +Y.
    On a curve (or a lateral jaywalk along ``right``) it faces the *opposite*
    horizontal, so walkers moonwalk, cars sit across the lane, and dashes
    cut the asphalt.
    """
    dx = float(direction.x)
    dy = float(direction.y)
    if dx * dx + dy * dy < 1e-16:
        return 0.0
    return math.atan2(-dx, dy)


def look_along(obj: bpy.types.Object, tangent: Vector) -> None:
    """Yaw the object so its local +Y faces `tangent` (Z-up)."""
    t = Vector((tangent.x, tangent.y, 0.0))
    if t.length < 1e-8:
        return
    t.normalize()
    if hasattr(obj, "rotation_mode"):
        obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, yaw_from_xy(t))


def _smootherstep(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _smootherstep_du(u: float) -> float:
    """d/du of smootherstep. Zero at the endpoints, peak in the middle."""
    u = min(1.0, max(0.0, float(u)))
    d = 1.0 - u
    return 30.0 * u * u * d * d


def _angle_lerp(a: float, b: float, k: float) -> float:
    """Lerp radians along the short arc. ``k`` in [0, 1]."""
    k = min(1.0, max(0.0, float(k)))
    d = (float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi
    return float(a) + d * k


def heading_from_tangent(tangent: Vector) -> float:
    return yaw_from_xy(tangent)


# ===========================================================================
# Actor wrapper
# ===========================================================================

@dataclass
class Actor:
    """Python-side kinematics twin of a Blender object.

    Dynamic actors live in Frenet coordinates `(s, lateral)` on the road
    spline. Lateral drift (`lat_target` / `lat_speed`) is how jaywalkers,
    crossing cars, and cut-ins move — never a straight world-space velocity,
    which would tunnel through buildings on a curve.
    """

    obj: bpy.types.Object
    instance_id: str
    class_name: str
    category: str
    velocity: Vector = field(default_factory=lambda: Vector((0.0, 0.0, 0.0)))
    speed: float = 0.0
    follow_spline: Optional[PathSpline] = None
    s: float = 0.0
    lateral: float = 0.0
    origin_z: float = 0.0
    behavior: str = "cruise"
    swerve_t: Optional[float] = None
    stop_t: Optional[float] = None
    hold_velocity: Optional[Vector] = None
    stopped: bool = False
    annotatable: bool = True
    threat_mode: str = "volume"  # "volume" | "footprint"
    # Sub-object whose AABB defines the *threat* point and Frenet extents,
    # when that differs from the silhouette used for the 2-D box. A street
    # tree is the motivating case: the 2-D box must cover the canopy, but
    # the thing a walker collides with is the trunk, and taking the threat
    # point from the canopy AABB would put the hazard 1.5 m off the path.
    threat_obj: Any = None
    gait: Any = None
    wheels: list = field(default_factory=list)  # (obj, radius_m)
    lat_speed: float = 0.0
    lat_target: Optional[float] = None
    lat_ease_t0: Optional[float] = None
    lat_ease_dur: float = 0.0
    lat_ease_from: Optional[float] = None
    heading_blend_s: float = 1.60
    _heading_yaw: Optional[float] = field(default=None, repr=False)
    corridor_pad: float = 0.35
    allow_sidewalk: bool = True
    # Organic path noise: fBm lateral drift so background crowds wander
    # instead of running on rails. Amplitude is metres, rate is Hz.
    wander_amp: float = 0.0
    wander_rate: float = 0.20
    wander_noise: Any = None
    # Deterministic sinusoidal weave (dangerous cyclist, out-of-control car).
    # Kept separate from the fBm drift because a weave must be reproducible
    # and bounded: its amplitude is exactly weave_amp, which is what lets an
    # injector aim the swing at the walker's line.
    weave_amp: float = 0.0
    weave_hz: float = 0.35
    weave_phase: float = 0.0
    _wander_prev: Optional[float] = field(default=None, repr=False)
    # Speed modulation for erratic actors (fraction of nominal speed).
    speed_noise: Any = None
    speed_amp: float = 0.0
    # Per-actor handle on the projection AABB cache, so the annotation loop
    # does not re-hash the object pointer once per actor per frame.
    _bounds: Any = field(default=None, repr=False)
    # Smooth turn onto the road: blend (speed, lat_speed) → (post_speed, post_lat_*).
    turn_t: Optional[float] = None
    turn_dt: float = 0.85
    post_speed: Optional[float] = None
    post_lat_speed: float = 0.0
    post_lat_target: Optional[float] = None
    _prev_loc: Optional[Vector] = field(default=None, repr=False)
    # Branch joints + leaf sprays. Trunk pose and Actor.velocity stay 0.
    foliage: list = field(default_factory=list)
    wind: Any = None
    # Backing vehicle: mesh faces −velocity so the boot, not the bumper, leads.
    look_flip: bool = False

    def world_location(self) -> Vector:
        return self.obj.matrix_world.translation.copy()

    def _tick_visuals(self, t: float, dt: float) -> None:
        if self.gait is not None:
            self.gait.apply(t, self.velocity.length if self.velocity.length > 1e-6 else abs(self.speed), self.stopped)
        if self.wheels and dt > 1e-8:
            dist = self.velocity.length * dt
            for wheel, radius in self.wheels:
                # Bottom of a +X-axis wheel must move −Y (local) for +Y travel.
                wheel.rotation_euler[0] -= dist / max(float(radius), 0.05)
        if self.foliage:
            env = gust = dx = 0.0
            dy = 1.0
            if self.wind is not None:
                env, gust, dx, dy = self.wind.state(t)
            if env > 1e-4:
                for part in self.foliage:
                    ax, ay, az = foliage_wind_euler(
                        t, part.phase, part.flutter, part.kind, env, gust, dx, dy,
                        getattr(part, "flex", 1.0),
                    )
                    rx, ry, rz = part.rest
                    part.obj.rotation_euler = (rx + ax, ry + ay, rz + az)

    def _lat_vel(self, t: float) -> float:
        """Signed d(lateral)/dt this frame (0 once the target is reached).

        When ``lat_ease_dur`` is set, the path from ``lat_ease_from`` to
        ``lat_target`` is a smootherstep over that window instead of a
        constant rate (the old integrator snapped heading with the velocity).
        """
        if self.lat_target is None:
            return float(self.lat_speed)
        if self.swerve_t is not None and t < self.swerve_t:
            return 0.0
        if self.lat_ease_dur > 1e-3 and self.lat_ease_from is not None:
            t0 = float(self.lat_ease_t0 or 0.0)
            if t < t0:
                return 0.0
            span = float(self.lat_target) - float(self.lat_ease_from)
            u = (t - t0) / max(float(self.lat_ease_dur), 1e-3)
            if u >= 1.0:
                delta = float(self.lat_target) - self.lateral
                if abs(delta) < 1e-4:
                    return 0.0
                return math.copysign(min(abs(delta) / 0.03, 6.0), delta)
            return span * _smootherstep_du(u) / max(float(self.lat_ease_dur), 1e-3)
        delta = float(self.lat_target) - self.lateral
        if abs(delta) < 1e-4:
            return 0.0
        return math.copysign(abs(self.lat_speed), delta)

    def _apply_heading(self, heading: Vector, dt: float) -> None:
        """Yaw the mesh toward the path, blended over ``heading_blend_s``."""
        desired = yaw_from_xy(Vector((heading.x, heading.y, 0.0)))
        tau = max(0.18, float(self.heading_blend_s) / 3.0)
        if self._heading_yaw is None or dt <= 1e-8:
            yaw = desired
        else:
            k = 1.0 - math.exp(-float(dt) / tau)
            yaw = _angle_lerp(self._heading_yaw, desired, k)
        self._heading_yaw = yaw
        if hasattr(self.obj, "rotation_mode"):
            self.obj.rotation_mode = "XYZ"
        self.obj.rotation_euler = (0.0, 0.0, yaw)

    def _lat_disturb(self, t: float) -> float:
        """Total lateral *displacement* from drift plus weave, in metres."""
        d = 0.0
        if self.wander_noise is not None and self.wander_amp > 1e-6:
            d += self.wander_amp * self.wander_noise.fbm(
                t * self.wander_rate, octaves=3, persistence=0.5, lacunarity=2.0
            )
        if self.weave_amp > 1e-6:
            d += self.weave_amp * math.sin(
                2.0 * math.pi * self.weave_hz * t + self.weave_phase
            )
        return d

    def _wander_rate(self, t: float, dt: float) -> float:
        """d(lateral)/dt from the lateral disturbance, in m/s.

        The disturbance is defined as a displacement, and the integrator
        consumes a rate, so this differentiates it. The first call seeds the
        previous sample and returns zero: differencing against an unset
        baseline would teleport the actor by the full amplitude on frame 0.
        """
        if dt <= 1e-8:
            return 0.0
        if self.weave_amp <= 1e-6 and (
            self.wander_noise is None or self.wander_amp <= 1e-6
        ):
            return 0.0
        d = self._lat_disturb(t)
        prev = self._wander_prev
        self._wander_prev = d
        if prev is None:
            return 0.0
        return (d - prev) / dt

    def _speed_scale(self, t: float) -> float:
        """Multiplicative speed jitter for erratic actors. Never negative."""
        if self.speed_noise is None or self.speed_amp <= 1e-6:
            return 1.0
        n = self.speed_noise.fbm(t * 0.55, octaves=2, persistence=0.5, lacunarity=2.0)
        return max(0.0, 1.0 + self.speed_amp * n)

    def _commanded_rates(self, t: float) -> tuple[float, float]:
        """(ds/dt, d(lateral)/dt), with a smoothstep blend if a turn is armed."""
        vlat_pre = self._lat_vel(t)
        if self.post_speed is None or self.turn_t is None or t < self.turn_t:
            return float(self.speed), vlat_pre
        u = (t - float(self.turn_t)) / max(float(self.turn_dt), 1e-3)
        u = min(1.0, max(0.0, u))
        u = u * u * (3.0 - 2.0 * u)
        vlat_post = 0.0
        if self.post_lat_target is not None:
            d = float(self.post_lat_target) - self.lateral
            if abs(d) > 1e-3:
                vlat_post = math.copysign(abs(self.post_lat_speed), d)
        else:
            vlat_post = float(self.post_lat_speed)
        ds = (1.0 - u) * float(self.speed) + u * float(self.post_speed)
        dlat = (1.0 - u) * vlat_pre + u * vlat_post
        return ds, dlat

    def update(self, t: float, dt: float, corridor: Optional[StreetCorridor] = None) -> None:
        if self.category == "static" or self.stopped:
            self.velocity = Vector((0.0, 0.0, 0.0))
            self._tick_visuals(t, dt)
            return

        if self.stop_t is not None and t >= self.stop_t:
            self.stopped = True
            self.velocity = Vector((0.0, 0.0, 0.0))
            self._tick_visuals(t, dt)
            return

        if self.follow_spline is not None:
            ds, vlat = self._commanded_rates(t)
            ds *= self._speed_scale(t)
            s_prev = self.s
            lat_prev = self.lateral
            self.s = self.s + ds * dt
            if abs(vlat) > 1e-8:
                step = vlat * dt
                lat_goal = self.lat_target
                if (
                    self.post_speed is not None
                    and self.turn_t is not None
                    and t >= float(self.turn_t)
                    and self.post_lat_target is not None
                ):
                    u = (t - float(self.turn_t)) / max(float(self.turn_dt), 1e-3)
                    if u >= 1.0:
                        lat_goal = self.post_lat_target
                if lat_goal is not None:
                    delta = float(lat_goal) - self.lateral
                    if abs(delta) <= abs(step):
                        self.lateral = float(lat_goal)
                        vlat = 0.0
                    else:
                        self.lateral += step
                else:
                    self.lateral += step

            # Organic drift is layered on top of the commanded lateral so it
            # perturbs a crossing without cancelling the target it is
            # crossing toward. It is folded into the reported velocity, so
            # TTC / CPA still see the true instantaneous motion.
            v_wander = self._wander_rate(t, dt)
            if v_wander != 0.0:
                self.lateral += v_wander * dt

            if corridor is not None:
                self.s, self.lateral = corridor.confine(
                    self.s, self.lateral, self.corridor_pad, self.allow_sidewalk
                )
            else:
                self.s = min(max(0.05, self.s), max(0.10, self.follow_spline.length - 0.05))

            # Velocity is the pose that actually stuck, not the command.
            # Confine used to zero the step then leave vlat in velocity —
            # the mesh froze against the kerb while TTC thought it was
            # still sliding, and heading snapped 90°.
            if dt > 1e-8:
                ds_act = (self.s - s_prev) / dt
                vlat_act = (self.lateral - lat_prev) / dt
            else:
                ds_act = ds
                vlat_act = vlat + v_wander

            p, tan, right = self.follow_spline.frame(self.s)
            loc = p + right * self.lateral
            if corridor is not None:
                loc.z = self.origin_z + corridor.ground_z(self.lateral)
            else:
                loc.z = self.origin_z
            self.obj.location = loc
            self.velocity = tan * ds_act + right * vlat_act
            if self.velocity.length > 0.12:
                heading = self.velocity
            elif self._heading_yaw is None:
                heading = tan if ds_act >= 0.0 else -tan
            else:
                heading = None
            if heading is not None:
                if self.look_flip:
                    heading = Vector((-heading.x, -heading.y, -heading.z))
                self._apply_heading(Vector((heading.x, heading.y, 0.0)), dt)
            self._tick_visuals(t, dt)
            return

        # Last-resort world-space step (should not be used for street actors).
        if self.hold_velocity is not None and (self.swerve_t is None or t >= self.swerve_t):
            self.obj.location = self.obj.location + self.hold_velocity * dt
            self.velocity = self.hold_velocity.copy()
            if corridor is not None:
                s, lat = corridor.world_to_sl(self.obj.location)
                s, lat = corridor.confine(s, lat, self.corridor_pad, self.allow_sidewalk)
                q = corridor.road.offset_point(s, lat, self.obj.location.z)
                self.obj.location = q
            if self.hold_velocity.length > 1e-6:
                self._apply_heading(self.hold_velocity, dt)
            self._tick_visuals(t, dt)
            return

        self.obj.location = self.obj.location + self.velocity * dt
        self._tick_visuals(t, dt)

    def finite_velocity(self, dt: float) -> Vector:
        loc = self.world_location()
        if self._prev_loc is None or dt <= 1e-8:
            self._prev_loc = loc.copy()
            return self.velocity.copy()
        v = (loc - self._prev_loc) / dt
        self._prev_loc = loc.copy()
        return v


# ===========================================================================
# Scene reset + lighting / weather
# ===========================================================================

def reset_blender_scene() -> bpy.types.Scene:
    """Wipe the default cube/light/camera and leftover datablocks."""
    # Ensure we are in object mode if a leftover context exists.
    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (
        bpy.data.meshes,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.materials,
        bpy.data.curves,
        bpy.data.worlds,
    ):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)

    scene = bpy.context.scene
    # Drop leftover child collections (keep the scene master).
    for child in list(scene.collection.children):
        scene.collection.children.unlink(child)
    # Collections themselves are datablocks; an unlinked one still holds a
    # reference to everything it contained until it is purged.
    for block in list(bpy.data.collections):
        if block.users == 0:
            bpy.data.collections.remove(block)
    # The explicit sweeps above only catch first-order orphans. Node groups,
    # images, and node trees hang off materials and need the recursive pass.
    purge_orphans()
    return scene


def release_episode(state: Optional["WorldState"] = None) -> int:
    """Tear an episode down and purge every datablock it owned.

    Called after the last frame is written, so the peak memory of episode
    *n + 1* does not include the corpse of episode *n*. Returns the number
    of datablocks freed, which the episode log prints as a leak check: a
    healthy pipeline reports a large number here and a flat RSS across
    thousands of episodes.
    """
    if state is not None:
        # Break the Python-side cycles first (Actor → obj → parent → Actor),
        # otherwise the objects still have users when the purge runs.
        for actor in list(state.actors):
            actor.obj = None  # type: ignore[assignment]
            actor.gait = None
            actor.wheels = []
            actor.follow_spline = None
            actor.threat_obj = None
            actor.foliage = []
            actor.wind = None
            actor._bounds = None
            actor.wander_noise = None
            actor.speed_noise = None
        state.actors.clear()
        state.collections.clear()
        state.materials.clear()

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    for child in list(scene.collection.children):
        scene.collection.children.unlink(child)
    scene.world = None
    return purge_orphans()


def _rand_color(
    rng: random.Random,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float]:
    lo, hi = bounds
    return (
        rng.uniform(lo[0], hi[0]),
        rng.uniform(lo[1], hi[1]),
        rng.uniform(lo[2], hi[2]),
    )


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    w = [max(0.0, float(weights[k])) for k in keys]
    total = sum(w)
    if total <= 0.0:
        return keys[0]
    x = rng.uniform(0.0, total)
    acc = 0.0
    for k, wi in zip(keys, w):
        acc += wi
        if x <= acc:
            return k
    return keys[-1]


def choose_environment(cfg: dict, rng: random.Random) -> dict:
    """Draw the episode's lighting, weather, dappling, and chaos dial.

    Six lighting states, not four. ``harsh_glare`` and ``overcast`` are the
    two that break a detector trained only on pleasant weather: one destroys
    contrast by saturating it, the other by removing it.
    """
    import os

    dr = cfg["domain_randomization"]
    forced = os.environ.get("BTP_LIGHTING", "").strip().lower()
    if forced in dr["lighting_weights"]:
        lighting = forced
    else:
        lighting = _weighted_choice(rng, dr["lighting_weights"])
    weather = _weighted_choice(rng, dr["weather_weights"])
    c_lo, c_hi = dr.get("material_chaos", (0.0, 0.0))
    wind_weights = dict(dr.get("wind_weights") or {"breeze": 1.0})
    forced_wind = os.environ.get("BTP_WIND", "").strip().lower()
    if forced_wind in wind_weights:
        wind = forced_wind
    else:
        wind = _weighted_choice(rng, wind_weights)
    w_lo, w_hi = (dr.get("wind_strength") or {}).get(wind, (0.28, 0.55))
    return {
        "lighting": lighting,
        "weather": weather,
        "elev_deg": rng.uniform(*dr["sun_elevation_deg"][lighting]),
        "azim_deg": rng.uniform(*dr["sun_azimuth_deg"]),
        "energy": rng.uniform(*dr["sun_energy"][lighting]),
        "dappled": rng.random() < float(dr.get("dappled_prob", 0.0)),
        "chaos": float(rng.uniform(float(c_lo), float(c_hi))),
        "wind": wind,
        "wind_strength": float(rng.uniform(float(w_lo), float(w_hi))),
        "wind_dir_deg": float(rng.uniform(0.0, 360.0)),
    }


def glare_azimuth_deg(tangent: Vector) -> float:
    """Sun azimuth that puts the disc in front of a walker facing `tangent`.

    A Blender sun with ``rotation_euler = (90° − e, 0, a)`` emits along

        d = (−sin a · cos e,  cos a · cos e,  −sin e)

    so the disc sits in the direction ``−d``. To place it down the walker's
    line of sight we need ``−d_xy ∝ tangent_xy``, i.e. ``d_xy ∝ −tangent_xy``:

        −sin a = −T_x  and  cos a = −T_y   ⇒   a = atan2(T_x, −T_y)
    """
    tx, ty = float(tangent.x), float(tangent.y)
    if tx * tx + ty * ty < 1e-12:
        return 0.0
    return math.degrees(math.atan2(tx, -ty)) % 360.0


def _apply_view_transform(scene: bpy.types.Scene, lighting: str) -> None:
    vs = getattr(scene, "view_settings", None)
    if vs is None:
        return
    try:
        vs.view_transform = "AgX"
    except Exception:
        pass
    try:
        vs.look = "None"
    except Exception:
        pass
    # Night + AgX crushes to black unless we lift exposure; noon needs less.
    # harsh_glare is pulled *down* so AgX rolls the highlights off into a
    # bloomed white instead of clipping the whole frame to paper.
    vs.exposure = {
        "night": 0.30,
        "dawn": 0.10,
        "dusk": 0.08,
        "noon": 0.00,
        "harsh_glare": -0.45,
        "overcast": 0.12,
    }.get(lighting, 0.0)
    if hasattr(vs, "gamma"):
        vs.gamma = 1.0


def apply_domain_randomization(
    scene: bpy.types.Scene,
    cfg: dict,
    rng: random.Random,
    lights_col: bpy.types.Collection,
    lamp_objects: list[bpy.types.Object],
    env: Optional[dict] = None,
) -> dict[str, str]:
    """Multiple-scattering sky + key sun + opposite fill + city-glow night."""
    env = env or choose_environment(cfg, rng)
    lighting = env["lighting"]
    weather = env["weather"]
    elev = float(env["elev_deg"])
    azim = float(env["azim_deg"])
    energy = float(env["energy"])
    night = lighting == "night"
    glare = lighting == "harsh_glare"
    overcast = lighting == "overcast"

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    if night:
        # Pitch black: the sun is a faint moon and the street is carried
        # almost entirely by the sparse, tinted streetlamps.
        sun_data.energy = max(float(energy) * 3.0, 0.25)
        sun_data.color = (0.42, 0.50, 0.82)
    elif glare:
        # Low, enormous, and warm. Nothing else in the frame survives it.
        sun_data.energy = float(energy)
        sun_data.color = (1.0, 0.86, 0.66)
    elif overcast:
        # The "sun" is the whole cloud deck: dim, white, and enormous, which
        # is what makes the shadows vanish rather than merely soften.
        sun_data.energy = float(energy)
        sun_data.color = (0.92, 0.94, 1.0)
    else:
        # Energy ∝ sin(elevation): dawn is warm and dimmer, noon is white.
        elev_w = max(0.18, math.sin(math.radians(max(elev, 0.0))))
        sun_data.energy = max(float(energy) * 1.35, 4.0) * (0.55 + 0.45 * elev_w)
        t = max(0.0, min(1.0, elev / 55.0))
        sun_data.color = (1.0, 0.68 + 0.30 * t, 0.40 + 0.55 * t)
    if hasattr(sun_data, "angle"):
        # Angular diameter drives shadow softness: a 0.53° disc is a crisp
        # sunny day, 60° is an overcast sky with no usable shadow terminator.
        sun_data.angle = math.radians(
            60.0 if overcast else (1.4 if night else (0.4 if glare else 0.53))
        )
    if hasattr(sun_data, "use_shadow"):
        sun_data.use_shadow = not overcast
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.rotation_euler = (math.radians(90.0 - elev), 0.0, math.radians(azim))
    sun_obj.location = Vector((0.0, 0.0, 40.0))
    _link(sun_obj, lights_col)

    fill = bpy.data.lights.new("SkyFill", type="SUN")
    if night:
        fill.energy = 0.35
    elif overcast:
        fill.energy = 5.5
    elif glare:
        fill.energy = 1.2
    else:
        fill.energy = 3.6 if lighting == "noon" else 2.4
    if night:
        fill.color = (0.30, 0.36, 0.62)
    elif overcast:
        fill.color = (0.86, 0.89, 0.96)
    else:
        fill.color = (0.70, 0.80, 0.95)
    if hasattr(fill, "angle"):
        fill.angle = math.radians(40.0)
    if hasattr(fill, "use_shadow"):
        fill.use_shadow = False
    fill_obj = bpy.data.objects.new("SkyFill", fill)
    fill_obj.rotation_euler = (math.radians(58.0), 0.0, math.radians((azim + 180.0) % 360.0))
    _link(fill_obj, lights_col)

    # Cool ground bounce so the shady side of a facade is not a black slab.
    bounce = bpy.data.lights.new("GroundBounce", type="SUN")
    if night:
        bounce.energy = 0.10
    elif overcast:
        bounce.energy = 1.6
    else:
        bounce.energy = 1.1
    bounce.color = (0.22, 0.22, 0.26) if night else (0.55, 0.50, 0.42)
    if hasattr(bounce, "angle"):
        bounce.angle = math.radians(60.0)
    if hasattr(bounce, "use_shadow"):
        bounce.use_shadow = False
    bounce_obj = bpy.data.objects.new("GroundBounce", bounce)
    bounce_obj.rotation_euler = (math.radians(165.0), 0.0, math.radians(azim))
    _link(bounce_obj, lights_col)

    apply_streetlamp_state(lamp_objects, cfg, night)

    _setup_world_shader(scene, lighting, weather, elev, azim, night)
    _apply_view_transform(scene, lighting)
    return {"lighting": lighting, "weather": weather}


def apply_streetlamp_state(
    lamp_objects: list[bpy.types.Object],
    cfg: dict,
    night: bool,
) -> None:
    """Switch the lamps on at night, honouring per-lamp burn-out and gain.

    Which lamps are dead and how bright the rest are is decided once at
    spawn time and stored on the object, because this runs twice per episode
    (once inside the lighting pass, once after `reveal_view_layer` un-hides
    everything) and the two calls must agree exactly.
    """
    base = float(cfg["world"]["streetlamp_energy_night"]) if night else 0.0
    for i, lamp in enumerate(lamp_objects):
        if lamp.data is None or lamp.type != "LIGHT":
            continue
        dead = bool(lamp.get("btp_lamp_dead", 0))
        gain = float(lamp.get("btp_lamp_gain", 1.0))
        lit = night and not dead
        lamp.data.energy = base * gain if lit else 0.0
        if hasattr(lamp.data, "use_shadow"):
            # Every other live lamp casts: keeps the shadow pool bounded.
            lamp.data.use_shadow = lit and (i % 2 == 0)
        lamp.hide_render = not lit
        lamp.hide_viewport = not lit


def _setup_world_shader(
    scene: bpy.types.Scene,
    lighting: str,
    weather: str,
    elev: float,
    azim: float,
    night: bool,
) -> None:
    """Sky Texture mixed with an analytic horizon/zenith gradient.

    For the world background, Geometry.Incoming is the camera ray. The
    elevation cosine μ = Incoming · (0,0,1) is 1 at the zenith and 0 on
    the horizon. A ColorRamp on μ is the discrete Hosek limb:

        L(μ) = L_zenith + (L_horizon − L_zenith) (1 − μ)^p

    At night the physical sky is nearly black, so the ramp carries a
    sodium-orange city glow on the horizon; by day it only tints.
    """
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    assert nt is not None
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    out.location = (640, 0)
    bg.location = (400, 40)
    sky.location = (-40, 160)
    for sky_id in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE", "PREETHAM"):
        try:
            sky.sky_type = sky_id
            break
        except Exception:
            continue
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = not night
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(max(elev, 2.0) if not night else 4.0)
    if hasattr(sky, "sun_rotation"):
        sky.sun_rotation = math.radians(azim)
    if hasattr(sky, "sun_intensity"):
        if night:
            sky.sun_intensity = 0.08
        elif lighting == "harsh_glare":
            sky.sun_intensity = 3.2  # the disc itself blooms out the frame
        elif lighting == "overcast":
            sky.sun_intensity = 0.10  # no visible disc through the deck
        else:
            sky.sun_intensity = 0.45 if weather == "heavy_smog" else 1.0
    if hasattr(sky, "air_density"):
        sky.air_density = 1.45 if weather == "heavy_smog" else (1.18 if weather == "light_fog" else 1.0)
    if hasattr(sky, "aerosol_density"):
        sky.aerosol_density = 3.2 if weather == "heavy_smog" else (1.5 if weather == "light_fog" else 1.0)
    if hasattr(sky, "dust_density"):
        sky.dust_density = 2.8 if weather == "heavy_smog" else (1.3 if weather == "light_fog" else 1.0)
    if hasattr(sky, "turbidity"):
        turb = 7.5 if weather == "heavy_smog" else (4.2 if weather == "light_fog" else 2.4)
        if lighting == "overcast":
            # High turbidity is what flattens the sky into a light box.
            turb = max(turb, 9.0)
        elif lighting == "harsh_glare":
            turb = max(turb, 5.0)  # haze around a low sun widens the glare
        sky.turbidity = turb
    if hasattr(sky, "ground_albedo"):
        sky.ground_albedo = 0.42 if lighting == "overcast" else 0.22

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-360, -80)
    zup = nt.nodes.new("ShaderNodeVectorMath")
    zup.location = (-160, -80)
    try:
        zup.operation = "DOT_PRODUCT"
    except Exception:
        pass
    if "Vector" in zup.inputs:
        # Incoming · world-up. Second vector socket is the constant (0,0,1).
        vecs = [s for s in zup.inputs if s.name == "Vector"]
        if len(vecs) >= 2:
            vecs[1].default_value = (0.0, 0.0, 1.0)
    try:
        nt.links.new(geom.outputs["Incoming"], zup.inputs[0])
    except Exception:
        pass

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (40, -80)
    if night:
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.18, 0.08, 0.04, 1.0)  # city glow
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.02, 0.03, 0.08, 1.0)  # zenith
        mid = ramp.color_ramp.elements.new(0.22)
        mid.color = (0.08, 0.05, 0.06, 1.0)
    elif lighting == "harsh_glare":
        # Washed-out warm horizon bleeding into a pale sky: the low sun is
        # scattering across the whole lower hemisphere.
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (1.00, 0.82, 0.55, 1.0)
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.62, 0.72, 0.92, 1.0)
        mid = ramp.color_ramp.elements.new(0.30)
        mid.color = (0.98, 0.90, 0.74, 1.0)
    elif lighting == "overcast":
        # Almost no gradient at all — that flatness is the whole point.
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.62, 0.64, 0.68, 1.0)
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.74, 0.76, 0.80, 1.0)
    elif lighting in ("dawn", "dusk"):
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.95, 0.42, 0.18, 1.0)
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.25, 0.38, 0.72, 1.0)
        mid = ramp.color_ramp.elements.new(0.28)
        mid.color = (0.85, 0.55, 0.35, 1.0)
    else:
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.55, 0.68, 0.88, 1.0)
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.18, 0.42, 0.88, 1.0)
    try:
        nt.links.new(zup.outputs["Value"], ramp.inputs["Fac"])
    except Exception:
        try:
            nt.links.new(zup.outputs[1], ramp.inputs["Fac"])
        except Exception:
            pass

    mix = nt.nodes.new("ShaderNodeMix")
    mix.location = (240, 40)
    try:
        mix.data_type = "RGBA"
    except Exception:
        pass
    mix_fac = {
        "night": 0.55,
        "dawn": 0.22,
        "dusk": 0.22,
        "harsh_glare": 0.30,
        "overcast": 0.78,  # the analytic flat grey dominates the sky model
    }.get(lighting, 0.12)
    fac_sock = None
    for s in mix.inputs:
        if s.name == "Factor" and s.type == "VALUE" and s.enabled:
            fac_sock = s
            break
    if fac_sock is not None:
        fac_sock.default_value = mix_fac
    # A = physical sky, B = analytic gradient.
    a_sock = b_sock = None
    for s in mix.inputs:
        if s.name == "A" and s.type == "RGBA" and s.enabled:
            a_sock = s
        if s.name == "B" and s.type == "RGBA" and s.enabled:
            b_sock = s
    try:
        if a_sock is not None:
            nt.links.new(sky.outputs["Color"], a_sock)
        if b_sock is not None:
            nt.links.new(ramp.outputs["Color"], b_sock)
        res = None
        for s in mix.outputs:
            if s.name == "Result" and s.type == "RGBA" and s.enabled:
                res = s
                break
        if res is not None:
            nt.links.new(res, bg.inputs["Color"])
        else:
            nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    except Exception:
        nt.links.new(sky.outputs["Color"], bg.inputs["Color"])

    if night:
        # Near-zero ambient: the void the streetlamps punch holes in.
        bg.inputs["Strength"].default_value = 0.045
    elif lighting == "overcast":
        # The sky *is* the light source in an overcast scene.
        bg.inputs["Strength"].default_value = 1.05
    elif lighting == "harsh_glare":
        bg.inputs["Strength"].default_value = 0.55
    elif lighting == "noon":
        bg.inputs["Strength"].default_value = 0.26 if weather == "clear" else 0.18
    else:
        bg.inputs["Strength"].default_value = 0.24 if weather != "heavy_smog" else 0.16
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def available_render_engines(scene: bpy.types.Scene) -> list[str]:
    prop = scene.render.bl_rna.properties.get("engine")
    if prop is None:
        return []
    return [item.identifier for item in prop.enum_items]


def pick_render_engine(scene: bpy.types.Scene, requested: str) -> str:
    avail = available_render_engines(scene)
    if requested in avail:
        return requested
    for cand in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"):
        if cand in avail:
            return cand
    return avail[0] if avail else requested


def reveal_view_layer(scene: bpy.types.Scene) -> None:
    """Collections created from Python can start excluded from the view layer."""

    def _walk(lc: Any) -> None:
        lc.exclude = False
        if hasattr(lc, "holdout"):
            lc.holdout = False
        if hasattr(lc, "indirect_only"):
            lc.indirect_only = False
        if hasattr(lc, "hide_viewport"):
            lc.hide_viewport = False
        for child in lc.children:
            _walk(child)

    try:
        _walk(bpy.context.view_layer.layer_collection)
    except Exception:
        pass
    for obj in scene.objects:
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_set(False)


def prepare_still_render(scene: bpy.types.Scene, png_compression: int = 1) -> None:
    """Blender 5.x starts with Sequencer + Compositor ON.

    The default VSE has zero strips. With ``use_sequencer=True`` the written
    PNG is the empty sequencer (pitch black) even though the 3D scene is fine
    — which is exactly what the annotation JSON vs. RGB mismatch showed.
    """
    scene.render.use_sequencer = False
    scene.render.use_compositing = False
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    scene.render.use_placeholder = False
    scene.render.image_settings.file_format = "PNG"
    if hasattr(scene.render.image_settings, "color_mode"):
        scene.render.image_settings.color_mode = "RGB"
    if hasattr(scene.render.image_settings, "compression"):
        scene.render.image_settings.compression = int(png_compression)


def _gpu_backend_report() -> str:
    """Which GL/Vulkan device EEVEE actually opened (set at process start)."""
    try:
        import gpu
        try:
            gpu.init()
        except Exception:
            pass
        plat = gpu.platform
        bits = []
        for fn in ("device_type_get", "renderer_get", "backend_type_get"):
            if hasattr(plat, fn):
                try:
                    bits.append(str(getattr(plat, fn)()))
                except Exception:
                    pass
        return " / ".join(bits) if bits else "unknown"
    except Exception as exc:
        return f"unavailable ({exc})"


def configure_eevee(scene: bpy.types.Scene, rcfg: dict) -> None:
    """EEVEE on Blender 4.x (EEVEE-Next) and 5.x (BLENDER_EEVEE)."""
    engine = pick_render_engine(scene, str(rcfg["engine"]))
    scene.render.engine = engine
    print(f"[render] engine={engine}  available={available_render_engines(scene)}")
    print(f"[render] gpu={_gpu_backend_report()}")

    scene.render.resolution_x = int(rcfg["resolution_x"])
    scene.render.resolution_y = int(rcfg["resolution_y"])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(rcfg["fps"])
    scene.render.image_settings.file_format = str(rcfg["filepath_format"])
    scene.render.image_settings.color_depth = str(rcfg["color_depth"])
    scene.render.film_transparent = bool(rcfg["film_transparent"])
    scene.render.use_file_extension = True
    scene.render.use_persistent_data = True
    scene.render.use_lock_interface = True
    scene.render.use_overwrite = True
    scene.render.use_placeholder = False
    prepare_still_render(scene, png_compression=int(rcfg.get("png_compression", 1)))

    eevee = getattr(scene, "eevee", None)
    if eevee is None:
        return

    def _set(name: str, value: Any) -> None:
        if hasattr(eevee, name):
            try:
                setattr(eevee, name, value)
            except Exception:
                pass

    taa = int(rcfg.get("taa_render_samples", 16))
    _set("taa_render_samples", taa)
    _set("taa_samples", taa)
    _set("use_taa_reprojection", True)
    _set("use_shadows", bool(rcfg.get("use_shadows", True)))
    _set("use_volumetric_shadows", False)
    _set("use_raytracing", bool(rcfg.get("use_raytracing", True)))
    # Blender 5 enum is SCREEN (not SCREEN_TRACE).
    _set("ray_tracing_method", "SCREEN")
    _set("use_fast_gi", True)
    _set("fast_gi_ray_count", int(rcfg.get("fast_gi_ray_count", 4)))
    _set("fast_gi_quality", float(rcfg.get("fast_gi_quality", 0.30)))
    _set("fast_gi_step_count", int(rcfg.get("fast_gi_step_count", 6)))
    _set("fast_gi_resolution", "2")
    _set("shadow_pool_size", str(rcfg.get("shadow_pool_size", "2048")))
    _set("gi_irradiance_pool_size", str(rcfg.get("gi_irradiance_pool_size", "32")))
    _set("indirect_light_intensity", 1.35)
    # Dense scatter (trees / grass) lives in the first 40 m of the walk.
    # Tight cascades keep the shadow map on the gait instead of wasting
    # pages on the far verge.
    _set("cascade_max_distance", float(rcfg.get("cascade_max_distance", 48.0)))
    _set("cascade_exponent", 0.85)
    _set("cascade_fade", 0.12)
    _set("shadow_max_resolution", "512")
    _set("volumetric_samples", int(rcfg.get("volumetric_samples", 16)))
    _set("volumetric_start", float(rcfg.get("volumetric_start", 0.1)))
    _set("volumetric_end", float(rcfg.get("volumetric_end", 80.0)))
    _set("volumetric_tile_size", str(rcfg.get("volumetric_tile_size", "8")))

    rt = getattr(eevee, "ray_tracing_options", None)
    if rt is not None:
        if hasattr(rt, "resolution_scale"):
            try:
                rt.resolution_scale = "2"
            except Exception:
                pass
        if hasattr(rt, "screen_trace_quality"):
            try:
                rt.screen_trace_quality = 0.25
            except Exception:
                pass
        if hasattr(rt, "use_denoise"):
            try:
                rt.use_denoise = True
            except Exception:
                pass


# ===========================================================================
# Geometry builders
# ===========================================================================

def _ribbon_mesh(
    name: str,
    spline: PathSpline,
    lat_left: float,
    lat_right: float,
    z: float,
    collection: bpy.types.Collection,
    mat: bpy.types.Material,
    z_amp: float = 0.0,
    rng: Optional[random.Random] = None,
) -> bpy.types.Object:
    """Extrude a constant-width strip along `spline` (quad strip).

    UVs are (s_metres / 4, lateral_01) so asphalt / pavers repeat every 4 m
    of walking distance rather than stretching over the whole street.

    ``z_amp`` is a visual only: a few millimetres of vertex jitter so a
    plaza's paving is not a perfectly flat slab. Actor ground height still
    comes from ``StreetCorridor.ground_z`` (the kerb step), not this mesh.
    """
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, ...]] = []
    span = max(1e-6, abs(lat_right - lat_left))
    amp = float(z_amp)
    for i, s in enumerate(spline._s):
        p, _t, r = spline.frame(s)
        l = p + r * lat_left
        rr = p + r * lat_right
        dz0 = dz1 = 0.0
        if amp > 1e-6 and rng is not None:
            dz0 = rng.uniform(-amp, amp)
            dz1 = rng.uniform(-amp, amp)
        verts.append((l.x, l.y, z + dz0))
        verts.append((rr.x, rr.y, z + dz1))
        uvs.append((s / 4.0, 0.0))
        uvs.append((s / 4.0, 1.0 * (abs(lat_right - lat_left) / span)))
        if i > 0:
            a = 2 * (i - 1)
            faces.append((a, a + 2, a + 3, a + 1))
    obj = create_mesh(name, verts, faces, collection, Vector((0, 0, 0)), mat)
    mesh = obj.data
    if mesh is not None and not mesh.uv_layers:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for loop in mesh.loops:
            uv_layer.data[loop.index].uv = uvs[loop.vertex_index]
    return obj


def _centerline_dash(
    name: str,
    spline: PathSpline,
    s0: float,
    s1: float,
    half_width: float,
    z: float,
    collection: bpy.types.Collection,
    mat: bpy.types.Material,
) -> bpy.types.Object:
    """Lane dash as a curved quad strip on the centreline (follows the Frenet frame)."""
    s0 = max(0.0, float(s0))
    s1 = min(float(s1), spline.length)
    if s1 - s0 < 0.15:
        s1 = s0 + 0.15
    n = max(2, int(math.ceil((s1 - s0) / 0.35)) + 1)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(n):
        s = s0 + (s1 - s0) * i / (n - 1)
        p, _t, r = spline.frame(s)
        a = p - r * half_width
        b = p + r * half_width
        verts.append((a.x, a.y, z))
        verts.append((b.x, b.y, z))
        if i > 0:
            k = 2 * (i - 1)
            faces.append((k, k + 2, k + 3, k + 1))
    return create_mesh(name, verts, faces, collection, Vector((0.0, 0.0, 0.0)), mat)


def offset_folds(
    spline: PathSpline,
    s: float,
    lat_near: float,
    lat_far: float,
    half_len: float,
    min_abs_lateral: float,
) -> bool:
    """True if a footprint at this offset folds back over the corridor.

    Offsetting a curve sideways by more than its radius of curvature makes
    the parallel curve self-intersect: on the *inside* of a 90° corner, a
    point placed 11 m out by the local Frenet frame at ``s`` comes back down
    on the far side of the road. ``spline.frame(s)`` cannot see this, because
    it only knows about ``s``; ``spline.project()`` can, because it asks
    where the point lies relative to the *whole* road.

    Symptom when unguarded: a building materialises across the pavement and
    the walker spends the episode nose-to-nose with a wall.

    The footprint is sampled at three arc stations and three lateral bands
    (the near face, the middle, and the far face), which is enough to catch
    a fold — the failure is monotonic in offset, so the far face folds first.
    """
    for ds in (-half_len, 0.0, half_len):
        ss = min(max(0.0, float(s) + ds), spline.length)
        for lt in (lat_near, 0.5 * (lat_near + lat_far), lat_far):
            p = spline.offset_point(ss, float(lt), z=0.0)
            _s2, lat2 = spline.project(p)
            if abs(lat2) < float(min_abs_lateral):
                return True
    return False


def _extrude_buildings(
    spline: PathSpline,
    lateral_sign: float,
    road_half: float,
    sidewalk_w: float,
    cfg: dict,
    rng: random.Random,
    collection: bpy.types.Collection,
    roughness: float,
    night: bool = False,
    gaps: Optional[list[tuple[float, float]]] = None,
) -> list[bpy.types.Object]:
    from materials import make_facade, make_facade_brick, make_facade_plaster

    objs: list[bpy.types.Object] = []
    gaps = list(gaps or [])
    s = 14.0
    setback = float(cfg.get("building_setback", 3.6))
    facade = road_half + sidewalk_w + setback
    facade_cache: dict[tuple, Any] = {}

    def _facade_mat(color: tuple[float, float, float]) -> Any:
        bucket = (
            round(color[0] * 4.0) / 4.0,
            round(color[1] * 4.0) / 4.0,
            round(color[2] * 4.0) / 4.0,
        )
        # Kind is part of the cache key so neighbouring lots can still differ.
        kind = rng.choice(("brick", "plaster", "windows"))
        key = (kind, bucket, bool(night))
        cached = facade_cache.get(key)
        if cached is not None:
            return cached
        # Force the chosen kind by calling the variant directly so the
        # cache hit is the same shader family, not a second random draw.
        seed = rng.random()
        name = f"bldg_{kind}_{len(facade_cache):03d}"
        if kind == "brick":
            mat = make_facade_brick(name, color, seed)
        elif kind == "plaster":
            mat = make_facade_plaster(name, color, seed)
        else:
            mat = make_facade(name, color, night, seed)
        facade_cache[key] = mat
        return mat

    def _skip_gap(s_now: float) -> float:
        for a, b in gaps:
            if a <= s_now <= b:
                return float(b) + 0.15
        return s_now

    while s < spline.length - 6.0:
        s = _skip_gap(s)
        if s >= spline.length - 6.0:
            break
        depth = rng.uniform(*cfg["building_depth"])
        height = rng.uniform(*cfg["building_height"])
        w_lo, w_hi = cfg.get("building_width", (5.0, 11.0))
        width = rng.uniform(float(w_lo), float(w_hi))
        gap = rng.uniform(*cfg["building_gap"])
        # Per-lot setback so the street wall is not a cloned plane.
        setback_j = rng.uniform(-0.55, 0.85)
        lot_facade = max(road_half + sidewalk_w + 0.35, facade + setback_j)
        mid = s + width * 0.5
        jumped = _skip_gap(mid)
        if jumped != mid:
            s = jumped
            continue
        # On the inside of a tight corner this offset folds across the road;
        # skip that station rather than dropping a wall onto the pavement.
        if offset_folds(
            spline,
            mid,
            lateral_sign * lot_facade,
            lateral_sign * (lot_facade + depth),
            width * 0.5,
            road_half + sidewalk_w - 0.10,
        ):
            s += width + gap
            continue
        p, tan, right = spline.frame(mid)
        center = p + right * (lateral_sign * (lot_facade + depth * 0.5))
        center.z = height * 0.5
        color = (
            rng.uniform(0.16, 0.38),
            rng.uniform(0.15, 0.34),
            rng.uniform(0.13, 0.30),
        )
        mat = _facade_mat(color)
        obj = create_box(
            f"building_{len(objs):03d}",
            (depth, width, height),
            center,
            collection,
            mat,
        )
        look_along(obj, tan)
        from materials import make_roof
        roof_c = (
            rng.uniform(0.08, 0.18),
            rng.uniform(0.08, 0.16),
            rng.uniform(0.08, 0.16),
        )
        roof_mat = make_roof(f"roof_{len(objs):03d}", roof_c)
        massing = rng.choice(("box", "box", "stepped", "l_plan", "arcade", "sloped"))
        if massing == "sloped":
            pitch = rng.uniform(0.22, 0.42)
            roof = create_box(
                f"building_{len(objs):03d}_roof",
                (depth + 0.20, width + 0.20, 0.55),
                Vector((0.0, 0.0, 0.0)),
                collection,
                roof_mat,
            )
            roof.parent = obj
            roof.location = Vector((0.0, 0.0, height * 0.5 + 0.10))
            roof.rotation_euler[0] = math.copysign(pitch, rng.choice((-1.0, 1.0)))
        else:
            roof = create_box(
                f"building_{len(objs):03d}_roof",
                (depth + 0.28, width + 0.28, 0.32),
                Vector((0.0, 0.0, 0.0)),
                collection,
                roof_mat,
            )
            roof.parent = obj
            roof.location = Vector((0.0, 0.0, height * 0.5 + 0.12))
        if massing == "stepped" and height > 7.0:
            cap_h = height * rng.uniform(0.22, 0.38)
            cap = create_box(
                f"building_{len(objs):03d}_step",
                (depth * 0.62, width * 0.70, cap_h),
                Vector((0.0, 0.0, 0.0)),
                collection,
                mat,
            )
            cap.parent = obj
            cap.location = Vector((depth * 0.08, 0.0, height * 0.5 + cap_h * 0.5))
        elif massing == "l_plan":
            wing_w = width * rng.uniform(0.32, 0.48)
            wing = create_box(
                f"building_{len(objs):03d}_wing",
                (depth * rng.uniform(0.45, 0.75), wing_w, height * rng.uniform(0.55, 0.90)),
                Vector((0.0, 0.0, 0.0)),
                collection,
                mat,
            )
            wing.parent = obj
            wing.location = Vector(
                (depth * 0.22, math.copysign(width * 0.38, rng.choice((-1.0, 1.0))), 0.0)
            )
        elif massing == "arcade" and height > 6.5:
            # Recessed ground-floor void (darker slab), not a new mesh family.
            dark = (
                color[0] * 0.35,
                color[1] * 0.35,
                color[2] * 0.32,
            )
            from materials import make_simple
            arch = create_box(
                f"building_{len(objs):03d}_arcade",
                (0.55, width * 0.92, min(3.2, height * 0.38)),
                Vector((0.0, 0.0, 0.0)),
                collection,
                make_simple(f"arcade_{len(objs):03d}", dark, 0.72),
            )
            arch.parent = obj
            arch.location = Vector((-depth * 0.5 + 0.18, 0.0, -height * 0.5 + 1.35))
        objs.append(obj)
        s += width + gap
    return objs


# ===========================================================================
# Procedural nature (instanced)
# ===========================================================================
#
# Every builder below returns bare ``(verts, faces)`` so it can be handed to
# MeshLibrary.mesh() and cached. A recursive tree is hundreds of objects but
# only a handful of datablocks (unit trunk, unit branch, a few leaf cards).

TREE_SHAPES: tuple[str, ...] = ("round", "conical", "columnar", "spreading", "bare")


def _tapered_tube_geom(
    r0: float, r1: float, height: float, segs: int = 8, rings: int = 3,
) -> tuple[list, list]:
    """Trunk: tapered tube along +Z with the origin at the base."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(rings + 1):
        t = i / rings
        z = height * t
        r = (1.0 - t) * r0 + t * r1
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            verts.append((r * math.cos(a), r * math.sin(a), z))

    def vid(i: int, j: int) -> int:
        return i * segs + (j % segs)

    for i in range(rings):
        for j in range(segs):
            faces.append((vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)))
    faces.append(tuple(reversed(range(segs))))
    faces.append(tuple(range(rings * segs, (rings + 1) * segs)))
    return verts, faces


def _blob_geom(
    rx: float, ry: float, rz: float, seed: int,
    rings: int = 7, segs: int = 10, jitter: float = 0.22,
) -> tuple[list, list]:
    """Canopy: UV sphere with per-vertex radial noise so it is not a ball.

    The noise is drawn from a locally seeded RNG rather than the episode RNG
    so the same shape key always produces the same datablock — otherwise the
    cache would return a mesh that does not match its key.
    """
    rnd = random.Random(int(seed))
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(rings + 1):
        th = math.pi * i / rings
        cz = math.cos(th)
        sr = math.sin(th)
        for j in range(segs):
            ph = 2.0 * math.pi * j / segs
            k = 1.0 + jitter * (rnd.random() * 2.0 - 1.0)
            verts.append((
                rx * sr * math.cos(ph) * k,
                ry * sr * math.sin(ph) * k,
                rz * cz * k,
            ))

    def vid(i: int, j: int) -> int:
        return i * segs + (j % segs)

    for i in range(rings):
        for j in range(segs):
            a, b = vid(i, j), vid(i, j + 1)
            c, d = vid(i + 1, j + 1), vid(i + 1, j)
            if i == 0:
                faces.append((a, c, d))
            elif i == rings - 1:
                faces.append((a, b, d))
            else:
                faces.append((a, b, c, d))
    return verts, faces


def _cone_geom(radius: float, height: float, segs: int = 10) -> tuple[list, list]:
    """Conifer canopy: cone with the origin at its base centre."""
    verts: list[tuple[float, float, float]] = [(0.0, 0.0, height)]
    for j in range(segs):
        a = 2.0 * math.pi * j / segs
        verts.append((radius * math.cos(a), radius * math.sin(a), 0.0))
    faces: list[tuple[int, ...]] = []
    for j in range(segs):
        faces.append((0, 1 + j, 1 + (j + 1) % segs))
    faces.append(tuple(range(1, segs + 1)))
    return verts, faces


def _disk_geom(radius: float, segs: int = 18, z: float = 0.0) -> tuple[list, list]:
    """Flat fan disk (puddles, crater mouths)."""
    verts: list[tuple[float, float, float]] = [(0.0, 0.0, z)]
    for j in range(segs):
        a = 2.0 * math.pi * j / segs
        verts.append((radius * math.cos(a), radius * math.sin(a), z))
    faces = [(0, 1 + j, 1 + (j + 1) % segs) for j in range(segs)]
    return verts, faces


def _box_geom(sx: float, sy: float, sz: float) -> tuple[list, list]:
    """Axis-aligned box, origin at the geometric centre. Shared by MeshLibrary."""
    hx, hy, hz = 0.5 * float(sx), 0.5 * float(sy), 0.5 * float(sz)
    verts = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5),
        (0, 4, 5, 1), (1, 5, 6, 2),
        (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    return verts, faces


def _shift_z(
    geom: tuple[list, list], dz: float,
) -> tuple[list, list]:
    verts, faces = geom
    return [(x, y, z + dz) for x, y, z in verts], faces


def _pyramid_geom() -> tuple[list, list]:
    """Unit square pyramid, origin at the geometric centre."""
    verts = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (0.0, 0.0, 0.5),
    ]
    faces = [
        (0, 1, 2, 3),
        (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4),
    ]
    return verts, faces


SHAPE_KINDS: tuple[str, ...] = (
    "cube", "sphere", "cylinder", "pyramid", "cone", "capsule", "lump",
)


def _shape_unit_geom(kind: str) -> tuple[list, list]:
    """Canonical 1 m threat-shape, origin at the centre, for MeshLibrary."""
    if kind == "cube":
        return _box_geom(1.0, 1.0, 1.0)
    if kind == "sphere":
        return _blob_geom(0.5, 0.5, 0.5, seed=0, rings=8, segs=12, jitter=0.0)
    if kind == "cylinder":
        return _shift_z(_tapered_tube_geom(0.5, 0.5, 1.0, segs=12, rings=2), -0.5)
    if kind == "pyramid":
        return _pyramid_geom()
    if kind == "cone":
        return _shift_z(_cone_geom(0.5, 1.0, segs=12), -0.5)
    if kind == "capsule":
        return _blob_geom(0.38, 0.38, 0.50, seed=1, rings=8, segs=12, jitter=0.0)
    # lump: irregular rock / bag — still one cached datablock.
    return _blob_geom(0.50, 0.42, 0.46, seed=2, rings=7, segs=10, jitter=0.28)


def _bind_object_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    """Per-object albedo on a linked mesh (does not rewrite the datablock slot)."""
    if obj.data is None:
        return
    if not obj.data.materials:
        obj.data.materials.append(mat)
    if obj.material_slots:
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = mat
    else:
        assign_mat(obj, mat)


def _crater_geom(
    radius: float, depth: float, rim: float = 0.06, segs: int = 18,
) -> tuple[list, list]:
    """Impact crater: raised rim ring, mouth ring, sunken apex.

    Unlike the pothole cylinder this is an actual bowl, so the silhouette
    under a low sun is a curved shadow rather than a flat black lid.
    """
    verts: list[tuple[float, float, float]] = [(0.0, 0.0, -abs(depth))]
    for j in range(segs):
        a = 2.0 * math.pi * j / segs
        verts.append((radius * math.cos(a), radius * math.sin(a), 0.0))
    for j in range(segs):
        a = 2.0 * math.pi * j / segs
        verts.append((radius * 1.16 * math.cos(a), radius * 1.16 * math.sin(a), rim))
    faces: list[tuple[int, ...]] = []
    for j in range(segs):
        k = (j + 1) % segs
        faces.append((0, 1 + j, 1 + k))                      # bowl
        faces.append((1 + j, 1 + segs + j, 1 + segs + k, 1 + k))  # rim skirt
    return verts, faces


def _grass_clump_geom(
    n_blades: int, radius: float, height: float, seed: int,
) -> tuple[list, list]:
    """A tuft of triangular blades fanning out of one point."""
    rnd = random.Random(int(seed))
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for _ in range(max(3, int(n_blades))):
        a = rnd.uniform(0.0, 2.0 * math.pi)
        r = radius * math.sqrt(rnd.random())
        bx, by = r * math.cos(a), r * math.sin(a)
        w = rnd.uniform(0.006, 0.016)
        h = height * rnd.uniform(0.55, 1.0)
        lean = rnd.uniform(0.0, 0.35) * h
        la = rnd.uniform(0.0, 2.0 * math.pi)
        base = len(verts)
        verts.append((bx - w, by, 0.0))
        verts.append((bx + w, by, 0.0))
        verts.append((bx + lean * math.cos(la), by + lean * math.sin(la), h))
        faces.append((base, base + 1, base + 2))
    return verts, faces


def _shape_ratio(shape: str, weber_ratio: float) -> float:
    """Weber–Penn ``ShapeRatio``. ``weber_ratio`` is 1 at the bole, 0 at the tip."""
    r = min(1.0, max(0.0, float(weber_ratio)))
    if shape == "conical":
        return 0.2 + 0.8 * r
    if shape == "columnar":
        return 0.50 + 0.50 * r
    if shape == "spreading":
        return 0.2 + 0.8 * math.sin(0.5 * math.pi * r)
    return 0.2 + 0.8 * math.sin(math.pi * r)


def _orient_y(
    p: tuple[float, float, float], dx: float, dy: float, dz: float,
) -> tuple[float, float, float]:
    """Rotate ``p`` so local +Y points at ``(dx, dy, dz)``."""
    target = Vector((float(dx), float(dy), float(dz)))
    if target.length < 1e-8:
        return p
    target.normalize()
    v = target.to_track_quat("Y", "Z") @ Vector(p)
    return (float(v.x), float(v.y), float(v.z))


def _leaf_spray_geom(
    seed: int, n: int, span: float, needle: bool = False,
) -> tuple[list, list]:
    """Many small kite leaves spaced along +Z (one twig's worth).

    Each leaf is two triangles plus back faces. They hang off the twig,
    they are not a ball at the origin — that was the old sprig, and it
    read as a green lump. Shared through MeshLibrary; per-twig variety
    is yaw of the parent joint.
    """
    rnd = random.Random(int(seed))
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    count = max(5, int(n))
    length = max(0.12, float(span))
    for i in range(count):
        # Bias toward the tip so the crown is a volume, not ivy on the bark.
        t = ((i + 0.5) / count) ** 0.62 + rnd.uniform(-0.02, 0.02)
        z = length * min(0.98, max(0.10, 0.12 + 0.86 * t))
        azim = i * 2.399963 + rnd.uniform(-0.22, 0.22)
        hang = rnd.uniform(0.22, 0.70)
        if needle:
            w = rnd.uniform(0.018, 0.034)
            tip = rnd.uniform(0.09, 0.16)
            mid = rnd.uniform(0.018, 0.034)
            droop = rnd.uniform(0.006, 0.016)
            outward = rnd.uniform(0.03, 0.08)
        else:
            w = rnd.uniform(0.060, 0.110)
            tip = rnd.uniform(0.18, 0.30)
            mid = rnd.uniform(0.050, 0.095)
            droop = rnd.uniform(0.014, 0.036)
            outward = rnd.uniform(0.07, 0.18)
        raw = (
            (0.0, 0.0, 0.0),
            (-w, mid, 0.002),
            (0.0, tip, -droop),
            (w, mid, 0.002),
        )
        dx, dy, dz = math.cos(azim), math.sin(azim), -0.20 - hang
        base = len(verts)
        ox0 = math.cos(azim) * outward
        oy0 = math.sin(azim) * outward
        for p in raw:
            ox, oy, oz = _orient_y(p, dx, dy, dz)
            verts.append((ox + ox0, oy + oy0, oz + z))
        faces.extend((
            (base, base + 1, base + 2),
            (base, base + 2, base + 3),
            (base, base + 2, base + 1),
            (base, base + 3, base + 2),
        ))
    return verts, faces


def _euler_align_z(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    """Euler XYZ that sends local +Z toward ``(dx, dy, dz)``."""
    v = Vector((float(dx), float(dy), float(dz)))
    if v.length < 1e-8:
        return (0.0, 0.0, 0.0)
    v.normalize()
    e = v.to_track_quat("Z", "Y").to_euler("XYZ")
    return (float(e.x), float(e.y), float(e.z))


def _tilt_z(split: float, azim: float) -> tuple[float, float, float]:
    """Rotate local +Z by ``split`` radians around an axis in the XY plane."""
    s, c = math.sin(split), math.cos(split)
    return _euler_align_z(s * math.sin(azim), -s * math.cos(azim), c)


def _socket(
    name: str,
    collection: bpy.types.Collection,
    parent: bpy.types.Object,
    location: Vector,
    scale: tuple[float, float, float],
    rotation: tuple[float, float, float],
) -> bpy.types.Object:
    """Unscaled-in-world animation joint (empty). Hide from the render."""
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.03
    obj.empty_display_type = "PLAIN_AXES"
    obj.hide_render = True
    obj.rotation_mode = "XYZ"
    _link(obj, collection)
    obj.parent = parent
    obj.location = location
    obj.scale = scale
    obj.rotation_euler = rotation
    return obj


def spawn_tree(
    lib: MeshLibrary,
    rng: random.Random,
    cfg: dict,
    collection: bpy.types.Collection,
    loc: Vector,
    counters: "_Counters",
    chaos: float = 0.0,
    shape: Optional[str] = None,
    max_canopy: Optional[float] = None,
    heading: Optional[float] = None,
    lean_to_road: bool = False,
    lat_sign: float = 0.0,
    wind: Any = None,
) -> Actor:
    """Weber–Penn street/park tree: recursive forks + individual triangle leaves.

    The trunk is a single unparented-to-foliage mesh (the TTC threat). Every
    limb is an unscaled empty (so wind rotation does not shear) plus a
    tapered tube. Children attach *along* the parent (monopodial) and the
    tip splits in two or three (dichotomous) — Weber & Penn 1995, Honda
    1971. Leaves are instanced twig sprays (many small triangles hung
    along the limb) on the last two wood levels.

    Shared datablocks: one trunk, one branch, a few leaf cards. Object
    count is high on purpose; the silhouette has to read as a tree from
    a 1.6 m camera.
    """
    from materials import make_bark, make_leaf

    tcfg = dict(cfg["world"].get("tree") or {})
    shapes = tuple(tcfg.get("shapes") or TREE_SHAPES)
    shape = shape or rng.choice(shapes)
    h_lo, h_hi = tcfg.get("trunk_height", (2.0, 4.8))
    r_lo, r_hi = tcfg.get("trunk_radius", (0.08, 0.28))
    c_lo, c_hi = tcfg.get("canopy_radius", (1.0, 3.0))
    lean_lo, lean_hi = tcfg.get("lean_deg", (0.0, 9.0))

    trunk_h = float(rng.uniform(float(h_lo), float(h_hi)))
    trunk_r = float(rng.uniform(float(r_lo), float(r_hi)))
    canopy_r = float(rng.uniform(float(c_lo), float(c_hi)))
    if max_canopy is not None:
        canopy_r = min(canopy_r, max(0.45, float(max_canopy)))
    instance_id = counters.next_id("tree")
    th = max(trunk_h, 1e-3)
    tr = max(trunk_r, 1e-3)
    bare = shape == "bare"
    needle = shape == "conical"
    small = canopy_r < 1.15
    if needle:
        max_depth = 2
        max_stems = 36 if small else 70
        n_length = 0.70
        base_size = 0.18
    elif shape == "columnar":
        max_depth = 4 if not small else 3
        max_stems = 48 if small else 110
        n_length = 0.28
        base_size = 0.30
    elif shape == "spreading":
        max_depth = 4 if not small else 3
        max_stems = 56 if small else 150
        n_length = 0.48
        base_size = 0.30
    else:
        max_depth = 4 if not small else 3
        max_stems = 48 if small else 140
        n_length = 0.40
        base_size = 0.34

    bark = lib.material("bark", lambda: make_bark("bark_shared"))
    leaf_palettes = (
        (0.10, 0.26, 0.07),  # spring
        (0.07, 0.18, 0.05),  # summer deep
        (0.16, 0.30, 0.08),  # lime
        (0.22, 0.20, 0.05),  # olive
        (0.40, 0.22, 0.05),  # autumn gold
        (0.48, 0.16, 0.04),  # autumn orange
        (0.36, 0.07, 0.05),  # autumn red
        (0.18, 0.14, 0.08),  # bronze
        (0.12, 0.16, 0.10),  # dusty evergreen
        (0.06, 0.12, 0.08),  # blue spruce
    )
    leaf_key = rng.randrange(0, len(leaf_palettes))
    leaf_base = leaf_palettes[leaf_key]
    leaf_mat = lib.material(
        f"leaf_{leaf_key}",
        lambda leaf_base=leaf_base, leaf_key=leaf_key: make_leaf(
            f"leaf_{leaf_key}", rng, chaos, base=leaf_base,
        ),
    )
    spray_prefix = f"spray_{'needle' if needle else 'broad'}_{leaf_key}"

    root = bpy.data.objects.new(instance_id, None)
    root.empty_display_size = 0.10
    root.empty_display_type = "PLAIN_AXES"
    root.rotation_mode = "XYZ"
    _link(root, collection)
    root.location = loc
    lean = math.radians(rng.uniform(float(lean_lo), float(lean_hi)))
    yaw = float(heading) if heading is not None else rng.uniform(0.0, 2.0 * math.pi)
    if lean_to_road and abs(lat_sign) > 1e-6:
        root.rotation_euler = (0.0, -math.copysign(lean, lat_sign), yaw)
    else:
        root.rotation_euler = (lean, 0.0, yaw)

    # The visible trunk is only the bole. Above that the stem *splits* —
    # a full-height pole with sticks glued on is what the last tree was.
    # Firs keep a full leader (monopodial); everything else forks.
    bole_h = th if needle else th * base_size
    trunk = lib.instance(
        f"{instance_id}_trunk",
        "trunk",
        lambda: _tapered_tube_geom(1.0, 0.72, 1.0, segs=10, rings=4),
        collection,
        Vector((0.0, 0.0, 0.0)),
        mat=bark,
        rotation_z=0.0,
        scale=(tr, tr, bole_h),
        smooth=True,
    )
    trunk.parent = root
    trunk.location = Vector((0.0, 0.0, 0.0))
    trunk.scale = (tr, tr, bole_h)
    trunk.rotation_euler = (0.0, 0.0, 0.0)

    foliage: list[FoliagePart] = []
    stem_i = [0]
    leaf_i = [0]
    rotate = math.radians(137.5)

    def _add_joint(obj: bpy.types.Object, kind: str, flex: float) -> None:
        rest = (
            float(obj.rotation_euler[0]),
            float(obj.rotation_euler[1]),
            float(obj.rotation_euler[2]),
        )
        foliage.append(FoliagePart(
            obj, rest,
            phase=rng.uniform(0.0, 2.0 * math.pi),
            flutter=rng.uniform(0.0, 2.0 * math.pi),
            kind=kind,
            flex=flex,
        ))

    def _wood(joint: bpy.types.Object, radius: float, length: float, tag: str) -> None:
        mesh = lib.instance(
            f"{instance_id}_{tag}",
            "branch",
            lambda: _tapered_tube_geom(1.0, 0.40, 1.0, segs=8, rings=3),
            collection,
            Vector((0.0, 0.0, 0.0)),
            mat=bark,
            rotation_z=0.0,
            scale=(radius, radius, length),
            smooth=True,
        )
        mesh.parent = joint
        mesh.location = Vector((0.0, 0.0, 0.0))
        mesh.scale = (radius, radius, length)
        mesh.rotation_euler = (0.0, 0.0, 0.0)

    def _place_sprays(joint: bpy.types.Object, length: float, dense: bool) -> None:
        if bare:
            return
        L = max(0.12, float(length))
        # One or two shared sprays cover the twig. Each mesh is many small
        # triangles already hung along +Z — not a pom-pom at the origin.
        if L <= 0.48:
            chunks = ((0.0, 0.36, 14, "s"),)
        elif L <= 0.82:
            chunks = ((0.0, 0.60, 20, "m"),)
        elif L <= 1.15:
            chunks = ((0.0, 0.92, 26, "l"),)
        else:
            chunks = ((0.0, 0.92, 24, "l"), (L - 0.88, 0.92, 22, "l"))
        extra = 6 if dense else 2
        for along, span, n, tag in chunks:
            leaf_i[0] += 1
            n_use = n + extra + (4 if needle else 0)
            key = f"{spray_prefix}_{tag}"
            yaw = rng.uniform(0.0, 2.0 * math.pi)
            s = min(1.12, max(0.68, L / span if L < span else 1.0))
            s *= rng.uniform(0.90, 1.10)
            spray = lib.instance(
                f"{instance_id}_f{leaf_i[0]}",
                key,
                lambda n_use=n_use, span=span: _leaf_spray_geom(
                    4300 + leaf_key * 10 + n_use, n_use, span, needle,
                ),
                collection,
                Vector((0.0, 0.0, 0.0)),
                mat=leaf_mat,
                rotation_z=0.0,
                scale=(s, s, s),
                smooth=False,
            )
            spray.parent = joint
            spray.location = Vector((0.0, 0.0, max(0.0, along)))
            spray.scale = (s, s, s)
            spray.rotation_euler = (0.0, 0.0, yaw)
            _add_joint(spray, "leaf", flex=1.0)

    def _down_angle(weber_r: float, lateral: bool) -> float:
        if needle:
            base, tip = 1.30, 0.85
        elif shape == "spreading":
            base, tip = 1.36, 0.55
        elif shape == "columnar":
            base, tip = 0.62, 0.28
        else:
            base, tip = 1.08, 0.40
        if not lateral:
            base *= 0.42
            tip *= 0.55
        ang = tip + (base - tip) * weber_r
        return max(0.12, ang + rng.uniform(-0.10, 0.10))

    def _grow(
        parent: bpy.types.Object,
        attach_z: float,
        eul: tuple[float, float, float],
        length: float,
        radius: float,
        depth: int,
    ) -> None:
        if length < 0.11 or radius < 0.007:
            return
        if stem_i[0] >= max_stems:
            if not bare and length > 0.14:
                leaf_i[0] += 1
                tip = _socket(
                    f"{instance_id}_x{leaf_i[0]}",
                    collection, parent,
                    Vector((0.0, 0.0, attach_z)),
                    (1.0, 1.0, 1.0), eul,
                )
                _place_sprays(tip, length, dense=True)
            return
        stem_i[0] += 1
        sid = stem_i[0]
        joint = _socket(
            f"{instance_id}_j{sid}",
            collection, parent,
            Vector((0.0, 0.0, attach_z)),
            (1.0, 1.0, 1.0), eul,
        )
        _wood(joint, radius, length, f"w{sid}")
        flex = 0.22 + 0.90 * (depth / max(max_depth, 1))
        _add_joint(joint, "branch", flex=flex)

        terminal = depth >= max_depth
        if not bare and (terminal or depth >= max_depth - 1):
            _place_sprays(joint, length, dense=terminal)
        if terminal:
            return

        remain = max_stems - stem_i[0]
        if needle:
            n_lat = 2 if depth == 1 and remain > 4 else 0
            n_apical = 0
        elif depth == 1:
            n_lat = 2 if small else 3
            n_apical = 2 if rng.random() < 0.62 else 3
        elif depth == 2:
            n_lat = 1 if small else 2
            n_apical = 2
        elif depth == 3:
            n_lat = 0
            n_apical = 2
        else:
            n_lat = 0
            n_apical = 2 if remain > 4 else 0
        n_lat = min(n_lat, max(0, remain - n_apical))
        n_apical = min(n_apical, max(0, remain - n_lat))

        rot0 = rng.uniform(0.0, 2.0 * math.pi)
        for i in range(n_lat):
            t = 0.32 + 0.52 * ((i + 0.5) / max(n_lat, 1))
            t += rng.uniform(-0.04, 0.04)
            t = min(0.88, max(0.24, t))
            wr = 1.0 - t
            child_len = (length - 0.45 * (t * length)) * rng.uniform(0.42, 0.62)
            child_len = max(0.16, min(child_len, canopy_r * 0.75))
            child_r = max(0.009, radius * rng.uniform(0.48, 0.64))
            azim = rot0 + i * rotate + rng.uniform(-0.20, 0.20)
            _grow(
                joint, t * length,
                _tilt_z(_down_angle(wr, True), azim),
                child_len, child_r, depth + 1,
            )

        for k in range(n_apical):
            azim = rot0 + (2.0 * math.pi * k) / max(n_apical, 1) + rng.uniform(-0.16, 0.16)
            child_len = length * rng.uniform(0.58, 0.82)
            child_len = max(0.16, min(child_len, canopy_r * 0.90))
            child_r = max(0.009, radius * rng.uniform(0.58, 0.76))
            # Wide enough crotch that the fork reads as a split, not a kink.
            split = 0.32 + rng.uniform(-0.06, 0.10)
            _grow(
                joint, length * rng.uniform(0.92, 0.99),
                _tilt_z(split, azim),
                child_len, child_r, depth + 1,
            )

    # First-order limbs along the trunk (parent = unscaled root).
    bole = base_size * th
    if needle:
        n_whorl = 4 if small else 6
        per = 4 if small else 6
        for w in range(n_whorl):
            t = 0.20 + 0.74 * (w / max(n_whorl - 1, 1))
            wr = (th - t * th) / max(th - bole, 1e-3)
            lat_len = max(0.22, canopy_r * _shape_ratio("conical", wr) * rng.uniform(0.90, 1.18))
            down = _down_angle(wr, True)
            az0 = w * 0.42 + rng.uniform(-0.08, 0.08)
            lat_r = max(0.010, tr * rng.uniform(0.22, 0.36))
            for k in range(per):
                _grow(
                    root, t * th,
                    _tilt_z(down, az0 + k * (2.0 * math.pi / per)),
                    lat_len * rng.uniform(0.88, 1.08),
                    lat_r, 1,
                )
    else:
        # Main crotch at the top of the bole: 2 or 3 thick leaders.
        n_tip = 2 if (small or rng.random() < 0.45) else 3
        rot0 = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(n_tip):
            azim = rot0 + (2.0 * math.pi * k) / n_tip + rng.uniform(-0.14, 0.14)
            tip_len = max(0.45, th * n_length * rng.uniform(0.95, 1.25))
            tip_len = min(tip_len, canopy_r * 1.05)
            _grow(
                root, bole_h * rng.uniform(0.94, 0.99),
                _tilt_z(0.38 + rng.uniform(-0.06, 0.10), azim),
                tip_len, max(0.018, tr * rng.uniform(0.60, 0.78)), 1,
            )
        # A few lower laterals off the upper bole so the fork is not a lone Y.
        n_low = 2 if small else (3 if shape == "columnar" else 4)
        for i in range(n_low):
            t = 0.52 + 0.40 * ((i + 0.4) / max(n_low, 1))
            wr = 1.0 - t
            child_len = th * n_length * _shape_ratio(shape, wr) * rng.uniform(0.70, 1.00)
            child_len = min(child_len, canopy_r * 0.85)
            child_len = max(0.32, child_len)
            azim = rot0 + 0.5 * rotate + i * rotate + rng.uniform(-0.16, 0.16)
            _grow(
                root, bole_h * min(0.96, max(0.40, t)),
                _tilt_z(_down_angle(wr, True), azim),
                child_len, max(0.014, tr * rng.uniform(0.40, 0.58)), 1,
            )

    _tag(root, instance_id, "tree")
    _tag(trunk, instance_id, "tree")
    return Actor(
        obj=root,
        instance_id=instance_id,
        class_name="tree",
        category="static",
        origin_z=float(loc.z),
        threat_mode="volume",
        threat_obj=trunk,
        foliage=foliage,
        wind=wind,
        corridor_pad=max(0.28, tr + 0.12),
    )


def spawn_streetlamp(
    loc: Vector,
    heading: float,
    collection: bpy.types.Collection,
    counters: "_Counters",
    height: float = 5.6,
    with_arm: bool = True,
    arm_sign: float = 1.0,
) -> Actor:
    """Annotatable lamp pole. TTC / splat use the pole, not the arm."""
    from materials import make_metal_paint

    instance_id = counters.next_id("streetlamp")
    h = max(2.4, float(height))
    mat = bpy.data.materials.get("lamp_pole")
    if mat is None:
        mat = make_principled("lamp_pole", (0.10, 0.10, 0.11), 0.55)
    pole = create_box(
        instance_id,
        (0.10, 0.10, h),
        loc + Vector((0.0, 0.0, h * 0.5)),
        collection,
        mat,
    )
    pole.rotation_mode = "XYZ"
    pole.rotation_euler = (0.0, 0.0, float(heading))
    if with_arm:
        arm_mat = make_metal_paint("lamp_arm", (0.12, 0.12, 0.13), 0.45)
        arm = create_box(
            instance_id + "_arm",
            (0.08, 0.85, 0.08),
            Vector((0.0, 0.0, 0.0)),
            collection,
            arm_mat,
        )
        arm.parent = pole
        arm.location = Vector((-float(arm_sign) * 0.40, 0.0, h * 0.5 - 0.04))
    _tag(pole, instance_id, "streetlamp")
    return Actor(
        obj=pole,
        instance_id=instance_id,
        class_name="streetlamp",
        category="static",
        origin_z=float(loc.z),
        threat_mode="volume",
        threat_obj=pole,
        corridor_pad=0.22,
    )


def scatter_grass(
    lib: MeshLibrary,
    rng: random.Random,
    cfg: dict,
    collection: bpy.types.Collection,
    road: PathSpline,
    lat_lo: float,
    lat_hi: float,
    n: int,
    ground_z: float = 0.0,
    chaos: float = 0.0,
    exclude_abs_lat: float = 0.0,
) -> int:
    """Scatter instanced grass tufts. Pure decor — never annotated.

    ``exclude_abs_lat`` keeps clumps off the carriageway (and, on paved
    biomes, off the sidewalk) so grass is not growing out of asphalt.
    """
    from materials import make_grass

    if n <= 0:
        return 0
    lo, hi = float(lat_lo), float(lat_hi)
    ex = max(0.0, float(exclude_abs_lat))
    bands: list[tuple[float, float]] = []
    if lo < -ex:
        bands.append((lo, min(hi, -ex)))
    if hi > ex:
        bands.append((max(lo, ex), hi))
    bands = [(a, b) for a, b in bands if b > a + 1e-3]
    if not bands:
        return 0
    gcfg = dict(cfg["world"].get("grass") or {})
    blades = int(gcfg.get("blades", 26))
    r_lo, r_hi = gcfg.get("clump_radius", (0.30, 0.85))
    h_lo, h_hi = gcfg.get("blade_height", (0.10, 0.42))
    variants = 4
    made = 0
    for i in range(int(n)):
        s = rng.uniform(1.0, max(2.0, road.length - 1.0))
        a, b = bands[i % len(bands)] if rng.random() < 0.5 else rng.choice(bands)
        lat = rng.uniform(a, b)
        v = rng.randrange(variants)
        mat = lib.material(
            f"grass_{v}",
            lambda v=v: make_grass(
                f"grass_blades_{v}",
                (rng.uniform(0.07, 0.20), rng.uniform(0.14, 0.34), rng.uniform(0.04, 0.14)),
            ),
        )
        lib.instance(
            f"grass_{i:04d}",
            f"grass_clump_{v}",
            lambda v=v: _grass_clump_geom(
                blades, float(r_hi), float(h_hi), seed=7000 + v
            ),
            collection,
            road.offset_point(s, lat, z=float(ground_z)),
            mat=mat,
            rotation_z=rng.uniform(0.0, 2.0 * math.pi),
            scale=(
                rng.uniform(float(r_lo) / float(r_hi), 1.0),
                rng.uniform(float(r_lo) / float(r_hi), 1.0),
                rng.uniform(float(h_lo) / float(h_hi), 1.0),
            ),
        )
        made += 1
    return made


# ===========================================================================
# Hazard / actor factories
# ===========================================================================

GROUND_CLASSES = (
    # size: (diameter, diameter, depth_below_pavement). Closed Z-cylinder
    # with the mouth a few millimetres above the ribbon (z-fight) and the
    # body buried so it reads as a hole, not a lid.
    ("pothole", "pothole", (1.05, 1.05, 0.42), "footprint"),
    ("trash_can", "trash_can", (0.45, 0.45, 0.85), "volume"),
    ("scooter", "scooter", (1.10, 0.40, 0.95), "volume"),
    ("barricade", "barricade", (0.30, 1.20, 1.05), "volume"),
    # Rich ground hazards. `crater` and `broken_slab` are genuine fall/trip
    # hazards and get the pothole's footprint treatment; `debris` is a solid
    # obstacle; `puddle` is a *distractor* (see spawn_ground_hazard).
    ("crater", "crater", (1.40, 1.40, 0.30), "footprint"),
    ("broken_slab", "broken_slab", (0.86, 0.86, 0.10), "footprint"),
    ("debris", "debris", (0.72, 0.72, 0.26), "volume"),
    ("puddle", "puddle", (1.30, 1.30, 0.02), "footprint"),
)

#: Hazards whose render mesh sinks below the pavement, so the 2-D box must
#: come from the mouth slab rather than the buried volume.
SUNKEN_CLASSES = frozenset({"pothole", "crater", "puddle", "broken_slab"})

HEAD_CLASSES = (
    ("tree_branch", "tree_branch", (1.80, 0.18, 0.18)),
    ("ac_unit", "ac_unit", (0.70, 0.55, 0.45)),
    ("sign", "sign", (1.10, 0.08, 0.70)),
    ("truck_door", "truck_door", (0.12, 1.00, 1.10)),
)

# Default sidewalk clutter — furniture on the *shop-front* half of the
# sidewalk, never floating boxes and never trip-holes (those stay injectors).
FURNITURE_CLASSES: tuple[str, ...] = ("trash_can", "scooter", "barricade", "puddle")


class _Counters:
    def __init__(self) -> None:
        self.n: dict[str, int] = {}

    def next_id(self, class_name: str) -> str:
        i = self.n.get(class_name, 0)
        self.n[class_name] = i + 1
        return f"{class_name}_{i:03d}"


def _tag(obj: bpy.types.Object, instance_id: str, class_name: str) -> None:
    obj["instance_id"] = instance_id
    obj["class_name"] = class_name
    obj["is_annotatable"] = True


def spawn_ground_hazard(
    class_name: str,
    loc: Vector,
    heading: float,
    collection: bpy.types.Collection,
    rng: random.Random,
    counters: _Counters,
    roughness: float,
    chaos: float = 0.0,
) -> Actor:
    from materials import chaos_albedo, make_chaos_surface, make_water

    spec = {c[0]: c for c in GROUND_CLASSES}[class_name]
    _cid, _cn, size, mode = spec
    instance_id = counters.next_id(class_name)
    ground = float(loc.z)
    if class_name == "pothole":
        # Closed black well. Origin is on the pavement (curb Z on a sidewalk,
        # z=0 on asphalt). The mouth sits 4 mm above the ribbon so it wins
        # the depth test; the body drops ~0.4 m so it is not a raised slab.
        # Do not boolean-cut the ribbon: open-mesh DIFFERENCE often no-ops
        # while still reporting success, which hid an uncapped well under
        # an intact sidewalk.
        loc = loc.copy()
        loc.z = ground
        radius = 0.5 * float(size[0]) * rng.uniform(0.92, 1.10)
        depth = max(0.28, float(size[2]) * rng.uniform(0.92, 1.10))
        mat = make_principled(instance_id, (0.018, 0.014, 0.012), 1.0)
        nt = getattr(mat, "node_tree", None)
        bsdf = nt.nodes.get("Principled BSDF") if nt is not None else None
        if bsdf is not None:
            for key, val in (
                ("Specular IOR Level", 0.0),
                ("Specular", 0.0),
                ("Metallic", 0.0),
            ):
                if key in bsdf.inputs:
                    bsdf.inputs[key].default_value = val
        obj = create_z_cylinder(
            instance_id,
            radius=radius,
            z_top=0.004,
            z_bottom=-depth,
            collection=collection,
            location=loc,
            mat=mat,
            segments=20,
            cap_top=True,
            cap_bottom=True,
        )
        if hasattr(obj, "visible_shadow"):
            obj.visible_shadow = False
    elif class_name == "crater":
        # A real bowl rather than a black lid: the rim catches the key light,
        # so it reads as depth under a low sun instead of as a decal.
        loc = loc.copy()
        loc.z = ground
        radius = 0.5 * float(size[0]) * rng.uniform(0.80, 1.25)
        depth = max(0.14, float(size[2]) * rng.uniform(0.8, 1.4))
        col = chaos_albedo(rng, (0.055, 0.045, 0.038), chaos, "dark")
        mat = make_chaos_surface(
            instance_id, col, rng, chaos * 0.5,
            base_roughness=0.96, allow_emission=False,
        )
        verts, faces = _crater_geom(radius, depth, rim=0.05 * rng.uniform(0.6, 1.6), segs=18)
        obj = create_mesh(instance_id, verts, faces, collection, loc, mat)
        if hasattr(obj, "visible_shadow"):
            obj.visible_shadow = False
    elif class_name == "broken_slab":
        # Lifted paving stone: a slab tilted about a horizontal axis so one
        # edge stands proud of the pavement. The classic urban trip hazard.
        loc = loc.copy()
        loc.z = ground + float(size[2]) * 0.35
        col = chaos_albedo(rng, (0.42, 0.41, 0.38), chaos, "matte")
        mat = make_chaos_surface(
            instance_id, col, rng, chaos * 0.6,
            base_roughness=0.88, allow_emission=False,
        )
        obj = create_box(instance_id, size, loc, collection, mat)
        obj.rotation_mode = "XYZ"
        tilt = math.radians(rng.uniform(7.0, 20.0)) * rng.choice((-1.0, 1.0))
        obj.rotation_euler = (tilt, 0.0, heading)
    elif class_name == "debris":
        # Rubble pile: one root chunk plus a few children, all boxes.
        loc = loc.copy()
        loc.z = ground + float(size[2]) * 0.5
        col = chaos_albedo(rng, (0.34, 0.30, 0.26), chaos, "matte")
        mat = make_chaos_surface(
            instance_id, col, rng, chaos, base_roughness=0.90, allow_emission=False,
        )
        obj = create_box(
            instance_id,
            (size[0] * 0.45, size[1] * 0.45, size[2]),
            loc, collection, mat,
        )
        for c in range(rng.randint(2, 5)):
            chunk = create_box(
                f"{instance_id}_c{c}",
                (
                    size[0] * rng.uniform(0.14, 0.38),
                    size[1] * rng.uniform(0.14, 0.38),
                    size[2] * rng.uniform(0.30, 0.95),
                ),
                Vector((0.0, 0.0, 0.0)),
                collection,
                mat,
            )
            chunk.parent = obj
            chunk.location = Vector((
                rng.uniform(-0.5, 0.5) * size[0],
                rng.uniform(-0.5, 0.5) * size[1],
                rng.uniform(-0.35, 0.25) * size[2],
            ))
            chunk.rotation_mode = "XYZ"
            chunk.rotation_euler = (
                rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.0, 3.14),
            )
    elif class_name == "puddle":
        # Standing water. Deliberately **not annotated**: it is a visual
        # distractor, not a hazard. Walking through a puddle is safe, but the
        # geometric threat model cannot express "dark patch that is fine to
        # step on" — a flat footprint object on the gait line always scores
        # as a fall-in. Omitting it from the label set is the honest answer,
        # and it still teaches the network that dark ground patches are not
        # automatically holes.
        loc = loc.copy()
        loc.z = ground + 0.006
        radius = 0.5 * float(size[0]) * rng.uniform(0.65, 1.45)
        mat = make_water(instance_id, rng, chaos)
        verts, faces = _disk_geom(radius, segs=20, z=0.0)
        obj = create_mesh(instance_id, verts, faces, collection, loc, mat)
        if hasattr(obj, "visible_shadow"):
            obj.visible_shadow = False
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (0.0, 0.0, heading)
        _tag(obj, instance_id, class_name)
        return Actor(
            obj=obj,
            instance_id=instance_id,
            class_name=class_name,
            category="static",
            origin_z=obj.location.z,
            threat_mode=mode,
            annotatable=False,
        )
    elif class_name == "trash_can":
        color = (0.15, 0.32, 0.18) if rng.random() < 0.5 else (0.25, 0.25, 0.28)
        loc = loc.copy()
        loc.z = ground + size[2] * 0.5
        obj = create_box(
            instance_id, size, loc, collection,
            make_chaos_surface(instance_id, color, rng, chaos, base_roughness=roughness),
        )
    elif class_name == "scooter":
        color = rng.choice(((0.05, 0.05, 0.06), (0.7, 0.15, 0.1), (0.1, 0.1, 0.7)))
        loc = loc.copy()
        loc.z = ground + size[2] * 0.5
        obj = create_box(
            instance_id, size, loc, collection,
            make_chaos_surface(instance_id, color, rng, chaos, base_roughness=roughness),
        )
    else:
        color = (0.75, 0.45, 0.05)
        loc = loc.copy()
        loc.z = ground + size[2] * 0.5
        obj = create_box(
            instance_id, size, loc, collection,
            make_chaos_surface(instance_id, color, rng, chaos, base_roughness=roughness),
        )
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, heading)
    _tag(obj, instance_id, class_name)
    return Actor(
        obj=obj,
        instance_id=instance_id,
        class_name=class_name,
        category="static",
        origin_z=obj.location.z,
        threat_mode=mode,
    )


def spawn_head_hazard(
    class_name: str,
    loc: Vector,
    heading: float,
    height: float,
    collection: bpy.types.Collection,
    rng: random.Random,
    counters: _Counters,
    roughness: float,
    chaos: float = 0.0,
) -> Actor:
    spec = {c[0]: c for c in HEAD_CLASSES}[class_name]
    size = spec[2]
    instance_id = counters.next_id(class_name)
    color = {
        "tree_branch": (0.22, 0.14, 0.06),
        "ac_unit": (0.55, 0.55, 0.58),
        "sign": (0.75, 0.12, 0.10),
        "truck_door": (0.15, 0.18, 0.45),
    }[class_name]
    from materials import make_chaos_surface

    loc = loc.copy()
    loc.z = height
    obj = create_box(
        instance_id, size, loc, collection,
        make_chaos_surface(instance_id, color, rng, chaos, base_roughness=roughness),
    )
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, heading)
    _tag(obj, instance_id, class_name)
    return Actor(
        obj=obj,
        instance_id=instance_id,
        class_name=class_name,
        category="static",
        origin_z=height,
        threat_mode="volume",
    )


def spawn_bench(
    loc: Vector,
    heading: float,
    collection: bpy.types.Collection,
    lib: MeshLibrary,
    rng: random.Random,
    counters: _Counters,
    chaos: float = 0.0,
) -> Actor:
    """Park bench the seated ego sits on. Linked instances, not unique meshes.

    Not annotated: sitting on it is the ego state, not a collision. The
    network still sees the silhouette (and must not key on "bench ⇒ SAFE").
    """
    from materials import chaos_albedo, make_chaos_surface

    instance_id = counters.next_id("bench")
    wood = chaos_albedo(rng, (0.32, 0.20, 0.10), chaos * 0.6, "natural")
    metal = chaos_albedo(rng, (0.12, 0.12, 0.13), chaos * 0.5, "dark")
    mat_wood = lib.material(
        f"bench_wood_{round(wood[0], 2)}_{round(wood[1], 2)}",
        lambda: make_chaos_surface(instance_id + "_wood", wood, rng, chaos, base_roughness=0.72),
    )
    mat_metal = lib.material(
        "bench_metal",
        lambda: make_chaos_surface(instance_id + "_metal", metal, rng, 0.0, base_roughness=0.45),
    )
    seat_h = 0.44
    root = lib.instance(
        instance_id,
        "bench_seat",
        lambda: _box_geom(1.55, 0.42, 0.06),
        collection,
        Vector((loc.x, loc.y, loc.z + seat_h)),
        mat=mat_wood,
        rotation_z=heading,
        scale=(1.0, 1.0, 1.0),
    )
    back = lib.instance(
        instance_id + "_back",
        "bench_back",
        lambda: _box_geom(1.55, 0.06, 0.42),
        collection,
        root.location.copy(),
        mat=mat_wood,
    )
    back.parent = root
    back.location = Vector((0.0, -0.20, 0.22))
    for i, x in enumerate((-0.68, 0.68)):
        for j, y in enumerate((-0.16, 0.16)):
            leg = lib.instance(
                f"{instance_id}_leg{i}{j}",
                "bench_leg",
                lambda: _box_geom(0.05, 0.05, seat_h),
                collection,
                root.location.copy(),
                mat=mat_metal,
            )
            leg.parent = root
            leg.location = Vector((x, y, -0.5 * seat_h))
    _tag(root, instance_id, "bench")
    return Actor(
        obj=root,
        instance_id=instance_id,
        class_name="bench",
        category="static",
        origin_z=seat_h,
        threat_mode="volume",
        annotatable=False,
    )


def _create_wheel(
    name: str,
    radius: float,
    width: float,
    collection: bpy.types.Collection,
    mat: Any,
    segments: int = 10,
) -> bpy.types.Object:
    """Cylinder along local +X (roll around X)."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    hw = width * 0.5
    n = max(8, segments)
    for i in range(n):
        a = 2.0 * math.pi * i / n
        y, z = radius * math.cos(a), radius * math.sin(a)
        verts.append((-hw, y, z))
        verts.append((hw, y, z))
    for i in range(n):
        a = 2 * i
        b = 2 * ((i + 1) % n)
        faces.append((a, b, b + 1, a + 1))
    faces.append(tuple(range(0, 2 * n, 2)))
    faces.append(tuple(reversed(range(1, 2 * n, 2))))
    return create_mesh(name, verts, faces, collection, Vector((0, 0, 0)), mat)


def spawn_vehicle(
    loc: Vector,
    heading: float,
    collection: bpy.types.Collection,
    color: tuple[float, float, float],
    roughness: float,
    counters: _Counters,
    rng: Optional[random.Random] = None,
    chaos: float = 0.0,
) -> Actor:
    from humanoid import create_superellipsoid
    from materials import make_chaos_car_paint, make_glass, make_rubber, make_simple

    instance_id = counters.next_id("vehicle")
    rng = rng or random.Random(hash(instance_id) & 0xFFFFFFFF)
    loc = loc.copy()
    loc.z = 0.38 + 0.36
    paint = make_chaos_car_paint(instance_id, color, rng, chaos)
    glass = make_glass(
        instance_id + "_glass",
        (rng.uniform(0.15, 0.65), rng.uniform(0.20, 0.70), rng.uniform(0.25, 0.75)),
    )
    rubber = make_rubber(instance_id + "_tire")
    dark = make_simple(instance_id + "_trim", (0.04, 0.04, 0.04), 0.55)

    # Barr superquadric hull (e < 1) → rounded sedan, not a shipping crate.
    body = create_superellipsoid(
        instance_id, (0.90, 2.05, 0.36), collection, paint,
        e1=0.45, e2=0.52, nu=8, nv=12,
    )
    body.location = loc
    cabin = create_superellipsoid(
        instance_id + "_cabin", (0.78, 0.82, 0.32), collection, glass,
        e1=0.55, e2=0.60, nu=6, nv=10,
    )
    cabin.parent = body
    cabin.location = Vector((0.0, -0.12, 0.42))
    bumper = create_box(instance_id + "_bumper", (1.84, 0.22, 0.28), loc, collection, dark)
    bumper.parent = body
    bumper.location = Vector((0.0, 2.05, -0.12))
    bumper_r = create_box(instance_id + "_bumper_r", (1.84, 0.18, 0.26), loc, collection, dark)
    bumper_r.parent = body
    bumper_r.location = Vector((0.0, -2.05, -0.12))

    wheels: list = []
    wr, ww = 0.32, 0.22
    for tag, xy in (
        ("fl", (0.78, 1.25)),
        ("fr", (-0.78, 1.25)),
        ("rl", (0.78, -1.25)),
        ("rr", (-0.78, -1.25)),
    ):
        w = _create_wheel(f"{instance_id}_{tag}", wr, ww, collection, rubber)
        w.parent = body
        w.rotation_mode = "XYZ"
        w.location = Vector((xy[0], xy[1], -0.38))
        wheels.append((w, wr))

    body.rotation_mode = "XYZ"
    body.rotation_euler = (0.0, 0.0, heading)
    shade_smooth(body)
    shade_smooth(cabin)
    _tag(body, instance_id, "vehicle")
    return Actor(
        obj=body,
        instance_id=instance_id,
        class_name="vehicle",
        category="vehicle",
        origin_z=loc.z,
        threat_mode="volume",
        wheels=wheels,
    )


def spawn_pedestrian(
    loc: Vector,
    heading: float,
    collection: bpy.types.Collection,
    color: tuple[float, float, float],
    roughness: float,
    counters: _Counters,
    rng: Optional[random.Random] = None,
    chaos: float = 0.0,
    child: bool = False,
) -> Actor:
    from humanoid import spawn_humanoid

    instance_id = counters.next_id("person")
    rng = rng or random.Random(hash(instance_id) & 0xFFFFFFFF)
    pelvis, rig, hip_z = spawn_humanoid(
        instance_id, loc, heading, collection, rng, color, chaos=chaos, child=child,
    )
    _tag(pelvis, instance_id, "person")
    return Actor(
        obj=pelvis,
        instance_id=instance_id,
        class_name="person",
        category="pedestrian",
        origin_z=hip_z,
        threat_mode="volume",
        gait=rig,
    )


def spawn_projectile(
    loc: Vector,
    collection: bpy.types.Collection,
    counters: _Counters,
) -> Actor:
    instance_id = counters.next_id("projectile")
    size = (0.28, 0.28, 0.28)
    color = (0.35, 0.22, 0.08)
    obj = create_box(instance_id, size, loc, collection, make_principled(instance_id, color, 0.7))
    _tag(obj, instance_id, "projectile")
    return Actor(
        obj=obj,
        instance_id=instance_id,
        class_name="projectile",
        category="projectile",
        origin_z=loc.z,
        threat_mode="volume",
        corridor_pad=0.20,
        allow_sidewalk=True,
    )


def spawn_threat_shape(
    loc: Vector,
    collection: bpy.types.Collection,
    counters: _Counters,
    rng: random.Random,
    kind: str = "cube",
    size: float = 0.45,
    z: Optional[float] = None,
    chaos: float = 0.0,
    lib: Optional[MeshLibrary] = None,
) -> Actor:
    """Generic approaching primitive — the detector should not overfit to cars.

    Cube / sphere / cylinder / pyramid / cone / capsule / lump, all unit
    meshes from MeshLibrary. Colour and aspect are randomised; the class
    name stays ``threat_<kind>`` so labels stay analysable.
    """
    from materials import make_chaos_surface

    if kind in ("shape", "random", "", "any"):
        kind = rng.choice(SHAPE_KINDS)
    if kind not in SHAPE_KINDS:
        kind = "cube"
    class_name = f"threat_{kind}"
    instance_id = counters.next_id(class_name)
    loc = loc.copy()
    loc.z = float(size) * 0.5 if z is None else float(z)
    color = rng.choice(
        (
            (0.72, 0.22, 0.12),
            (0.18, 0.42, 0.55),
            (0.78, 0.62, 0.12),
            (0.22, 0.22, 0.24),
            (0.45, 0.18, 0.48),
            (0.12, 0.55, 0.28),
            (0.62, 0.40, 0.18),
        )
    )
    ax = 1.0 + 0.55 * chaos * (rng.random() * 2.0 - 1.0)
    ay = 1.0 + 0.55 * chaos * (rng.random() * 2.0 - 1.0)
    az = 1.0 + 0.55 * chaos * (rng.random() * 2.0 - 1.0)
    sx = size * max(0.40, ax)
    sy = size * max(0.40, ay)
    sz = size * max(0.40, az)
    if kind == "capsule":
        sx = sy = size * max(0.40, 0.55 * ax)
        sz = size * max(0.55, az)
    mat = make_chaos_surface(
        instance_id, color, rng, chaos, base_roughness=rng.uniform(0.22, 0.78),
    )
    if lib is not None:
        obj = lib.instance(
            instance_id,
            f"threat_{kind}",
            lambda k=kind: _shape_unit_geom(k),
            collection,
            loc,
            mat=None,
            rotation_z=rng.uniform(0.0, 2.0 * math.pi),
            scale=(sx, sy, sz),
            smooth=kind != "cube",
        )
        _bind_object_material(obj, mat)
    else:
        verts, faces = _shape_unit_geom(kind)
        verts = [(x * sx, y * sy, z_ * sz) for x, y, z_ in verts]
        obj = create_mesh(instance_id, verts, faces, collection, loc, mat)
        obj.rotation_mode = "XYZ"
        obj.rotation_euler[2] = rng.uniform(0.0, 2.0 * math.pi)
    _tag(obj, instance_id, class_name)
    return Actor(
        obj=obj,
        instance_id=instance_id,
        class_name=class_name,
        category="dynamic",
        origin_z=loc.z,
        threat_mode="volume",
        corridor_pad=max(0.18, size * 0.5 + 0.05),
        allow_sidewalk=True,
    )


def spawn_threat_cube(
    loc: Vector,
    collection: bpy.types.Collection,
    counters: _Counters,
    rng: random.Random,
    size: float = 0.45,
    z: Optional[float] = None,
    chaos: float = 0.0,
    lib: Optional[MeshLibrary] = None,
) -> Actor:
    """Cube-shaped looming primitive. Prefer :func:`spawn_threat_shape`."""
    return spawn_threat_shape(
        loc, collection, counters, rng,
        kind="cube", size=size, z=z, chaos=chaos, lib=lib,
    )


def spawn_bicycle(
    loc: Vector,
    heading: float,
    collection: bpy.types.Collection,
    color: tuple[float, float, float],
    roughness: float,
    counters: _Counters,
    rng: Optional[random.Random] = None,
    chaos: float = 0.0,
) -> Actor:
    from materials import make_chaos_surface, make_rubber, make_simple

    instance_id = counters.next_id("bicycle")
    rng = rng or random.Random(hash(instance_id) & 0xFFFFFFFF)
    loc = loc.copy()
    loc.z = 0.55
    frame_mat = make_chaos_surface(
        instance_id + "_frame", color, rng, chaos, base_roughness=roughness,
    )
    rubber = make_rubber(instance_id + "_tire")
    dark = make_simple(instance_id + "_dark", (0.06, 0.06, 0.07), 0.55)

    root = create_box(instance_id, (0.06, 1.05, 0.08), loc, collection, frame_mat)
    downtube = create_box(instance_id + "_down", (0.05, 0.72, 0.05), loc, collection, frame_mat)
    downtube.parent = root
    downtube.location = Vector((0.0, 0.05, -0.18))
    downtube.rotation_euler[0] = math.radians(18.0)
    seat = create_box(instance_id + "_seat", (0.16, 0.28, 0.05), loc, collection, dark)
    seat.parent = root
    seat.location = Vector((0.0, -0.28, 0.18))
    bars = create_box(instance_id + "_bars", (0.52, 0.05, 0.05), loc, collection, dark)
    bars.parent = root
    bars.location = Vector((0.0, 0.42, 0.22))
    rider = create_box(instance_id + "_rider", (0.32, 0.28, 0.42), loc, collection, frame_mat)
    rider.parent = root
    rider.location = Vector((0.0, -0.06, 0.42))

    wheels: list = []
    wr, ww = 0.33, 0.05
    for tag, y in (("f", 0.48), ("r", -0.48)):
        w = _create_wheel(f"{instance_id}_{tag}", wr, ww, collection, rubber)
        w.parent = root
        w.rotation_mode = "XYZ"
        w.location = Vector((0.0, y, -0.22))
        wheels.append((w, wr))

    root.rotation_mode = "XYZ"
    root.rotation_euler = (0.0, 0.0, heading)
    _tag(root, instance_id, "bicycle")
    return Actor(
        obj=root,
        instance_id=instance_id,
        class_name="bicycle",
        category="bicycle",
        origin_z=loc.z,
        threat_mode="volume",
        wheels=wheels,
        corridor_pad=0.40,
        allow_sidewalk=True,
    )


# ===========================================================================
# World state
# ===========================================================================

@dataclass
class WorldState:
    road: PathSpline
    sidewalk: PathSpline
    sidewalk_lateral: float
    actors: list[Actor]
    environment: dict
    walk_speed: float
    counters: _Counters
    collections: dict
    materials: dict
    corridor: StreetCorridor
    road_half: float
    sidewalk_w: float
    s_cross: float


# ===========================================================================
# Generator
# ===========================================================================

class WorldGenerator:
    def __init__(self, cfg: dict, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self.state: Optional[WorldState] = None
        self._scenario = "safe_walk"
        self._building_gaps: list[tuple[float, float]] = []
        self._s_cross = 11.0
        self._compose: Optional[ComposeSession] = None
        self.biome = "street"
        self.force_ego_mode = ""
        self.chaos = 0.0
        self.lib = MeshLibrary()
        self._wind: Optional[WindField] = None
        self._sites: list[tuple[float, float, float]] = []
        # Populated by freeze(); invalidated whenever an actor is appended.
        self._movers: Optional[list[Actor]] = None
        self._annotatable: Optional[list[Actor]] = None

    # -- biome -------------------------------------------------------------

    def prepare_biome(
        self,
        requested: str = "auto",
        names: Sequence[str] | None = None,
    ) -> str:
        """Pick the biome and fold its overrides into ``cfg['world']``.

        Every biome is the *same* Frenet corridor with different widths,
        ground cover, and props. Keeping one substrate is what lets all 30+
        scenario injectors, the occupancy solver, and the annotation loop
        work unchanged in a park: a "lane" in a park is simply a lateral
        band of a 3 m gravel path.

        Must run before :meth:`prepare_scenario`, because a sparse scenario
        overrides the biome's traffic counts and not the other way round.
        """
        wcfg = self.cfg["world"]
        table = dict(wcfg.get("biomes") or {})
        name = str(requested or "auto").strip().lower()
        if name in ("", "auto", "random"):
            weights = dict(wcfg.get("biome_weights") or {"street": 1.0})
            weights = {k: v for k, v in weights.items() if k in table} or {"street": 1.0}
            if names and any(n in CLEAR_CENTER_SCENARIOS for n in names):
                keep = {k: v for k, v in weights.items() if k in CLEAR_CENTER_BIOMES}
                if keep:
                    weights = keep
            name = _weighted_choice(self.rng, weights)
        if name not in table:
            known = ", ".join(sorted(table)) or "street"
            raise ValueError(f"unknown biome {name!r}. Known: {known}")
        self.biome = name
        for key, val in (table.get(name) or {}).items():
            wcfg[key] = val
        self._jitter_layout()
        return name

    def _jitter_layout(self) -> None:
        """Per-episode width / setback jitter so biomes are not cloned streets."""
        wcfg = self.cfg["world"]
        j = dict((self.cfg.get("domain_randomization") or {}).get("layout_jitter") or {})
        if not j:
            return

        def _scale(key: str, frac: float, lo: float = 0.35) -> None:
            if key not in wcfg:
                return
            raw = wcfg[key]
            if isinstance(raw, (int, float)):
                wcfg[key] = max(lo, float(raw) * (1.0 + self.rng.uniform(-frac, frac)))

        _scale("road_width", float(j.get("road_width", 0.0)), 2.4)
        _scale("sidewalk_width", float(j.get("sidewalk_width", 0.0)), 0.85)
        _scale("building_setback", float(j.get("building_setback", 0.0)), 0.35)
        _scale("lane_offset", float(j.get("lane_offset", 0.0)), 0.55)
        med0 = float(wcfg.get("median_width", 0.0) or 0.0)
        if med0 >= 1.4:
            _scale("median_width", float(j.get("median_width", 0.0)), 1.4)
        # Keep the travel lane outside a planted median.
        med = float(wcfg.get("median_width", 0.0) or 0.0)
        if med >= 1.4:
            wcfg["lane_offset"] = max(float(wcfg.get("lane_offset", 1.75)), 0.5 * med + 1.55)

    def _apply_chaos_crowd(self) -> None:
        """High appearance chaos also means a busier street (counts, not meshes)."""
        c = max(0.0, min(1.0, float(self.chaos)))
        k = 0.55 + 1.30 * c
        wcfg = self.cfg["world"]
        for key in ("n_background_pedestrians", "n_background_vehicles"):
            pair = wcfg.get(key)
            if not pair:
                continue
            lo, hi = int(pair[0]), int(pair[1])
            wcfg[key] = (max(0, int(round(lo * k))), max(0, int(round(hi * k))))
        poisson = wcfg.get("poisson")
        if isinstance(poisson, dict) and "n_ground_static" in poisson:
            lo, hi = [int(v) for v in poisson["n_ground_static"]]
            poisson["n_ground_static"] = (
                max(0, int(round(lo * k))),
                max(0, int(round(hi * k))),
            )

    def _draw_walk_speed(self) -> float:
        """Stroll / walk / hurry. Seated still pins the rig to 0 downstream."""
        g = self.cfg.get("gait") or {}
        weights = dict(g.get("pace_weights") or {})
        bands = dict(g.get("pace_speed") or {})
        if weights and bands:
            keys = [k for k in weights if k in bands]
            if keys:
                pace = _weighted_choice(self.rng, {k: weights[k] for k in keys})
                lo, hi = bands[pace]
                return float(self.rng.uniform(float(lo), float(hi)))
        return float(self.rng.uniform(float(g.get("walk_speed_min", 0.70)), float(g.get("walk_speed_max", 2.05))))

    def prepare_scenario(self, scenario: str | Sequence[str]) -> None:
        """Call before `build()` so intersections / sparse streets exist in mesh.

        ``scenario`` is one name or the already-resolved compound list. A gap
        opens if *any* component crosses the ribbon; ``empty_street`` still
        sparsifies background traffic when it appears in a mix.
        """
        names = [scenario] if isinstance(scenario, str) else [str(n) for n in scenario]
        names = [n for n in names if n]
        self._scenario = compose_slug(names) if names else "safe_walk"
        self._s_cross = 11.0
        self._building_gaps = []
        if any(n in CROSS_EGO_SCENARIOS for n in names):
            self.force_ego_mode = "crosswalk"
        elif any(n == "hasty_look" for n in names):
            self.force_ego_mode = "hasty"
        elif any(n in CLEAR_CENTER_SCENARIOS for n in names):
            # Look down the empty sidewalk so the image centre stays cold
            # until a side actor actually cuts in.
            self.force_ego_mode = "walk"
        else:
            self.force_ego_mode = ""
        n_cross = sum(1 for n in names if n in CROSS_GAP_SCENARIOS)
        if n_cross:
            # One crosser: original 5–20 m window. Compounds stagger extra
            # crossers further down the ribbon, so open a longer gap.
            hi = 20.0 if n_cross <= 1 else 28.0
            self._building_gaps.append((4.0, hi))
        if any(n in SPARSE_SCENARIOS for n in names):
            self.cfg["world"]["n_background_pedestrians"] = (0, 0)
            self.cfg["world"]["n_background_vehicles"] = (0, 0)
            self.cfg["world"]["poisson"]["n_ground_static"] = (0, 2)
            self.cfg["world"]["poisson"]["n_head_hazards"] = (0, 0)
        if any(n in CLEAR_CENTER_SCENARIOS for n in names):
            self._prepare_clear_center()

    def _prepare_clear_center(self) -> None:
        """Empty sidewalk ahead: no clutter in the gaze, cars stay on asphalt."""
        wcfg = self.cfg["world"]
        poisson = wcfg.setdefault("poisson", {})
        poisson["n_ground_static"] = (0, 0)
        poisson["n_head_hazards"] = (0, 0)
        wcfg["n_median_trees"] = (0, 0)
        wcfg["n_path_trees"] = (0, 0)
        wcfg["median_width"] = 0.0
        if float(wcfg.get("road_width", 7.0) or 0.0) < 5.2:
            wcfg["road_width"] = 7.0
            wcfg["lane_offset"] = 1.75
            wcfg["curb_height"] = 0.12
            wcfg["lane_paint"] = True
            wcfg["buildings"] = True
            wcfg.pop("ground", None)
        types = tuple(wcfg.get("path_types") or ("straight", "gentle_curve"))
        kept = tuple(t for t in types if t in ("straight", "gentle_curve"))
        wcfg["path_types"] = kept or ("straight",)

    # -- build -------------------------------------------------------------

    def build(self) -> WorldState:
        scene = reset_blender_scene()
        configure_eevee(scene, self.cfg["render"])

        wcfg = self.cfg["world"]
        dr = self.cfg["domain_randomization"]
        rng = self.rng
        # A fresh library per episode: the previous one's datablocks were
        # just freed by reset_blender_scene, so holding them would be a
        # use-after-free waiting to happen.
        self.lib = MeshLibrary()
        self._wind = None
        self._sites = []

        col_world = ensure_collection("WORLD", scene)
        col_haz = ensure_collection("HAZARDS", scene)
        col_act = ensure_collection("ACTORS", scene)
        col_lights = ensure_collection("LIGHTS", scene)

        path_type = rng.choice(tuple(wcfg["path_types"]))
        length = rng.uniform(wcfg["path_length_min"], wcfg["path_length_max"])
        road = PathSpline.generate(path_type, length, rng, ds=float(wcfg["sample_ds"]))

        sidewalk_sign = rng.choice((-1.0, 1.0))
        road_half = float(wcfg["road_width"]) * 0.5
        sw = float(wcfg["sidewalk_width"])
        ground_kind = str(wcfg.get("ground", "paved"))
        # Sit closer to the curb than the facade. A 24 mm lens at 1.3 m from a
        # building face fills the entire frame with one wall (looks "empty").
        # Parks have no kerb: the walker is on the gravel path itself.
        if ground_kind == "grass":
            sidewalk_lateral = sidewalk_sign * rng.uniform(0.18, max(0.35, road_half * 0.55))
        else:
            sidewalk_lateral = sidewalk_sign * (road_half + sw * 0.38)
        sidewalk = road.offset_spline(sidewalk_lateral)

        from materials import (
            chaos_albedo, make_asphalt, make_concrete_tiles, make_grass, make_simple,
        )

        env_choice = choose_environment(self.cfg, rng)
        night = env_choice["lighting"] == "night"
        if env_choice["lighting"] == "harsh_glare":
            # Aim the disc down the walker's line of sight; a blinding sun
            # behind the camera is just a bright day.
            env_choice["azim_deg"] = glare_azimuth_deg(road.tangent(3.0))
        self.chaos = float(env_choice.get("chaos", 0.0))
        chaos = self.chaos
        self._apply_chaos_crowd()
        self._wind = WindField(
            self.rng.randrange(1, 2**31),
            strength=float(env_choice.get("wind_strength", 0.40)),
            direction=math.radians(float(env_choice.get("wind_dir_deg", 0.0))),
            label=str(env_choice.get("wind", "breeze")),
        )
        roughness_sw = rng.uniform(*dr["roughness"])

        curb_h = float(wcfg["curb_height"])
        verge_w = float(wcfg.get("verge_width", 4.5))

        cobble = (
            ground_kind != "grass"
            and rng.random() < float(wcfg.get("cobble_prob", 0.0) or 0.0)
        )
        if ground_kind == "grass":
            # Park: a gravel/dirt path on open ground, no asphalt.
            mat_road = make_concrete_tiles(
                "path", chaos_albedo(rng, (0.42, 0.37, 0.30), chaos * 0.5, "matte"),
                tile_m=rng.uniform(0.22, 0.55),
            )
        elif cobble:
            cobble_box = dr.get("cobble_color", ((0.28, 0.22, 0.18), (0.52, 0.42, 0.32)))
            mat_road = make_concrete_tiles(
                "road_cobble",
                chaos_albedo(rng, _rand_color(rng, cobble_box), chaos * 0.45, "matte"),
                tile_m=rng.uniform(0.16, 0.34),
            )
        else:
            mat_road = make_asphalt(
                "road",
                chaos_albedo(rng, _rand_color(rng, dr["road_color"]), chaos * 0.45, "matte"),
                seed=rng.random(),
            )
        mat_sw = make_concrete_tiles(
            "sidewalk",
            chaos_albedo(rng, _rand_color(rng, dr["sidewalk_color"]), chaos * 0.45, "matte"),
            tile_m=rng.uniform(0.28, 0.85),
        )
        mat_curb = make_simple("curb", chaos_albedo(rng, (0.42, 0.42, 0.40), chaos * 0.4), 0.7)
        mat_grass = make_grass("verge", (
            rng.uniform(0.10, 0.18),
            rng.uniform(0.18, 0.30),
            rng.uniform(0.06, 0.12),
        ))

        _ribbon_mesh("road_surface", road, -road_half, road_half, 0.0, col_world, mat_road)
        if ground_kind == "grass":
            # Open park: no kerb, no paving. Grass runs from the path edge
            # out to the verge so the camera is in a field, not a street
            # with the asphalt painted green.
            for sign in (-1.0, 1.0):
                g0 = sign * road_half
                g1 = sign * (road_half + verge_w)
                va, vb = (g0, g1) if g0 < g1 else (g1, g0)
                _ribbon_mesh(
                    f"verge_{'R' if sign > 0 else 'L'}", road, va, vb, 0.02,
                    col_world, mat_grass,
                )
        else:
            slab_amp = 0.012 if self.biome == "plaza" else 0.006
            for sign in (-1.0, 1.0):
                inner = sign * road_half
                outer = sign * (road_half + sw)
                a, b = (inner, outer) if inner < outer else (outer, inner)
                _ribbon_mesh(
                    f"sidewalk_{'R' if sign > 0 else 'L'}", road, a, b, curb_h,
                    col_world, mat_sw, z_amp=slab_amp, rng=rng,
                )
                if curb_h > 0.01:
                    _ribbon_mesh(
                        f"curb_{'R' if sign > 0 else 'L'}",
                        road, inner - 0.06 * sign, inner + 0.06 * sign,
                        curb_h * 0.5, col_world, mat_curb,
                    )
                g0 = sign * (road_half + sw)
                g1 = sign * (road_half + sw + verge_w)
                va, vb = (g0, g1) if g0 < g1 else (g1, g0)
                _ribbon_mesh(
                    f"verge_{'R' if sign > 0 else 'L'}", road, va, vb, 0.02,
                    col_world, mat_grass,
                )

        median_w = float(wcfg.get("median_width", 0.0) or 0.0)
        if median_w >= 1.4 and ground_kind != "grass":
            mh = 0.5 * median_w
            _ribbon_mesh(
                "median_plant", road, -mh, mh, 0.03, col_world, mat_grass,
            )

        # Painted dashes: 3 m mark, 3 m gap, 0.12 m wide, sampled on the spline
        # so they follow a corner instead of a 3 m chord across the lane.
        if bool(wcfg.get("lane_paint", True)):
            mat_paint = make_simple("lane_paint", (0.86, 0.78, 0.18), 0.45)
            dash_s = 2.0
            while dash_s < road.length - 2.0:
                _centerline_dash(
                    f"dash_{int(dash_s):03d}",
                    road,
                    dash_s,
                    min(dash_s + 3.0, road.length - 0.5),
                    0.06,
                    0.016,
                    col_world,
                    mat_paint,
                )
                dash_s += 6.0

        if bool(wcfg.get("buildings", True)):
            bldg_rough = rng.uniform(*dr["roughness"])
            for side in tuple(wcfg.get("building_sides", (-1.0, 1.0))):
                _extrude_buildings(
                    road, float(side), road_half, sw, wcfg, rng, col_world, bldg_rough,
                    night=night, gaps=self._building_gaps,
                )

        counters = _Counters()
        # Trees first so lamps / furniture / cars can keep clear of trunks.
        tree_actors = self._scatter_trees(road, road_half, sw, verge_w, col_haz, counters)
        lamp_objects, lamp_actors = self._spawn_streetlamps(
            road, road_half, sw, col_world, col_lights, counters,
        )
        n_grass = self.rng.randint(*[int(v) for v in wcfg.get("n_grass_clumps", (0, 0))])
        if n_grass > 0:
            # Grass stays on the verge (and park fields), never the asphalt
            # or the walking slab.
            exclude = road_half + (0.12 if ground_kind == "grass" else sw + 0.08)
            scatter_grass(
                self.lib, rng, self.cfg, col_world, road,
                lat_lo=-(road_half + sw + verge_w * 0.95),
                lat_hi=(road_half + sw + verge_w * 0.95),
                n=n_grass, ground_z=0.02, chaos=chaos,
                exclude_abs_lat=exclude,
            )

        if bool(env_choice.get("dappled", False)):
            self._spawn_canopy_gobo(road, road_half + sw + verge_w, col_world)

        env = apply_domain_randomization(scene, self.cfg, rng, col_lights, lamp_objects, env=env_choice)
        env = dict(env)
        env["biome"] = self.biome
        env["chaos"] = round(chaos, 4)
        env["cobble"] = bool(cobble)
        env["median_width"] = round(float(wcfg.get("median_width", 0.0) or 0.0), 3)
        env["dappled"] = bool(env_choice.get("dappled", False))
        env["wind"] = str(env_choice.get("wind", "breeze"))
        env["wind_strength"] = round(float(env_choice.get("wind_strength", 0.4)), 3)
        env["wind_dir_deg"] = round(float(env_choice.get("wind_dir_deg", 0.0)), 1)

        actors: list[Actor] = []
        actors.extend(tree_actors)
        actors.extend(lamp_actors)
        actors.extend(self._scatter_ground(road, sidewalk_lateral, sw, col_haz, counters, roughness_sw))
        actors.extend(self._scatter_head(road, sidewalk_lateral, col_haz, counters, roughness_sw))
        actors.extend(self._spawn_background_vehicles(road, sidewalk_lateral, col_act, counters))
        actors.extend(self._spawn_background_pedestrians(road, sidewalk_lateral, col_act, counters))

        walk_speed = self._draw_walk_speed()
        reveal_view_layer(scene)
        # reveal_view_layer un-hides every object; put daytime lamps back to sleep.
        apply_streetlamp_state(lamp_objects, self.cfg, night)
        prepare_still_render(scene)

        corridor = StreetCorridor(
            road=road,
            road_half=road_half,
            sidewalk_w=sw,
            max_abs_lateral=road_half + sw - float(wcfg.get("corridor_margin", 0.22)),
            curb_height=float(wcfg["curb_height"]),
        )
        self._movers = None
        self._annotatable = None
        self.state = WorldState(
            road=road,
            sidewalk=sidewalk,
            sidewalk_lateral=sidewalk_lateral,
            actors=actors,
            environment=env,
            walk_speed=walk_speed,
            counters=counters,
            collections={"world": col_world, "hazards": col_haz, "actors": col_act, "lights": col_lights},
            materials={"road": mat_road, "sidewalk": mat_sw},
            corridor=corridor,
            road_half=road_half,
            sidewalk_w=sw,
            s_cross=self._s_cross,
        )
        for actor in actors:
            if actor.follow_spline is not None:
                actor.s, actor.lateral = corridor.confine(
                    actor.s, actor.lateral, actor.corridor_pad, actor.allow_sidewalk
                )
                loc = road.offset_point(
                    actor.s, actor.lateral, actor.origin_z + corridor.ground_z(actor.lateral)
                )
                actor.obj.location = loc
        return self.state

    def create_camera(self) -> bpy.types.Object:
        scene = bpy.context.scene
        cam_data = bpy.data.cameras.new(self.cfg["camera"]["name"])
        cam_obj = bpy.data.objects.new(self.cfg["camera"]["name"], cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
        return cam_obj

    def place_ego_props(self, rig: Any) -> None:
        """World props that belong to the ego, not to a scenario injector.

        A seated walker needs a bench at the eye's (s, lateral). Spawned
        *after* injectors so occupancy does not treat the bench as a
        competing static hazard on the gait line.
        """
        if self.state is None:
            return
        prof = getattr(rig, "profile", None)
        if prof is None or getattr(prof, "mode", "walk") != "seated":
            return
        s = self._cam_s(rig, 0.0)
        lat = float(rig.lateral_at(0.0) if callable(getattr(rig, "lateral_at", None)) else rig.lateral)
        # Sit the slats just behind the HMD so the camera is not inside the mesh.
        loc = self.state.road.offset_point(
            max(0.4, s - 0.18), lat, z=self.state.corridor.ground_z(lat),
        )
        heading = heading_from_tangent(self.state.road.tangent(s))
        actor = spawn_bench(
            loc, heading, self.state.collections["world"], self.lib,
            self.rng, self.state.counters, self.chaos,
        )
        actor.s = s
        actor.lateral = lat
        self._append(actor)

    # -- per-frame ---------------------------------------------------------

    def freeze(self) -> None:
        """Partition actors once, after the last injector has run.

        The per-frame loop then walks only the actors that can actually move
        and only the actors that can actually be annotated, instead of
        re-filtering the full list 150 times per episode. Static clutter is
        the majority of the list in a cluttered biome, and its `update()` is
        a no-op that still costs a Python call and a Vector allocation.
        """
        assert self.state is not None
        actors = self.state.actors
        self._movers = [
            a for a in actors
            if a.category != "static"
            or a.stop_t is not None
            or a.gait is not None
            or a.wheels
            or a.foliage
        ]
        self._annotatable = [a for a in actors if a.annotatable]

    def update(self, t: float, dt: float) -> None:
        assert self.state is not None
        movers = self._movers
        if movers is None:
            self.freeze()
            movers = self._movers or []
        corridor = self.state.corridor
        for actor in movers:
            actor.update(t, dt, corridor)

    def annotatable(self) -> list[Actor]:
        assert self.state is not None
        cached = self._annotatable
        if cached is None:
            self.freeze()
            cached = self._annotatable or []
        return cached

    # -- scenario injection (Part 4.3) ------------------------------------

    def _scenario_handlers(self) -> dict:
        """Canonical name → injector. Built once per ``inject_scenarios`` call."""
        nm = float(self.cfg["scenarios"]["near_miss_cpa_target"])
        cr = float(self.cfg["scenarios"]["critical_cpa_target"])
        return {
            "safe_walk": self._inject_none,
            "empty_street": self._inject_none,
            "periph_empty": self._inject_none,
            "periph_car_side": self._inject_periph_car_side,
            "periph_parked": self._inject_periph_parked,
            "periph_ped_side": self._inject_periph_ped_side,
            "periph_car_turn": lambda r: self._inject_periph_car_swerve(r, runoff=False),
            "periph_car_runoff": lambda r: self._inject_periph_car_swerve(r, runoff=True),
            "periph_ped_cut": lambda r: self._inject_through_cross(
                r, kind="person", cpa=cr, from_left=self.rng.random() < 0.5,
                speed=self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"]),
            ),
            "periph_child_cut": self._inject_child_dart,
            "oncoming_pedestrian": lambda r: self._inject_oncoming(
                r, kind="person", obj_speed=self.rng.uniform(0.95, 1.30),
                tau=3.0, dlat="opposite",
            ),
            "parallel_pedestrian": lambda r: self._inject_parallel_person(r),
            "cyclist_same_way": lambda r: self._inject_cyclist_same_way(r),
            "car_pass_far": lambda r: self._inject_car_lane(
                r, far=True, obj_speed=self.rng.uniform(5.5, 8.0), tau=3.2,
            ),
            "car_approaching": lambda r: self._inject_car_lane(
                r, far=False, obj_speed=self.rng.uniform(5.0, 7.0), tau=3.0,
            ),
            "distant_jaywalk": lambda r: self._inject_distant_jaywalk(r),
            "pothole_offset": lambda r: self._inject_pothole(
                r, lead=float(self.cfg["scenarios"].get("pothole_offset_lead_m", 8.0)), dlat=1.80,
            ),
            "parked_car_opposite": lambda r: self._inject_parked_car(r),
            "crossing_street": self._inject_none,
            "cyclist_overtake": self._inject_cyclist_overtake,
            "group_crossing": self._inject_group_crossing,
            "scooter_from_sidewalk": lambda r: self._inject_through_cross(
                r, kind="bicycle", cpa=nm, from_left=self.rng.random() < 0.5,
                speed=self.rng.uniform(2.4, 4.2),
            ),
            "parked_car_door": self._inject_parked_car_door,
            "crossing_car_side": lambda r: self._inject_crossing_car(r, far=False),
            "crossing_head_on": lambda r: self._inject_crossing_car(r, far=True),
            "backing_vehicle": self._inject_backing_vehicle,
            "near_miss_pass": lambda r: self._inject_oncoming(
                r, kind="person", obj_speed=self.rng.uniform(0.95, 1.30),
                tau=float(self.cfg["scenarios"]["oncoming_tau"]), dlat=nm,
            ),
            "jaywalker_offset": lambda r: self._inject_through_cross(
                r, kind="person", cpa=nm, from_left=True,
                speed=self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"]),
            ),
            "jaywalker_from_left": lambda r: self._inject_through_cross(
                r, kind="person", cpa=nm, from_left=True,
                speed=self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"]),
            ),
            "jaywalker_from_right": lambda r: self._inject_through_cross(
                r, kind="person", cpa=nm, from_left=False,
                speed=self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"]),
            ),
            "jaywalker_turn_away": lambda r: self._inject_jaywalk_turn(
                r, toward=False, cpa=nm,
            ),
            "cyclist_near_miss": lambda r: self._inject_oncoming(
                r, kind="bicycle",
                obj_speed=self.rng.uniform(*self.cfg["world"]["bicycle_speed"]),
                tau=2.5, dlat=nm,
            ),
            "car_near_miss_lane": lambda r: self._inject_oncoming(
                r, kind="vehicle", obj_speed=self.rng.uniform(4.8, 6.5),
                tau=2.8, dlat=nm, pad=1.05,
            ),
            "car_cross_front": lambda r: self._inject_through_cross(
                r, kind="vehicle", cpa=nm, from_left=self.rng.random() < 0.5,
                speed=self.rng.uniform(*self.cfg["world"]["cross_car_speed"]),
            ),
            "cube_near_miss": lambda r: self._inject_oncoming(
                r, kind="cube", obj_speed=self.rng.uniform(*self.cfg["world"]["cube_speed"]),
                tau=2.6, dlat=nm, cube_size=self.rng.uniform(0.35, 0.60),
            ),
            "shape_near_miss": lambda r: self._inject_oncoming(
                r, kind="shape",
                obj_speed=self.rng.uniform(*self.cfg["world"].get("shape_speed", self.cfg["world"]["cube_speed"])),
                tau=2.6, dlat=nm, cube_size=self.rng.uniform(0.35, 0.65),
            ),
            "pothole_near": lambda r: self._inject_pothole(
                r, lead=float(self.cfg["scenarios"].get("pothole_near_lead_m", 7.2)), dlat=0.85,
            ),
            "jaywalker": lambda r: self._inject_through_cross(
                r, kind="person", cpa=cr, from_left=self.rng.random() < 0.5,
                speed=self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"]),
            ),
            "jaywalker_turn_toward": lambda r: self._inject_jaywalk_turn(
                r, toward=True, cpa=cr,
            ),
            "sudden_stop": self._inject_sudden_stop,
            "swerve_vehicle": lambda r: self._inject_cut_in(
                r,
                t0=float(self.cfg["scenarios"]["swerve_trigger_s"]),
                t_hit=3.4,
                v_car=5.5,
                cpa=cr,
                behavior="swerve",
            ),
            "pothole_on_path": lambda r: self._inject_pothole(
                r, lead=float(self.cfg["scenarios"].get("pothole_on_path_lead_m", 7.8)), dlat=0.0,
            ),
            "tree_on_path": lambda r: self._inject_tree(
                r, lead=float(self.cfg["scenarios"].get("tree_on_path_lead_m", 7.5)), dlat=0.0,
            ),
            "tree_near": lambda r: self._inject_tree(
                r, lead=float(self.cfg["scenarios"].get("tree_near_lead_m", 7.2)), dlat=0.80,
            ),
            "lamp_on_path": lambda r: self._inject_lamp(
                r, lead=float(self.cfg["scenarios"].get("lamp_on_path_lead_m", 7.5)), dlat=0.0,
            ),
            "lamp_near": lambda r: self._inject_lamp(
                r, lead=float(self.cfg["scenarios"].get("lamp_near_lead_m", 7.2)), dlat=0.80,
            ),
            "hasty_look": self._inject_none,
            "cube_head_on": lambda r: self._inject_oncoming(
                r, kind="cube", obj_speed=self.rng.uniform(*self.cfg["world"]["cube_speed"]),
                tau=2.4, dlat=cr, cube_size=self.rng.uniform(0.40, 0.70),
            ),
            "cube_from_left": lambda r: self._inject_through_cross(
                r, kind="cube", cpa=cr, from_left=True,
                speed=self.rng.uniform(*self.cfg["world"]["cube_speed"]),
                cube_size=self.rng.uniform(0.35, 0.65),
            ),
            "cube_from_right": lambda r: self._inject_through_cross(
                r, kind="cube", cpa=cr, from_left=False,
                speed=self.rng.uniform(*self.cfg["world"]["cube_speed"]),
                cube_size=self.rng.uniform(0.35, 0.65),
            ),
            "cube_on_path": lambda r: self._inject_static_shapes(r, kind="cube", n=1),
            "shape_head_on": lambda r: self._inject_oncoming(
                r, kind="shape",
                obj_speed=self.rng.uniform(*self.cfg["world"].get("shape_speed", self.cfg["world"]["cube_speed"])),
                tau=2.4, dlat=cr, cube_size=self.rng.uniform(0.40, 0.75),
            ),
            "shape_from_left": lambda r: self._inject_through_cross(
                r, kind="shape", cpa=cr, from_left=True,
                speed=self.rng.uniform(*self.cfg["world"].get("shape_speed", self.cfg["world"]["cube_speed"])),
                cube_size=self.rng.uniform(0.35, 0.70),
            ),
            "shape_from_right": lambda r: self._inject_through_cross(
                r, kind="shape", cpa=cr, from_left=False,
                speed=self.rng.uniform(*self.cfg["world"].get("shape_speed", self.cfg["world"]["cube_speed"])),
                cube_size=self.rng.uniform(0.35, 0.70),
            ),
            "shapes_on_path": lambda r: self._inject_static_shapes(r, kind="shape", n=0),
            "car_cut_in": lambda r: self._inject_cut_in(
                r,
                t0=float(self.cfg["scenarios"]["cut_in_trigger_s"]),
                t_hit=2.9,
                v_car=6.2,
                cpa=cr,
                behavior="cut_in",
            ),
            "cyclist_head_on": lambda r: self._inject_oncoming(
                r, kind="bicycle",
                obj_speed=self.rng.uniform(*self.cfg["world"]["bicycle_speed"]),
                tau=2.3, dlat=cr,
            ),
            "head_level_projectile": self._inject_head_cube,
            "car_cross_critical": lambda r: self._inject_through_cross(
                r, kind="vehicle", cpa=cr, from_left=self.rng.random() < 0.5,
                speed=self.rng.uniform(*self.cfg["world"]["cross_car_speed"]),
            ),
            # --- erratic / chaotic actors -------------------------------
            "cyclist_weaving": lambda r: self._inject_weaving(
                r, kind="bicycle",
                speed=self.rng.uniform(*self.cfg["world"]["bicycle_speed"]),
                amplitude=self.rng.uniform(0.9, 1.9),
                hz=self.rng.uniform(0.28, 0.62),
                lane=self._near_lane(),
                pad=0.40,
            ),
            "car_erratic_swerve": lambda r: self._inject_weaving(
                r, kind="vehicle",
                speed=self.rng.uniform(4.5, 8.0),
                amplitude=self.rng.uniform(1.0, 2.2),
                hz=self.rng.uniform(0.18, 0.40),
                lane=self._near_lane(),
                pad=1.05,
            ),
            "car_runs_off_road": self._inject_run_off_road,
            "child_darting": self._inject_child_dart,
        }

    def inject_scenario(self, scenario: str, rig: Any) -> str:
        """Force an edge-case so the episode lands in the requested bucket.

        ``scenario`` may be a single name, an alias, or a comma/plus list.
        Every injector is Frenet-constrained: actors change `s` and `lateral`
        on the road spline and are clamped to the street corridor.
        """
        names = pick_scenarios(self.rng, self.cfg, scenario)
        return self.inject_scenarios(names, rig)

    def inject_scenarios(self, names: Sequence[str], rig: Any) -> str:
        """Instantiate every requested injector into the same street.

        Occupancy is reserved in Frenet ``(s, lateral)`` so compounds do not
        spawn inside each other or inside nearby background traffic. This
        runs once per episode (before the two-phase sim/render loop).
        """
        assert self.state is not None
        resolved = [resolve_scenario_name(n, self.rng, self.cfg) for n in names]
        resolved = [n for n in resolved if n]
        if not resolved:
            resolved = ["safe_walk"]

        handlers = self._scenario_handlers()
        unknown = [n for n in resolved if n not in handlers]
        if unknown:
            known = ", ".join(sorted(handlers))
            raise ValueError(f"unknown scenario {unknown!r}. Known: {known}")

        s0 = self._cam_s(rig, 0.0)
        look = float(self.cfg.get("scenarios", {}).get("compose", {}).get("look_ahead_m", 32.0))
        s_hi = min(self.state.road.length - 5.0, s0 + max(18.0, look))
        session = ComposeSession.from_cfg(
            resolved, s_lo=s0 + 3.2, s_hi=s_hi, cfg=self.cfg,
        )
        session.seed_from_actors(self.state.actors, s0)
        self._compose = session
        try:
            for key in sort_for_inject(resolved):
                session.begin(key)
                if is_noop(key):
                    continue
                handlers[key](rig)
        finally:
            self._compose = None
        return compose_slug(resolved)

    # -- internals: scatter / background ----------------------------------

    def _occupy(self, s: float, lat: float, radius: float) -> None:
        self._sites.append((float(s), float(lat), float(radius)))

    def _site_free(self, s: float, lat: float, radius: float) -> bool:
        for ps, pl, pr in self._sites:
            ds = float(s) - ps
            dl = float(lat) - pl
            need = float(radius) + pr
            if ds * ds + dl * dl < need * need:
                return False
        return True

    def _carriage_free(
        self,
        s: float,
        lat: float,
        half_s: float,
        half_lat: float,
        road_half: float,
    ) -> bool:
        """True if (s, lat) on the carriageway is clear of in-road trunks."""
        for ps, pl, _pr in self._sites:
            if abs(pl) > float(road_half) + 0.35:
                continue
            if abs(float(s) - ps) < half_s and abs(float(lat) - pl) < half_lat:
                return False
        return True

    def _tree_free(self, s: float, lat: float, along: float, xy: float = 3.8) -> bool:
        """Keep crowns from overlapping; extra along-track gap in the same strip."""
        for ps, pl, _pr in self._sites:
            ds = abs(float(s) - ps)
            dl = abs(float(lat) - pl)
            if ds * ds + dl * dl < xy * xy:
                return False
            if dl < 2.2 and ds < along:
                return False
        return True

    def _count_range(self, key: str, default: tuple[int, int] = (0, 0)) -> int:
        lo, hi = [int(v) for v in self.cfg["world"].get(key, default)]
        if hi <= 0:
            return 0
        lo = max(0, min(lo, hi))
        return self.rng.randint(lo, hi)

    def _scatter_ground(
        self,
        road: PathSpline,
        sidewalk_lateral: float,
        sidewalk_w: float,
        collection: bpy.types.Collection,
        counters: _Counters,
        roughness: float,
    ) -> list[Actor]:
        """Shop-front furniture only. Trip holes stay scenario injectors."""
        wcfg = self.cfg["world"]
        n_lo, n_hi = wcfg["poisson"]["n_ground_static"]
        n = self.rng.randint(int(n_lo), int(n_hi))
        if n <= 0:
            return []
        samples = poisson_disk_strip(
            length=road.length,
            half_width=sidewalk_w * 0.12,
            radius=float(wcfg["poisson"]["static_radius"]),
            rng=self.rng,
            n_max=n,
            k_candidates=int(wcfg["poisson"]["k_candidates"]),
            s_min=8.0,
            s_max=road.length - 6.0,
        )
        actors: list[Actor] = []
        road_half = float(wcfg["road_width"]) * 0.5
        curb = float(wcfg["curb_height"])
        # Outer half of the sidewalk (facade / shop-front), not the gait line.
        furniture_lat = math.copysign(road_half + sidewalk_w * 0.78, sidewalk_lateral)
        for s, dlat in samples:
            lat = furniture_lat + dlat
            if not self._site_free(s, lat, 1.1):
                continue
            cls_name = self.rng.choice(FURNITURE_CLASSES)
            gz = curb if abs(lat) >= road_half - 0.08 else 0.0
            loc = road.offset_point(s, lat, z=gz)
            heading = heading_from_tangent(road.tangent(s))
            actor = spawn_ground_hazard(
                cls_name, loc, heading, collection, self.rng, counters,
                roughness, self.chaos,
            )
            actor.s = s
            actor.lateral = lat
            self._occupy(s, lat, 1.0)
            actors.append(actor)
        return actors

    def _scatter_head(
        self,
        road: PathSpline,
        sidewalk_lateral: float,
        collection: bpy.types.Collection,
        counters: _Counters,
        roughness: float,
    ) -> list[Actor]:
        """Head-level hazards sit on the *walker's* column (1.2–1.8 m AGL)."""
        wcfg = self.cfg["world"]
        n_lo, n_hi = wcfg["poisson"]["n_head_hazards"]
        n = self.rng.randint(int(n_lo), int(n_hi))
        samples = poisson_disk_strip(
            length=road.length,
            half_width=0.35,
            radius=float(wcfg["poisson"]["head_hazard_radius"]),
            rng=self.rng,
            n_max=n,
            k_candidates=int(wcfg["poisson"]["k_candidates"]),
            s_min=12.0,
            s_max=road.length - 8.0,
        )
        hlo, hhi = wcfg["head_hazard_height"]
        curb = float(wcfg["curb_height"])
        actors: list[Actor] = []
        for s, dlat in samples:
            cls_name = self.rng.choice([c[0] for c in HEAD_CLASSES])
            loc = road.offset_point(s, sidewalk_lateral + dlat, z=0.0)
            height = self.rng.uniform(hlo, hhi) + curb
            heading = heading_from_tangent(road.tangent(s))
            actors.append(
                spawn_head_hazard(
                    cls_name, loc, heading, height, collection, self.rng, counters,
                    roughness, self.chaos,
                )
            )
        return actors

    def _scatter_trees(
        self,
        road: PathSpline,
        road_half: float,
        sw: float,
        verge_w: float,
        collection: bpy.types.Collection,
        counters: _Counters,
    ) -> list[Actor]:
        """Planting-strip and (optional) planted-median trees.

        Trunks never sit on the carriageway or the walking slab. The old
        curb/median roles planted just inside the kerb / on the centreline;
        those laterals are rejected. A median tree is allowed only inside a
        planted ``median_width`` strip that driving lanes do not use.
        """
        wcfg = self.cfg["world"]
        open_ground = str(wcfg.get("ground", "paved")) == "grass"
        setback = float(wcfg.get("building_setback", 3.6))
        facade = road_half + sw + setback
        have_buildings = bool(wcfg.get("buildings", True)) and not open_ground
        median_w = float(wcfg.get("median_width", 0.0) or 0.0)
        have_median = (not open_ground) and median_w >= 1.4
        roles: list[str] = []
        roles.extend("plant" for _ in range(self._count_range("n_trees")))
        if open_ground:
            roles.extend("path" for _ in range(self._count_range("n_path_trees")))
        else:
            # "curb" is now a second planting-strip band (sidewalk-adjacent),
            # never a pit on the asphalt.
            roles.extend("curb" for _ in range(self._count_range("n_street_trees")))
            if have_median:
                roles.extend("median" for _ in range(self._count_range("n_median_trees")))
        if not roles:
            return []

        actors: list[Actor] = []
        curb = float(wcfg["curb_height"])
        s_lo, s_hi = 4.0, max(6.0, road.length - 4.0)
        walk_edge = road_half + (0.20 if open_ground else sw * 0.55)

        def _on_drive_or_walk(lat_v: float, role_name: str) -> bool:
            a = abs(float(lat_v))
            if role_name == "median":
                return a > 0.5 * median_w - 0.22
            if a < road_half + 0.08:
                return True
            if not open_ground and a < walk_edge:
                return True
            return False

        for role in roles:
            placed = False
            for _attempt in range(16):
                s = self.rng.uniform(s_lo, s_hi)
                if role == "plant" and have_buildings:
                    sides = tuple(wcfg.get("building_sides", (-1.0, 1.0)))
                    sign = float(self.rng.choice(sides))
                else:
                    sign = self.rng.choice((-1.0, 1.0))
                if role == "plant":
                    if open_ground:
                        inner = road_half + max(0.45, sw * 0.55)
                        outer = road_half + sw + max(0.5, verge_w * 0.92)
                        lat = self.rng.uniform(inner, outer) * sign
                        max_c = 2.8
                    elif have_buildings:
                        # Sit in the planting strip, crown just shy of the wall.
                        canopy_guess = self.rng.uniform(1.0, 2.4)
                        lat_abs = facade - canopy_guess - 0.40
                        min_lat = road_half + sw + 0.50
                        if lat_abs < min_lat:
                            canopy_guess = max(0.55, facade - 0.40 - min_lat)
                            lat_abs = min_lat
                        lat = sign * (lat_abs + self.rng.uniform(-0.12, 0.12))
                        max_c = canopy_guess
                    else:
                        lat = sign * (road_half + sw + self.rng.uniform(0.6, 1.8))
                        max_c = 2.2
                    if offset_folds(
                        road, s, lat, lat, max(0.7, max_c * 0.55),
                        road_half + (0.15 if open_ground else sw - 0.10),
                    ):
                        continue
                    along = 7.5
                elif role == "curb":
                    # Planting strip, sidewalk-adjacent — not the carriageway.
                    if setback < 1.05 and have_buildings:
                        continue
                    lat = sign * (road_half + sw + self.rng.uniform(0.35, min(1.15, max(0.45, setback * 0.45))))
                    max_c = 1.35
                    along = 8.0
                    if offset_folds(
                        road, s, lat, lat, 0.80,
                        road_half + sw - 0.10,
                    ):
                        continue
                elif role == "path":
                    # Park: off the gravel gait, in the verge.
                    lat = sign * (road_half + self.rng.uniform(0.70, max(1.4, verge_w * 0.55)))
                    max_c = 1.55
                    along = 6.0
                else:
                    if not have_median:
                        continue
                    half = 0.5 * median_w
                    lat = self.rng.uniform(-max(0.08, half - 0.35), max(0.08, half - 0.35))
                    max_c = min(1.35, half - 0.15)
                    along = 10.0
                if _on_drive_or_walk(lat, role):
                    continue
                if not self._tree_free(s, lat, along=along):
                    continue
                z = curb if abs(lat) >= road_half - 0.08 else 0.03
                loc = road.offset_point(s, lat, z=z)
                heading = heading_from_tangent(road.tangent(s))
                actor = spawn_tree(
                    self.lib, self.rng, self.cfg, collection, loc, counters,
                    self.chaos,
                    max_canopy=max_c,
                    heading=heading,
                    lean_to_road=(role in ("plant", "curb") and not open_ground),
                    lat_sign=lat,
                    wind=self._wind,
                )
                actor.s = s
                actor.lateral = lat
                self._occupy(s, lat, 2.2)
                actors.append(actor)
                placed = True
                break
            if not placed:
                continue
        return actors

    def _spawn_canopy_gobo(
        self,
        road: PathSpline,
        half_width: float,
        collection: bpy.types.Collection,
    ) -> Optional[bpy.types.Object]:
        """Overhead alpha-punched sheet that dapples the whole corridor.

        Modelling every leaf of an avenue of plane trees would be thousands
        of instances for an effect that only ever reaches the ground as a
        shadow. One alpha-hashed sheet at canopy height produces the same
        moving pattern of light and shade for the cost of a quad strip.

        The sheet is hidden from camera rays where the build supports it, so
        it contributes shadows only and can never appear as a ceiling.
        """
        from materials import make_canopy_gobo

        mat = make_canopy_gobo("canopy_gobo", self.rng, self.chaos)
        z = self.rng.uniform(6.5, 10.5)
        w = max(6.0, float(half_width) * self.rng.uniform(1.1, 1.8))
        obj = _ribbon_mesh("canopy_gobo", road, -w, w, z, collection, mat)
        for attr in ("visible_camera", "visible_diffuse", "visible_glossy"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, False)
                except Exception:
                    pass
        if hasattr(obj, "visible_shadow"):
            obj.visible_shadow = True
        return obj

    def _spawn_streetlamps(
        self,
        road: PathSpline,
        road_half: float,
        sw: float,
        world_col: bpy.types.Collection,
        lights_col: bpy.types.Collection,
        counters: _Counters,
    ) -> tuple[list[bpy.types.Object], list[Actor]]:
        n = int(self.cfg["world"]["n_streetlamps"])
        h = float(self.cfg["world"]["streetlamp_height"])
        lamps: list[bpy.types.Object] = []
        actors: list[Actor] = []
        if n <= 0:
            return lamps, actors
        from materials import make_emissive
        for i in range(n):
            s = (i + 0.5) * road.length / n
            sign = -1.0 if i % 2 == 0 else 1.0
            lat = sign * (road_half + sw * 0.88)
            if not self._site_free(s, lat, 1.6):
                continue
            tan = road.tangent(s)
            right = road.right(s)
            base = road.offset_point(s, lat, z=float(self.cfg["world"]["curb_height"]))
            self._occupy(s, lat, 1.4)
            actor = spawn_streetlamp(
                base,
                heading_from_tangent(tan),
                world_col,
                counters,
                height=h,
                with_arm=True,
                arm_sign=sign,
            )
            actor.s = s
            actor.lateral = lat
            actors.append(actor)
            pole = actor.obj
            bulb_world = base + right * (-sign * 0.70) + Vector((0.0, 0.0, h + 0.02))
            create_box(
                f"{actor.instance_id}_bulb",
                (0.26, 0.26, 0.12),
                bulb_world,
                world_col,
                make_emissive(f"{actor.instance_id}_bulb", (1.0, 0.84, 0.52), 28.0),
            )
            light = bpy.data.lights.new(f"lamp_{i:02d}", type="SPOT")
            light.energy = 0.0
            light.color = self.rng.choice((
                (1.00, 0.72, 0.34),
                (1.00, 0.84, 0.58),
                (0.78, 0.86, 1.00),
                (0.96, 0.97, 1.00),
                (0.62, 1.00, 0.78),
            ))
            light.spot_size = math.radians(self.rng.uniform(58.0, 96.0))
            light.spot_blend = self.rng.uniform(0.28, 0.62)
            if hasattr(light, "shadow_soft_size"):
                light.shadow_soft_size = 0.45
            if hasattr(light, "use_shadow"):
                light.use_shadow = False
            lobj = bpy.data.objects.new(f"lamp_{i:02d}", light)
            lobj["btp_lamp_dead"] = int(self.rng.random() < 0.28)
            lobj["btp_lamp_gain"] = float(self.rng.uniform(0.55, 1.5))
            lobj.location = bulb_world
            lobj.rotation_euler = (0.0, 0.0, 0.0)
            _link(lobj, lights_col)
            lamps.append(lobj)
        return lamps, actors

    def _spawn_background_vehicles(
        self,
        road: PathSpline,
        sidewalk_lateral: float,
        collection: bpy.types.Collection,
        counters: _Counters,
    ) -> list[Actor]:
        wcfg = self.cfg["world"]
        dr = self.cfg["domain_randomization"]
        n_lo, n_hi = wcfg["n_background_vehicles"]
        n = self.rng.randint(int(n_lo), int(n_hi))
        actors: list[Actor] = []
        used_s: list[float] = []
        # Opposite / far lane so D_cpa stays > 1.5 m versus the sidewalk walker.
        # Median trees push the travel lane outward so hulls do not sit inside
        # a crown.
        road_half = float(wcfg["road_width"]) * 0.5
        lane_abs = self._drive_lane_abs()
        if any(abs(lat) < road_half * 0.45 and rad >= 3.0 for _s, lat, rad in self._sites):
            lane_abs = max(lane_abs, road_half * 0.62)
            lane_abs = min(lane_abs, max(1.15, road_half - 1.15))
        lane = -math.copysign(lane_abs, sidewalk_lateral)
        for _ in range(n):
            s = self.rng.uniform(4.0, max(5.0, road.length - 15.0))
            if any(abs(s - u) < 8.0 for u in used_s):
                continue
            if not self._carriage_free(s, lane, 3.4, 1.35, road_half):
                continue
            used_s.append(s)
            speed = self.rng.uniform(*wcfg["vehicle_speed"])
            # Half the cars travel against the path parameter (oncoming).
            direction = self.rng.choice((-1.0, 1.0))
            loc = road.offset_point(s, lane, z=0.0)
            heading = heading_from_tangent(road.tangent(s) * direction)
            color = self.rng.choice(list(dr["vehicle_palette"]))
            actor = spawn_vehicle(
                loc, heading, collection, color, self.rng.uniform(*dr["roughness"]),
                counters, self.rng, self.chaos,
            )
            actor.follow_spline = road
            actor.s = s
            actor.lateral = lane
            actor.speed = speed * direction
            actor.behavior = "cruise"
            actor.allow_sidewalk = False
            actor.corridor_pad = 1.05
            # Lane keeping is imperfect: a small drift inside the lane.
            self._add_wander(actor, (0.05, 0.20), (0.08, 0.22), speed_amp=(0.0, 0.12))
            actors.append(actor)
        return actors

    def _spawn_background_pedestrians(
        self,
        road: PathSpline,
        sidewalk_lateral: float,
        collection: bpy.types.Collection,
        counters: _Counters,
    ) -> list[Actor]:
        wcfg = self.cfg["world"]
        dr = self.cfg["domain_randomization"]
        n_lo, n_hi = wcfg["n_background_pedestrians"]
        n = self.rng.randint(int(n_lo), int(n_hi))
        actors: list[Actor] = []
        for _ in range(n):
            # Opposite sidewalk, or same sidewalk but far ahead / behind.
            if self.rng.random() < 0.55:
                lat = -sidewalk_lateral
            else:
                lat = sidewalk_lateral + self.rng.choice((-0.55, 0.55))
            s = self.rng.uniform(10.0, max(12.0, road.length - 8.0))
            if not self._site_free(s, lat, 1.6):
                continue
            direction = self.rng.choice((-1.0, 1.0))
            speed = self.rng.uniform(*wcfg["pedestrian_speed"])
            loc = road.offset_point(s, lat, z=0.0)
            heading = heading_from_tangent(road.tangent(s) * direction)
            color = self.rng.choice(list(dr["pedestrian_palette"]))
            actor = spawn_pedestrian(
                loc, heading, collection, color, self.rng.uniform(*dr["roughness"]),
                counters, self.rng, self.chaos,
            )
            actor.follow_spline = road
            actor.s = s
            actor.lateral = lat
            actor.speed = speed * direction
            actor.behavior = "cruise"
            actor.allow_sidewalk = True
            actor.corridor_pad = 0.35
            # People do not walk on rails: fBm drift plus a speed wobble.
            self._add_wander(actor, (0.10, 0.42), (0.10, 0.34), speed_amp=(0.05, 0.30))
            actors.append(actor)
        return actors

    def _add_wander(
        self,
        actor: Actor,
        amp: tuple[float, float],
        rate: tuple[float, float],
        speed_amp: tuple[float, float] = (0.0, 0.0),
    ) -> Actor:
        """Attach independent fBm streams for lateral drift and speed.

        Each actor gets its own Perlin seed, so a crowd decorrelates instead
        of swaying in unison. Kept off scenario actors whose CPA is the whole
        point of the episode — see `_bind`.
        """
        seed = self.rng.randrange(1, 2**31)
        actor.wander_amp = float(self.rng.uniform(*amp))
        actor.wander_rate = float(self.rng.uniform(*rate))
        actor.wander_noise = Perlin1D(seed)
        hi = float(speed_amp[1])
        if hi > 1e-6:
            actor.speed_amp = float(self.rng.uniform(float(speed_amp[0]), hi))
            actor.speed_noise = Perlin1D(seed + 7919)
        return actor

    # -- internals: forced collisions (Frenet, corridor-clamped) ----------

    def _append(self, actor: Actor) -> Actor:
        assert self.state is not None
        self.state.actors.append(actor)
        # Any new actor invalidates the frozen partitions.
        self._movers = None
        self._annotatable = None
        return actor

    def _palette_ped(self) -> tuple[float, float, float]:
        return self.rng.choice(list(self.cfg["domain_randomization"]["pedestrian_palette"]))

    def _palette_car(self) -> tuple[float, float, float]:
        return self.rng.choice(list(self.cfg["domain_randomization"]["vehicle_palette"]))

    def _cam_sl(self, rig: Any, t: float) -> tuple[float, float]:
        """Ego (s, lateral) at time `t`, following the rig's own schedule.

        Reading `lateral_at` rather than the static `rig.lateral` is what
        makes intercepts land on a diagonal crossing: otherwise every
        injector aims at the kerb the walker started from.
        """
        assert self.state is not None
        getter = getattr(rig, "lateral_at", None)
        if callable(getter):
            lat = float(getter(float(t)))
        else:
            lat = float(getattr(rig, "lateral", self.state.sidewalk_lateral))
        return self._cam_s(rig, t), lat

    def _drive_lane_abs(self) -> float:
        """Travel-lane |lateral|, always outside a planted median."""
        wcfg = self.cfg["world"]
        road_half = float(wcfg["road_width"]) * 0.5
        lane = float(wcfg["lane_offset"])
        med = float(wcfg.get("median_width", 0.0) or 0.0)
        if med >= 1.4:
            lane = max(lane, 0.5 * med + 1.55)
        return min(lane, max(1.05, road_half - 1.05))

    def _near_lane(self) -> float:
        assert self.state is not None
        return math.copysign(self._drive_lane_abs(), self.state.sidewalk_lateral)

    def _far_lane(self) -> float:
        return -self._near_lane()

    def _resolve_lat(self, dlat: Any, L: float, pad: float) -> float:
        """`dlat` is 'opposite', a signed offset from the walker, or an absolute."""
        assert self.state is not None
        if dlat == "opposite":
            lat = -L
        elif dlat == "near_lane":
            lat = self._near_lane()
        elif dlat == "far_lane":
            lat = self._far_lane()
        else:
            # Positive: toward the road (smaller |lateral|). Negative: toward facade.
            delta = float(dlat)
            if abs(L) > 1e-6:
                lat = L - math.copysign(delta, L)
            else:
                lat = L - delta
        return self.state.corridor.confine(0.0, lat, pad, True)[1]

    def _kind_pad(self, kind: str) -> float:
        if kind in SHAPE_KINDS or kind in ("shape", "random"):
            return 0.28
        return {"person": 0.35, "vehicle": 1.05, "bicycle": 0.40, "cube": 0.25}.get(kind, 0.30)

    def _spawn_kind(
        self,
        kind: str,
        s: float,
        lat: float,
        heading_sign: float,
        cube_size: float = 0.45,
        cube_z: Optional[float] = None,
        child: bool = False,
    ) -> Actor:
        assert self.state is not None
        col = self.state.collections["actors"]
        loc = self.state.road.offset_point(s, lat, z=0.0)
        heading = heading_from_tangent(self.state.road.tangent(s) * heading_sign)
        rough = self.rng.uniform(*self.cfg["domain_randomization"]["roughness"])
        if kind == "person":
            actor = spawn_pedestrian(
                loc, heading, col, self._palette_ped(), rough,
                self.state.counters, self.rng, self.chaos, child=child,
            )
            actor.corridor_pad = 0.24 if child else 0.35
            actor.allow_sidewalk = True
            return actor
        if kind == "vehicle":
            actor = spawn_vehicle(
                loc, heading, col, self._palette_car(), 0.45,
                self.state.counters, self.rng, self.chaos,
            )
            actor.corridor_pad = 1.05
            actor.allow_sidewalk = True
            return actor
        if kind == "bicycle":
            actor = spawn_bicycle(
                loc, heading, col, self._palette_car(), rough,
                self.state.counters, self.rng, self.chaos,
            )
            actor.corridor_pad = 0.40
            actor.allow_sidewalk = True
            return actor
        shape_kind = "cube"
        if kind in SHAPE_KINDS:
            shape_kind = kind
        elif kind in ("shape", "random"):
            shape_kind = "random"
        actor = spawn_threat_shape(
            loc, col, self.state.counters, self.rng,
            kind=shape_kind, size=cube_size, z=cube_z, chaos=self.chaos,
            lib=self.lib,
        )
        actor.allow_sidewalk = True
        return actor

    def _bind(
        self,
        actor: Actor,
        s: float,
        lat: float,
        *,
        speed: float = 0.0,
        lat_speed: float = 0.0,
        lat_target: Optional[float] = None,
        stop_t: Optional[float] = None,
        swerve_t: Optional[float] = None,
        behavior: str = "cruise",
        allow_sidewalk: bool = True,
        pad: Optional[float] = None,
        turn_t: Optional[float] = None,
        turn_dt: float = 1.60,
        post_speed: Optional[float] = None,
        post_lat_speed: float = 0.0,
        post_lat_target: Optional[float] = None,
        look_flip: bool = False,
    ) -> Actor:
        assert self.state is not None
        if pad is not None:
            actor.corridor_pad = float(pad)
        actor.allow_sidewalk = allow_sidewalk
        actor.follow_spline = self.state.road
        actor.s, actor.lateral = self.state.corridor.confine(
            s, lat, actor.corridor_pad, allow_sidewalk
        )
        if self._compose is not None:
            hs, hl = extents_for(actor.class_name, actor.corridor_pad)
            placed = self._compose.reserve(
                s=actor.s,
                lat=actor.lateral,
                half_s=hs,
                half_lat=hl,
                ds=float(speed),
                lat_speed=abs(float(lat_speed)),
                lat_target=lat_target,
                class_name=str(actor.class_name),
                behavior=behavior,
                allow_flip_lat=lat_target is not None,
                allow_lane_flip=(
                    actor.class_name == "vehicle"
                    and behavior in ("oncoming", "cruise", "parked")
                ),
                turn_t=turn_t,
                turn_dt=turn_dt,
                post_speed=post_speed,
                post_lat_speed=abs(float(post_lat_speed)),
                post_lat_target=post_lat_target,
                swerve_t=swerve_t,
            )
            actor.s, actor.lateral = self.state.corridor.confine(
                placed.s, placed.lat, actor.corridor_pad, allow_sidewalk
            )
            if placed.lat_target is not None:
                lat_target = placed.lat_target
        actor.speed = float(speed)
        actor.lat_speed = abs(float(lat_speed))
        actor.lat_target = lat_target
        actor.stop_t = stop_t
        actor.swerve_t = swerve_t
        actor.behavior = behavior
        actor.turn_t = turn_t
        actor.turn_dt = float(turn_dt)
        actor.post_speed = post_speed
        actor.post_lat_speed = abs(float(post_lat_speed))
        actor.post_lat_target = post_lat_target
        if lat_target is not None and behavior not in ("through", "dart"):
            # Full-span smootherstep starts at rest. A through-crosser on the
            # FOV edge would sit still for a second then lurch — skip it so
            # they enter already walking. Cut-in / merge still ease.
            actor.lat_ease_from = float(actor.lateral)
            actor.lat_ease_t0 = float(swerve_t) if swerve_t is not None else 0.0
            dist = abs(float(lat_target) - float(actor.lateral))
            speed = abs(float(lat_speed))
            dur = dist / max(speed, 0.08) if speed > 1e-8 else 2.0
            actor.lat_ease_dur = max(1.50, float(dur))
        if look_flip:
            actor.look_flip = True
        loc = self.state.road.offset_point(
            actor.s, actor.lateral, actor.origin_z + self.state.corridor.ground_z(actor.lateral)
        )
        actor.obj.location = loc
        tan = self.state.road.tangent(actor.s)
        _p, _t, right = self.state.road.frame(actor.s)
        vlat = 0.0
        if lat_target is not None:
            vlat = math.copysign(actor.lat_speed, float(lat_target) - actor.lateral)
        elif abs(lat_speed) > 1e-8:
            vlat = float(lat_speed)
        heading = tan * actor.speed + right * vlat
        if heading.length < 1e-4:
            heading = tan if actor.speed >= 0.0 else -tan
        if actor.look_flip:
            heading = -heading
        look_along(actor.obj, heading)
        return self._append(actor)

    def _inject_none(self, rig: Any) -> None:
        return

    def _gait_lat(self) -> float:
        assert self.state is not None
        return float(self.state.sidewalk_lateral)

    def _cam_s(self, rig: Any, t: float) -> float:
        """Road arc-length of the ego at time `t` (same parameter actors use).

        Prefers the rig's integrated arc table so hesitation and speed
        modulation are accounted for; a rig without one is a constant-speed
        walk and gets the closed form.
        """
        assert self.state is not None
        getter = getattr(rig, "arc_length_at", None)
        if callable(getter):
            return self.state.corridor.clamp_s(float(getter(float(t))))
        s0 = float(getattr(rig, "sidewalk_s0", 3.0))
        return self.state.corridor.clamp_s(s0 + float(rig.walk_speed) * float(t))

    def _ego_speed(self, rig: Any) -> float:
        """Nominal ego ground speed used to size intercepts.

        May be exactly 0 (seated / hesitation). Callers that convert a TTC
        into an along-track spawn must use ``max(v_ego + v_obj, v_obj)`` so
        a stationary camera still gets the object placed ``v_obj * tau``
        metres ahead rather than on top of the HMD. Nothing here floors
        the value — a fake 0.25 m/s would desynchronise the intercept from
        the kinematics the annotator later writes.
        """
        return max(0.0, abs(float(getattr(rig, "walk_speed", 0.0) or 0.0)))

    def _frustum_half_width(self, depth_m: float, frac: float = 0.90) -> float:
        """Half-width of the image at `depth_m` along the gait, as a lateral offset."""
        hfov = float(self.cfg["camera"].get("hfov_deg") or 73.7)
        half = math.radians(max(28.0, hfov) * 0.5)
        return max(1.20, float(depth_m) * math.tan(half * float(frac)))

    def _visible_lead(self, preferred: float, *, near: float = 3.6, far: float = 9.2) -> float:
        """Clamp along-track spawn so the actor stays in the walking VFOV."""
        look = float(self.cfg.get("scenarios", {}).get("compose", {}).get("look_ahead_m", 32.0))
        hi = min(float(far), max(float(near) + 1.0, look * 0.42))
        return min(max(float(preferred), float(near)), hi)

    def _inject_through_cross(
        self,
        rig: Any,
        *,
        kind: str,
        cpa: float,
        from_left: bool,
        speed: float,
        cube_size: float = 0.45,
        cube_z: Optional[float] = None,
    ) -> None:
        """Enter one side of the frame and walk/roll all the way out the other.

        Stays on the street ribbon. Lateral target is the far FOV/corridor edge,
        not a 2 m shuffle that dies in the middle of the road.
        """
        assert self.state is not None
        pad = self._kind_pad(kind)
        s0 = self._cam_s(rig, 0.0)
        L = self._gait_lat()
        lim = self.state.corridor.lateral_limit(pad, True)
        walk = max(0.85, float(speed))
        # Place the crossing a few metres ahead so they occupy the frame for seconds.
        tan_b = math.tan(math.radians(max(28.0, float(self.cfg["camera"].get("hfov_deg") or 73.7)) * 0.5) * 0.90)
        k = 1.0 - float(rig.walk_speed) * tan_b / max(walk, 0.1)
        depth = 1.7 / k if k > 0.18 else 5.2
        depth = min(max(depth, 3.8), 7.0)
        depth = depth + 1.1 * max(0.0, float(cpa) - 0.25)
        if self._compose is not None:
            extra, from_left = self._compose.take_cross_layout(from_left)
            depth = depth + extra
        near = 6.6 if kind == "vehicle" else 3.8
        far = 10.5 if kind == "vehicle" else 9.0
        depth = self._visible_lead(depth, near=near, far=far)
        hw = self._frustum_half_width(depth, 0.92)
        left = max(-lim, L - hw)
        right = min(lim, L + hw)
        if from_left:
            lat_start = left
            lat_end = min(lim, right + 0.55)
        else:
            lat_start = right
            lat_end = max(-lim, left - 0.55)
        # Small miss offset so critical vs near-miss is a graze, not a teleport.
        # The path still runs edge-to-edge; CPA comes from when they pass the gait line.
        s = s0 + depth
        heading_sign = 1.0 if from_left else -1.0
        actor = self._spawn_kind(
            kind, s, lat_start, heading_sign=heading_sign,
            cube_size=cube_size, cube_z=cube_z,
        )
        self._bind(
            actor, s, lat_start,
            speed=0.0,
            lat_speed=walk,
            lat_target=lat_end,
            behavior="through",
            allow_sidewalk=True,
            pad=pad,
        )

    def _inject_jaywalk_turn(self, rig: Any, *, toward: bool, cpa: float) -> None:
        """Walk in from the side, then turn smoothly onto the sidewalk.

        `toward=True`  — after the turn they walk at you (oncoming on the gait).
        `toward=False` — after the turn they walk with you (ahead, pulling away).
        Heading rotates with velocity via a smoothstep blend (~0.85 s), not a snap.
        """
        assert self.state is not None
        pad = 0.35
        s0 = self._cam_s(rig, 0.0)
        L = self._gait_lat()
        lim = self.state.corridor.lateral_limit(pad, True)
        walk = self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"])
        from_left = self.rng.random() < 0.5
        depth = 5.4
        if self._compose is not None:
            extra, from_left = self._compose.take_cross_layout(from_left)
            depth = depth + extra
        depth = self._visible_lead(depth, near=4.0, far=8.8)
        hw = min(self._frustum_half_width(depth, 0.88), lim * 0.95)
        lat_start = max(-lim, L - hw) if from_left else min(lim, L + hw)
        # Merge onto the gait line, with a small CPA offset.
        miss = float(cpa)
        if toward:
            lat_merge = L + (-miss if from_left else miss)
            post_speed = -walk
        else:
            # Shoulder off the gait so the merge is a near-miss, then pull away.
            lat_merge = L + math.copysign(max(miss, 1.05), -1.0 if from_left else 1.0)
            post_speed = float(rig.walk_speed) + 0.35
        lat_merge = max(-lim, min(lim, lat_merge))
        dist = abs(lat_merge - lat_start)
        t_arrive = dist / max(walk, 0.08)
        turn_t = max(0.25, t_arrive - 0.40)
        s = s0 + depth
        heading_sign = 1.0 if from_left else -1.0
        actor = self._spawn_kind("person", s, lat_start, heading_sign=heading_sign)
        self._bind(
            actor, s, lat_start,
            speed=0.0,
            lat_speed=walk,
            lat_target=lat_merge,
            behavior="turn_toward" if toward else "turn_away",
            allow_sidewalk=True,
            pad=pad,
            turn_t=turn_t,
            turn_dt=1.60,
            post_speed=post_speed,
            post_lat_speed=0.15,
            post_lat_target=lat_merge,
        )

    def _inject_oncoming(
        self,
        rig: Any,
        *,
        kind: str,
        obj_speed: float,
        tau: float,
        dlat: Any,
        pad: Optional[float] = None,
        cube_size: float = 0.45,
        cube_z: Optional[float] = None,
        allow_sidewalk: bool = True,
    ) -> None:
        """Actor approaching along −s (toward the camera) at `obj_speed`."""
        assert self.state is not None
        use_pad = float(pad) if pad is not None else self._kind_pad(kind)
        s0, L = self._cam_sl(rig, 0.0)
        lat = self._resolve_lat(dlat, L, use_pad)
        v_close = float(self._ego_speed(rig)) + float(obj_speed)
        tau = float(tau)
        if self._compose is not None:
            tau = tau + self._compose.take_group_offset("along") / max(v_close, 1.0)
        # Seated ego: closing speed is just obj_speed, so spawn at v_obj * tau.
        s = s0 + max(v_close, float(obj_speed)) * tau
        s = min(max(2.0, s), self.state.road.length - 4.0)
        actor = self._spawn_kind(
            kind, s, lat, heading_sign=-1.0, cube_size=cube_size, cube_z=cube_z,
        )
        self._bind(
            actor, s, lat,
            speed=-float(obj_speed),
            behavior="oncoming",
            allow_sidewalk=allow_sidewalk,
            pad=use_pad,
        )

    def _inject_cyclist_overtake(self, rig: Any) -> None:
        s0, L = self._cam_sl(rig, 0.0)
        lat = self._resolve_lat(0.85, L, 0.40)
        s = self._visible_lead(s0 + 2.2 - s0, near=1.6, far=4.2) + s0
        if self._compose is not None:
            s = s + self._compose.take_group_offset("along")
        speed = max(float(rig.walk_speed) + 2.4, 3.6)
        actor = self._spawn_kind("bicycle", s, lat, heading_sign=1.0)
        self._bind(
            actor, s, lat,
            speed=speed,
            behavior="overtake",
            allow_sidewalk=True,
            pad=0.40,
        )

    def _inject_group_crossing(self, rig: Any) -> None:
        n = self.rng.randint(2, 3)
        cpa0 = float(self.cfg["scenarios"]["near_miss_cpa_target"])
        for i in range(n):
            self._inject_through_cross(
                rig,
                kind="person",
                cpa=cpa0 + 0.28 * i,
                from_left=(i % 2 == 0),
                speed=self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"]),
            )

    def _inject_parked_car_door(self, rig: Any) -> None:
        """Parked car on the near gutter; an open door occupies the gait."""
        assert self.state is not None
        s0, L = self._cam_sl(rig, 0.0)
        s = s0 + self._visible_lead(6.4, near=4.8, far=8.5)
        if self._compose is not None:
            s = s + self._compose.take_group_offset("static")
        lat_car = self._resolve_lat(1.25, L, 1.05)
        actor = self._spawn_kind("vehicle", s, lat_car, heading_sign=1.0)
        actor.category = "static"
        self._bind(
            actor, s, lat_car,
            speed=0.0,
            behavior="parked",
            allow_sidewalk=False,
            pad=1.05,
        )
        door_lat = L + math.copysign(0.12, L if L else 1.0)
        door = self._spawn_kind(
            "cube", s + 0.35, door_lat, heading_sign=1.0,
            cube_size=0.42, cube_z=0.85,
        )
        door.category = "static"
        self._bind(
            door, s + 0.35, door_lat,
            speed=0.0,
            behavior="static",
            allow_sidewalk=True,
            pad=0.22,
        )

    def _inject_backing_vehicle(self, rig: Any) -> None:
        """Car ahead, facing away, rolling back toward the walker."""
        s0, _L = self._cam_sl(rig, 0.0)
        lat = self._near_lane()
        v = self.rng.uniform(1.6, 2.8)
        s = s0 + self._visible_lead(6.8, near=5.0, far=8.8)
        if self._compose is not None:
            s = s + self._compose.take_group_offset("along")
        actor = self._spawn_kind("vehicle", s, lat, heading_sign=1.0)
        actor.look_flip = True
        self._bind(
            actor, s, lat,
            speed=-v,
            behavior="backing",
            allow_sidewalk=False,
            pad=1.05,
            look_flip=True,
        )

    def _inject_parallel_person(self, rig: Any) -> None:
        s0, L = self._cam_sl(rig, 0.0)
        lat = self._resolve_lat(0.80, L, 0.35)
        s = s0 + 4.0
        if self._compose is not None:
            s = s + self._compose.take_group_offset("along")
        actor = self._spawn_kind("person", s, lat, heading_sign=1.0)
        self._bind(
            actor, s, lat,
            speed=float(rig.walk_speed),
            behavior="parallel",
            allow_sidewalk=True,
            pad=0.35,
        )

    def _inject_cyclist_same_way(self, rig: Any) -> None:
        s0, _L = self._cam_sl(rig, 0.0)
        lat = self._far_lane()
        s = s0 + 7.0
        if self._compose is not None:
            s = s + self._compose.take_group_offset("along")
        speed = self.rng.uniform(*self.cfg["world"]["bicycle_speed"])
        actor = self._spawn_kind("bicycle", s, lat, heading_sign=1.0)
        self._bind(
            actor, s, lat,
            speed=speed,
            behavior="cruise",
            allow_sidewalk=False,
            pad=0.40,
        )

    def _inject_periph_car_side(self, rig: Any) -> None:
        """Oncoming (or same-way) car that stays in the far driving lane.

        Spawn is close enough that the hull sits on a FOV edge, not a speck
        at the vanishing point — that is what keeps the image centre cold.
        """
        assert self.state is not None
        s0, _L = self._cam_sl(rig, 0.0)
        lat = self._far_lane()
        v = float(self.rng.uniform(*self.cfg["world"]["vehicle_speed"]))
        oncoming = self.rng.random() < 0.75
        if oncoming:
            # Far enough to stay in frame for a few seconds, close enough
            # that the hull sits on a FOV side — not a vanishing-point speck.
            depth = self._visible_lead(13.4, near=10.5, far=16.5)
            sign = -1.0
        else:
            depth = self._visible_lead(7.6, near=5.6, far=10.5)
            sign = 1.0
        if self._compose is not None:
            depth = depth + min(2.2, abs(self._compose.take_group_offset("along")))
        s = min(max(2.0, s0 + depth), self.state.road.length - 6.0)
        actor = self._spawn_kind("vehicle", s, lat, heading_sign=sign)
        self._bind(
            actor, s, lat,
            speed=sign * v,
            behavior="cruise",
            allow_sidewalk=False,
            pad=1.05,
        )

    def _inject_periph_parked(self, rig: Any) -> None:
        """Parked car in a gutter — visible beside the empty sidewalk."""
        s0, _L = self._cam_sl(rig, 0.0)
        depth = self._visible_lead(7.4, near=5.6, far=10.0)
        if self._compose is not None:
            depth = depth + self._compose.take_group_offset("along")
        s = s0 + depth
        lat = self._far_lane() if self.rng.random() < 0.55 else self._near_lane()
        actor = self._spawn_kind("vehicle", s, lat, heading_sign=1.0)
        actor.category = "static"
        self._bind(
            actor, s, lat,
            speed=0.0,
            behavior="parked",
            allow_sidewalk=False,
            pad=1.05,
        )

    def _inject_periph_ped_side(self, rig: Any) -> None:
        """Person on the opposite sidewalk — never on the gait line."""
        s0, L = self._cam_sl(rig, 0.0)
        lat = self._resolve_lat("opposite", L, 0.35)
        depth = self._visible_lead(6.8, near=4.8, far=9.6)
        if self._compose is not None:
            depth = depth + self._compose.take_group_offset("along")
        s = s0 + depth
        sign = -1.0 if self.rng.random() < 0.40 else 1.0
        speed = float(self.rng.uniform(*self.cfg["world"]["pedestrian_speed"]))
        actor = self._spawn_kind("person", s, lat, heading_sign=sign)
        self._bind(
            actor, s, lat,
            speed=sign * speed,
            behavior="cruise",
            allow_sidewalk=True,
            pad=0.35,
        )

    def _inject_periph_car_swerve(self, rig: Any, *, runoff: bool) -> None:
        """Car visible in the near driving lane, then turns onto the gait.

        A far-lane start walks out of the HFOV before the turn is visible.
        Near-lane + ~12 m lead keeps the hull on the road side of the frame
        for a beat, then the swerve brings it into the image centre.
        """
        assert self.state is not None
        s0, L = self._cam_sl(rig, 0.0)
        lane = self._near_lane()
        v_car = (
            self.rng.uniform(5.8, 7.6) if runoff else self.rng.uniform(5.0, 6.6)
        )
        depth = self._visible_lead(12.2, near=10.0, far=14.5)
        if self._compose is not None:
            depth = depth + min(2.2, abs(self._compose.take_group_offset("along")))
        s = min(max(2.0, s0 + depth), self.state.road.length - 5.0)
        t0 = self.rng.uniform(0.35, 0.70)
        closing = float(v_car) + float(self._ego_speed(rig))
        t_hit = max(t0 + 1.70, min(3.10, depth / max(closing, 1.0)))
        if runoff:
            lim = self.state.corridor.lateral_limit(1.05, True)
            lat_target = math.copysign(lim, L if L else 1.0)
            behavior = "run_off_road"
        else:
            cpa = float(self.cfg["scenarios"]["critical_cpa_target"])
            lat_target = L + math.copysign(cpa, 1.0 if L >= 0.0 else -1.0)
            behavior = "cut_in"
        ease = max(1.60, min(2.40, t_hit - t0))
        t0 = max(0.12, t_hit - ease)
        rate = abs(lat_target - lane) / max(0.35, ease)
        actor = self._spawn_kind("vehicle", s, lane, heading_sign=-1.0)
        self._bind(
            actor, s, lane,
            speed=-float(v_car),
            lat_speed=rate,
            lat_target=lat_target,
            swerve_t=t0,
            behavior=behavior,
            allow_sidewalk=True,
            pad=1.05,
        )

    def _inject_crossing_car(self, rig: Any, *, far: bool) -> None:
        """Traffic on the carriageway while the ego crosses.

        The car stays in a driving lane and rolls along the road (oncoming).
        It must not spawn on the sidewalk or slide laterally with the walker —
        that was the through-cross path, which reads as a car on the footpath
        turning with the camera.
        """
        assert self.state is not None
        s0, _L = self._cam_sl(rig, 0.0)
        if self._compose is not None:
            far = self._compose.prefer_far_lane(far)
        lat = self._far_lane() if far else self._near_lane()
        v = float(self.rng.uniform(*self.cfg["world"]["cross_car_speed"]))
        depth = self._visible_lead(8.4, near=6.6, far=11.5)
        if self._compose is not None:
            depth = depth + min(2.4, abs(self._compose.take_group_offset("along")))
        s = min(max(2.0, s0 + depth), self.state.road.length - 6.0)
        actor = self._spawn_kind("vehicle", s, lat, heading_sign=-1.0)
        self._bind(
            actor, s, lat,
            speed=-v,
            behavior="oncoming",
            allow_sidewalk=False,
            pad=1.05,
        )

    def _inject_car_lane(self, rig: Any, *, far: bool, obj_speed: float, tau: float) -> None:
        assert self.state is not None
        s0, _L = self._cam_sl(rig, 0.0)
        if self._compose is not None:
            far = self._compose.prefer_far_lane(far)
        lat = self._far_lane() if far else self._near_lane()
        v_close = float(self._ego_speed(rig)) + float(obj_speed)
        tau = float(tau)
        if self._compose is not None:
            tau = tau + self._compose.take_group_offset("along") / max(v_close, 1.0)
        s = min(max(2.0, s0 + max(v_close, float(obj_speed)) * tau), self.state.road.length - 6.0)
        actor = self._spawn_kind("vehicle", s, lat, heading_sign=-1.0)
        self._bind(
            actor, s, lat,
            speed=-float(obj_speed),
            behavior="oncoming",
            allow_sidewalk=False,
            pad=1.05,
        )

    def _inject_distant_jaywalk(self, rig: Any) -> None:
        assert self.state is not None
        s0, L = self._cam_sl(rig, 0.0)
        pad = 0.35
        lim = self.state.corridor.lateral_limit(pad, True)
        s = min(s0 + 16.0, self.state.road.length - 8.0)
        if self._compose is not None:
            extra, _side = self._compose.take_cross_layout(True)
            s = min(s + extra, self.state.road.length - 8.0)
        speed = self.rng.uniform(*self.cfg["scenarios"]["cross_person_speed"])
        lat_start = max(-lim, min(lim, -0.35 * lim))
        actor = self._spawn_kind("person", s, lat_start, heading_sign=1.0)
        self._bind(
            actor, s, lat_start,
            speed=0.0,
            lat_speed=speed,
            lat_target=0.35 * lim,
            behavior="cross",
            allow_sidewalk=True,
            pad=pad,
        )

    def _inject_static_shapes(
        self,
        rig: Any,
        *,
        kind: str = "shape",
        n: int = 0,
    ) -> None:
        """Stationary primitives on the gait line (cube-on-path, mixed shapes)."""
        assert self.state is not None
        s0, L = self._cam_sl(rig, 0.0)
        count = int(n) if int(n) > 0 else self.rng.randint(2, 4)
        lead0 = 5.4
        if self._compose is not None:
            lead0 = lead0 + self._compose.take_group_offset("static")
        for i in range(count):
            s = s0 + lead0 + i * 2.55
            lat = L + self.rng.uniform(-0.10, 0.10)
            pad = self._kind_pad(kind)
            actor = self._spawn_kind(
                kind, s, lat, heading_sign=1.0,
                cube_size=self.rng.uniform(0.38, 0.72),
            )
            actor.category = "static"
            self._bind(
                actor, s, lat,
                speed=0.0,
                behavior="static",
                allow_sidewalk=True,
                pad=pad,
            )

    def _inject_pothole(self, rig: Any, lead: float, dlat: float) -> None:
        assert self.state is not None
        s0, L = self._cam_sl(rig, 0.0)
        lead = float(lead)
        if self._compose is not None:
            lead = lead + self._compose.take_group_offset("static")
        s = s0 + lead
        lat = self._resolve_lat(dlat, L, 0.20)
        # On-gait trip hazard: pothole / crater / lifted slab. Offset clutter
        # can also be a debris pile (still a shin-strike, different silhouette).
        if abs(float(dlat)) < 0.35:
            cls_name = self.rng.choice(("pothole", "crater", "broken_slab"))
        else:
            cls_name = self.rng.choice(("pothole", "crater", "broken_slab", "debris"))
        if self._compose is not None:
            hs, hl = extents_for(cls_name, 0.20)
            placed = self._compose.reserve(
                s=s,
                lat=lat,
                half_s=hs,
                half_lat=hl,
                class_name=cls_name,
                behavior="static",
            )
            s, lat = placed.s, placed.lat
        loc = self.state.road.offset_point(s, lat, z=self.state.corridor.ground_z(lat))
        heading = heading_from_tangent(self.state.road.tangent(s))
        actor = spawn_ground_hazard(
            cls_name,
            loc,
            heading,
            self.state.collections["hazards"],
            self.rng,
            self.state.counters,
            0.95,
            self.chaos,
        )
        actor.s = s
        actor.lateral = lat
        self._append(actor)

    def _inject_tree(self, rig: Any, lead: float, dlat: float) -> None:
        """Trunk on (or just off) the gait. Background scatter still stays off-slab."""
        assert self.state is not None
        s0, L = self._cam_sl(rig, 0.0)
        lead = float(lead)
        if self._compose is not None:
            lead = lead + self._compose.take_group_offset("static")
        s = s0 + lead
        lat = self._resolve_lat(dlat, L, 0.28)
        if self._compose is not None:
            hs, hl = extents_for("tree", 0.28)
            placed = self._compose.reserve(
                s=s, lat=lat, half_s=hs, half_lat=hl,
                class_name="tree", behavior="static",
            )
            s, lat = placed.s, placed.lat
        loc = self.state.road.offset_point(s, lat, z=self.state.corridor.ground_z(lat))
        heading = heading_from_tangent(self.state.road.tangent(s))
        actor = spawn_tree(
            self.lib, self.rng, self.cfg, self.state.collections["hazards"], loc,
            self.state.counters, self.chaos,
            max_canopy=1.15,
            heading=heading,
            wind=self._wind,
        )
        actor.s = s
        actor.lateral = lat
        self._append(actor)

    def _inject_lamp(self, rig: Any, lead: float, dlat: float) -> None:
        """Pole on (or just off) the gait — same TTC path as a bollard."""
        assert self.state is not None
        s0, L = self._cam_sl(rig, 0.0)
        lead = float(lead)
        if self._compose is not None:
            lead = lead + self._compose.take_group_offset("static")
        s = s0 + lead
        lat = self._resolve_lat(dlat, L, 0.22)
        if self._compose is not None:
            hs, hl = extents_for("streetlamp", 0.22)
            placed = self._compose.reserve(
                s=s, lat=lat, half_s=hs, half_lat=hl,
                class_name="streetlamp", behavior="static",
            )
            s, lat = placed.s, placed.lat
        loc = self.state.road.offset_point(s, lat, z=self.state.corridor.ground_z(lat))
        heading = heading_from_tangent(self.state.road.tangent(s))
        h = float(self.cfg["world"]["streetlamp_height"])
        actor = spawn_streetlamp(
            loc, heading, self.state.collections["hazards"],
            self.state.counters, height=h, with_arm=True,
            arm_sign=1.0 if lat >= 0.0 else -1.0,
        )
        actor.s = s
        actor.lateral = lat
        self._append(actor)

    def _inject_parked_car(self, rig: Any) -> None:
        s0, _L = self._cam_sl(rig, 0.0)
        s = s0 + 9.0
        if self._compose is not None:
            s = s + self._compose.take_group_offset("along")
        lat = self._far_lane()
        actor = self._spawn_kind("vehicle", s, lat, heading_sign=1.0)
        actor.category = "static"
        self._bind(
            actor, s, lat,
            speed=0.0,
            behavior="parked",
            allow_sidewalk=False,
            pad=1.05,
        )

    def _inject_sudden_stop(self, rig: Any) -> None:
        lead = float(self.cfg["scenarios"]["sudden_stop_lead_m"])
        stop_t = float(self.cfg["scenarios"]["sudden_stop_trigger_s"])
        s0, L = self._cam_sl(rig, 0.0)
        if self._compose is not None:
            lead = lead + self._compose.take_group_offset("along")
        s = s0 + lead
        actor = self._spawn_kind("person", s, L, heading_sign=1.0)
        self._bind(
            actor, s, L,
            speed=float(rig.walk_speed),
            stop_t=stop_t,
            behavior="sudden_stop",
            allow_sidewalk=True,
            pad=0.35,
        )

    def _inject_cut_in(
        self,
        rig: Any,
        *,
        t0: float,
        t_hit: float,
        v_car: float,
        cpa: float,
        behavior: str,
    ) -> None:
        """Oncoming car in the near lane, then drifts onto the sidewalk."""
        assert self.state is not None
        s_hit, L = self._cam_sl(rig, t_hit)
        lane = self._near_lane()
        tau_extra = 0.0
        if self._compose is not None:
            tau_extra = self._compose.take_group_offset("along") / max(float(v_car), 1.0)
        s_car0 = s_hit + float(v_car) * (float(t_hit) + tau_extra)
        s_car0 = min(max(2.0, s_car0), self.state.road.length - 5.0)
        lat_target = L + math.copysign(float(cpa), 1.0 if L >= 0.0 else -1.0)
        ease = max(1.60, min(2.40, float(t_hit) - 0.20))
        t0 = max(0.12, float(t_hit) - ease)
        rate = abs(lat_target - lane) / max(0.35, ease)
        actor = self._spawn_kind("vehicle", s_car0, lane, heading_sign=-1.0)
        self._bind(
            actor, s_car0, lane,
            speed=-float(v_car),
            lat_speed=rate,
            lat_target=lat_target,
            swerve_t=float(t0),
            behavior=behavior,
            allow_sidewalk=True,
            pad=1.05,
        )

    def _scale_person(self, actor: Actor, k: float) -> Actor:
        """Uniformly rescale a humanoid and keep its feet on the ground.

        ``origin_z`` is the hip height in world metres, and the pelvis is the
        object origin, so scaling the object about that origin lifts the feet
        by ``hip_z (1 − k)``. Rescaling ``origin_z`` by the same factor is
        what puts them back down.
        """
        k = max(0.35, float(k))
        actor.obj.scale = (k, k, k)
        actor.origin_z = float(actor.origin_z) * k
        actor.corridor_pad = max(0.18, float(actor.corridor_pad) * k)
        if actor.gait is not None:
            # Step frequency is derived from stature; keep the two in step or
            # a scaled-down adult moonwalks.
            actor.gait.stature = float(getattr(actor.gait, "stature", 1.7)) * k
        return actor

    def _inject_run_off_road(self, rig: Any) -> None:
        """Vehicle leaves the carriageway entirely and mounts the pavement.

        Distinct from `car_cut_in`: the lateral target is past the walker,
        at the building line, so the car does not settle next to them — it
        crosses their whole corridor at speed. This is the single most
        dangerous urban event for a blind pedestrian and the taxonomy has to
        see it as a genuine CRITICAL rather than a grazing near-miss.
        """
        assert self.state is not None
        t_hit = self.rng.uniform(2.2, 3.2)
        v_car = self.rng.uniform(6.5, 10.5)
        t0 = max(0.4, t_hit - self.rng.uniform(1.1, 1.9))
        s_hit, L = self._cam_sl(rig, t_hit)
        lane = self._near_lane()
        tau_extra = 0.0
        if self._compose is not None:
            tau_extra = self._compose.take_group_offset("along") / max(v_car, 1.0)
        s_car0 = min(
            max(2.0, s_hit + v_car * (t_hit + tau_extra)), self.state.road.length - 5.0
        )
        # Overshoot past the walker toward the facade.
        lim = self.state.corridor.lateral_limit(1.05, True)
        lat_target = math.copysign(lim, L if L else 1.0)
        ease = max(1.60, min(2.50, t_hit - 0.20))
        t0 = max(0.12, t_hit - ease)
        rate = abs(lat_target - lane) / max(0.35, ease)
        actor = self._spawn_kind("vehicle", s_car0, lane, heading_sign=-1.0)
        self._bind(
            actor, s_car0, lane,
            speed=-v_car,
            lat_speed=rate,
            lat_target=lat_target,
            swerve_t=t0,
            behavior="run_off_road",
            allow_sidewalk=True,
            pad=1.05,
        )

    def _inject_weaving(
        self,
        rig: Any,
        *,
        kind: str,
        speed: float,
        amplitude: float,
        hz: float,
        lane: Optional[float] = None,
        pad: Optional[float] = None,
    ) -> None:
        """Oncoming actor that swings side to side across the corridor.

        The weave is phased so the actor is at the *centre* of its swing when
        it reaches the walker, which is when a swing toward them is maximally
        surprising. Amplitude is exact (a sine, not noise), so the injector
        can guarantee the swing actually reaches the gait line.
        """
        assert self.state is not None
        use_pad = float(pad) if pad is not None else self._kind_pad(kind)
        s0, L = self._cam_sl(rig, 0.0)
        base = self._near_lane() if lane is None else float(lane)
        v_close = self._ego_speed(rig) + float(speed)
        tau = self.rng.uniform(2.4, 3.6)
        if self._compose is not None:
            tau = tau + self._compose.take_group_offset("along") / max(v_close, 1.0)
        s = min(max(2.0, s0 + max(v_close, float(speed)) * tau), self.state.road.length - 4.0)
        actor = self._spawn_kind(kind, s, base, heading_sign=-1.0)
        self._bind(
            actor, s, base,
            speed=-float(speed),
            behavior="weaving",
            allow_sidewalk=True,
            pad=use_pad,
        )
        actor.weave_amp = float(amplitude)
        actor.weave_hz = float(hz)
        # Phase so sin(2π f τ + φ) = 0 with a positive slope at the encounter.
        actor.weave_phase = -2.0 * math.pi * float(hz) * tau
        # Bias the swing toward the walker's side of the corridor.
        actor.lateral = base
        if abs(L) > 1e-6 and math.copysign(1.0, base) != math.copysign(1.0, L):
            actor.weave_phase += math.pi

    def _inject_child_dart(self, rig: Any) -> None:
        """A child bolting across the walker's path.

        True child anthropometry (larger head fraction, ~1.1–1.3 m stature),
        not a uniformly scaled adult. A detector tuned on 1.7 m rectangles
        systematically underestimates the threat of a smaller, faster body.
        """
        assert self.state is not None
        pad = 0.24
        t_hit = self.rng.uniform(1.70, 2.60)
        s_hit, L = self._cam_sl(rig, t_hit)
        lim = self.state.corridor.lateral_limit(pad, True)
        speed = self.rng.uniform(1.9, 3.1)
        from_left = self.rng.random() < 0.5
        if self._compose is not None:
            _extra, from_left = self._compose.take_cross_layout(from_left)
        # Start far enough that they reach the gait at t_hit (chest intercept).
        dist = min(abs(lim - L) * 0.95, speed * t_hit)
        dist = max(1.6, dist)
        lat_start = L - dist if from_left else L + dist
        lat_start = max(-lim, min(lim, lat_start))
        speed = abs(lat_start - L) / max(0.55, t_hit)
        lat_end = L + (0.55 if from_left else -0.55)
        lat_end = max(-lim, min(lim, lat_end))
        s = s_hit
        actor = self._spawn_kind(
            "person", s, lat_start, heading_sign=1.0 if from_left else -1.0,
            child=True,
        )
        self._bind(
            actor, s, lat_start,
            speed=0.0,
            lat_speed=speed,
            lat_target=lat_end,
            behavior="dart",
            allow_sidewalk=True,
            pad=pad,
        )

    def _inject_head_cube(self, rig: Any) -> None:
        tau = float(self.cfg["scenarios"]["projectile_ttc"])
        speed = float(self.cfg["scenarios"]["projectile_speed"])
        z = float(self.cfg["camera"]["eye_height_m"])
        cr = float(self.cfg["scenarios"]["critical_cpa_target"])
        self._inject_oncoming(
            rig, kind="cube", obj_speed=speed, tau=tau, dlat=cr,
            cube_size=0.28, cube_z=z, pad=0.20,
        )
