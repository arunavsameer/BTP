"""Articulated pedestrian built from anthropometric surfaces + Winter gait.

Why not a box
-------------
A rectangular torso has a constant silhouette under yaw — the detector
then keys on that rectangle. We loft elliptical cross-sections whose
half-width / half-depth follow Drillis & Contini stations (fractions of
stature H), and we hang limbs as tapered capsules with a mild sinusoidal
muscle belly.

    head        0.130 H          thigh       0.245 H
    neck        0.052 H          shank       0.246 H
    torso       0.288 H          foot L      0.152 H
    upper arm   0.186 H          biacromial  0.259 H
    forearm     0.146 H          bi-hip      0.191 H

Barr superquadric (head, pelvis, hands)
---------------------------------------
    x = a |cos v|^{e1} |cos u|^{e2} sgn
    y = b |cos v|^{e1} |sin u|^{e2} sgn
    z = c |sin v|^{e1}              sgn
u ∈ [0, 2π], v ∈ [−π/2, π/2].  (e1, e2) = (1, 1) is a sphere;
(0.5, 0.6) is a rounded box — hips without a hard edge.

Gait (Winter 1991, two-harmonic sagittal oscillator)
----------------------------------------------------
φ = 2π f t + φ₀,   f = |v| / (0.41 H)     (Grieve & Gear step length)

    θ_hip  =  A_hip  sin(φ + {0, π})
    θ_knee = −A_knee [max(0, sin(φ + α + {0, π}))]^p     (no hyperextension)
    θ_sh   = −A_arm  sin(φ + {0, π})                      (antiphase arms)
    θ_ank  =  A_ank  sin(φ + β)

Pelvis (applied *after* look_along so heading is preserved):

    list  (Y)  = A_list sin(φ)
    pitch (X)  = A_pitch sin(2φ)
    yaw   (Z) += A_yaw  sin(φ)
    bob   (Z)  = A_bob |sin(φ)|            (2× step, always up at DS)

Thorax yaw is −0.65 × pelvic yaw (reciprocal trunk rotation).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from mathutils import Vector


def _link(obj: Any, collection: Any) -> Any:
    collection.objects.link(obj)
    return obj


def _assign(obj: Any, mat: Any) -> None:
    if obj.data is None or mat is None:
        return
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _smooth(obj: Any) -> Any:
    mesh = obj.data
    if mesh is None:
        return obj
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def _mesh(name: str, verts: list[tuple[float, float, float]],
          faces: list[tuple[int, ...]], collection: Any, mat: Any) -> Any:
    import bpy

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    _link(obj, collection)
    _assign(obj, mat)
    return _smooth(obj)


def _sgn_pow(x: float, e: float) -> float:
    ax = abs(x)
    if ax < 1e-12:
        return 0.0
    return math.copysign(ax ** e, x)


def create_superellipsoid(
    name: str,
    radii: tuple[float, float, float],
    collection: Any,
    mat: Any,
    e1: float = 0.7,
    e2: float = 0.75,
    nu: int = 12,
    nv: int = 14,
) -> Any:
    """Barr superquadric, origin at the centre, +Z up."""
    a, b, c = radii
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(nu + 1):
        v = -0.5 * math.pi + math.pi * i / nu
        cv, sv = math.cos(v), math.sin(v)
        for j in range(nv):
            u = 2.0 * math.pi * j / nv
            cu, su = math.cos(u), math.sin(u)
            verts.append((
                a * _sgn_pow(cv, e1) * _sgn_pow(cu, e2),
                b * _sgn_pow(cv, e1) * _sgn_pow(su, e2),
                c * _sgn_pow(sv, e1),
            ))

    def vid(i: int, j: int) -> int:
        return i * nv + (j % nv)

    for i in range(nu):
        for j in range(nv):
            p00, p01 = vid(i, j), vid(i, j + 1)
            p11, p10 = vid(i + 1, j + 1), vid(i + 1, j)
            if i == 0:
                faces.append((p00, p11, p10))
            elif i == nu - 1:
                faces.append((p00, p01, p10))
            else:
                faces.append((p00, p01, p11, p10))
    return _mesh(name, verts, faces, collection, mat)


def create_uv_sphere(
    name: str,
    radius: float,
    collection: Any,
    mat: Any,
    rings: int = 12,
    segs: int = 14,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Any:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    sx, sy, sz = scale
    for i in range(rings + 1):
        th = math.pi * i / rings
        y = math.cos(th)
        r = math.sin(th)
        for j in range(segs):
            ph = 2.0 * math.pi * j / segs
            verts.append((
                sx * radius * r * math.cos(ph),
                sy * radius * r * math.sin(ph),
                sz * radius * y,
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
    return _mesh(name, verts, faces, collection, mat)


def _lerp_keys(u: float, keys: list[tuple[float, float]]) -> float:
    u = min(1.0, max(0.0, u))
    for (u0, v0), (u1, v1) in zip(keys, keys[1:]):
        if u <= u1:
            t = 0.0 if u1 <= u0 else (u - u0) / (u1 - u0)
            # Smoothstep so stations don't crease.
            t = t * t * (3.0 - 2.0 * t)
            return v0 + (v1 - v0) * t
    return keys[-1][1]


def create_torso_loft(
    name: str,
    height: float,
    H: float,
    collection: Any,
    mat: Any,
    width_scale: float = 1.0,
    depth_scale: float = 1.0,
    rings: int = 11,
    segs: int = 16,
) -> Any:
    """Elliptical loft, origin at the *bottom centre* (pelvis junction), +Z up.

    Half-width / half-depth are fractions of stature, then scaled by body
    type. A two-harmonic sagittal offset mimics lumbar lordosis / thoracic
    kyphosis so the chest sits slightly forward of the waist.
    """
    # Stations: (u, half-width / H) — waist → chest → axilla → neck.
    w_keys = [(0.00, 0.100), (0.16, 0.094), (0.40, 0.122), (0.68, 0.116), (1.00, 0.050)]
    d_keys = [(0.00, 0.074), (0.22, 0.080), (0.48, 0.096), (0.78, 0.078), (1.00, 0.048)]
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(rings + 1):
        u = i / rings
        z = height * u
        rx = _lerp_keys(u, w_keys) * H * width_scale
        ry = _lerp_keys(u, d_keys) * H * depth_scale
        # Spinal curve in the sagittal plane (character faces +Y).
        y_off = H * (0.014 * math.sin(math.pi * u) - 0.009 * math.sin(2.0 * math.pi * u))
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            # Slightly flatten the back (negative Y) so it isn't a perfect ellipse.
            back = 0.92 if math.sin(a) < 0.0 else 1.04
            verts.append((rx * math.cos(a), y_off + ry * back * math.sin(a), z))

    def vid(i: int, j: int) -> int:
        return i * segs + (j % segs)

    for i in range(rings):
        for j in range(segs):
            faces.append((vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)))
    # Close neck and waist so EEVEE doesn't leak light through the shell.
    faces.append(tuple(reversed(range(segs))))
    faces.append(tuple(range(rings * segs, (rings + 1) * segs)))
    return _mesh(name, verts, faces, collection, mat)


def create_capsule(
    name: str,
    length: float,
    r0: float,
    r1: float,
    collection: Any,
    mat: Any,
    segs: int = 14,
    rings: int = 8,
    cap_n: int = 4,
    belly: float = 0.10,
) -> Any:
    """Tapered capsule along −Z, origin at the proximal joint.

    Shaft radius r(t) = lerp(r0, r1, t) · (1 + belly sin(π t)).
    Distal end is a hemisphere so knees / elbows read as joints, not tubes.
    """
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    # Shaft (including the equator of the distal cap).
    for i in range(rings + 1):
        t = i / rings
        z = -length * t
        rad = ((1.0 - t) * r0 + t * r1) * (1.0 + belly * math.sin(math.pi * t))
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            verts.append((rad * math.cos(a), rad * math.sin(a), z))
    # Distal hemisphere past z = −length.
    for k in range(1, cap_n + 1):
        phi = 0.5 * math.pi * k / cap_n
        rr = r1 * math.cos(phi)
        zz = -length - r1 * math.sin(phi)
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            verts.append((rr * math.cos(a), rr * math.sin(a), zz))

    n_rings = rings + cap_n
    def vid(i: int, j: int) -> int:
        return i * segs + (j % segs)

    for i in range(n_rings):
        for j in range(segs):
            faces.append((vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)))
    # Proximal and distal caps.
    faces.append(tuple(reversed(range(segs))))
    faces.append(tuple(range(n_rings * segs, (n_rings + 1) * segs)))
    return _mesh(name, verts, faces, collection, mat)


def create_column(
    name: str,
    length: float,
    r0: float,
    r1: float,
    collection: Any,
    mat: Any,
    segs: int = 10,
    rings: int = 4,
) -> Any:
    """Tapered tube along +Z, origin at the bottom centre (neck, posts)."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for i in range(rings + 1):
        t = i / rings
        z = length * t
        rad = (1.0 - t) * r0 + t * r1
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            verts.append((rad * math.cos(a), rad * math.sin(a), z))

    def vid(i: int, j: int) -> int:
        return i * segs + (j % segs)

    for i in range(rings):
        for j in range(segs):
            faces.append((vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)))
    faces.append(tuple(reversed(range(segs))))
    faces.append(tuple(range(rings * segs, (rings + 1) * segs)))
    return _mesh(name, verts, faces, collection, mat)


def create_foot(
    name: str,
    length: float,
    width: float,
    height: float,
    collection: Any,
    mat: Any,
    segs: int = 10,
    rings: int = 5,
) -> Any:
    """Tapered shoe: elliptical sections from heel to toe, origin at ankle."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    # Heel sits slightly behind the ankle; toes along +Y.
    for i in range(rings + 1):
        t = i / rings
        # Hermite: heel wide, mid-foot widest, toe narrow.
        w = width * (0.78 + 0.32 * math.sin(math.pi * t) - 0.28 * t * t)
        h = height * (1.0 - 0.45 * t)
        y = -0.22 * length + t * length
        z = -height * 0.55
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            verts.append((0.5 * w * math.cos(a), y + 0.08 * h * math.sin(a), z + 0.5 * h * math.sin(a)))

    def vid(i: int, j: int) -> int:
        return i * segs + (j % segs)

    for i in range(rings):
        for j in range(segs):
            faces.append((vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)))
    faces.append(tuple(reversed(range(segs))))
    faces.append(tuple(range(rings * segs, (rings + 1) * segs)))
    return _mesh(name, verts, faces, collection, mat)


def _parent(child: Any, parent: Any, local: Vector) -> None:
    child.parent = parent
    child.location = local
    child.rotation_euler = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Walk cycle
# ---------------------------------------------------------------------------

@dataclass
class WalkRig:
    pelvis: Any
    torso: Any
    left_thigh: Any
    right_thigh: Any
    left_shin: Any
    right_shin: Any
    left_arm: Any
    right_arm: Any
    left_fore: Any
    right_fore: Any
    left_foot: Any
    right_foot: Any
    stature: float
    phase: float
    a_hip: float = 0.40
    a_knee: float = 0.88
    a_arm: float = 0.44
    a_elbow: float = 0.38
    a_ank: float = 0.22
    a_list: float = 0.055
    a_pitch: float = 0.035
    a_yaw: float = 0.070
    a_bob: float = 0.018
    knee_phase: float = 0.35
    knee_pow: float = 1.15

    def step_freq(self, speed: float) -> float:
        step = max(0.45, 0.41 * self.stature)
        return max(0.8, abs(float(speed)) / step)

    def apply(self, t: float, speed: float, stopped: bool) -> None:
        if stopped or abs(speed) < 0.08:
            self._pose(0.0, 0.0, 0.0, 0.0)
            self.pelvis.rotation_euler[0] = 0.0
            self.pelvis.rotation_euler[1] = 0.0
            self.torso.rotation_euler[2] = 0.0
            return
        f = self.step_freq(speed)
        phi = 2.0 * math.pi * f * t + self.phase
        hip_l = self.a_hip * math.sin(phi)
        hip_r = self.a_hip * math.sin(phi + math.pi)
        sw_l = max(0.0, math.sin(phi + self.knee_phase)) ** self.knee_pow
        sw_r = max(0.0, math.sin(phi + math.pi + self.knee_phase)) ** self.knee_pow
        knee_l = -self.a_knee * sw_l
        knee_r = -self.a_knee * sw_r
        arm_l = -self.a_arm * math.sin(phi)
        arm_r = -self.a_arm * math.sin(phi + math.pi)
        elb_l = -self.a_elbow * (0.40 + 0.60 * sw_r)
        elb_r = -self.a_elbow * (0.40 + 0.60 * sw_l)
        ank_l = self.a_ank * math.sin(phi + 0.6)
        ank_r = self.a_ank * math.sin(phi + math.pi + 0.6)
        self._pose(hip_l, hip_r, knee_l, knee_r, arm_l, arm_r, elb_l, elb_r, ank_l, ank_r)

        # look_along owns heading (euler Z) and zeroes X/Y just before this.
        self.pelvis.rotation_euler[0] = self.a_pitch * math.sin(2.0 * phi)
        self.pelvis.rotation_euler[1] = self.a_list * math.sin(phi)
        self.pelvis.rotation_euler[2] += self.a_yaw * math.sin(phi)
        self.torso.rotation_euler[2] = -0.65 * self.a_yaw * math.sin(phi)
        self.pelvis.location.z += self.a_bob * abs(math.sin(phi))

    def _pose(
        self,
        hip_l: float,
        hip_r: float,
        knee_l: float,
        knee_r: float,
        arm_l: float = 0.0,
        arm_r: float = 0.0,
        elb_l: float = 0.0,
        elb_r: float = 0.0,
        ank_l: float = 0.0,
        ank_r: float = 0.0,
    ) -> None:
        self.left_thigh.rotation_euler[0] = hip_l
        self.right_thigh.rotation_euler[0] = hip_r
        self.left_shin.rotation_euler[0] = knee_l
        self.right_shin.rotation_euler[0] = knee_r
        self.left_arm.rotation_euler[0] = arm_l
        self.right_arm.rotation_euler[0] = arm_r
        self.left_fore.rotation_euler[0] = elb_l
        self.right_fore.rotation_euler[0] = elb_r
        self.left_foot.rotation_euler[0] = ank_l
        self.right_foot.rotation_euler[0] = ank_r


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SKIN_TONES = (
    (0.62, 0.42, 0.32),
    (0.48, 0.32, 0.22),
    (0.78, 0.58, 0.46),
    (0.36, 0.24, 0.18),
    (0.70, 0.52, 0.40),
    (0.55, 0.38, 0.28),
)

HAIR_TONES = (
    (0.08, 0.06, 0.05),
    (0.18, 0.10, 0.05),
    (0.28, 0.18, 0.08),
    (0.05, 0.05, 0.06),
    (0.35, 0.28, 0.18),
)


def spawn_humanoid(
    instance_id: str,
    loc: Vector,
    heading: float,
    collection: Any,
    rng: random.Random,
    cloth_color: tuple[float, float, float],
    chaos: float = 0.0,
    child: bool = False,
) -> tuple[Any, WalkRig, float]:
    """Build the figure. Returns (pelvis_root, rig, hip_height).

    `chaos` widens the anthropometry well past the adult 5th–95th percentile
    band: children, very tall adults, and extreme build ratios. The point is
    that "pedestrian" must not be a fixed silhouette — the network has to
    key on articulated bipedal motion, not on a 1.7 m rectangle. Proportions
    stay internally consistent (limbs remain fractions of stature), so the
    Winter gait and the 0.41·H step length are still valid at every size.

    `child=True` switches to a ~7-year-old station set (larger head, shorter
    legs as a fraction of H) instead of uniformly scaling an adult.
    """
    from materials import make_patterned_cloth, make_simple, make_skin

    c = 0.0 if chaos < 0.0 else (1.0 if chaos > 1.0 else float(chaos))
    if child:
        H = rng.uniform(1.10, 1.32)
        w_sc = rng.uniform(0.82, 1.12)
        d_sc = rng.uniform(0.85, 1.10)
        hip_sc = rng.uniform(0.88, 1.15)
        head_frac, torso_frac = 0.090, 0.340
        thigh_frac, shank_frac = 0.220, 0.220
    else:
        # Stature: the tame band widens toward child (1.15 m) and very tall (2.05 m).
        h_lo = 1.58 - 0.43 * c
        h_hi = 1.84 + 0.21 * c
        H = rng.uniform(h_lo, h_hi)
        w_sc = rng.uniform(0.88 - 0.30 * c, 1.16 + 0.52 * c)
        d_sc = rng.uniform(0.90 - 0.28 * c, 1.12 + 0.48 * c)
        hip_sc = rng.uniform(0.92 - 0.26 * c, 1.18 + 0.42 * c)
        head_frac, torso_frac = 0.068, 0.300
        thigh_frac, shank_frac = 0.245, 0.246

    head_r = head_frac * H
    torso_h = torso_frac * H
    pelvis_h = 0.100 * H
    thigh_l = thigh_frac * H
    shank_l = shank_frac * H
    foot_h = 0.038 * H
    uarm_l = 0.186 * H
    farm_l = 0.146 * H
    neck_l = 0.048 * H
    sh_w = 0.259 * H * (0.7 + 0.3 * w_sc)
    hip_w = 0.191 * H * hip_sc
    hip_z = thigh_l + shank_l + foot_h

    skin_c = rng.choice(SKIN_TONES)
    hair_c = rng.choice(HAIR_TONES)
    pants_c = (
        rng.uniform(0.07, 0.22),
        rng.uniform(0.07, 0.20),
        rng.uniform(0.09, 0.24),
    )
    mat_skin = make_skin(f"{instance_id}_skin", skin_c)
    mat_shirt = make_patterned_cloth(f"{instance_id}_shirt", cloth_color, rng, c, seed=rng.random())
    mat_pants = make_patterned_cloth(f"{instance_id}_pants", pants_c, rng, c, seed=rng.random())
    mat_shoe = make_simple(f"{instance_id}_shoe", (0.07, 0.07, 0.075), 0.72)
    mat_hair = make_simple(f"{instance_id}_hair", hair_c, 0.85)

    pelvis = create_superellipsoid(
        instance_id,
        (0.50 * hip_w, 0.072 * H * d_sc, 0.50 * pelvis_h),
        collection,
        mat_pants,
        e1=0.58,
        e2=0.68,
        nu=10,
        nv=14,
    )
    pelvis.location = Vector((loc.x, loc.y, float(loc.z) + hip_z))
    pelvis.rotation_mode = "XYZ"
    pelvis.rotation_euler = (0.0, 0.0, heading)

    torso = create_torso_loft(
        instance_id + "_torso", torso_h, H, collection, mat_shirt,
        width_scale=w_sc, depth_scale=d_sc,
    )
    _parent(torso, pelvis, Vector((0.0, 0.0, pelvis_h * 0.38)))

    neck = create_column(
        instance_id + "_neck", neck_l, 0.034 * H, 0.028 * H, collection, mat_skin,
        segs=10, rings=3,
    )
    _parent(neck, torso, Vector((0.0, 0.010 * H, torso_h - 0.008 * H)))

    head = create_superellipsoid(
        instance_id + "_head",
        (0.92 * head_r, 1.05 * head_r, 1.12 * head_r),
        collection,
        mat_skin,
        e1=0.90,
        e2=0.95,
        nu=12,
        nv=16,
    )
    _parent(head, neck, Vector((0.0, 0.014 * H, neck_l + 0.92 * head_r)))

    hair = create_uv_sphere(
        instance_id + "_hair", head_r * 1.02, collection, mat_hair,
        rings=8, segs=12, scale=(1.0, 1.08, 0.72),
    )
    _parent(hair, head, Vector((0.0, -0.008 * H, 0.028 * H)))

    ear_r = 0.018 * H
    for tag, sx in (("_lear", -1.0), ("_rear", 1.0)):
        ear = create_uv_sphere(
            instance_id + tag, ear_r, collection, mat_skin,
            rings=6, segs=8, scale=(0.45, 1.0, 1.25),
        )
        _parent(ear, head, Vector((sx * 0.88 * head_r, 0.0, -0.01 * H)))

    # Legs.
    r_th = 0.058 * H * (0.85 + 0.15 * hip_sc)
    r_sh = 0.040 * H
    l_thigh = create_capsule(instance_id + "_lthigh", thigh_l, r_th, r_th * 0.78, collection, mat_pants, belly=0.12)
    r_thigh = create_capsule(instance_id + "_rthigh", thigh_l, r_th, r_th * 0.78, collection, mat_pants, belly=0.12)
    _parent(l_thigh, pelvis, Vector((-hip_w * 0.28, 0.0, -pelvis_h * 0.28)))
    _parent(r_thigh, pelvis, Vector((hip_w * 0.28, 0.0, -pelvis_h * 0.28)))

    l_shin = create_capsule(instance_id + "_lshin", shank_l, r_th * 0.74, r_sh, collection, mat_pants, belly=0.08)
    r_shin = create_capsule(instance_id + "_rshin", shank_l, r_th * 0.74, r_sh, collection, mat_pants, belly=0.08)
    _parent(l_shin, l_thigh, Vector((0.0, 0.0, -thigh_l)))
    _parent(r_shin, r_thigh, Vector((0.0, 0.0, -thigh_l)))

    l_foot = create_foot(instance_id + "_lfoot", 0.152 * H, 0.072 * H, foot_h, collection, mat_shoe)
    r_foot = create_foot(instance_id + "_rfoot", 0.152 * H, 0.072 * H, foot_h, collection, mat_shoe)
    _parent(l_foot, l_shin, Vector((0.0, 0.01 * H, -shank_l)))
    _parent(r_foot, r_shin, Vector((0.0, 0.01 * H, -shank_l)))

    # Arms + deltoid caps so the sleeve doesn't read as a stick on a box.
    r_ua = 0.036 * H
    for tag, side in (("_ldelt", -1.0), ("_rdelt", 1.0)):
        delt = create_uv_sphere(instance_id + tag, 0.042 * H, collection, mat_shirt, rings=7, segs=10)
        _parent(delt, torso, Vector((side * sh_w * 0.46, 0.0, torso_h * 0.86)))

    l_arm = create_capsule(instance_id + "_larm", uarm_l, r_ua, r_ua * 0.82, collection, mat_shirt, belly=0.10)
    r_arm = create_capsule(instance_id + "_rarm", uarm_l, r_ua, r_ua * 0.82, collection, mat_shirt, belly=0.10)
    _parent(l_arm, torso, Vector((-sh_w * 0.48, 0.0, torso_h * 0.84)))
    _parent(r_arm, torso, Vector((sh_w * 0.48, 0.0, torso_h * 0.84)))

    l_fore = create_capsule(instance_id + "_lfore", farm_l, r_ua * 0.80, 0.026 * H, collection, mat_skin, belly=0.06)
    r_fore = create_capsule(instance_id + "_rfore", farm_l, r_ua * 0.80, 0.026 * H, collection, mat_skin, belly=0.06)
    _parent(l_fore, l_arm, Vector((0.0, 0.0, -uarm_l)))
    _parent(r_fore, r_arm, Vector((0.0, 0.0, -uarm_l)))

    hand_r = (0.028 * H, 0.016 * H, 0.040 * H)
    l_hand = create_superellipsoid(instance_id + "_lhand", hand_r, collection, mat_skin, e1=0.7, e2=0.8, nu=6, nv=8)
    r_hand = create_superellipsoid(instance_id + "_rhand", hand_r, collection, mat_skin, e1=0.7, e2=0.8, nu=6, nv=8)
    _parent(l_hand, l_fore, Vector((0.0, 0.0, -farm_l - 0.01 * H)))
    _parent(r_hand, r_fore, Vector((0.0, 0.0, -farm_l - 0.01 * H)))

    rig = WalkRig(
        pelvis=pelvis,
        torso=torso,
        left_thigh=l_thigh,
        right_thigh=r_thigh,
        left_shin=l_shin,
        right_shin=r_shin,
        left_arm=l_arm,
        right_arm=r_arm,
        left_fore=l_fore,
        right_fore=r_fore,
        left_foot=l_foot,
        right_foot=r_foot,
        stature=H,
        phase=rng.uniform(0.0, 2.0 * math.pi),
        # Gait style is part of the morphology: a shuffling stride and a
        # striding one look nothing alike at the same walking speed.
        a_hip=rng.uniform(0.34 - 0.16 * c, 0.46 + 0.22 * c),
        a_knee=rng.uniform(0.74 - 0.30 * c, 0.98 + 0.22 * c),
        a_arm=rng.uniform(0.34 - 0.28 * c, 0.52 + 0.34 * c),
        a_bob=rng.uniform(0.014, 0.022 + 0.020 * c),
    )
    rig.a_list = rng.uniform(0.055 - 0.035 * c, 0.055 + 0.055 * c)
    rig.a_yaw = rng.uniform(0.070 - 0.045 * c, 0.070 + 0.065 * c)
    return pelvis, rig, hip_z
