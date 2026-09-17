"""3D world AABB → 2D pixel bounding box via the camera projection pipeline.

Pipeline (Part 5.1). View/projection are built once per frame. Object-space
AABBs are cached; each frame only transforms them by ``matrix_world``.

  1. 8 AABB corners in object space (`obj.bound_box`) → world via matrix_world.
  2. View matrix  = inverse(camera.matrix_world)     → camera space.
  3. Drop corners with camera-space Z > 0 (behind the lens; Blender cameras
     look down local −Z, so the visible half-space is Z < 0).
  4. Projection matrix from focal length / sensor / resolution (or, when a
     live camera object is available, `Object.calc_matrix_camera`, which is
     the matrix EEVEE-Next itself uses).
  5. clip = P @ (x, y, z, 1);  NDC = clip.xyz / clip.w  ∈ [−1, 1]².
  6. Pixel map (origin top-left, 1920×1080 by default):

         u = (NDC_x + 1) * (W / 2)
         v = (1 − NDC_y) * (H / 2)

  7. bbox = [min u, min v, max u, max v] over surviving corners.

Frustum-culled objects (all corners off-screen, or all behind the camera)
return None and are omitted from that frame's JSON. Partial overlap is
clamped and flagged `truncated`. A physics ray from the camera origin to
the object centroid sets `occluded` if something else is hit first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from mathutils import Matrix, Vector


@dataclass
class BoundingBox2D:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    truncated: bool
    occluded: bool
    n_front_corners: int

    def as_dict(self) -> dict:
        return {
            "xmin": int(self.xmin),
            "ymin": int(self.ymin),
            "xmax": int(self.xmax),
            "ymax": int(self.ymax),
        }


def construct_projection_matrix(
    lens_mm: float,
    sensor_width_mm: float,
    res_x: int,
    res_y: int,
    clip_start: float,
    clip_end: float,
    sensor_fit: str = "HORIZONTAL",
) -> Matrix:
    """Analytic OpenGL-style perspective matrix matching Blender's convention.

    The camera looks down local −Z. At the near plane the half-extents are

        right = near * (sensor_w / 2) / lens
        top   = near * (sensor_h / 2) / lens

    with sensor_h derived from the render aspect when `sensor_fit` is
    HORIZONTAL (the pipeline default).
    """
    aspect = float(res_x) / float(max(1, res_y))
    if str(sensor_fit).upper() == "VERTICAL":
        sensor_h = float(sensor_width_mm)
        sensor_w = sensor_h * aspect
    else:
        sensor_w = float(sensor_width_mm)
        sensor_h = sensor_w / aspect

    near = float(clip_start)
    far = float(clip_end)
    right = near * (sensor_w * 0.5) / float(lens_mm)
    top = near * (sensor_h * 0.5) / float(lens_mm)
    left = -right
    bottom = -top

    # Standard symmetric OpenGL frustum (row-major mathutils.Matrix).
    a = 2.0 * near / (right - left)
    b = 2.0 * near / (top - bottom)
    c = (right + left) / (right - left)
    d = (top + bottom) / (top - bottom)
    e = -(far + near) / (far - near)
    f = -(2.0 * far * near) / (far - near)
    return Matrix((
        (a, 0.0, c, 0.0),
        (0.0, b, d, 0.0),
        (0.0, 0.0, e, f),
        (0.0, 0.0, -1.0, 0.0),
    ))


def camera_view_matrix(cam_obj: Any) -> Matrix:
    """World → camera. Equivalent to inverse(camera.matrix_world)."""
    return cam_obj.matrix_world.inverted()


def camera_projection_matrix(
    cam_obj: Any,
    depsgraph: Any,
    res_x: int,
    res_y: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Matrix:
    """Prefer the matrix EEVEE actually uses; fall back to the analytic form."""
    try:
        return cam_obj.calc_matrix_camera(
            depsgraph,
            x=int(res_x),
            y=int(res_y),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
        )
    except TypeError:
        # Pre-2.8 positional signature, kept as a last resort.
        return cam_obj.calc_matrix_camera(res_x, res_y, scale_x, scale_y)
    except Exception:
        cam = cam_obj.data
        return construct_projection_matrix(
            lens_mm=cam.lens,
            sensor_width_mm=cam.sensor_width,
            res_x=res_x,
            res_y=res_y,
            clip_start=cam.clip_start,
            clip_end=cam.clip_end,
            sensor_fit=cam.sensor_fit,
        )


# Spatial overlay uses inscribed boxes, not the outer union AABB.
INNER_BOX_SCALE = 0.85


def _inset_local_corners(
    locals_: tuple[Vector, ...],
    scale: float = INNER_BOX_SCALE,
) -> tuple[Vector, ...]:
    """Shrink an object-space AABB toward its centre. Scale 1 keeps it."""
    if not locals_ or scale >= 0.999:
        return locals_
    n = float(len(locals_))
    cx = sum(c.x for c in locals_) / n
    cy = sum(c.y for c in locals_) / n
    cz = sum(c.z for c in locals_) / n
    centre = Vector((cx, cy, cz))
    s = float(scale)
    return tuple(centre + (c - centre) * s for c in locals_)


class LocalBoundCache:
    """Object-space AABB corners of a root mesh and every MESH descendant.

    Limb meshes do not deform — they rotate. Capturing ``bound_box`` once and
    transforming by ``matrix_world`` each frame avoids ``evaluated_get`` and
    a child walk through Blender's RNA.
    """

    __slots__ = ("parts",)

    def __init__(self, root: Any) -> None:
        self.parts: list[tuple[Any, tuple[Vector, ...]]] = []
        stack = [root]
        while stack:
            o = stack.pop()
            if getattr(o, "type", None) == "MESH" and getattr(o, "bound_box", None):
                self.parts.append((o, tuple(Vector(c) for c in o.bound_box)))
            stack.extend(getattr(o, "children", ()))
        if not self.parts:
            if getattr(root, "bound_box", None):
                self.parts.append((root, tuple(Vector(c) for c in root.bound_box)))

    def world_corners(self) -> list[Vector]:
        out: list[Vector] = []
        for o, locals_ in self.parts:
            mw = o.matrix_world
            out.extend(mw @ c for c in locals_)
        if out:
            return out
        if self.parts:
            return [self.parts[0][0].matrix_world.translation.copy()]
        return []

    def inner_world_corner_sets(
        self,
        scale: float = INNER_BOX_SCALE,
        only_obj: Any = None,
    ) -> list[list[Vector]]:
        """One inscribed 8-corner box per MESH part (cabin, wheel, limb, …).

        ``only_obj`` restricts to that mesh (tree trunk, lamp pole) so child
        arms / canopy never inflate the heat splat.
        """
        sets: list[list[Vector]] = []
        for o, locals_ in self.parts:
            if only_obj is not None and o is not only_obj:
                continue
            inset = _inset_local_corners(locals_, scale)
            mw = o.matrix_world
            sets.append([mw @ c for c in inset])
        if not sets and only_obj is not None and getattr(only_obj, "bound_box", None):
            inset = _inset_local_corners(
                tuple(Vector(c) for c in only_obj.bound_box), scale,
            )
            mw = only_obj.matrix_world
            sets.append([mw @ c for c in inset])
        return sets


_BOUND_BY_PTR: dict[int, LocalBoundCache] = {}


def clear_bound_caches() -> None:
    """Drop AABB caches. Call once per episode after the previous scene is wiped."""
    _BOUND_BY_PTR.clear()


def bound_cache_for(obj: Any) -> LocalBoundCache:
    pid = int(obj.as_pointer())
    cache = _BOUND_BY_PTR.get(pid)
    if cache is not None:
        return cache
    cache = LocalBoundCache(obj)
    _BOUND_BY_PTR[pid] = cache
    return cache


def world_aabb_corners(obj: Any) -> list[Vector]:
    """World-space AABB corners of `obj` **and its mesh children**.

    Articulated pedestrians and wheeled cars parent visual parts to a root.
    Blender's ``Object.bound_box`` is only the root datablock, so a pelvis-only
    box would under-cover a walking figure. We union every MESH descendant.
    """
    return bound_cache_for(obj).world_corners()


def world_aabb_minmax(obj: Any) -> tuple[Vector, Vector]:
    return _minmax_corners(world_aabb_corners(obj))


def _minmax_corners(corners: list[Vector]) -> tuple[Vector, Vector]:
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def footprint_mouth_corners(obj: Any, half_height: float = 0.03) -> list[Vector]:
    """Thin world slab at a pothole's pavement mouth (ignore the buried well).

    The render mesh drops ~0.4 m below the ribbon so it looks like a hole.
    Using that full AABB for the 2-D box inflates the box downward and can
    miss the screen when only the mouth is in frame. The object's origin is
    on the pavement; we take the local XY radius from ``bound_box``.
    """
    xs = [float(v[0]) for v in obj.bound_box]
    ys = [float(v[1]) for v in obj.bound_box]
    r = max(abs(min(xs)), abs(max(xs)), abs(min(ys)), abs(max(ys)), 0.20)
    h = float(half_height)
    local = (
        Vector((-r, -r, -h)),
        Vector((r, -r, -h)),
        Vector((r, r, -h)),
        Vector((-r, r, -h)),
        Vector((-r, -r, h)),
        Vector((r, -r, h)),
        Vector((r, r, h)),
        Vector((-r, r, h)),
    )
    mw = obj.matrix_world
    return [mw @ p for p in local]


def threat_point_from_corners(corners: list[Vector], cam_z: float, mode: str = "volume") -> Vector:
    mn, mx = _minmax_corners(corners)
    cx = 0.5 * (mn.x + mx.x)
    cy = 0.5 * (mn.y + mx.y)
    if mode == "footprint":
        return Vector((cx, cy, float(cam_z)))
    cz = min(max(float(cam_z), mn.z), mx.z)
    return Vector((cx, cy, cz))


def threat_point(obj: Any, cam_z: float, mode: str = "volume") -> Vector:
    """Single point attached to the object used for TTC / CPA.

    A naive origin (feet at Z=0 vs camera at Z=1.6) inflates D_cpa by ~1.6 m
    and would make a direct hit look like a miss. We therefore pick a point
    that lives on the object's world AABB at the most relevant height:

    * ``footprint`` (potholes): XY of the AABB centre, Z = camera height so
      the object occupies the walker's vertical column.
    * ``volume`` (people, cars, branches): XY of the AABB centre, Z clamped
      into the object's own slab (preferring the camera height). A car roof
      at 1.5 m then has only a 0.1 m vertical residual against a 1.6 m camera.
    """
    return threat_point_from_corners(world_aabb_corners(obj), cam_z, mode)


def project_corners(
    corners_world: list[Vector],
    view: Matrix,
    proj: Matrix,
    res_x: int,
    res_y: int,
) -> tuple[list[tuple[float, float]], int]:
    """Project world corners. Returns (pixel list, n_front).

    A corner is discarded when it is behind the camera (cam-space Z ≥ 0) or
    when the homogeneous w is non-positive after projection (degenerate).
    """
    pixels: list[tuple[float, float]] = []
    n_front = 0
    for p in corners_world:
        cam = view @ Vector((p.x, p.y, p.z, 1.0))
        # Visible half-space: camera-space Z < 0.
        if cam.z >= 0.0:
            continue
        n_front += 1
        clip = proj @ cam
        if clip.w <= 1e-8:
            continue
        ndc_x = clip.x / clip.w
        ndc_y = clip.y / clip.w
        u = (ndc_x + 1.0) * (res_x * 0.5)
        v = (1.0 - ndc_y) * (res_y * 0.5)
        pixels.append((u, v))
    return pixels, n_front


def _bbox_from_pixels(
    pixels: list[tuple[float, float]],
    res_x: int,
    res_y: int,
    n_front: int,
) -> Optional[BoundingBox2D]:
    if not pixels:
        return None
    us = [p[0] for p in pixels]
    vs = [p[1] for p in pixels]
    xmin, xmax = min(us), max(us)
    ymin, ymax = min(vs), max(vs)

    # Frustum cull: the AABB of the projected corners misses the screen.
    if xmax < 0.0 or ymax < 0.0 or xmin > res_x or ymin > res_y:
        return None

    truncated = (
        xmin < 0.0
        or ymin < 0.0
        or xmax > res_x
        or ymax > res_y
        or n_front < 8
    )
    xmin_i = int(math.floor(max(0.0, min(xmin, float(res_x)))))
    ymin_i = int(math.floor(max(0.0, min(ymin, float(res_y)))))
    xmax_i = int(math.ceil(max(0.0, min(xmax, float(res_x)))))
    ymax_i = int(math.ceil(max(0.0, min(ymax, float(res_y)))))

    if xmax_i <= xmin_i or ymax_i <= ymin_i:
        return None

    return BoundingBox2D(
        xmin=xmin_i,
        ymin=ymin_i,
        xmax=xmax_i,
        ymax=ymax_i,
        truncated=truncated,
        occluded=False,
        n_front_corners=n_front,
    )


_GROUND_PREFIXES = ("road_surface", "sidewalk_", "curb_", "centerline", "lamp_pole", "dash_")


def _is_self_or_child(hit_obj: Any, target_obj: Any) -> bool:
    walker = hit_obj
    while walker is not None:
        if walker == target_obj:
            return True
        walker = walker.parent
    return False


def _is_ignorable_ground(hit_obj: Any) -> bool:
    """Road / sidewalk ribbons must not occlude the pothole sitting on them."""
    name = getattr(hit_obj, "name", "")
    return any(name.startswith(p) for p in _GROUND_PREFIXES)


def raycast_occluded(
    scene: Any,
    depsgraph: Any,
    cam_obj: Any,
    target_obj: Any,
    epsilon: float = 0.08,
    centroid: Optional[Vector] = None,
) -> bool:
    """True if a ray from the camera origin hits something before `target_obj`.

    Uses `Scene.ray_cast` on the evaluated depsgraph (Blender 4.x). The
    target itself (and its children) do not count as occluders. Ground-plane
    ribbons are stepped through so a pothole is not flagged occluded by the
    sidewalk it sits on. A hit whose distance is within `epsilon` of the
    target centroid is treated as a self-hit.
    """
    origin = cam_obj.matrix_world.translation
    if centroid is None:
        corners = world_aabb_corners(target_obj)
        if not corners:
            return False
        centroid = sum(corners, Vector((0.0, 0.0, 0.0))) / float(len(corners))
    to_target = centroid - origin
    dist = to_target.length
    if dist < 1e-6:
        return False
    direction = to_target.normalized()

    start = origin.copy()
    remaining = dist
    for _ in range(10):
        if remaining <= float(epsilon):
            return False
        hit, loc, _n, _idx, hit_obj, _mat = scene.ray_cast(
            depsgraph, start, direction, distance=remaining
        )
        if not hit or hit_obj is None:
            return False
        if _is_self_or_child(hit_obj, target_obj):
            return False
        hit_dist_from_cam = (loc - origin).length
        if hit_dist_from_cam >= dist - float(epsilon):
            return False
        if _is_ignorable_ground(hit_obj):
            step = max(0.03, (loc - start).length + 0.03)
            start = start + direction * step
            remaining = dist - (start - origin).length
            continue
        return True
    return False


def project_from_corners(
    corners: list[Vector],
    view: Matrix,
    proj: Matrix,
    res_x: int,
    res_y: int,
) -> Optional[BoundingBox2D]:
    pixels, n_front = project_corners(corners, view, proj, res_x, res_y)
    if n_front == 0:
        return None
    return _bbox_from_pixels(pixels, res_x, res_y, n_front)


def inset_world_corners(
    corners: list[Vector],
    scale: float = INNER_BOX_SCALE,
) -> list[Vector]:
    """Inset a world-space corner set about its centroid (pothole mouth)."""
    if not corners or scale >= 0.999:
        return list(corners)
    n = float(len(corners))
    acc = Vector((0.0, 0.0, 0.0))
    for c in corners:
        acc += c
    centre = acc / n
    s = float(scale)
    return [centre + (c - centre) * s for c in corners]


def project_inner_boxes(
    corner_sets: list[list[Vector]],
    view: Matrix,
    proj: Matrix,
    res_x: int,
    res_y: int,
) -> list[tuple[int, int, int, int]]:
    """Project inscribed part boxes. Empty / behind-camera parts are dropped."""
    out: list[tuple[int, int, int, int]] = []
    for corners in corner_sets:
        if not corners:
            continue
        bbox = project_from_corners(corners, view, proj, res_x, res_y)
        if bbox is None:
            continue
        out.append((bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))
    return out


def project_object(
    obj: Any,
    cam_obj: Any,
    scene: Any,
    depsgraph: Any,
    res_x: int,
    res_y: int,
    occlusion_epsilon: float = 0.08,
    do_occlusion: bool = True,
    view: Optional[Matrix] = None,
    proj: Optional[Matrix] = None,
    corners: Optional[list[Vector]] = None,
) -> Optional[BoundingBox2D]:
    """Full Part-5 extraction for one annotatable object. None ⇒ drop frame."""
    if view is None:
        view = camera_view_matrix(cam_obj)
    if proj is None:
        proj = camera_projection_matrix(cam_obj, depsgraph, res_x, res_y)
    if corners is None:
        corners = world_aabb_corners(obj)
    if not corners:
        return None

    bbox = project_from_corners(corners, view, proj, res_x, res_y)
    if bbox is None:
        return None

    if do_occlusion:
        centroid = sum(corners, Vector((0.0, 0.0, 0.0))) / float(len(corners))
        bbox.occluded = raycast_occluded(
            scene, depsgraph, cam_obj, obj,
            epsilon=occlusion_epsilon, centroid=centroid,
        )
    return bbox
