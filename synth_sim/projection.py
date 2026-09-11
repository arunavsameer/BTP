"""Analytical 3D OBB → 2D AABB projection, near-plane clipping, occlusion tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import glm
import numpy as np

from actors import Actor
from config import CameraIntrinsicsConfig, SimulationConfig


def mat4_to_numpy(m: glm.mat4) -> np.ndarray:
    """Convert a GLM column-major mat4 into a row-major numpy matrix for `@` products."""
    out = np.zeros((4, 4), dtype=np.float64)
    for col in range(4):
        colv = m[col]
        out[0, col] = float(colv.x)
        out[1, col] = float(colv.y)
        out[2, col] = float(colv.z)
        out[3, col] = float(colv.w)
    return out


def perspective_gl(cfg: CameraIntrinsicsConfig) -> glm.mat4:
    """OpenGL clip-space projection matching the pinhole K of Section 3.5.

    Mathematical layout (row-major):
        [ 2n/(r-l)  0  (r+l)/(r-l)  0 ]
        [ 0  2n/(t-b)  (t+b)/(t-b)  0 ]
        [ 0  0  -(f+n)/(f-n)  -2fn/(f-n) ]
        [ 0  0  -1  0 ]
    GLM stores columns, so P[row, col] lives at m[col][row].
    """
    n = cfg.near
    f = cfg.far
    r = n * math.tan(math.radians(cfg.fov_x_deg) * 0.5)
    l = -r
    t = r / cfg.aspect
    b = -t
    p = glm.mat4(0.0)
    p[0][0] = (2.0 * n) / (r - l)
    p[1][1] = (2.0 * n) / (t - b)
    p[2][0] = (r + l) / (r - l)
    p[2][1] = (t + b) / (t - b)
    p[2][2] = -(f + n) / (f - n)
    p[3][2] = -(2.0 * f * n) / (f - n)
    p[2][3] = -1.0
    return p


def pinhole_k(cfg: CameraIntrinsicsConfig) -> np.ndarray:
    return np.array(
        [[cfg.fx, 0.0, cfg.cx], [0.0, cfg.fy, cfg.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass
class BoundingBox2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    visible: bool
    behind: bool = False

    def clamp(self, w: int, h: int) -> "BoundingBox2D":
        return BoundingBox2D(
            xmin=max(0.0, min(self.xmin, float(w - 1))),
            ymin=max(0.0, min(self.ymin, float(h - 1))),
            xmax=max(0.0, min(self.xmax, float(w - 1))),
            ymax=max(0.0, min(self.ymax, float(h - 1))),
            visible=self.visible,
            behind=self.behind,
        )

    def as_int(self) -> Tuple[int, int, int, int]:
        return int(math.floor(self.xmin)), int(math.floor(self.ymin)), int(math.ceil(self.xmax)), int(math.ceil(self.ymax))


def obb_corners_world(actor: Actor) -> np.ndarray:
    ex, ey, ez = float(actor.extents.x), float(actor.extents.y), float(actor.extents.z)
    local = np.array(
        [
            [-0.5 * ex, -0.5 * ey, -0.5 * ez],
            [0.5 * ex, -0.5 * ey, -0.5 * ez],
            [-0.5 * ex, 0.5 * ey, -0.5 * ez],
            [0.5 * ex, 0.5 * ey, -0.5 * ez],
            [-0.5 * ex, -0.5 * ey, 0.5 * ez],
            [0.5 * ex, -0.5 * ey, 0.5 * ez],
            [-0.5 * ex, 0.5 * ey, 0.5 * ez],
            [0.5 * ex, 0.5 * ey, 0.5 * ez],
        ],
        dtype=np.float64,
    )
    model = mat4_to_numpy(actor.model_matrix())
    homo = np.concatenate([local, np.ones((8, 1))], axis=1)
    world = (model @ homo.T).T
    return world[:, :3]


def _clip_polygon_near(poly: np.ndarray, near: float) -> np.ndarray:
    """Sutherland–Hodgman clip of view-space points against the plane z = -near.

    Inside (in front of the near plane): z <= -near.
    """
    if len(poly) == 0:
        return poly
    out: List[np.ndarray] = []
    nvert = len(poly)

    def inside(p: np.ndarray) -> bool:
        return float(p[2]) <= -near + 1e-9

    def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        za, zb = float(a[2]), float(b[2])
        denom = zb - za
        t = 0.0 if abs(denom) < 1e-12 else (-near - za) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return a + t * (b - a)

    for i in range(nvert):
        cur = poly[i]
        prev = poly[(i - 1) % nvert]
        cin, pin = inside(cur), inside(prev)
        if cin:
            if not pin:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif pin:
            out.append(intersect(prev, cur))
    if not out:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(out, axis=0)


def project_points_view_to_screen(
    view_pts: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """view_pts (N,3) → pixel (N,2) with OpenGL NDC (y-up) converted to image (y-down)."""
    if len(view_pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    homo = np.concatenate([view_pts, np.ones((len(view_pts), 1))], axis=1)
    clip = (proj @ homo.T).T
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] + 1.0) * 0.5 * width
    v = (1.0 - ndc[:, 1]) * 0.5 * height
    return np.stack([u, v], axis=1)


def project_obb_to_2d(
    actor: Actor,
    view_matrix: glm.mat4,
    proj_matrix: glm.mat4,
    screen_dims: Tuple[int, int],
    near: float = 0.1,
) -> BoundingBox2D:
    """Project the 8 OBB corners with near-plane clipping to a tight 2D AABB."""
    w, h = screen_dims
    view = mat4_to_numpy(view_matrix)
    proj = mat4_to_numpy(proj_matrix)
    world = obb_corners_world(actor)
    homo = np.concatenate([world, np.ones((8, 1))], axis=1)
    view_pts = (view @ homo.T).T[:, :3]
    if np.all(view_pts[:, 2] >= -near):
        return BoundingBox2D(0, 0, 0, 0, visible=False, behind=True)

    # Clip the silhouette by clipping each of the 12 cube edges, then hull the result.
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    clipped: List[np.ndarray] = []
    for a, b in edges:
        poly = _clip_polygon_near(np.stack([view_pts[a], view_pts[b]], axis=0), near)
        for p in poly:
            clipped.append(p)
    if not clipped:
        return BoundingBox2D(0, 0, 0, 0, visible=False, behind=True)
    pts = np.stack(clipped, axis=0)
    pix = project_points_view_to_screen(pts, proj, w, h)
    xmin = float(np.min(pix[:, 0]))
    xmax = float(np.max(pix[:, 0]))
    ymin = float(np.min(pix[:, 1]))
    ymax = float(np.max(pix[:, 1]))
    # Completely off-screen?
    if xmax < 0 or ymax < 0 or xmin > w - 1 or ymin > h - 1:
        return BoundingBox2D(xmin, ymin, xmax, ymax, visible=False, behind=False)
    return BoundingBox2D(xmin, ymin, xmax, ymax, visible=True, behind=False)


def truncation_ratio(box: BoundingBox2D, width: int, height: int) -> Tuple[float, bool]:
    if not box.visible or box.behind:
        return 1.0, True
    raw_w = max(box.xmax - box.xmin, 1e-6)
    raw_h = max(box.ymax - box.ymin, 1e-6)
    raw_area = raw_w * raw_h
    c = box.clamp(width, height)
    cw = max(c.xmax - c.xmin, 0.0)
    ch = max(c.ymax - c.ymin, 0.0)
    vis_area = cw * ch
    trunc = float(np.clip(1.0 - vis_area / raw_area, 0.0, 1.0))
    return trunc, trunc > 0.02


def _ray_aabb_t(origin: np.ndarray, direction: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> Optional[float]:
    inv = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    t0 = (bmin - origin) * inv
    t1 = (bmax - origin) * inv
    tmin = float(np.max(np.minimum(t0, t1)))
    tmax = float(np.min(np.maximum(t0, t1)))
    if tmax < 0.0 or tmin > tmax:
        return None
    t = tmin if tmin >= 0.0 else tmax
    if t < 0.0:
        return None
    return float(t)


def compute_occlusion_and_truncation(
    actor: Actor,
    depth_buffer: np.ndarray,
    view_matrix: glm.mat4,
    proj_matrix: glm.mat4,
    cfg: SimulationConfig,
    box: Optional[BoundingBox2D] = None,
    instance_mask: Optional[np.ndarray] = None,
) -> Tuple[float, bool, float, bool, BoundingBox2D]:
    """Grid-sample the 2D box, compare linear depth (and optional instance IDs).

    Returns (occlusion_ratio, is_occluded, truncation_ratio, is_truncated, clamped_box).
    """
    w, h = cfg.camera.width, cfg.camera.height
    if box is None:
        box = project_obb_to_2d(actor, view_matrix, proj_matrix, (w, h), near=cfg.camera.near)
    trunc, is_trunc = truncation_ratio(box, w, h)
    if not box.visible or box.behind:
        return 1.0, True, trunc, is_trunc, box

    clamped = box.clamp(w, h)
    if clamped.xmax <= clamped.xmin or clamped.ymax <= clamped.ymin:
        return 1.0, True, trunc, True, clamped

    m = max(3, int(cfg.occlusion_grid))
    xs = np.linspace(clamped.xmin, clamped.xmax, m)
    ys = np.linspace(clamped.ymin, clamped.ymax, m)
    view = mat4_to_numpy(view_matrix)
    model = mat4_to_numpy(actor.model_matrix())
    inv_model = np.linalg.inv(model)
    cam_world = np.linalg.inv(view)[:3, 3]
    ex, ey, ez = 0.5 * float(actor.extents.x), 0.5 * float(actor.extents.y), 0.5 * float(actor.extents.z)
    bmin = np.array([-ex, -ey, -ez])
    bmax = np.array([ex, ey, ez])
    fx, fy = cfg.camera.fx, cfg.camera.fy
    cx, cy = cfg.camera.cx, cfg.camera.cy
    inv_view_r = np.linalg.inv(view)[:3, :3]

    visible = 0
    total = 0
    for u in xs:
        for v in ys:
            ui = int(np.clip(round(u), 0, w - 1))
            vi = int(np.clip(round(v), 0, h - 1))
            total += 1
            if instance_mask is not None:
                if int(instance_mask[vi, ui]) == int(actor.instance_id):
                    visible += 1
                    continue
                # If the pixel is this instance, it is visible; otherwise it may still be
                # a background miss due to the coarse box. Fall through to depth.
            # Pixel ray in camera space (OpenGL: camera looks down -Z).
            x_n = (u - cx) / fx
            y_n = -((v - cy) / fy)
            dir_cam = np.array([x_n, y_n, -1.0], dtype=np.float64)
            dir_cam = dir_cam / max(np.linalg.norm(dir_cam), 1e-9)
            dir_world = inv_view_r @ dir_cam
            origin_local = (inv_model @ np.array([cam_world[0], cam_world[1], cam_world[2], 1.0]))[:3]
            dir_local = (inv_model[:3, :3] @ dir_world)
            t_hit = _ray_aabb_t(origin_local, dir_local, bmin, bmax)
            if t_hit is None:
                continue
            z_analytic = t_hit  # Euclidean metres along the ray
            z_buf = float(depth_buffer[vi, ui])
            if z_buf <= 1e-6:
                visible += 1
                continue
            if z_buf >= z_analytic - 0.08:
                visible += 1

    if total == 0:
        occ = 1.0
    else:
        occ = float(np.clip(1.0 - visible / float(total), 0.0, 1.0))
    is_occ = occ >= cfg.occlusion_flag_threshold
    return occ, is_occ, trunc, is_trunc, clamped
