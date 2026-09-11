"""Headless ModernGL context, MRT g-buffer, sun shadow map, sky pass."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import glm
import numpy as np

from config import SimulationConfig
from geometry import Mesh, generate_sky_dome
from projection import perspective_gl
from shaders import (
    FRAGMENT_SHADER_FLOAT_ID,
    FRAGMENT_SHADER_UINT,
    SHADOW_FRAGMENT,
    SHADOW_VERTEX,
    SKY_FRAGMENT,
    SKY_VERTEX,
    VERTEX_SHADER,
)


def mat_bytes(m: glm.mat4 | glm.mat3) -> bytes:
    if hasattr(m, "to_bytes"):
        return m.to_bytes()  # type: ignore[no-any-return]
    return bytes(m)


def create_headless_context(require: int = 330):
    import moderngl

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    errors: List[str] = []
    for kwargs in (
        {"standalone": True, "backend": "egl", "require": require},
        {"standalone": True, "backend": "egl"},
        {"standalone": True, "require": require},
        {"standalone": True},
    ):
        try:
            ctx = moderngl.create_context(**kwargs)
            return ctx, kwargs
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kwargs}: {exc}")
    try:
        ctx = moderngl.create_context(require=require)
        return ctx, {"fallback": "current"}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"current: {exc}")
        joined = "\n".join(errors)
        raise RuntimeError(
            "Failed to create a headless OpenGL context. Install mesa/libglvnd and "
            "moderngl/glcontext. Attempts:\n" + joined
        ) from exc


def _normal_matrix(model: glm.mat4) -> glm.mat3:
    return glm.transpose(glm.inverse(glm.mat3(model)))


def _set_if(prog, name: str, fn) -> None:
    if name in prog:
        fn(prog[name])


@dataclass
class GPUMesh:
    vao: object
    count: int


class OffscreenRenderer:
    """MRT framebuffer plus an optional directional shadow map and sky dome."""

    def __init__(self, cfg: SimulationConfig) -> None:
        import moderngl

        self.cfg = cfg
        self.w = cfg.camera.width
        self.h = cfg.camera.height
        self.ctx, self.ctx_info = create_headless_context(330)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.depth_func = "<"
        self.ctx.viewport = (0, 0, self.w, self.h)

        self.color_tex = self.ctx.texture((self.w, self.h), 4, dtype="f1")
        self.depth_tex = self.ctx.texture((self.w, self.h), 1, dtype="f4")
        self.depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.depth_rbo = self.ctx.depth_renderbuffer((self.w, self.h))

        self.id_is_uint = False
        self.id_tex = None
        self.prog = None
        self._init_gbuffer()

        self.shadow_prog = self.ctx.program(vertex_shader=SHADOW_VERTEX, fragment_shader=SHADOW_FRAGMENT)
        self.sky_prog = self.ctx.program(vertex_shader=SKY_VERTEX, fragment_shader=SKY_FRAGMENT)
        self.shadow_res = int(cfg.shadow_resolution)
        self.shadow_enabled = False
        self._init_shadow()

        self._mesh_cache: Dict[Tuple[int, int], GPUMesh] = {}
        self.projection = perspective_gl(cfg.camera)
        self.light_vp = glm.mat4(1.0)
        self.sky_mesh = generate_sky_dome(radius=80.0)

    def _init_gbuffer(self) -> None:
        attempts: List[Tuple[str, str]] = [
            ("u2", FRAGMENT_SHADER_UINT),
            ("u4", FRAGMENT_SHADER_UINT),
            ("f4", FRAGMENT_SHADER_FLOAT_ID),
        ]
        last_err: Optional[Exception] = None
        for dtype, frag in attempts:
            try:
                id_tex = self.ctx.texture((self.w, self.h), 1, dtype=dtype)
                id_tex.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
                prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=frag)
                fbo = self.ctx.framebuffer(
                    color_attachments=(self.color_tex, self.depth_tex, id_tex),
                    depth_attachment=self.depth_rbo,
                )
                fbo.use()
                fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
                self.id_tex = id_tex
                self.prog = prog
                self.fbo = fbo
                self.id_is_uint = dtype.startswith("u")
                self.id_dtype = dtype
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise RuntimeError(f"Could not allocate MRT g-buffer: {last_err}") from last_err

    def _init_shadow(self) -> None:
        try:
            self.shadow_tex = self.ctx.depth_texture((self.shadow_res, self.shadow_res))
            self.shadow_tex.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
            self.shadow_tex.repeat_x = False
            self.shadow_tex.repeat_y = False
            self.shadow_fbo = self.ctx.framebuffer(depth_attachment=self.shadow_tex)
            self.shadow_enabled = True
        except Exception:
            self.shadow_enabled = False
            self.shadow_tex = self.ctx.depth_texture((4, 4))
            self.shadow_fbo = None

    def upload_mesh(self, mesh: Mesh, program) -> GPUMesh:
        key = (id(mesh), id(program))
        cached = self._mesh_cache.get(key)
        if cached is not None:
            return cached
        if mesh.vertices.size == 0 or mesh.indices.size == 0:
            gpu = GPUMesh(vao=None, count=0)
            self._mesh_cache[key] = gpu
            return gpu
        vbo = self.ctx.buffer(mesh.vertex_bytes)
        ibo = self.ctx.buffer(mesh.index_bytes)
        if program is self.shadow_prog or program is self.sky_prog:
            vao = self.ctx.vertex_array(
                program,
                [(vbo, "3f 20x", "in_position")],
                index_buffer=ibo,
                index_element_size=4,
            )
        else:
            vao = self.ctx.vertex_array(
                program,
                [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
                index_buffer=ibo,
                index_element_size=4,
            )
        gpu = GPUMesh(vao=vao, count=int(len(mesh.indices)))
        self._mesh_cache[key] = gpu
        return gpu

    def compute_light_vp(self, sun_dir: glm.vec3, focus: glm.vec3, extent: float = 32.0) -> glm.mat4:
        L = glm.normalize(sun_dir)
        eye = focus + L * 55.0
        up = glm.vec3(0.0, 1.0, 0.0)
        if abs(float(glm.dot(up, L))) > 0.92:
            up = glm.vec3(1.0, 0.0, 0.0)
        view = glm.lookAt(eye, focus, up)
        proj = glm.ortho(-extent, extent, -extent * 0.7, extent * 0.7, 1.0, 130.0)
        self.light_vp = proj * view
        return self.light_vp

    def begin_shadow(self) -> None:
        if not self.shadow_enabled:
            return
        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_res, self.shadow_res)
        self.shadow_fbo.clear(depth=1.0)

    def draw_shadow(self, mesh: Mesh, model: glm.mat4) -> None:
        if not self.shadow_enabled:
            return
        gpu = self.upload_mesh(mesh, self.shadow_prog)
        if gpu.vao is None:
            return
        mvp = self.light_vp * model
        self.shadow_prog["u_light_mvp"].write(mat_bytes(mvp))
        gpu.vao.render()

    def begin_frame(self, sky: Tuple[float, float, float]) -> None:
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.w, self.h)
        self.fbo.clear(sky[0], sky[1], sky[2], 1.0, depth=1.0)
        if self.id_is_uint:
            zeros = np.zeros((self.h, self.w), dtype=np.uint16 if self.id_dtype == "u2" else np.uint32)
        else:
            zeros = np.zeros((self.h, self.w), dtype=np.float32)
        self.id_tex.write(zeros.tobytes())
        self.depth_tex.write(np.zeros((self.h, self.w), dtype=np.float32).tobytes())

    def draw_sky(
        self,
        view: glm.mat4,
        sun_dir: glm.vec3,
        sun_color: glm.vec3,
        sun_elevation: float,
    ) -> None:
        gpu = self.upload_mesh(self.sky_mesh, self.sky_prog)
        if gpu.vao is None:
            return
        self.ctx.disable(self.ctx.DEPTH_TEST)
        self.sky_prog["u_view"].write(mat_bytes(view))
        self.sky_prog["u_projection"].write(mat_bytes(self.projection))
        self.sky_prog["u_sun_dir"].value = (float(sun_dir.x), float(sun_dir.y), float(sun_dir.z))
        self.sky_prog["u_sun_color"].value = (float(sun_color.x), float(sun_color.y), float(sun_color.z))
        self.sky_prog["u_sun_elevation"].value = float(sun_elevation)
        if "u_turbidity" in self.sky_prog:
            self.sky_prog["u_turbidity"].value = 2.5
        gpu.vao.render()
        self.ctx.enable(self.ctx.DEPTH_TEST)

    def set_frame_uniforms(
        self,
        view: glm.mat4,
        sun_dir: glm.vec3,
        sun_color: glm.vec3,
        ambient: float,
        sky: glm.vec3,
        fog: float,
        cam_pos: glm.vec3,
        wetness: float,
        sun_elevation: float = 45.0,
    ) -> None:
        p = self.prog
        p["u_view"].write(mat_bytes(view))
        p["u_projection"].write(mat_bytes(self.projection))
        p["u_sun_dir"].value = (float(sun_dir.x), float(sun_dir.y), float(sun_dir.z))
        p["u_sun_color"].value = (float(sun_color.x), float(sun_color.y), float(sun_color.z))
        p["u_ambient"].value = float(ambient)
        p["u_sky_color"].value = (float(sky.x), float(sky.y), float(sky.z))
        p["u_fog_density"].value = float(fog)
        p["u_cam_pos"].value = (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z))
        p["u_wetness"].value = float(wetness)
        _set_if(p, "u_sun_elevation", lambda u: setattr(u, "value", float(sun_elevation)))
        _set_if(p, "u_light_vp", lambda u: u.write(mat_bytes(self.light_vp)))
        _set_if(p, "u_shadow_enabled", lambda u: setattr(u, "value", 1 if self.shadow_enabled else 0))
        if self.shadow_tex is not None and "u_shadow_map" in p:
            self.shadow_tex.use(0)
            p["u_shadow_map"].value = 0

    def draw(
        self,
        mesh: Mesh,
        model: glm.mat4,
        albedo: Tuple[float, float, float],
        instance_id: int,
        specularity: float = 0.1,
        shininess: float = 16.0,
        style: int = 0,
        wetness: float = 0.0,
    ) -> None:
        gpu = self.upload_mesh(mesh, self.prog)
        if gpu.vao is None:
            return
        p = self.prog
        p["u_model"].write(mat_bytes(model))
        p["u_normal_matrix"].write(mat_bytes(_normal_matrix(model)))
        p["u_albedo"].value = (float(albedo[0]), float(albedo[1]), float(albedo[2]))
        p["u_specularity"].value = float(specularity)
        _set_if(p, "u_shininess", lambda u: setattr(u, "value", float(shininess)))
        p["u_instance_id"].value = int(instance_id)
        p["u_style"].value = int(style)
        p["u_wetness"].value = float(wetness)
        gpu.vao.render()

    def read_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = np.frombuffer(self.color_tex.read(), dtype=np.uint8).reshape(self.h, self.w, 4)
        rgb = np.flipud(rgb)[:, :, :3].copy()
        depth = np.frombuffer(self.depth_tex.read(), dtype=np.float32).reshape(self.h, self.w)
        depth = np.flipud(depth).copy()
        if self.id_is_uint:
            dt = np.uint16 if self.id_dtype == "u2" else np.uint32
            inst = np.frombuffer(self.id_tex.read(), dtype=dt).reshape(self.h, self.w)
            inst = np.flipud(inst).astype(np.uint16, copy=True)
        else:
            raw = np.frombuffer(self.id_tex.read(), dtype=np.float32).reshape(self.h, self.w)
            inst = np.flipud(np.rint(raw).astype(np.uint16))
        return rgb, depth, inst

    def release(self) -> None:
        try:
            self.fbo.release()
            self.color_tex.release()
            self.depth_tex.release()
            self.id_tex.release()
            self.depth_rbo.release()
            self.prog.release()
            if getattr(self, "shadow_fbo", None) is not None:
                self.shadow_fbo.release()
            if getattr(self, "shadow_tex", None) is not None:
                self.shadow_tex.release()
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass
