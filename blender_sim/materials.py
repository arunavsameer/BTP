"""Procedural PBR materials — no image textures, EEVEE-safe node graphs.

Spatial scales are metres. We sample **Object** coordinates (local metres)
rather than UVs so a building cube with no unwrap still gets storeys and
window bays, and a road ribbon still gets aggregate at a physical size.

Why the previous graphs looked flat
-----------------------------------
Blender 4+/5 Mix nodes have *stacked* sockets that share a name (``A`` is
a float, a vector, *and* a colour). ``node.inputs["A"]`` is the float.
``ShaderNodeTexNoise`` outputs ``Factor``, not ``Fac``. Both bugs made
every layered colour mix a no-op, so asphalt / cloth / tiles collapsed
to a constant albedo.

Math notes
----------
Asphalt roughness is anticorrelated with albedo: oil-dark Voronoi cells
are glossier (typical of bitumen). Facade windows are a 2-D lattice in
the object's (Y, Z) plane — bay × storey — with a hashed occupancy so
adjacent buildings do not blink in sync at night.

Appearance chaos (anti-overfitting)
-----------------------------------
The network must learn *looming motion*, not "red car" or "grey pothole".
Every surface therefore draws a **family** — natural, saturated, neon,
chrome, matte, pastel, or dark — and a matching BSDF envelope, so the same
mesh can render as brushed chrome in one episode and matte neon pink in the
next. ``chaos`` in ``[0, 1]`` interpolates between the original tame palette
(0.0, byte-identical to the pre-chaos pipeline) and full anarchy (1.0).

Colour families are drawn in HSV and converted with :mod:`colorsys` so
saturation and value can be controlled independently of hue; sampling three
independent RGB channels instead would concentrate the draws around grey.
"""

from __future__ import annotations

import colorsys
import math
import random
from typing import Any, Optional, Sequence


def _input(node: Any, name: str, typ: Optional[str] = None) -> Any:
    matches = [s for s in node.inputs if s.name == name]
    if typ:
        typed = [s for s in matches if s.type == typ]
        enabled = [s for s in typed if getattr(s, "enabled", True)]
        if enabled:
            return enabled[0]
        if typed:
            return typed[0]
    enabled = [s for s in matches if getattr(s, "enabled", True)]
    if enabled:
        return enabled[0]
    return matches[0] if matches else None


def _output(node: Any, name: str, typ: Optional[str] = None) -> Any:
    matches = [s for s in node.outputs if s.name == name]
    if typ:
        typed = [s for s in matches if s.type == typ]
        enabled = [s for s in typed if getattr(s, "enabled", True)]
        if enabled:
            return enabled[0]
        if typed:
            return typed[0]
    enabled = [s for s in matches if getattr(s, "enabled", True)]
    if enabled:
        return enabled[0]
    return matches[0] if matches else None


def _set(node: Any, name: str, value: Any, typ: Optional[str] = None) -> None:
    sock = _input(node, name, typ)
    if sock is None:
        return
    try:
        sock.default_value = value
    except Exception:
        pass


def _link(nt: Any, a: Any, a_out: str, b: Any, b_in: str,
          a_typ: Optional[str] = None, b_typ: Optional[str] = None) -> None:
    src = _output(a, a_out, a_typ)
    dst = _input(b, b_in, b_typ)
    if src is None or dst is None:
        return
    try:
        nt.links.new(src, dst)
    except Exception:
        pass


def _link_to(nt: Any, a: Any, a_out: str, dst: Any, a_typ: Optional[str] = None) -> None:
    src = _output(a, a_out, a_typ)
    if src is None or dst is None:
        return
    try:
        nt.links.new(src, dst)
    except Exception:
        pass


def _new_mat(name: str) -> tuple[Any, Any, Any, Any]:
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out.location = (420, 0)
    bsdf.location = (80, 0)
    _link(nt, bsdf, "BSDF", out, "Surface")
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    return mat, nt, bsdf, out


def _object_coords(nt: Any, scale: Sequence[float], loc_x: float = -780) -> Any:
    """Texture Coordinate (Object, metres) → Mapping. Returns Mapping."""
    tex = nt.nodes.new("ShaderNodeTexCoord")
    maps = nt.nodes.new("ShaderNodeMapping")
    tex.location = (loc_x, 0)
    maps.location = (loc_x + 200, 0)
    _link(nt, tex, "Object", maps, "Vector")
    _set(maps, "Scale", (float(scale[0]), float(scale[1]), float(scale[2])))
    return maps


def _mix_rgba(nt: Any, loc: tuple[float, float] = (0, 0)) -> Any:
    mix = nt.nodes.new("ShaderNodeMix")
    mix.location = loc
    try:
        mix.data_type = "RGBA"
    except Exception:
        pass
    return mix


def _bump_from(nt: Any, height_node: Any, height_out: str, bsdf: Any,
               strength: float = 0.35, distance: float = 0.04) -> None:
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (40, -280)
    _set(bump, "Strength", float(strength))
    _set(bump, "Distance", float(distance))
    _link(nt, height_node, height_out, bump, "Height")
    _link(nt, bump, "Normal", bsdf, "Normal")


def make_asphalt(name: str, albedo: tuple[float, float, float], seed: float = 0.0) -> Any:
    """Worn bitumen: dark base, Voronoi stones, glossy oil patches, bump."""
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (1.0, 1.0, 1.0))

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-360, 160)
    _set(noise, "Scale", 12.0 + 4.0 * (seed % 1.0))
    _set(noise, "Detail", 10.0)
    _set(noise, "Roughness", 0.55)
    _link(nt, maps, "Vector", noise, "Vector")

    voro = nt.nodes.new("ShaderNodeTexVoronoi")
    voro.location = (-360, -80)
    try:
        voro.feature = "F1"
    except Exception:
        pass
    _set(voro, "Scale", 7.5)
    _set(voro, "Randomness", 0.85)
    _link(nt, maps, "Vector", voro, "Vector")

    mix = _mix_rgba(nt, (-80, 80))
    _set(mix, "A", (*albedo, 1.0), "RGBA")
    _set(mix, "B", (albedo[0] * 0.35, albedo[1] * 0.35, albedo[2] * 0.32, 1.0), "RGBA")
    _link(nt, voro, "Distance", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")

    # Roughness: dry ~0.92, oil patches (low noise) ~0.45.
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-80, -200)
    ramp.color_ramp.elements[0].position = 0.30
    ramp.color_ramp.elements[0].color = (0.42, 0.42, 0.42, 1.0)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (0.94, 0.94, 0.94, 1.0)
    _link(nt, noise, "Factor", ramp, "Fac")
    _link(nt, ramp, "Color", bsdf, "Roughness")
    _set(bsdf, "Specular IOR Level", 0.28)
    _bump_from(nt, noise, "Factor", bsdf, strength=0.45, distance=0.03)
    return mat


def make_concrete_tiles(name: str, albedo: tuple[float, float, float], tile_m: float = 0.40) -> Any:
    """Sidewalk pavers: Brick lattice in metres + fine noise + grout bump."""
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (1.0 / tile_m, 1.0 / tile_m, 1.0 / tile_m))

    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.location = (-360, 40)
    _set(brick, "Scale", 1.0)
    _set(brick, "Mortar Size", 0.08)
    _set(brick, "Mortar Smooth", 0.10)
    _set(brick, "Bias", 0.12)
    _set(brick, "Brick Width", 1.0)
    _set(brick, "Row Height", 1.0)
    _set(brick, "Color1", (min(1.0, albedo[0] * 1.18), min(1.0, albedo[1] * 1.14), min(1.0, albedo[2] * 1.08), 1.0))
    _set(brick, "Color2", (albedo[0] * 0.62, albedo[1] * 0.60, albedo[2] * 0.56, 1.0))
    _set(brick, "Mortar", (0.12, 0.11, 0.10, 1.0))
    _link(nt, maps, "Vector", brick, "Vector")

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-360, -240)
    _set(noise, "Scale", 28.0)
    _set(noise, "Detail", 8.0)
    _link(nt, maps, "Vector", noise, "Vector")

    mix = _mix_rgba(nt, (-80, 20))
    _set(mix, "Factor", 0.22, "VALUE")
    _link(nt, brick, "Color", mix, "A", b_typ="RGBA")
    _set(mix, "B", (albedo[0] * 0.55, albedo[1] * 0.54, albedo[2] * 0.50, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Roughness", 0.82)
    _set(bsdf, "Specular IOR Level", 0.12)
    # Invert brick Factor so grout (mortar=0) becomes the tall bump.
    inv = nt.nodes.new("ShaderNodeInvert")
    inv.location = (-80, -200)
    _link(nt, brick, "Factor", inv, "Color")
    _bump_from(nt, inv, "Color", bsdf, strength=0.55, distance=0.012)
    return mat


def make_grass(name: str, albedo: tuple[float, float, float] = (0.14, 0.22, 0.08)) -> Any:
    """Verge: two-tone 1/f grass + strong micro-bump."""
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (1.4, 1.4, 1.4))

    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", 22.0)
    _set(noise, "Detail", 12.0)
    _set(noise, "Roughness", 0.7)
    _link(nt, maps, "Vector", noise, "Vector")

    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*albedo, 1.0), "RGBA")
    _set(mix, "B", (albedo[0] * 0.45, albedo[1] * 0.70, albedo[2] * 0.35, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Roughness", 0.92)
    _set(bsdf, "Specular IOR Level", 0.04)
    _bump_from(nt, noise, "Factor", bsdf, strength=0.65, distance=0.025)
    return mat


def _math(nt: Any, op: str, loc: tuple[float, float], b: Optional[float] = None) -> Any:
    node = nt.nodes.new("ShaderNodeMath")
    node.location = loc
    try:
        node.operation = op
    except Exception:
        pass
    if b is not None:
        ins = [s for s in node.inputs if s.name == "Value"]
        if len(ins) >= 2:
            ins[1].default_value = float(b)
    return node


def _math_in(node: Any, index: int = 0) -> Any:
    ins = [s for s in node.inputs if s.name == "Value"]
    return ins[index] if index < len(ins) else ins[0]


def make_facade(name: str, wall: tuple[float, float, float], night: bool, seed: float = 0.0) -> Any:
    """Rectangular window lattice in the object's (Y, Z) metre plane.

    Building box: X=depth, Y=width, Z=height. Street faces are ±X, so a
    window is the product of two 1-D pulse trains

        win = 1[εx < {Y/bay} < 1-εx] · 1[εz < {Z/H} < 1-εz]

    Occupancy at night is a per-cell hash, not a smooth Noise field
    (smooth noise painted leopard-print emission across the elevation):

        h = fract(sin(i·12.9898 + j·78.233 + seed)·43758.5453)
        emit = win · 1[h > 0.52]
    """
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (1.0, 1.0, 1.0))
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-720, 40)
    _link(nt, maps, "Vector", sep, "Vector")

    bay_m, storey_m = 1.70, 3.20

    def _pulse_axis(coord_out: str, period: float, lo: float, hi: float, x0: float) -> Any:
        """1[lo < {coord/period} < hi] along one object axis."""
        div = _math(nt, "DIVIDE", (x0, 40), period)
        _link_to(nt, sep, coord_out, _math_in(div, 0))
        fr = _math(nt, "FRACT", (x0 + 140, 40))
        _link_to(nt, div, "Value", _math_in(fr, 0))
        gt = _math(nt, "GREATER_THAN", (x0 + 280, 80), lo)
        lt = _math(nt, "LESS_THAN", (x0 + 280, 0), hi)
        _link_to(nt, fr, "Value", _math_in(gt, 0))
        _link_to(nt, fr, "Value", _math_in(lt, 0))
        both = _math(nt, "MULTIPLY", (x0 + 420, 40))
        _link_to(nt, gt, "Value", _math_in(both, 0))
        _link_to(nt, lt, "Value", _math_in(both, 1))
        return div, both

    y_div, y_pulse = _pulse_axis("Y", bay_m, 0.22, 0.80, -700)
    x_div, x_pulse = _pulse_axis("X", bay_m, 0.22, 0.80, -700)
    z_div, z_pulse = _pulse_axis("Z", storey_m, 0.30, 0.82, -700)
    # Reposition the three pulse stacks so they don't sit on top of each other.
    for node, dy in ((y_div, 220), (y_pulse, 220), (x_div, 40), (x_pulse, 40), (z_div, -140), (z_pulse, -140)):
        node.location = (node.location.x, dy)

    win_front = _math(nt, "MULTIPLY", (-80, 180))  # ±X facade: YZ
    win_side = _math(nt, "MULTIPLY", (-80, -40))   # ±Y facade: XZ
    _link_to(nt, y_pulse, "Value", _math_in(win_front, 0))
    _link_to(nt, z_pulse, "Value", _math_in(win_front, 1))
    _link_to(nt, x_pulse, "Value", _math_in(win_side, 0))
    _link_to(nt, z_pulse, "Value", _math_in(win_side, 1))

    # Object-space normal (not world): look_along yaws the mesh, so world
    # ±X is not the street facade. TexCoord.Normal is local.
    ntex = nt.nodes.new("ShaderNodeTexCoord")
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ntex.location = (-80, 360)
    nsep.location = (80, 360)
    _link(nt, ntex, "Normal", nsep, "Vector")
    ax = _math(nt, "ABSOLUTE", (240, 400))
    ay = _math(nt, "ABSOLUTE", (240, 320))
    _link_to(nt, nsep, "X", _math_in(ax, 0))
    _link_to(nt, nsep, "Y", _math_in(ay, 0))
    den = _math(nt, "ADD", (400, 360))
    _link_to(nt, ax, "Value", _math_in(den, 0))
    _link_to(nt, ay, "Value", _math_in(den, 1))
    wgt = _math(nt, "DIVIDE", (540, 360))
    _link_to(nt, ax, "Value", _math_in(wgt, 0))
    _link_to(nt, den, "Value", _math_in(wgt, 1))
    win = nt.nodes.new("ShaderNodeMix")
    try:
        win.data_type = "FLOAT"
    except Exception:
        pass
    win.location = (80, 80)
    _link_to(nt, wgt, "Value", _input(win, "Factor", "VALUE"))
    _link_to(nt, win_side, "Value", _input(win, "A", "VALUE"))
    _link_to(nt, win_front, "Value", _input(win, "B", "VALUE"))
    win_out = _output(win, "Result", "VALUE")

    mix = _mix_rgba(nt, (280, 80))
    _set(mix, "A", (*wall, 1.0), "RGBA")
    _set(mix, "B", (0.04, 0.05, 0.06, 1.0), "RGBA")
    try:
        nt.links.new(win_out, _input(mix, "Factor", "VALUE"))
    except Exception:
        pass
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")

    # roughness = 0.78 - 0.62*win  → wall 0.78, glass 0.16
    rmix = _math(nt, "MULTIPLY", (280, -160), 0.62)
    rsub = _math(nt, "SUBTRACT", (440, -160))
    rsub_ins = [s for s in rsub.inputs if s.name == "Value"]
    rsub_ins[0].default_value = 0.78
    try:
        nt.links.new(win_out, _math_in(rmix, 0))
    except Exception:
        pass
    _link_to(nt, rmix, "Value", rsub_ins[1] if len(rsub_ins) > 1 else rsub_ins[0])
    _link(nt, rsub, "Value", bsdf, "Roughness")
    _set(bsdf, "Specular IOR Level", 0.42)
    _set(bsdf, "Metallic", 0.0)

    if night:
        y_fl = _math(nt, "FLOOR", (-380, -200))
        z_fl = _math(nt, "FLOOR", (-380, -280))
        _link_to(nt, y_div, "Value", _math_in(y_fl, 0))
        _link_to(nt, z_div, "Value", _math_in(z_fl, 0))
        y_s = _math(nt, "MULTIPLY", (-220, -200), 12.9898)
        z_s = _math(nt, "MULTIPLY", (-220, -280), 78.233)
        _link_to(nt, y_fl, "Value", _math_in(y_s, 0))
        _link_to(nt, z_fl, "Value", _math_in(z_s, 0))
        add1 = _math(nt, "ADD", (-60, -240))
        _link_to(nt, y_s, "Value", _math_in(add1, 0))
        _link_to(nt, z_s, "Value", _math_in(add1, 1))
        add2 = _math(nt, "ADD", (80, -240), 17.0 * (seed % 1.0) + 1.3)
        _link_to(nt, add1, "Value", _math_in(add2, 0))
        sine = _math(nt, "SINE", (220, -240))
        _link_to(nt, add2, "Value", _math_in(sine, 0))
        mulh = _math(nt, "MULTIPLY", (360, -240), 43758.5453)
        _link_to(nt, sine, "Value", _math_in(mulh, 0))
        fr = _math(nt, "FRACT", (500, -240))
        _link_to(nt, mulh, "Value", _math_in(fr, 0))
        lit = _math(nt, "GREATER_THAN", (640, -240), 0.52)
        _link_to(nt, fr, "Value", _math_in(lit, 0))
        gate = _math(nt, "MULTIPLY", (780, -160))
        try:
            nt.links.new(win_out, _math_in(gate, 0))
        except Exception:
            pass
        _link_to(nt, lit, "Value", _math_in(gate, 1))
        emit = _math(nt, "MULTIPLY", (920, -160), 4.5)
        _link_to(nt, gate, "Value", _math_in(emit, 0))
        _set(bsdf, "Emission Color", (1.0, 0.80, 0.48, 1.0))
        _link(nt, emit, "Value", bsdf, "Emission Strength")
    else:
        _set(bsdf, "Emission Color", (0.45, 0.52, 0.60, 1.0))
        _set(bsdf, "Emission Strength", 0.03)
    return mat


def make_car_paint(name: str, color: tuple[float, float, float]) -> Any:
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (0.8, 0.8, 0.8))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", 6.0)
    _set(noise, "Detail", 4.0)
    _link(nt, maps, "Vector", noise, "Vector")
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*color, 1.0), "RGBA")
    _set(mix, "B", (min(1.0, color[0] * 1.15), min(1.0, color[1] * 1.15), min(1.0, color[2] * 1.12), 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Metallic", 0.78)
    _set(bsdf, "Roughness", 0.22)
    _set(bsdf, "Coat Weight", 1.0)
    _set(bsdf, "Coat Roughness", 0.05)
    _set(bsdf, "Coat IOR", 1.5)
    _set(bsdf, "Specular IOR Level", 0.55)
    return mat


def make_glass(name: str, tint: tuple[float, float, float] = (0.45, 0.55, 0.62)) -> Any:
    mat, _nt, bsdf, _out = _new_mat(name)
    _set(bsdf, "Base Color", (*tint, 1.0))
    _set(bsdf, "Transmission Weight", 0.82)
    _set(bsdf, "Roughness", 0.06)
    _set(bsdf, "IOR", 1.45)
    _set(bsdf, "Alpha", 0.62)
    _set(bsdf, "Metallic", 0.05)
    _set(bsdf, "Specular IOR Level", 0.6)
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "BLEND"
        except Exception:
            pass
    return mat


def make_rubber(name: str) -> Any:
    mat, _nt, bsdf, _out = _new_mat(name)
    _set(bsdf, "Base Color", (0.025, 0.025, 0.025, 1.0))
    _set(bsdf, "Roughness", 0.94)
    _set(bsdf, "Specular IOR Level", 0.04)
    return mat


def make_skin(name: str, color: tuple[float, float, float]) -> Any:
    """Lambertian skin + SSS. A tiny noise keeps the face from looking plastic."""
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (9.0, 9.0, 9.0))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", 14.0)
    _set(noise, "Detail", 6.0)
    _link(nt, maps, "Vector", noise, "Vector")
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*color, 1.0), "RGBA")
    _set(mix, "B", (color[0] * 0.82, color[1] * 0.78, color[2] * 0.72, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Roughness", 0.52)
    _set(bsdf, "Specular IOR Level", 0.38)
    _set(bsdf, "Subsurface Weight", 0.22)
    _set(bsdf, "Subsurface Radius", (1.0, 0.35, 0.18))
    _set(bsdf, "Subsurface Scale", 0.12)
    _bump_from(nt, noise, "Factor", bsdf, strength=0.12, distance=0.004)
    return mat


def make_cloth(name: str, color: tuple[float, float, float], seed: float = 0.0) -> Any:
    """Woven fabric: fine 1/f noise + sheen."""
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (14.0, 14.0, 14.0))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", 8.0 + 4.0 * (seed % 1.0))
    _set(noise, "Detail", 10.0)
    _link(nt, maps, "Vector", noise, "Vector")
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*color, 1.0), "RGBA")
    _set(mix, "B", (color[0] * 0.62, color[1] * 0.62, color[2] * 0.62, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Roughness", 0.84)
    _set(bsdf, "Sheen Weight", 0.40)
    _set(bsdf, "Sheen Roughness", 0.38)
    _bump_from(nt, noise, "Factor", bsdf, strength=0.25, distance=0.006)
    return mat


def make_metal_paint(name: str, color: tuple[float, float, float], roughness: float = 0.4) -> Any:
    mat, _nt, bsdf, _out = _new_mat(name)
    _set(bsdf, "Base Color", (*color, 1.0))
    _set(bsdf, "Metallic", 0.55)
    _set(bsdf, "Roughness", roughness)
    return mat


def make_emissive(name: str, color: tuple[float, float, float], strength: float) -> Any:
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    _set(emit, "Color", (*color, 1.0))
    _set(emit, "Strength", float(strength))
    _link(nt, emit, "Emission", out, "Surface")
    return mat


def make_bark(name: str) -> Any:
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (4.0, 1.2, 4.0))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", 12.0)
    _set(noise, "Detail", 10.0)
    _set(noise, "Roughness", 0.7)
    _link(nt, maps, "Vector", noise, "Vector")
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (0.22, 0.12, 0.06, 1.0), "RGBA")
    _set(mix, "B", (0.10, 0.06, 0.03, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Roughness", 0.92)
    _bump_from(nt, noise, "Factor", bsdf, strength=0.7, distance=0.02)
    return mat


def make_roof(name: str, color: tuple[float, float, float] = (0.12, 0.12, 0.13)) -> Any:
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (2.2, 2.2, 2.2))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", 18.0)
    _set(noise, "Detail", 6.0)
    _link(nt, maps, "Vector", noise, "Vector")
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*color, 1.0), "RGBA")
    _set(mix, "B", (color[0] * 0.55, color[1] * 0.55, color[2] * 0.55, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Roughness", 0.88)
    _bump_from(nt, noise, "Factor", bsdf, strength=0.4, distance=0.015)
    return mat


def make_simple(name: str, color: tuple[float, float, float], roughness: float = 0.6) -> Any:
    mat, _nt, bsdf, _out = _new_mat(name)
    _set(bsdf, "Base Color", (*color, 1.0))
    _set(bsdf, "Roughness", float(roughness))
    return mat


# ===========================================================================
# Appearance chaos
# ===========================================================================

#: Surface families. Each one is a joint distribution over (hue, saturation,
#: value) *and* the BSDF envelope, because "neon" is not only a colour — it
#: is also dielectric, smooth-ish, and sometimes self-lit.
CHAOS_FAMILIES: tuple[str, ...] = (
    "natural",
    "saturated",
    "neon",
    "chrome",
    "matte",
    "pastel",
    "dark",
)

#: Family weights at chaos = 1. At chaos = 0 everything collapses to natural.
_FAMILY_WEIGHTS: dict[str, float] = {
    "natural": 0.26,
    "saturated": 0.20,
    "neon": 0.14,
    "chrome": 0.12,
    "matte": 0.14,
    "pastel": 0.08,
    "dark": 0.06,
}


def _lerp(a: float, b: float, u: float) -> float:
    return float(a) + (float(b) - float(a)) * float(u)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _rgb_to_hsv(c: Sequence[float]) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(_clamp01(c[0]), _clamp01(c[1]), _clamp01(c[2]))


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(float(h) % 1.0, _clamp01(s), _clamp01(v))
    return (r, g, b)


def pick_family(rng: random.Random, chaos: float) -> str:
    """Draw a surface family. ``chaos = 0`` always returns ``natural``."""
    c = _clamp01(chaos)
    if c <= 1e-6 or rng.random() > c:
        return "natural"
    keys = list(_FAMILY_WEIGHTS)
    total = sum(_FAMILY_WEIGHTS[k] for k in keys)
    x = rng.uniform(0.0, total)
    acc = 0.0
    for k in keys:
        acc += _FAMILY_WEIGHTS[k]
        if x <= acc:
            return k
    return keys[-1]


def chaos_albedo(
    rng: random.Random,
    base: Sequence[float],
    chaos: float,
    family: Optional[str] = None,
) -> tuple[float, float, float]:
    """Randomise an albedo inside ``family``, blended toward ``base`` by chaos.

    The result is always interpolated back toward the caller's ``base`` by
    ``1 - chaos`` so that a low chaos dial still produces a recognisable
    street rather than a discontinuous jump to full randomness.
    """
    c = _clamp01(chaos)
    fam = family or pick_family(rng, c)
    h0, s0, v0 = _rgb_to_hsv(base)
    if fam == "natural":
        h = h0 + rng.uniform(-0.05, 0.05)
        s = s0 * rng.uniform(0.75, 1.25)
        v = v0 * rng.uniform(0.75, 1.30)
    elif fam == "saturated":
        h = rng.random()
        s = rng.uniform(0.70, 1.00)
        v = rng.uniform(0.35, 0.85)
    elif fam == "neon":
        h = rng.random()
        s = rng.uniform(0.88, 1.00)
        v = rng.uniform(0.80, 1.00)
    elif fam == "chrome":
        h = rng.random()
        s = rng.uniform(0.00, 0.10)
        v = rng.uniform(0.72, 0.98)
    elif fam == "matte":
        h = rng.random()
        s = rng.uniform(0.05, 0.35)
        v = rng.uniform(0.22, 0.62)
    elif fam == "pastel":
        h = rng.random()
        s = rng.uniform(0.14, 0.36)
        v = rng.uniform(0.78, 1.00)
    else:  # dark
        h = rng.random()
        s = rng.uniform(0.00, 0.45)
        v = rng.uniform(0.02, 0.14)
    r, g, b = _hsv_to_rgb(h, s, v)
    br, bg, bb = _clamp01(base[0]), _clamp01(base[1]), _clamp01(base[2])
    return (_lerp(br, r, c), _lerp(bg, g, c), _lerp(bb, b, c))


def apply_surface_chaos(
    bsdf: Any,
    rng: random.Random,
    chaos: float,
    family: str = "natural",
    *,
    allow_emission: bool = True,
    base_roughness: Optional[float] = None,
) -> None:
    """Randomise the BSDF envelope: metallic, roughness, coat, sheen, IOR.

    Ranges are family-conditioned so the shading agrees with the albedo:
    chrome is metallic and smooth, matte is dielectric and rough, neon is a
    bright dielectric that may self-illuminate.
    """
    c = _clamp01(chaos)
    if c <= 1e-6:
        if base_roughness is not None:
            _set(bsdf, "Roughness", float(base_roughness))
        return

    if family == "chrome":
        metallic = rng.uniform(0.85, 1.00)
        rough = rng.uniform(0.02, 0.28)
    elif family == "matte":
        metallic = 0.0
        rough = rng.uniform(0.72, 1.00)
    elif family == "neon":
        metallic = rng.uniform(0.0, 0.12)
        rough = rng.uniform(0.12, 0.55)
    elif family == "dark":
        metallic = rng.uniform(0.0, 0.55)
        rough = rng.uniform(0.30, 0.95)
    else:
        metallic = rng.uniform(0.0, 1.0) if rng.random() < 0.35 else rng.uniform(0.0, 0.15)
        rough = rng.uniform(0.08, 0.95)

    if base_roughness is not None:
        rough = _lerp(float(base_roughness), rough, c)
    metallic = _lerp(0.0, metallic, c)

    _set(bsdf, "Metallic", float(metallic))
    _set(bsdf, "Roughness", float(_clamp01(rough)))
    _set(bsdf, "Specular IOR Level", float(rng.uniform(0.18, 0.72)))
    _set(bsdf, "IOR", float(rng.uniform(1.32, 1.72)))

    # Clear coat: a wet/lacquered top layer over anything.
    if rng.random() < 0.30 * c:
        _set(bsdf, "Coat Weight", float(rng.uniform(0.35, 1.0)))
        _set(bsdf, "Coat Roughness", float(rng.uniform(0.02, 0.35)))
        _set(bsdf, "Coat IOR", float(rng.uniform(1.35, 1.75)))
    if rng.random() < 0.22 * c:
        _set(bsdf, "Sheen Weight", float(rng.uniform(0.15, 0.85)))
        _set(bsdf, "Sheen Roughness", float(rng.uniform(0.15, 0.75)))
    if rng.random() < 0.30 * c:
        _set(bsdf, "Anisotropic", float(rng.uniform(0.15, 0.85)))
        _set(bsdf, "Anisotropic Rotation", float(rng.uniform(0.0, 1.0)))
    if allow_emission and family == "neon" and rng.random() < 0.45 * c:
        _set(bsdf, "Emission Color", (*_hsv_to_rgb(rng.random(), 0.95, 1.0), 1.0))
        _set(bsdf, "Emission Strength", float(rng.uniform(0.6, 4.5)))


def make_chaos_surface(
    name: str,
    base_color: Sequence[float],
    rng: random.Random,
    chaos: float = 0.0,
    *,
    base_roughness: float = 0.6,
    family: Optional[str] = None,
    allow_emission: bool = True,
) -> Any:
    """Generic randomised Principled surface for props and clutter."""
    fam = family or pick_family(rng, chaos)
    col = chaos_albedo(rng, base_color, chaos, fam)
    mat, _nt, bsdf, _out = _new_mat(name)
    _set(bsdf, "Base Color", (*col, 1.0))
    apply_surface_chaos(
        bsdf, rng, chaos, fam,
        allow_emission=allow_emission,
        base_roughness=base_roughness,
    )
    return mat


# ---------------------------------------------------------------------------
# Procedural pattern generators (cloth, liveries, signage)
# ---------------------------------------------------------------------------

#: Every pattern is a scalar field in object space that drives a two-colour
#: mix. Names match the node they are built from.
PATTERN_KINDS: tuple[str, ...] = (
    "plain",
    "noise",
    "stripe",
    "checker",
    "camo",
    "dots",
    "magic",
)


def _pattern_factor(
    nt: Any,
    maps: Any,
    kind: str,
    rng: random.Random,
) -> tuple[Any, str, Optional[str]]:
    """Build the pattern node. Returns (node, output_name, output_type)."""
    if kind == "stripe":
        node = nt.nodes.new("ShaderNodeTexWave")
        node.location = (-360, 120)
        for attr, val in (
            ("wave_type", rng.choice(("BANDS", "RINGS"))),
            ("bands_direction", rng.choice(("X", "Y", "Z", "DIAGONAL"))),
            ("wave_profile", rng.choice(("SIN", "SAW", "TRI"))),
        ):
            try:
                setattr(node, attr, val)
            except Exception:
                pass
        _set(node, "Scale", rng.uniform(1.5, 14.0))
        _set(node, "Distortion", rng.uniform(0.0, 6.0))
        _link(nt, maps, "Vector", node, "Vector")
        return node, "Fac", "VALUE"

    if kind == "checker":
        node = nt.nodes.new("ShaderNodeTexChecker")
        node.location = (-360, 120)
        _set(node, "Scale", rng.uniform(3.0, 26.0))
        _link(nt, maps, "Vector", node, "Vector")
        return node, "Fac", "VALUE"

    if kind == "dots":
        node = nt.nodes.new("ShaderNodeTexVoronoi")
        node.location = (-360, 120)
        try:
            node.feature = "F1"
        except Exception:
            pass
        _set(node, "Scale", rng.uniform(4.0, 22.0))
        _set(node, "Randomness", rng.uniform(0.3, 1.0))
        _link(nt, maps, "Vector", node, "Vector")
        return node, "Distance", "VALUE"

    if kind == "magic":
        node = nt.nodes.new("ShaderNodeTexMagic")
        node.location = (-360, 120)
        try:
            node.turbulence_depth = rng.randint(1, 5)
        except Exception:
            pass
        _set(node, "Scale", rng.uniform(2.0, 12.0))
        _set(node, "Distortion", rng.uniform(0.5, 4.0))
        _link(nt, maps, "Vector", node, "Vector")
        return node, "Fac", "VALUE"

    if kind == "camo":
        # Noise through a constant-interpolation ramp: hard-edged blotches.
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.location = (-560, 120)
        _set(noise, "Scale", rng.uniform(2.5, 9.0))
        _set(noise, "Detail", rng.uniform(2.0, 8.0))
        _link(nt, maps, "Vector", noise, "Vector")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.location = (-360, 120)
        try:
            ramp.color_ramp.interpolation = "CONSTANT"
        except Exception:
            pass
        ramp.color_ramp.elements[0].position = rng.uniform(0.30, 0.45)
        ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        ramp.color_ramp.elements[1].position = rng.uniform(0.52, 0.68)
        ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        _link(nt, noise, "Factor", ramp, "Fac")
        return ramp, "Color", None

    # "noise" and "plain" both fall through to a smooth 1/f field; "plain"
    # simply uses a low contrast mix at the call site.
    node = nt.nodes.new("ShaderNodeTexNoise")
    node.location = (-360, 120)
    _set(node, "Scale", rng.uniform(4.0, 26.0))
    _set(node, "Detail", rng.uniform(4.0, 12.0))
    _set(node, "Roughness", rng.uniform(0.35, 0.8))
    _link(nt, maps, "Vector", node, "Vector")
    return node, "Factor", "VALUE"


def make_patterned_cloth(
    name: str,
    color: Sequence[float],
    rng: random.Random,
    chaos: float = 0.0,
    seed: float = 0.0,
) -> Any:
    """Garment fabric with a procedural print.

    At ``chaos = 0`` this is the original two-tone weave. Above it the shirt
    can be striped, checked, blotched, spotted, or psychedelic, with the two
    print colours drawn from independent families — the point is that a
    pedestrian's silhouette must be learnable without their texture.
    """
    c = _clamp01(chaos)
    kind = "noise" if c <= 1e-6 else rng.choice(PATTERN_KINDS)
    fam_a = pick_family(rng, c)
    fam_b = pick_family(rng, c)
    col_a = chaos_albedo(rng, color, c, fam_a)
    dim = (color[0] * 0.62, color[1] * 0.62, color[2] * 0.62)
    col_b = chaos_albedo(rng, dim, c, fam_b)

    mat, nt, bsdf, _out = _new_mat(name)
    scale = 14.0 if c <= 1e-6 else rng.uniform(4.0, 26.0)
    maps = _object_coords(nt, (scale, scale, scale))
    try:
        maps.inputs["Rotation"].default_value = (
            rng.uniform(0.0, 0.6) * c,
            rng.uniform(0.0, 0.6) * c,
            rng.uniform(0.0, math.pi) * c,
        )
    except Exception:
        pass

    node, out_name, out_typ = _pattern_factor(nt, maps, kind, rng)
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*col_a, 1.0), "RGBA")
    _set(mix, "B", (*col_b, 1.0), "RGBA")
    _link(nt, node, out_name, mix, "Factor", a_typ=out_typ, b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")

    _set(bsdf, "Roughness", 0.84)
    _set(bsdf, "Sheen Weight", 0.40)
    _set(bsdf, "Sheen Roughness", 0.38)
    apply_surface_chaos(
        bsdf, rng, c, fam_a, allow_emission=True, base_roughness=0.84,
    )
    # Weave bump always comes from a fine noise, never from the print.
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.location = (-360, -240)
    _set(fine, "Scale", 8.0 + 4.0 * (float(seed) % 1.0))
    _set(fine, "Detail", 10.0)
    _link(nt, maps, "Vector", fine, "Vector")
    _bump_from(nt, fine, "Factor", bsdf, strength=0.25, distance=0.006)
    return mat


def make_chaos_car_paint(
    name: str,
    color: Sequence[float],
    rng: random.Random,
    chaos: float = 0.0,
) -> Any:
    """Vehicle livery: factory metallic, matte wrap, chrome, or neon.

    Keeps a flake noise on the base colour so the body still reads as a
    curved metal panel under EEVEE's screen-space reflections.
    """
    c = _clamp01(chaos)
    if c <= 1e-6:
        return make_car_paint(name, tuple(color))
    fam = pick_family(rng, c)
    col = chaos_albedo(rng, color, c, fam)
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (0.8, 0.8, 0.8))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", rng.uniform(3.0, 28.0))
    _set(noise, "Detail", rng.uniform(2.0, 8.0))
    _link(nt, maps, "Vector", noise, "Vector")
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*col, 1.0), "RGBA")
    flake = chaos_albedo(rng, col, c * 0.5, fam)
    _set(mix, "B", (*flake, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    apply_surface_chaos(bsdf, rng, c, fam, base_roughness=0.22)
    # A car is a lacquered panel far more often than a random prop.
    if rng.random() < 0.55:
        _set(bsdf, "Coat Weight", float(rng.uniform(0.4, 1.0)))
        _set(bsdf, "Coat Roughness", float(rng.uniform(0.02, 0.20)))
    return mat


def make_water(
    name: str,
    rng: random.Random,
    chaos: float = 0.0,
    tint: Sequence[float] = (0.03, 0.045, 0.05),
) -> Any:
    """Standing water: near-mirror with a noise-rippled roughness.

    EEVEE resolves this through screen-space reflection, so the puddle picks
    up the sky and the facades without needing refraction support. A little
    transmission is requested when the build exposes it, but the look does
    not depend on it.
    """
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (2.4, 2.4, 2.4))
    ripple = nt.nodes.new("ShaderNodeTexNoise")
    ripple.location = (-360, -60)
    _set(ripple, "Scale", rng.uniform(9.0, 40.0))
    _set(ripple, "Detail", rng.uniform(4.0, 12.0))
    _set(ripple, "Roughness", rng.uniform(0.4, 0.8))
    _link(nt, maps, "Vector", ripple, "Vector")

    # Roughness stays glassy; the ramp only breaks up the mirror.
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-140, -60)
    lo = rng.uniform(0.01, 0.05)
    hi = lo + rng.uniform(0.03, 0.22) * (0.4 + 0.6 * _clamp01(chaos))
    ramp.color_ramp.elements[0].position = 0.30
    ramp.color_ramp.elements[0].color = (lo, lo, lo, 1.0)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (hi, hi, hi, 1.0)
    _link(nt, ripple, "Factor", ramp, "Fac")
    _link(nt, ramp, "Color", bsdf, "Roughness")

    col = chaos_albedo(rng, tint, _clamp01(chaos) * 0.5, "dark")
    _set(bsdf, "Base Color", (*col, 1.0))
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Specular IOR Level", 1.0)
    _set(bsdf, "IOR", 1.33)
    _set(bsdf, "Transmission Weight", rng.uniform(0.10, 0.45))
    _bump_from(nt, ripple, "Factor", bsdf, strength=0.12, distance=0.004)
    for attr in ("use_raytrace_refraction", "use_screen_refraction"):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, True)
            except Exception:
                pass
    return mat


def make_foliage(
    name: str,
    rng: random.Random,
    chaos: float = 0.0,
    base: Sequence[float] = (0.10, 0.22, 0.06),
) -> Any:
    """Tree canopy: two-tone leaf mass with translucency and a rough surface.

    Autumn / dead / exotic canopies come out of the chaos dial, so a tree is
    not a reliable green blob for the detector to key on.
    """
    c = _clamp01(chaos)
    if rng.random() < 0.30 * c:
        leaf = chaos_albedo(rng, base, c, rng.choice(("saturated", "neon", "dark")))
    else:
        leaf = chaos_albedo(rng, base, c * 0.6, "natural")
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (2.2, 2.2, 2.2))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    _set(noise, "Scale", rng.uniform(6.0, 22.0))
    _set(noise, "Detail", rng.uniform(4.0, 12.0))
    _link(nt, maps, "Vector", noise, "Vector")
    mix = _mix_rgba(nt, (-40, 20))
    _set(mix, "A", (*leaf, 1.0), "RGBA")
    _set(mix, "B", (leaf[0] * 0.45, leaf[1] * 0.72, leaf[2] * 0.38, 1.0), "RGBA")
    _link(nt, noise, "Factor", mix, "Factor", a_typ="VALUE", b_typ="VALUE")
    _link(nt, mix, "Result", bsdf, "Base Color", a_typ="RGBA")
    _set(bsdf, "Roughness", rng.uniform(0.68, 0.95))
    _set(bsdf, "Specular IOR Level", 0.12)
    # Backlit leaves: cheap translucency keeps the canopy from reading solid.
    _set(bsdf, "Subsurface Weight", rng.uniform(0.05, 0.25))
    _set(bsdf, "Subsurface Radius", (0.35, 0.9, 0.25))
    _set(bsdf, "Subsurface Scale", rng.uniform(0.02, 0.10))
    _bump_from(nt, noise, "Factor", bsdf, strength=0.55, distance=0.02)
    return mat


def make_canopy_gobo(name: str, rng: random.Random, chaos: float = 0.0) -> Any:
    """Alpha-punched sheet used as an overhead shadow gobo.

    Dappled light is a *shadow* effect, so the material has to be visible to
    the shadow pass while being invisible to primary rays wherever alpha is
    0. EEVEE resolves that with the dithered (alpha-hashed) render method
    plus transparent shadows; both are set defensively because the property
    names moved between 4.x and 5.x.
    """
    mat, nt, bsdf, _out = _new_mat(name)
    maps = _object_coords(nt, (1.0, 1.0, 1.0))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-360, 0)
    _set(noise, "Scale", rng.uniform(0.35, 1.4))
    _set(noise, "Detail", rng.uniform(4.0, 10.0))
    _set(noise, "Roughness", rng.uniform(0.45, 0.75))
    _link(nt, maps, "Vector", noise, "Vector")

    # Constant ramp → hard leaf/gap edges, which is what makes the shadow
    # read as foliage rather than as a soft blur.
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-140, 0)
    try:
        ramp.color_ramp.interpolation = "CONSTANT"
    except Exception:
        pass
    cut = rng.uniform(0.42, 0.60)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = cut
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    _link(nt, noise, "Factor", ramp, "Fac")
    _link(nt, ramp, "Color", bsdf, "Alpha")

    leaf = chaos_albedo(rng, (0.08, 0.16, 0.05), _clamp01(chaos) * 0.6, "natural")
    _set(bsdf, "Base Color", (*leaf, 1.0))
    _set(bsdf, "Roughness", 0.9)
    for attr, val in (
        ("surface_render_method", "DITHERED"),
        ("blend_method", "HASHED"),
        ("shadow_method", "HASHED"),
        ("use_transparent_shadow", True),
        ("use_backface_culling", False),
    ):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, val)
            except Exception:
                pass
    return mat
