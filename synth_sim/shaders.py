"""GLSL sources: MRT g-buffer, sun shadow pass, and atmospheric sky.

Material `u_style` codes:
    0 generic    1 plaster facade   2 sidewalk   3 asphalt
    4 foliage    5 vehicle paint    6 pedestrian 7 brick
    8 bark       9 metal           10 concrete  11 grass
   12 lane paint 13 curb concrete
"""

from __future__ import annotations

VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat3 u_normal_matrix;
uniform mat4 u_light_vp;

out vec3 v_world_pos;
out vec3 v_view_pos;
out vec3 v_world_n;
out vec2 v_uv;
out vec4 v_light_pos;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    vec4 view  = u_view  * world;
    v_world_pos = world.xyz;
    v_view_pos  = view.xyz;
    v_world_n   = normalize(u_normal_matrix * in_normal);
    v_uv        = in_uv;
    v_light_pos = u_light_vp * world;
    gl_Position = u_projection * view;
}
"""

SHADOW_VERTEX = """
#version 330 core
layout(location = 0) in vec3 in_position;
uniform mat4 u_light_mvp;
void main() {
    gl_Position = u_light_mvp * vec4(in_position, 1.0);
}
"""

SHADOW_FRAGMENT = """
#version 330 core
void main() {}
"""

SKY_VERTEX = """
#version 330 core
layout(location = 0) in vec3 in_position;
uniform mat4 u_view;
uniform mat4 u_projection;
out vec3 v_dir;
void main() {
    v_dir = in_position;
    mat4 view_rot = u_view;
    view_rot[3] = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 clip = u_projection * view_rot * vec4(in_position, 1.0);
    gl_Position = clip.xyww;
}
"""

SKY_FRAGMENT = """
#version 330 core
in vec3 v_dir;
uniform vec3  u_sun_dir;
uniform vec3  u_sun_color;
uniform float u_sun_elevation;
uniform float u_turbidity;
out vec4 out_color;

vec3 atmosphere(vec3 dir) {
    vec3 d = normalize(dir);
    vec3 L = normalize(u_sun_dir);
    float el = clamp(u_sun_elevation / 90.0, 0.0, 1.0);
    float turb = max(u_turbidity, 1.5);
    // Preetham-lite: Rayleigh (1+mu^2) + strongly peaked Mie.
    float mu = clamp(dot(d, L), -1.0, 1.0);
    float mu2 = mu * mu;
    float rayleigh = 0.75 * (1.0 + mu2);
    float g = 0.76;
    float mie = (1.0 - g * g) / max(pow(1.0 + g * g - 2.0 * g * mu, 1.5), 1e-4);
    vec3 ray_c = mix(vec3(0.18, 0.28, 0.62), vec3(0.22, 0.42, 0.88), el);
    vec3 mie_c = mix(vec3(0.55, 0.28, 0.12), u_sun_color, el);
    float h = clamp(d.y, -0.15, 1.0);
    float air = exp(-max(h, 0.0) * 0.65) * (0.55 + 0.45 * el);
    vec3 zenith  = mix(vec3(0.04, 0.05, 0.10), vec3(0.14, 0.32, 0.72), el);
    vec3 horizon = mix(vec3(0.42, 0.18, 0.08), vec3(0.70, 0.76, 0.86), el);
    vec3 ground  = mix(vec3(0.08, 0.07, 0.06), vec3(0.24, 0.26, 0.22), el);
    float hz = smoothstep(-0.06, 0.22, d.y);
    vec3 col = mix(horizon, zenith, pow(clamp(d.y, 0.0, 1.0), 0.55));
    col = mix(ground, col, step(0.0, d.y) * hz + (1.0 - step(0.0, d.y)) * 0.0);
    if (d.y < 0.0)
        col = mix(horizon, ground, pow(-d.y, 0.42));
    col += ray_c * rayleigh * 0.18 * air;
    col += mie_c * mie * 0.015 * el / turb;
    col += u_sun_color * 2.2 * pow(max(mu, 0.0), 220.0) * el;
    col += u_sun_color * 5.0 * smoothstep(0.9993, 0.99996, mu) * el;
    // Horizon haze
    col += horizon * exp(-abs(d.y) * 8.0) * 0.22;
    return col;
}

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 c = aces(atmosphere(v_dir) * 0.92);
    c = pow(c, vec3(1.0 / 2.2));
    c = (c - 0.5) * 1.08 + 0.48;
    out_color = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


def _fragment_body(id_decl: str, id_write: str) -> str:
    return f"""
#version 330 core

in vec3 v_world_pos;
in vec3 v_view_pos;
in vec3 v_world_n;
in vec2 v_uv;
in vec4 v_light_pos;

uniform vec3  u_albedo;
uniform float u_specularity;
uniform float u_shininess;
uniform vec3  u_sun_dir;
uniform vec3  u_sun_color;
uniform float u_ambient;
uniform vec3  u_sky_color;
uniform float u_fog_density;
uniform vec3  u_cam_pos;
uniform uint  u_instance_id;
uniform int   u_style;
uniform float u_wetness;
uniform float u_sun_elevation;
uniform sampler2D u_shadow_map;
uniform int   u_shadow_enabled;

layout(location = 0) out vec4  out_color;
layout(location = 1) out float out_depth;
{id_decl}

float hash13(vec3 p) {{
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}}

float value_noise(vec3 p) {{
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i);
    float n100 = hash13(i + vec3(1,0,0));
    float n010 = hash13(i + vec3(0,1,0));
    float n110 = hash13(i + vec3(1,1,0));
    float n001 = hash13(i + vec3(0,0,1));
    float n101 = hash13(i + vec3(1,0,1));
    float n011 = hash13(i + vec3(0,1,1));
    float n111 = hash13(i + vec3(1,1,1));
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z);
}}

float fbm(vec3 p) {{
    float a = 0.5;
    float s = 0.0;
    for (int i = 0; i < 5; ++i) {{
        s += a * value_noise(p);
        p = p * 2.03 + 17.1;
        a *= 0.5;
    }}
    return s;
}}

float shadow_term(vec3 N, vec3 L) {{
    if (u_shadow_enabled < 1) return 1.0;
    vec3 proj = v_light_pos.xyz / max(v_light_pos.w, 1e-5);
    proj = proj * 0.5 + 0.5;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0)
        return 1.0;
    float bias = max(0.0035 * (1.0 - dot(N, L)), 0.0012);
    float shadow = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    for (int x = -2; x <= 2; ++x) {{
        for (int y = -2; y <= 2; ++y) {{
            float closest = texture(u_shadow_map, proj.xy + vec2(x, y) * texel).r;
            shadow += (proj.z - bias > closest) ? 0.0 : 1.0;
        }}
    }}
    return shadow / 25.0;
}}

vec3 bump_normal(vec3 p, vec3 N, float scale, float amp) {{
    float e = 0.045;
    float h = fbm(p * scale);
    float hx = fbm((p + vec3(e, 0.0, 0.0)) * scale) - h;
    float hz = fbm((p + vec3(0.0, 0.0, e)) * scale) - h;
    vec3 q = normalize(cross(N, vec3(0.0, 1.0, 0.0)));
    if (length(q) < 0.12) q = normalize(cross(N, vec3(1.0, 0.0, 0.0)));
    vec3 b = normalize(cross(N, q));
    return normalize(N - amp * (hx * q + hz * b) / e);
}}

vec3 shade_albedo(inout float roughness, inout float metal) {{
    vec3 a = u_albedo;
    vec3 w = v_world_pos;
    vec2 fu = v_uv;
    if (fu.x < -50.0) {{
        float night = 1.0 - clamp(u_sun_elevation / 16.0, 0.0, 1.0);
        float glow = step(0.55, hash13(floor(w * 0.35)));
        a = vec3(0.035, 0.05, 0.065) + u_sun_color * 0.10;
        a += vec3(0.55, 0.40, 0.16) * night * glow * 0.65;
        metal = 0.42;
        roughness = 0.08;
        return a;
    }}
    if (u_style == 1) {{
        float stain = fbm(w * 0.32) * 0.20;
        a = u_albedo * (0.86 + 0.14 * fbm(w * 1.8) - stain);
        float fy = fract((w.y - 0.40) / 3.15);
        float fx = fract(fu.x / 2.35);
        float win = step(0.20, fy) * step(fy, 0.80) * step(0.16, fx) * step(fx, 0.84);
        float night = 1.0 - clamp(u_sun_elevation / 18.0, 0.0, 1.0);
        vec3 glass = vec3(0.07, 0.11, 0.15) + u_sun_color * 0.18;
        glass += vec3(0.62, 0.46, 0.18) * night * step(0.52, hash13(floor(vec3(fu.x, w.y, 2.0) / vec3(2.35, 3.15, 1.0))));
        a = mix(a, glass, win);
        metal = mix(0.0, 0.38, win);
        roughness = mix(0.70, 0.10, win);
    }} else if (u_style == 2) {{
        float tile = max(step(0.935, fract(w.x * 1.55)), step(0.935, fract(w.z * 1.55)));
        float n = fbm(w * 4.2);
        a = mix(u_albedo * (0.82 + 0.22 * n), u_albedo * 0.38, tile);
        float dirt = smoothstep(0.52, 0.94, fbm(w * 0.42));
        a *= (1.0 - 0.22 * dirt);
        a = mix(a, vec3(0.28, 0.24, 0.18), smoothstep(0.78, 0.96, fbm(w * 0.18)) * 0.25);
        roughness = 0.80 - 0.28 * u_wetness;
    }} else if (u_style == 3) {{
        float n = fbm(vec3(w.x, 0.0, w.z) * 1.6);
        float agg = fbm(vec3(w.x, 0.0, w.z) * 6.0);
        float crack = step(0.93, fbm(vec3(w.x, 0.0, w.z) * 0.55));
        a = u_albedo * (0.82 + 0.12 * n + 0.05 * agg) * (1.0 - 0.20 * u_wetness);
        a = mix(a, a * 0.42, crack * 0.65);
        float oil = smoothstep(0.76, 0.96, fbm(vec3(w.xz * 0.14, 3.1)));
        a = mix(a, vec3(0.045, 0.05, 0.045), oil * 0.40);
        roughness = mix(0.88, 0.16, u_wetness);
        metal = 0.05 + 0.22 * u_wetness;
    }} else if (u_style == 4) {{
        float barkish = 1.0 - smoothstep(0.22, 0.62, abs(normalize(v_world_n).y));
        float rings = 0.5 + 0.5 * sin(w.y * 16.0 + fbm(w * 2.2) * 4.0);
        vec3 bark = mix(vec3(0.22, 0.14, 0.08), vec3(0.38, 0.24, 0.12), rings);
        float n = fbm(w * 4.2);
        vec3 leaf = mix(u_albedo * 0.48, u_albedo * 1.22, n);
        leaf *= 0.70 + 0.30 * clamp(v_world_n.y, 0.0, 1.0);
        a = mix(leaf, bark, barkish * 0.85);
        roughness = mix(0.86, 0.78, barkish);
    }} else if (u_style == 5) {{
        float flake = fbm(w * 10.0);
        float panel = max(step(0.97, fract(fu.x * 0.55)), step(0.97, fract(fu.y * 0.8)));
        a = u_albedo * (0.88 + 0.12 * flake);
        a = mix(a, a * 0.55, panel * 0.5);
        metal = 0.62;
        roughness = 0.22;
    }} else if (u_style == 6) {{
        float cloth = fbm(w * 7.5);
        a = u_albedo * (0.88 + 0.14 * cloth);
        float skin = step(1.48, w.y) * step(w.y, 1.82);
        a = mix(a, vec3(0.62, 0.42, 0.32) * (0.9 + 0.1 * cloth), skin * 0.85);
        roughness = mix(0.72, 0.45, skin);
    }} else if (u_style == 7) {{
        float bx = fract(fu.x * 3.15 + floor(w.y * 2.45) * 0.5);
        float by = fract(w.y * 2.45);
        float mortar = max(step(0.88, bx), step(0.84, by));
        vec3 brick = u_albedo * (0.78 + 0.22 * hash13(floor(vec3(fu.x * 3.15, w.y * 2.45, 2.0))));
        a = mix(brick, vec3(0.40, 0.38, 0.35), mortar);
        float fy = fract((w.y - 0.40) / 3.15);
        float fx = fract(fu.x / 2.35);
        float win = step(0.20, fy) * step(fy, 0.80) * step(0.16, fx) * step(fx, 0.84);
        a = mix(a, vec3(0.06, 0.09, 0.13) + u_sun_color * 0.14, win);
        roughness = mix(0.82, 0.12, win);
        metal = mix(0.0, 0.32, win);
    }} else if (u_style == 8) {{
        float rings = 0.5 + 0.5 * sin(w.y * 18.0 + fbm(w * 2.0) * 4.0);
        a = mix(u_albedo * 0.62, u_albedo * 1.12, rings);
        roughness = 0.84;
    }} else if (u_style == 9) {{
        a = u_albedo * (0.82 + 0.18 * fbm(w * 12.0));
        metal = 0.78;
        roughness = 0.28;
    }} else if (u_style == 10) {{
        float panel = max(step(0.965, fract(fu.x * 0.38)), step(0.965, fract(w.y * 0.28)));
        a = mix(u_albedo * (0.88 + 0.12 * fbm(w * 1.3)), u_albedo * 0.50, panel);
        float fy = fract((w.y - 0.40) / 3.40);
        float fx = fract(fu.x / 2.60);
        float win = step(0.16, fy) * step(fy, 0.84) * step(0.12, fx) * step(fx, 0.88);
        a = mix(a, vec3(0.11, 0.15, 0.19), win);
        roughness = mix(0.68, 0.09, win);
    }} else if (u_style == 11) {{
        float blades = 0.5 + 0.5 * sin(w.x * 42.0 + fbm(w * 3.5) * 5.0);
        float clump = fbm(w * 0.55);
        vec3 lush = mix(u_albedo * 0.45, u_albedo * 1.25, blades);
        lush = mix(lush, u_albedo * vec3(0.55, 0.75, 0.35), clump);
        float dirt = smoothstep(0.64, 0.92, fbm(w * 0.28));
        a = mix(lush, vec3(0.30, 0.22, 0.14), dirt * 0.55);
        a *= 0.75 + 0.25 * clamp(v_world_n.y, 0.0, 1.0);
        roughness = 0.90;
    }} else if (u_style == 12) {{
        a = u_albedo * (0.92 + 0.08 * fbm(w * 6.0));
        roughness = 0.42;
        metal = 0.0;
    }} else if (u_style == 13) {{
        float chip = step(0.90, fbm(w * 5.5));
        a = u_albedo * (0.80 + 0.20 * fbm(w * 3.0));
        a = mix(a, a * 0.55, chip);
        roughness = 0.76;
    }} else {{
        a = u_albedo * (0.88 + 0.12 * fbm(w * 2.2));
        roughness = 0.66;
    }}
    return a;
}}

vec3 aces(vec3 x) {{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}}

void main() {{
    vec3 Ngeo = normalize(v_world_n);
    float bump_amp = 0.18;
    float bump_sc = 2.4;
    if (u_style == 2) {{ bump_amp = 0.28; bump_sc = 4.0; }}
    if (u_style == 3) {{ bump_amp = 0.22; bump_sc = 3.2; }}
    if (u_style == 4) {{ bump_amp = 0.45; bump_sc = 3.6; }}
    if (u_style == 7) {{ bump_amp = 0.32; bump_sc = 5.0; }}
    if (u_style == 11) {{ bump_amp = 0.55; bump_sc = 6.0; }}
    if (u_style == 5 || u_style == 9 || u_style == 12) {{ bump_amp = 0.06; }}
    vec3 N = bump_normal(v_world_pos, Ngeo, bump_sc, bump_amp);
    vec3 L = normalize(u_sun_dir);
    vec3 V = normalize(u_cam_pos - v_world_pos);
    vec3 H = normalize(L + V);
    float roughness = 0.6;
    float metal = 0.0;
    vec3 albedo = shade_albedo(roughness, metal);

    float ndl = max(dot(N, L), 0.0);
    float ndv = max(dot(N, V), 0.0);
    float ndh = max(dot(N, H), 0.0);
    float a = max(roughness * roughness, 0.02);
    float a2 = a * a;
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159 * denom * denom, 1e-5);
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
    vec3 F0 = mix(vec3(0.04), albedo, metal);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    vec3 spec = (D * G * F) / max(4.0 * ndv * ndl, 1e-4);
    vec3 diff = (1.0 - F) * (1.0 - metal) * albedo / 3.14159;

    float sh = shadow_term(Ngeo, L);
    float wrap = ndl * 0.90 + 0.10;
    float ao = mix(0.42, 1.0, clamp(Ngeo.y * 0.5 + 0.5, 0.0, 1.0));
    ao *= 0.72 + 0.28 * fbm(v_world_pos * 1.05);
    vec3 hemi = mix(u_sky_color * 0.06, u_sky_color * 0.38, clamp(N.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 fill = albedo * u_sky_color * 0.07 * (1.0 - 0.20 * sh);
    vec3 bounce = albedo * vec3(0.055, 0.058, 0.062);
    vec3 lit = albedo * (u_ambient * 0.28 * ao + hemi * 0.32 * ao) + fill + bounce
             + (diff + spec * (0.75 + 0.90 * u_specularity)) * u_sun_color * wrap * sh * 3.15;

    float wet_f = pow(1.0 - ndv, 5.0) * u_wetness;
    lit += u_sun_color * wet_f * 0.22 * sh;
    if (u_style == 3 || u_style == 2) {{
        vec3 R = reflect(-V, N);
        float puddle = smoothstep(0.62, 0.90, fbm(vec3(v_world_pos.xz * 0.22, 1.7))) * u_wetness;
        lit += u_sky_color * puddle * pow(max(R.y, 0.0), 3.0) * 0.28;
    }}

    float dist = length(v_view_pos);
    float height_fog = exp(-u_fog_density * 0.22 * max(v_world_pos.y, 0.0));
    float fog = exp(-u_fog_density * dist) * mix(1.0, 0.82, 1.0 - height_fog);
    vec3 fog_col = mix(u_sky_color * 0.85, u_sun_color, 0.10);
    vec3 color = mix(fog_col, lit, fog);
    color = aces(color * 0.82);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    color = (color - 0.5) * 1.12 + 0.5;

    out_color = vec4(color, 1.0);
    out_depth = max(-v_view_pos.z, 0.0);
    {id_write}
}}
"""


FRAGMENT_SHADER_UINT = _fragment_body(
    "layout(location = 2) out uint  out_instance_id;",
    "out_instance_id = u_instance_id;",
)

FRAGMENT_SHADER_FLOAT_ID = _fragment_body(
    "layout(location = 2) out float out_instance_id;",
    "out_instance_id = float(u_instance_id);",
)
