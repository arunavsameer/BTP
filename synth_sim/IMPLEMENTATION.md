# EgocentricAccessibilitySim-Scratch — Full Implementation Notes

**Package root:** `/storage/BTP/synth_sim/`  
**Generator name:** `EgocentricAccessibilitySim-Scratch`  
**Current version:** `2.3.0` (see `config.SimulationConfig.generator_version`)  
**Purpose of this file:** future-context dump of *why* the simulator exists, *what* it produces, *how* every module fits together, the maths, the CLI, the gotchas, and how to re-run it. Read this before changing spawn logic, shaders, or labels. **§9 is the output/annotation spec** (how every PNG, NPY, and JSON field is measured and what it means).

This is **not** Blender / Unity / Unreal / Godot. It is a **headless, from-scratch** rasterizer: procedural meshes + ModernGL + EGL, written so a forehead-mounted RGB camera for a blind / visually-impaired wearable can be trained on synthetic street hazards with haptic-relevant labels (SAFE / NEAR_MISS / CRITICAL_THREAT).

---

## 1. Task and need

### 1.1 The product problem

A wearable sits on the **forehead** and looks forward while the wearer walks a sidewalk. It must warn (haptics) about:

- people and vehicles that will enter the stride envelope soon (time-to-collision / closest-point-of-approach);
- static clutter on the walking lane (bins, bollards, parked scooters);
- ground holes (potholes);
- head-height strikers (branches, pipes, signs, awnings).

Collecting this with a real camera on a real visually-impaired user is slow, unsafe, and class-imbalanced (true CRITICAL events are rare). The simulator exists to **mint labeled RGB + depth + instance segmentation + per-object kinematics** under a known world frame.

### 1.2 Hard constraints (do not violate)

| Constraint | Why |
|---|---|
| No Blender / Unity / Unreal / Godot / GUI | Must run on a GPU box without a display; reproducible in scripts. |
| Headless EGL | Offscreen MRT; `moderngl.create_context(standalone=True, backend='egl')`. |
| World frame **+X right, +Y up, +Z forward** | Matches the wearable spec. `glm.lookAt` is **wrong** here (it mirrors X). |
| Local-space actor meshes + model matrix | Baking world positions into vertices then also applying a model matrix **double-transforms** (already bitten us on head hazards / potholes). |
| No face culling on the street | Floor quads were culled and the road vanished. Cull stays **off**. |
| Dataset labels are **instantaneous per object** plus an episode **realized** class | The CPA frame is often *not* NEAR_MISS because \(\mathbf{P}\cdot\mathbf{V}\approx 0\). Episode class = worst label seen in the clip. |
| Videos ≥ 15–20 s | Default **20 s** (600 frames @ 30 Hz). `--seconds` clamps to ≥ 15 unless `--frames` is set. |

### 1.3 What “good” looks like

- A person (the camera) **walks a sidewalk** with inverted-pendulum gait and head tremor.
- The street is **populated**: walkers both ways, pairs, standers, people **crossing the road**, cars both directions, parked cars, cyclists, scooters.
- Scripted **beats** realize SAFE / NEAR_MISS / CRITICAL at a chosen frame \(k^\*\).
- Output is **per-episode folders**, as frames, MP4, or both.
- Geometry is procedural but **not a pile of axis-aligned boxes**: lofted cars, recessed glass, cambered road, PBR-ish shading, shadows, sky.

---

## 2. Coordinate systems (read this first)

### 2.1 World

Right-handed:

- \(+X\): right (lateral)
- \(+Y\): up
- \(+Z\): forward along the street on a straight block

Units are **metres**. Time is seconds. Angles in code are radians unless a name ends in `_deg`.

### 2.2 Frenet / street frame

The street centerline is a Catmull–Rom spline parameterized by arclength \(s\).

At each sample:

- \(\mathbf{T}\): unit tangent (direction of travel, mostly \(+Z\))
- \(\mathbf{N} = \mathbf{T} \times \mathbf{e}_y\): unit **left** normal (world-left of travel)
- \(\mathbf{U} = \mathbf{N} \times \mathbf{T}\): up (parallel-transported)

A point at arclength \(s\) and signed lateral \(L\) is

\[
\mathbf{p}(s,L) = \mathbf{c}(s) + L\,\mathbf{N}(s)
\]

- \(L > 0\): left of centerline  
- \(L < 0\): right of centerline  

**Camera walks the right sidewalk:** `camera_side = -1`, so

```text
sidewalk_lateral = side * (0.5 * road_width + curb_width + 0.5 * sidewalk_width)
                 = -(0.5 w_road + w_curb + 0.5 w_walk)
```

(`geometry.sidewalk_center_lateral`)

### 2.3 Actor local space vs OBB

- `Actor.position` is the **OBB center** (used for TTC, 3D boxes, projection).
- Dynamic meshes (pedestrian, vehicle, cyclist, scooter, dustbin, bollard, hydrant, planter) are **Y-centered** via `geometry._center_y` so visual feet/tires sit at `position.y - extents.y/2`.
- Trees, lamps, benches, street extrusion stay **ground-relative** (origin at y = 0) and are placed with a model matrix, not as centered actors (trees are actors at `y = 0` with large extents — the OBB is imperfect; the mesh is correct).
- Buildings are centered cuboids placed at `y = 0.5 * height`.

**Do not** bake a world translation into the mesh *and* translate again with `model_matrix()`.

### 2.4 Camera / OpenGL view

The wearable looks along world \(+Z\). OpenGL cameras look along **\-Z**.

`glm.lookAt(eye, eye+Z, up)` flips **X** (image left/right swapped). We use a custom matrix (`biomechanics.view_matrix_z_forward`):

- \(\mathbf{f} = \mathrm{normalize}(\mathrm{forward})\)
- \(\mathbf{r} = \mathrm{normalize}(\mathrm{up} \times \mathbf{f})\)
- \(\mathbf{u} = \mathbf{f} \times \mathbf{r}\)
- Rotation rows encode \((\mathbf{r}, \mathbf{u}, -\mathbf{f})\) so clip-space X stays world-right.

`CameraPose.target = eye + forward` (unit look point). Shadow focus is `eye + 16 * forward`.

### 2.5 Image / depth

- RGB is **uint8**, top-left origin after `np.flipud` (GL reads bottom-up).
- Linear depth MRT is **view-space metres** `max(-v_view_pos.z, 0)` (distance along camera \-Z, not Euclidean ray length).
- Instance ID is `uint16` (preferred `u2` texture; fallback `u4` then `f4`).
- Pixel Y is down. Projection NDC Y-up is converted in `projection.project_points_view_to_screen`.

Full measurement / loading notes: **§9**.

---

## 3. Architecture and data flow

```text
CLI (main.py)
  └─ SimulationConfig (config.py)
       │
       ├─ CollisionDirector.build_episode
       │     street spline + meshes
       │     buildings, lamps, benches, hydrants, planters, markings
       │     furniture actors (bins, bollards, parked scooters, trees)
       │     pick target class + scenario name + k*
       │
       ├─ BiomechanicalCameraRig.precompute(frames)
       │     inverted-pendulum path on right sidewalk
       │
       ├─ CollisionDirector.finalize_with_camera
       │     street life (crowd, crossers, traffic)
       │     scripted threat aimed at camera(k*)
       │
       └─ for frame in 0..F-1
             advance_actor(...)          # Frenet / crossing / Euler
             shadow pass (sun ortho)
             sky + lit MRT
             read RGB / depth / ID
             TTC/CPA + OBB→AABB + occlusion
             EpisodeSink.submit → PNG/NPY/JSON and/or MP4
```

### 3.1 Two-phase spawn (important)

`build_episode` does **not** know the camera trajectory yet. It only builds the static world and furniture.

`finalize_with_camera(world, poses, rng)` runs **after** gait precompute so people/cars can be placed relative to `s_cam(0)` and `s_cam(T)` and threats can invert onto \(\mathbf{p}_\mathrm{cam}(t^\*)\).

If you add a new threat, put it in `finalize_with_camera`, not `build_episode`.

### 3.2 Determinism

| RNG | Seed |
|---|---|
| World / director | `seed + 7919 * episode` |
| Finalize / street life | `seed + 104729 * episode` (from `main.render_episode`) |
| Camera gait | `seed + 13 * episode` |
| Instance IDs | `reset_instance_ids(100 + episode * 1000)` |

Same `--seed` + `--start-episode` should reproduce the same geometry and kinematics. GPU rasterization is deterministic enough for labels; video codecs are not bit-exact.

---

## 4. File map

All application code lives in `/storage/BTP/synth_sim/` (flat package, `sys.path` insert in `main.py`).

| File | Role |
|---|---|
| `main.py` | CLI, config assembly, per-frame render/annotate loop |
| `config.py` | Dataclasses: camera, gait, street, Poisson radii, threat thresholds, population, domain rand, class IDs |
| `geometry.py` | Splines, street extrusion, all procedural meshes, Poisson scatter |
| `biomechanics.py` | Perlin fBM, custom view matrix, inverted-pendulum forehead rig |
| `actors.py` | `Actor`, environment (sun/Kelvin), palettes, instance IDs, `make_actor` |
| `collision_engine.py` | TTC/CPA, threat class, Frenet motion, director, scenarios, street life |
| `projection.py` | GL perspective, OBB→2D AABB, near clip, occlusion/truncation |
| `shaders.py` | GLSL: MRT lit, shadow, sky |
| `core_gl.py` | EGL context, FBO MRT, shadow map, VAO cache, draw/readback |
| `dataset_writer.py` | Per-episode folders, threaded frames, ffmpeg/OpenCV MP4, JSON schema |
| `requirements.txt` | Python deps |
| `setup_arch.sh` | Arch bootstrap (mesa/EGL + venv). **Commands in that script still mention old 5 s defaults; use §11 of this file.** |
| `.venv/` | Project venv (`--system-site-packages`) |
| `dataset_output/` | Default write root (may contain **old flat** `rgb/` from v2.0 — do not mix layouts) |
| `IMPLEMENTATION.md` | This document |

There is **no** `tests/` tree. Validation has been smoke renders + class-count prints.

---

## 5. What each file does (detail)

### 5.1 `config.py`

Frozen (or top-level) dataclasses. Change numbers here rather than sprinkling magic constants.

- `CameraIntrinsicsConfig` — pinhole \(f_x = W / (2\tan(\mathrm{fov}_x/2))\), square pixels \(f_y=f_x\), principal point center. `fov_y_rad` derived from aspect.
- `GaitConfig` — walk speed band, step frequency, heave/sway amplitudes, cervical noise, clip limits for explore angles.
- `StreetConfig` — road/sidewalk widths, **160 m** length, spline resolution 0.45 m, control-point wander.
- `PoissonRadii` — minimum spacing for furniture scatter.
- `ThreatThresholds` — TTC/CPA gates (see §7).
- `PopulationConfig` — how many walkers / crossers / cars / cyclists.
- `KinematicEnvelopes` — clamp speeds when inverting trajectories.
- `DomainRandomization` — sun, wetness, fog.
- `SimulationConfig` — fps 30, default 600 frames, ratios SAFE 0.35 / NEAR_MISS 0.30 / CRITICAL 0.35, shadow 2048, writer workers, output mode.
- `CLASS_NAME_IDS` — semantic class integers (not the same as **instance** IDs).
- `DYNAMIC_CLASSES` / `HAZARD_CLASSES` — sets used conceptually; labeling uses `classify_threat` + class_name strings.

### 5.2 `geometry.py`

**Vertex layout (interleaved float32, 8 floats):** `px py pz  nx ny nz  u v`.

`MeshBuilder.add_face` computes a geometric normal if none is given. Cuboid UVs are **metric** (metres along the face) so facade shaders stay aligned after yaw.

Important generators:

- `catmull_rom_point` / `build_street_spline` / `StreetSpline.frame_at`
- `generate_street_mesh` → `road`, `curb`, `sidewalk`, `verge`, plus the spline object
  - road **camber**: crown +0.042 m at center, gutters at 0
  - curb **bevel** (~0.06 m) instead of a 90° wall
  - grass verge ~1.85 m outside the sidewalk
- `generate_ground_plane` — subdivided, slight sinusoid so the horizon is not a slab
- `generate_lofted_box` — independent top/bottom rectangles (car hoods, cabins, roofs)
- `generate_beveled_box` — three overlapping boxes ≈ chamfer
- `_center_y`, `_as_glass` (sentinel UV \(u=-100\) → shader glass), `_displace_along_normal`
- Trees, lamps, benches, hydrants, planters, vehicles (sedan/van/truck), pedestrians (stride pose), cyclist, scooter, dustbin, bollard
- `generate_building_mesh` — setbacks, storefront panes, recessed glass, balconies, HVAC, downspouts, optional hip roof
- `generate_road_markings` — dashed center + solid edges at y ≈ 0.012
- `poisson_disk_2d` (Bridson) / `scatter_on_sidewalk`

Street mesh vertices are already in **world** space (identity model). Actor/building meshes are local.

### 5.3 `biomechanics.py`

- `Perlin1D` + `fbm` — Ken Perlin fade \(6t^5-15t^4+10t^3\), 1-D gradients ±1.
- `SidewalkPath.sample(s)` — interpolates Frenet samples and **adds** `lateral_offset` (the walking corridor).
- `BiomechanicalCameraRig` — see §6.1. Starts at `s0 = 8` m so the wearer is not spawned on the spline endpoint.

### 5.4 `actors.py`

- `Actor.step` — Euler: \(v \leftarrow v+a\Delta t\), clamp speed, \(p \leftarrow p+v\Delta t\), optional yaw from \(v_{xz}\).
- `Actor.apply_social_force` — Helbing relaxation + exponential repulsion (legacy; spline followers skip this).
- `EnvironmentState.sun_direction` — \(\mathbf{L}=(\cos el\sin az,\;\sin el,\;\cos el\cos az)\).
- `kelvin_to_rgb` — Tanner Helland.
- `make_actor` — assigns albedo, style (vehicle 5, ped 6, tree/head 4, metal 9, …).
- Global `_NEXT_ID` via `next_instance_id` / `reset_instance_ids`.

### 5.5 `collision_engine.py`

The largest file. Sections:

1. TTC/CPA + `classify_threat` + `invert_constant_velocity`
2. `StaticProp` / `EpisodeWorld`
3. Frenet helpers: `bind_spline_motion`, `bind_crossing`, `sync_path_pose`, `advance_actor`
4. `CollisionDirector` — world build, furniture, infra, street life, threats
5. `kinematics_vs_camera` — horizontal TTC for locomotion; **full 3D** for `head_obstacle`, `sign`, `awning`
6. `episode_realized_class` — CRITICAL > NEAR_MISS > SAFE

Scenario name lists: `SAFE_SCENARIOS`, `NEAR_SCENARIOS`, `CRITICAL_SCENARIOS` (see §8).

### 5.6 `projection.py`

- `perspective_gl` — OpenGL frustum matching the pinhole. GLM is column-major; \(P_{2,3}=-1\) lives at `p[2][3]`, the \(-2fn/(f-n)\) term at `p[3][2]`. Getting this index wrong flattened the world.
- `obb_corners_world` — 8 corners from `extents` (full size, so half-extents = 0.5 * extents).
- Near-plane clip of cube **edges** (Sutherland–Hodgman on z = −near).
- Truncation = 1 − (clamped area / raw area).
- Occlusion: \(7\times7\) grid (configurable) over the 2D box. Prefer instance-ID match; else ray–AABB vs linear depth with 8 cm bias.

### 5.7 `shaders.py`

Three programs:

1. **Main** — world/view/clip, `v_light_pos = u_light_vp * world`. Fragment: value-noise fBM, Cook–Torrance GGX, hemisphere + bounce, 5×5 PCF shadows, wet Fresnel / puddles, height+distance fog, ACES + gamma + slight contrast. MRT: RGB, linear depth, instance ID (uint or float).
2. **Shadow** — depth-only, `u_light_mvp`.
3. **Sky** — dome, `gl_Position = clip.xyww` (always far). Preetham-lite Rayleigh + Mie + sun disc.

`u_style` codes: 0 generic, 1 plaster+windows, 2 sidewalk tiles, 3 asphalt, 4 foliage/bark hybrid, 5 vehicle, 6 clothing/skin, 7 brick+windows, 8 bark/wood, 9 metal, 10 concrete panels, 11 grass, 12 lane paint, 13 curb.

Glass submeshes: `v_uv.x < -50` (from `_as_glass`).

### 5.8 `core_gl.py`

- Tries EGL standalone contexts in order; last resort “current” GL.
- Color RGBA8, depth `R32F`, ID `R16UI` (fallback).
- Dual VAO cache key `(id(mesh), id(program))`. Shadow/sky bind `3f 20x` (position only).
- Shadow: depth texture, ortho around focus, `glm.lookAt` is OK here (consistent for both pass and sampling).
- `begin_frame` clears color/depth/ID. Sky drawn with depth test off, then scene.
- Readback flips Y.

### 5.9 `dataset_writer.py`

- `DatasetWriter.begin_episode(i)` → `EpisodeSink` under `root/episode_XXXXX/`.
- `frames` / `both`: `rgb/`, `depth/`, `segmentation/`, `annotations/`.
- `video` / `both`: `video.mp4`. Annotations always written (even video-only).
- Frames: thread pool, PNG RGB, `npy` float32 depth, `I;16` instance PNG.
- Video: **ffmpeg first** (`libx264`, CRF 18, yuv420p, raw RGB pipe), then OpenCV fourcc `mp4v`/`avc1`/`XVID`/`H264`. Video writes are **synchronous** (order). Frames are async.
- `build_annotation` only fills metadata / camera / environment / objects; `main.annotate_frame_full` adds `episode` and computes kinematics, boxes, visibility. **See §9 for the meaning of every field.**

### 5.10 `main.py`

Orchestrator only. `SKIP_ANNOTATE` drops ground/road/sidewalk/curb/building from JSON objects (they still render and occupy instance IDs 1–5).

Per frame: shadow casters (skip `class_name=="ground"`, which includes the grass verge), sky, props, actors, read, annotate, submit.

Episode print: scenario, **target** class, **worst realized** class, raster fps.

---

## 6. Maths (implementation-faithful)

### 6.1 Gait / forehead camera

Progress along the sidewalk:

\[
s(t)=s_0+v_0\left(t+\frac{\alpha_v}{\omega}\sin(\omega t)\right),\quad \omega=4\pi f_\mathrm{step}
\]

Closed form of \(v(t)=v_0(1+\alpha_v\cos(4\pi f_\mathrm{step}t))\). Defaults: \(v_0\in[1.1,1.5]\) m/s, \(f_\mathrm{step}=1.8\) Hz, \(\alpha_v=0.08\), \(s_0=8\) m.

In 20 s the wearer travels ~22–30 m — hence a **160 m** street so oncoming cars still have room.

Lateral sway (stride frequency \(f_\mathrm{stride}=f_\mathrm{step}/2\)):

\[
\mathrm{sway}=a_\mathrm{sway}\sin(2\pi f_\mathrm{stride}t+\phi)+\eta_x
\]

Heave (always positive, heel-strike):

\[
\mathrm{heave}=a_\mathrm{heave}\lvert\sin(2\pi f_\mathrm{step}t)\rvert+\eta_y
\]

Eye height: \(h_\mathrm{nominal}+\mathrm{heave}\) with \(h_\mathrm{nominal}\sim\mathcal{N}(1.60,0.10^2)\).

Cervical Euler (then clipped):

\[
\begin{aligned}
\theta_\mathrm{pitch}&=a_p\sin(4\pi f_\mathrm{step}t)+\beta_p\,n_p(t)\\
\theta_\mathrm{yaw}&=a_y\sin(2\pi f_\mathrm{stride}t)+\beta_y\,n_y(t)\\
\theta_\mathrm{roll}&=a_r\cos(2\pi f_\mathrm{stride}t)+\beta_r\,n_r(t)
\end{aligned}
\]

\(n_\cdot\) are Perlin fBM. Intrinsic order **yaw → pitch → roll**. Optical axis:

\[
\mathbf{f}=(\sin\psi\cos\theta,\;-\sin\theta,\;\cos\psi\cos\theta),\quad \psi=\mathrm{heading}+\theta_\mathrm{yaw}
\]

Heading is \(\mathrm{atan2}(T_x,T_z)\).

### 6.2 Catmull–Rom street

Hermite on \(P_1,P_2\) with tangents \(T_1=\tau(P_2-P_0)\), \(T_2=\tau(P_3-P_1)\), \(\tau=0.5\). Control points wander in \(X\) up to `lateral_wander` (3.5 m) while \(Z\) is the long axis.

### 6.3 TTC and CPA

Relative state \(\mathbf{P}=\mathbf{p}_i-\mathbf{p}_c\), \(\mathbf{V}=\mathbf{v}_i-\mathbf{v}_c\).

Converging iff \(\mathbf{P}\cdot\mathbf{V}<0\).

\[
t_\mathrm{TTC}=t_\mathrm{CPA}=-\frac{\mathbf{P}\cdot\mathbf{V}}{\lVert\mathbf{V}\rVert^2},\qquad
d_\mathrm{CPA}=\lVert\mathbf{P}+\mathbf{V}\,t_\mathrm{CPA}\rVert
\]

If not converging, TTC \(=+\infty\) and \(t_\mathrm{CPA}=0\) (distance now).

Locomotion uses **horizontal** \((P_x,0,P_z)\) so a tall person does not get a bogus 3D miss. Head obstacles use full 3D so a bar at y = 1.62 m still collapses TTC.

### 6.4 Threat labels (per object, per frame)

Priority:

1. **CRITICAL_THREAT** if converging and \(t<2.0\) s and \(d<0.45\) m
2. **NEAR_MISS** if converging and \(t\in[1.8,3.5]\) and \(d\in[0.4,1.2]\)
3. Static (\(\lVert v\rVert<0.15\)): CRITICAL if converging, \(d<0.45\), \(t<3.5\); NEAR_MISS if \(d\le 1.2\), \(t\le 4\); else **SAFE_STATIC**
4. Else SAFE_DYNAMIC if not converging or (\(t>4\) and \(d>1.8\))
5. Extra NEAR_MISS if converging, \(t<3.5\), \(d<1.5\)
6. Else SAFE_DYNAMIC

Episode **realized** class (`episode_realized_class` / `worst` in the log): any CRITICAL in any frame → CRITICAL; else any NEAR_MISS → NEAR_MISS; else SAFE.

JSON also stores `target_class` (what the director aimed for) and `realized_class` **on that frame**. The printed `worst=` is max over the episode.

### 6.5 Trajectory inversion

To hit \(\mathbf{p}^\*\) at duration \(\Delta = k^\*\Delta t\):

\[
\mathbf{v}=\mathrm{clamp\_speed}\!\left(\frac{\mathbf{p}^\*-\mathbf{p}_\mathrm{spawn}}{\Delta}\right)
\]

then often \(\mathbf{p}_\mathrm{spawn}\leftarrow\mathbf{p}^\*-\mathbf{v}\Delta\) so arrival is exact after the speed clamp.

Oncoming corridor threats instead spawn **ahead along \(+\mathbf{T}\)** and walk \(-\mathbf{T}\) with a **lateral offset** so they do not occupy the camera voxel on earlier frames.

### 6.6 Frenet agents (20 s clips)

Spline follow:

\[
s\leftarrow s+\dot s\,\Delta t,\quad
\mathbf{p}=\mathbf{c}(s)+L\mathbf{N}(s),\quad
\mathbf{v}=\dot s\,\mathbf{T}(s)
\]

Pedestrians / cyclists / scooters **reverse** \(\dot s\) at the ends of the street. Cars **stop**.

Crossing at fixed \(s\):

\[
u=\mathrm{clip}(t/T,0,1),\quad L=(1-u)L_0+u L_1,\quad
\mathbf{v}=\frac{L_1-L_0}{T}\mathbf{N}
\]

Y is 0.86 m on the roadway and \(0.86+h_\mathrm{curb}\) on sidewalks. When \(u=1\), the agent binds to the destination sidewalk and walks with stored `_path_ds`.

### 6.7 Social force (only free-Euler peds)

Not used for `_follow_spline` / `_crossing` agents (almost everyone now).

\[
\mathbf{F}=\frac{\mathbf{v}_\mathrm{des}-\mathbf{v}}{\tau}+\sum_j a\exp\!\left(\frac{r_i+r_j-d_{ij}}{b}\right)\hat{\mathbf{n}}_{ij}
\]

\(\tau=0.45\), \(a=1.8\), \(b=0.45\).

### 6.8 Lighting (fragment)

GGX normal distribution \(D\), Smith-Schlick \(G\), Schlick Fresnel \(F\) with \(F_0=\mathrm{mix}(0.04,\mathrm{albedo},\mathrm{metal})\).

\[
k_\mathrm{spec}=\frac{DGF}{4\,n\cdot v\,n\cdot l},\quad
k_\mathrm{diff}=(1-F)(1-m)\frac{\mathrm{albedo}}{\pi}
\]

Shadows: project `v_light_pos` to [0,1], 5×5 PCF vs depth map, bias \(\max(0.0035(1-n\cdot L),0.0012)\).

Bump: fBM finite differences, projected onto a tangent frame.

Fog: \(\exp(-\rho\,\lVert\mathbf{x}_\mathrm{view}\rVert)\) with a mild height term. \(\rho\in[0.0012,0.0045]\) (was much higher; it milked the image).

Tonemap: ACES fitted curve, gamma 2.2, then `(c-0.5)*1.12+0.5`.

Sky: zenith/horizon/ground lerp + Rayleigh \(0.75(1+\mu^2)\) + Henyey–Greenstein-ish Mie + sun disc.

Sun color: Kelvin → RGB (Tanner Helland).

### 6.9 Pinhole / projection

\[
f_x=\frac{W}{2\tan(\mathrm{fov}_x/2)},\quad
u=f_x\frac{X}{Z}+c_x
\]

in camera space the optical axis is **\-Z**, so the GL projection uses a standard frustum with near 0.1 m, far 100 m, horizontal FOV 90°.

### 6.10 Poisson-disk furniture

Bridson in the \((s,\text{across-walk})\) rectangle, cell size \(r/\sqrt{2}\), \(k=28\) candidates, then lift through the Frenet frame.

---

## 7. Episode construction (director)

### 7.1 Target class

Multinomial with `class_ratios`. Then a scenario name is drawn from the matching list.

\(k^\*\sim\mathrm{Unif}\{0.40F,\ldots,0.80F-1\}\) so the beat is at **8–16 s** of a 20 s clip.

### 7.2 Static world

Instance IDs reserved for terrain:

| ID | Object | style |
|---|---|---|
| 1 | ground plane | 11 grass |
| 2 | road | 3 asphalt |
| 3 | curb | 13 |
| 4 | sidewalk | 2 |
| 5 | verge (if present) | 11 |

Then `next_instance_id()` from 100+ for buildings, lamps, benches, markings, hydrants, planters, actors.

Buildings: both sides of the street, random width/depth/height, styles `{1,7,10}` (plaster / brick / concrete), yaw = street heading.

Infra: road paint (style 12), lamps every ~11–16 m (arm toward roadway), benches on outer walk, hydrants in the **gutter** (`±(0.5 w_road + 0.18)` — not in the walking lane), planters on the outer walk.

Furniture (actors, not StaticProp): dustbins, bollards, parked scooters, trees. On SAFE/NEAR_MISS the **camera corridor** is avoided (`|L - sidewalk_lateral| < 1.25` skipped).

### 7.3 Street life (every episode)

Called from `finalize_with_camera` **before** the scripted beat.

| Population | Range | Notes |
|---|---|---|
| Walkers | 9–16 | Both sidewalks, both directions |
| Standers | 2–5 | Building line |
| Pairs | 1–3 pairs | Same \(\dot s\), ±0.32 m lateral |
| Crossers | 3–6 | 5.2–8 s to cross; staggered phase |
| Moving cars | 6–10 | Lanes at \(L=\pm 0.28\,w_\mathrm{road}\), 7.5–13.5 m/s, min 14 m gap |
| Parked cars | 4–7 (+2 if `parked_street`) | Curb \(L=\pm(0.5w-1.15)\) |
| Cyclists | 1–3 | Road, 4–7 m/s |
| Moving scooters | 2 | Opposite sidewalk |

**SAFE corridor rules** (so ambient life does not steal the class):

- ~78% of walkers on the **opposite** sidewalk.
- Camera-side walkers: **same direction only**, building line, \(s \ge s_\mathrm{cam}(0)+12\).
- No oncoming on the wearer’s walk center.
- Pairs always opposite-side when SAFE.
- Standers skipped on camera side inside \([s_0-2,s_T+3]\).
- Ambient crossers: \(s > s_T+5\) (ahead of the whole walk) or \(s < s_0-3\) (behind). If they finish on the camera sidewalk, they attach to the **building line**, not the corridor.

Same-direction cars spawn behind the wearer (overtake) or ahead (recede). Oncoming cars spawn down-street toward the camera.

### 7.4 Scripted scenarios

| Name | Target | What happens |
|---|---|---|
| `sidewalk_stroll` | SAFE | Life only; wearer walks |
| `opposite_flow` | SAFE | Life only (more opposite-side traffic by chance) |
| `far_crosswalk` | SAFE | Extra 3–5 people crossing ~16 m ahead |
| `parked_street` | SAFE | +2 parked cars |
| `near_miss_ped` | NEAR | Oncoming ped, lateral offset in `[0.6,1.2]` m |
| `near_miss_cyclist` | NEAR | Same with cyclist envelope |
| `jaywalker` | NEAR or CRIT | Ped walks laterally toward \(\mathbf{p}_\mathrm{cam}(t^\*)\); parked van as cover |
| `close_overtake` | NEAR | Cyclist passes on the road-side of the sidewalk, Frenet-locked |
| `approaching_group` | NEAR | 2–3 oncoming peds with alternating near-miss offsets |
| `curb_stepout` | NEAR | Ped on building line steps laterally into the stride envelope |
| `critical_inversion` | CRIT | Oncoming ped, offset `[0,0.35]` m |
| `sidewalk_incursion` | CRIT | Car steers from the near lane onto \(\mathbf{p}_\mathrm{cam}(t^\*)\) |
| `sudden_brake` | CRIT | Lead ped matches gait, then \(\ddot s = -3\,\mathrm{m/s}^2\) from `k* - 0.7s` |
| `head_overhang` | CRIT | Horizontal pipe mesh at y = 1.62 m on the camera position at \(t^\*\) |
| `pothole_stepin` | CRIT | Crater mesh at camera XZ, surface = curb height |
| `crosswalk_conflict` | CRIT | Cross group at \(s(t^\*)\) **plus** critical jaywalker |

Unknown names fall back to inverted ped.

`k*` is stored on `EpisodeWorld` and in every JSON `episode.k_star`.

---

## 8. Rendering pipeline (per frame)

1. `compute_light_vp(sun_dir, focus)` — ortho ~32 m, eye = focus + 55 L.
2. `begin_shadow` — clear depth 1.
3. Draw all non-ground props + all actor meshes into the shadow map.
4. `begin_frame(sky)` — clear MRT; zero ID/depth textures.
5. `draw_sky` — hemisphere, depth test off.
6. `set_frame_uniforms` — view, sun, fog, wetness, `u_light_vp`, bind shadow map to unit 0.
7. Draw static props then actors (`style`, `wetness`).
8. `read_targets` → RGB uint8 HxWx3, depth float32 HxW, instance uint16 HxW.
9. Annotate; write.

Face culling **disabled**. Depth func `<`.

Material styles on props/actors are listed in §5.7. Vehicles/scooters get extra wetness; road `wetness_scale=1`.

---

## 9. Dataset output — what you get, how it is measured, what it tells you

This section is the source of truth for anyone training on the dump. Every tensor and JSON field is defined here: **units, origin, how the number is computed, and what you should (and should not) infer from it.**

Code path per frame (`main.render_episode` → `annotate_frame_full` → `EpisodeSink.submit`):

1. GPU MRT readback → RGB, linear depth, instance ID (`core_gl.read_targets`).
2. CPU kinematics + boxes + occlusion for each **Actor** (`collision_engine.kinematics_vs_camera`, `projection.*`).
3. JSON assembled by `dataset_writer.build_annotation` plus an `episode` block.
4. Disk write (threaded PNG/NPY/JSON; synchronous MP4).

There is **no** KITTI / COCO / nuScenes converter in-tree. Downstream code should read these files as specified below.

---

### 9.1 Layout (v2.1+)

```text
<output>/
  episode_00000/
    rgb/frame_0000.png              # uint8 RGB, only if frames|both
    depth/frame_0000.npy            # float32 HxW metres, only if frames|both
    segmentation/frame_0000.png     # 16-bit instance IDs, only if frames|both
    annotations/frame_0000.json     # ALWAYS written (even video-only)
    video.mp4                       # only if video|both
  episode_00001/
    ...
```

| Mode | RGB / depth / seg | JSON | MP4 |
|---|---|---|---|
| `frames` | yes | yes | no |
| `video` | **no** | yes | yes |
| `both` (default) | yes | yes | yes |

Frame index is zero-padded to 4 digits (`frame_0000` … `frame_0599` for a 20 s / 30 Hz clip). Episode index is 5 digits. `frame_id` inside JSON is `ep{episode:05d}_f{frame:04d}`.

`--output-mode video` still writes `annotations/` because that is the only place threat labels live. It does **not** write depth or segmentation; if you need those, use `frames` or `both`.

**Do not** dump into an old v2.0 directory that has a flat `rgb/` at the root unless you are fine mixing layouts.

**Same-index correspondence:** `rgb/frame_K.png`, `depth/frame_K.npy`, `segmentation/frame_K.png`, `annotations/frame_K.json`, and MP4 frame *K* are the same instant \(t = K / 30\).

---

### 9.2 Image conventions (shared by RGB, depth, seg)

After `np.flipud` on the GL readback (`core_gl.read_targets`):

- Shape is `(H, W)` or `(H, W, 3)` with **row 0 = top of the image**, column 0 = left.
- Pixel \((u, v)\) = (column, row). \(u\) increases right, \(v\) increases **down**.
- This matches the 2D boxes in JSON (`xmin/ymin/xmax/ymax` in the same pixel frame).
- OpenGL NDC is y-up; `projection.project_points_view_to_screen` converts with \(v = (1 - \mathrm{ndc}_y)\,H/2\).

Default resolution is whatever you passed (`--width` `--height`, default 1280×720). Intrinsics in JSON always match that resolution.

Sky / cleared pixels: RGB is the sky-clear colour (then overwritten by the sky dome where it draws); depth is **0**; instance ID is **0**. Treat ID 0 as background.

---

### 9.3 RGB (`rgb/frame_XXXX.png`)

| Property | Value |
|---|---|
| Format | PNG, 8-bit RGB (Pillow `mode="RGB"`, no optimize) |
| Colour | Tone-mapped sRGB-ish: Cook–Torrance lighting → ACES fit → \(\gamma=2.2\) → contrast `(c-0.5)*1.12+0.5` |
| Not | linear radiance, not a raw sensor Bayer, no EXIF, no camera response |

**How it is measured.** The main fragment shader writes `out_color` after fog, ACES, and gamma (`shaders.py`). The colour attachment is RGBA8. Readback drops alpha. Face culling is off, so both sides of thin geometry can shade.

**What it tells you.** This is the forehead-camera appearance the wearable would (idealized) see: wet asphalt, glass, shadows, gait shake. Use it for detection / segmentation / monocular depth *appearance*. Do **not** treat pixel intensity as illuminance. Video (`video.mp4`) is the same RGB **lossy-encoded**; prefer the PNGs for training if you have them (CRF 18 H.264 still smears thin poles and lane paint).

---

### 9.4 Depth (`depth/frame_XXXX.npy`)

| Property | Value |
|---|---|
| Format | NumPy `.npy`, `float32`, shape `(H, W)`, C-contiguous |
| Unit | **metres** |
| Quantity | **Linear view-space \(Z\)**: \(\max(-z_\mathrm{view}, 0)\), i.e. distance along the camera **optical axis**, not Euclidean ray length |
| Invalid / sky | `0.0` (FBO depth colour target is zeroed each frame; sky does not write this attachment) |
| Range | Practically 0 … far plane (100 m). Nothing is written beyond the far clip. |

**How it is measured.** Vertex shader stores `v_view_pos = (u_view * world).xyz`. Fragment writes:

```text
out_depth = max(-v_view_pos.z, 0.0)
```

The camera looks along world forward, which is encoded as OpenGL **−Z** in view space (`view_matrix_z_forward`). So \(-z_\mathrm{view}\) is “metres in front of the optical centre, along the look axis.” A point 5 m dead-ahead has depth ≈ 5. The same point 5 m away at 45° off-axis has depth ≈ \(5/\sqrt{2}\) (the \(Z\) component), **not** 5.

This is **not**:

- OpenGL window depth \(z\in[0,1]\) (that lives on a separate depth renderbuffer and is discarded);
- inverse-depth or disparity;
- LiDAR range \(\lVert\mathbf{x}_\mathrm{cam}\rVert\).

**What it tells you.** Per-pixel metric structure of the scene as seen by the pinhole. Use it to:

- supervise monocular depth (remember: **Z-depth**, convert if your loss wants range);
- lift a pixel to a camera-frame point (below);
- run the occlusion test (JSON `visibility` already did this for you).

**Lift a pixel to a 3-D point in camera / world.** With JSON intrinsics and this \(Z\):

\[
X = (u - c_x)\,Z / f_x,\quad
Y = -\,(v - c_y)\,Z / f_y,\quad
Z_\mathrm{cam} = -Z
\]

Camera frame here is OpenGL: \(+X\) right, \(+Y\) up, look \(-\!Z\). Image \(v\) is down, hence the minus on \(Y\). World point:

\[
\mathbf{x}_w = R^\top \mathbf{x}_\mathrm{cam} + \mathbf{t}
\]

where \(R,\mathbf{t}\) come from inverting `pose.view` (not stored in JSON — reconstruct from `world_position` + Euler **is incomplete**; see §9.7.3). For a point cloud, invert the 4×4 view if you log it yourself, or use the stored `world_position` only as the eye and accept that cervical Euler in JSON is **not** the full world rotation (heading is missing).

Load:

```python
depth = np.load("episode_00000/depth/frame_0000.npy")  # (H, W) float32 metres
```

---

### 9.5 Instance segmentation (`segmentation/frame_XXXX.png`)

| Property | Value |
|---|---|
| Format | PNG `I;16` (16-bit unsigned, one channel) |
| Meaning | **Instance ID**, not semantic class ID |
| Shape | `(H, W)`, same alignment as RGB |
| 0 | Background / sky / uncleared |
| 1–5 | Reserved terrain (see table) |
| ≥ 100 | Buildings, lamps, actors, … (`reset_instance_ids(100 + episode * 1000)`) |

**How it is measured.** Each draw call sets `u_instance_id`. The third MRT attachment is `R16UI` (`id_dtype=u2`) or fallback `u4`/`f4`. The CPU array is always saved as `uint16`.

Reserved terrain IDs (every episode):

| ID | What | In JSON `objects`? |
|---|---|---|
| 0 | Sky / empty | no |
| 1 | Ground plane (grass) | no (`ground` skipped) |
| 2 | Road asphalt | no |
| 3 | Curb | no |
| 4 | Sidewalk | no |
| 5 | Grass verge (if present) | no |

Everything else (buildings, lane paint, lamps, benches, hydrants, planters, pedestrians, cars, trees, potholes, …) gets a unique ID from `next_instance_id()`.

**Critical split: raster vs JSON.**

- **Segmentation PNG** contains *every* rasterized mesh that wrote an ID, including `StaticProp` (buildings, lamps, benches, hydrants, planters, road markings) and terrain.
- **JSON `objects`** only lists **`Actor`** instances, and even then skips `class_name ∈ {ground, road, sidewalk, curb, building}` and far/behind filters (§9.8).

So a lamp that fills 200 pixels has an ID in the PNG and **no** row in `objects`. A pedestrian has both, and `objects[].instance_id` **equals** the PNG value.

`config.CLASS_NAME_IDS` (background=0, vehicle=5, pedestrian=6, …) is a **planned semantic taxonomy**. It is **not** written to disk. Do not confuse those integers with instance IDs (vehicle instance 1142 is not class 5).

**What it tells you.** Pixel-perfect instance masks for anything that was drawn. Join to JSON:

```python
from PIL import Image
seg = np.array(Image.open("segmentation/frame_0120.png"))  # uint16 or int32
# for each obj in annotation["objects"]:
#     mask = seg == obj["instance_id"]
```

For semantic maps, map `class_name` → an integer yourself. Terrain IDs 1–5 have no JSON row; hard-code those five.

**Gotchas.**

- Lane-paint markings reuse `class_name="road"` as a StaticProp with their **own** instance ID (not 2).
- Lamps are `sign`, hydrants `bollard`, benches/planters often `dustbin` — same strings as furniture actors, but those infra pieces are **not** in JSON.
- Two meshes never share an instance ID inside one episode (IDs reset per episode). Across episodes, IDs restart and will collide if you concatenate without namespacing (`episode` + `instance_id`).

---

### 9.6 Video (`video.mp4`)

| Property | Value |
|---|---|
| Container | MP4 |
| Preferred codec | ffmpeg `libx264`, CRF 18, `veryfast`, `yuv420p`, no audio |
| Fallback | OpenCV `VideoWriter` fourcc `mp4v` / `avc1` / `XVID` / `H264` (BGR) |
| Frame rate | `SimulationConfig.fps` (30) |
| Resolution | same as RGB |

**How it is measured.** The same `uint8` RGB array that would become a PNG is piped, in order, into the encoder (`dataset_writer.VideoSink`). ffmpeg is tried **first**.

**What it tells you.** A watchable clip of the episode. Chroma 4:2:0 will blur colour edges; do not evaluate segmentation on decoded video. Use it for qualitative review and for models that ingest video, not as a substitute for PNG + NPY.

---

### 9.7 JSON annotations (`annotations/frame_XXXX.json`) — field by field

One file per frame, pretty-printed (`indent=2`). Built in `dataset_writer.build_annotation` plus `ann["episode"]` in `main.annotate_frame_full`.

#### 9.7.1 `dataset_metadata`

| Field | What it is |
|---|---|
| `generator` | `"EgocentricAccessibilitySim-Scratch"` |
| `version` | `SimulationConfig.generator_version` (currently `2.3.0`) |
| `coordinate_system` | Literal: right-handed **+X right, +Y up, +Z forward**, metres |

Tells you which world frame every `world_*` vector uses. If you mix with a dataset that is +Z up or OpenCV camera frame, rotate first.

#### 9.7.2 `frame_id`, `timestamp`

| Field | Measurement |
|---|---|
| `frame_id` | `ep{EEEEE}_f{FFFF}` — unique string per frame in a run |
| `timestamp` | \(t = \mathrm{frame\_index}\times\Delta t\), \(\Delta t=1/30\) s, **seconds from the start of this episode** (not wall clock, not Unix time) |

Frame 0 is \(t=0\). The scripted beat is at \(t^\* = k^\* \Delta t\).

#### 9.7.3 `camera_rig`

All of this is the **forehead wearable**, not a vehicle IMU.

| Field | How measured | What it tells you |
|---|---|---|
| `world_position` `[x,y,z]` | `CameraPose.position`: sidewalk sample + lateral sway + (`h_nominal` + heel-strike heave). Metres. | Optical centre in the world. \(y\) is eye height (~1.5–1.8 m), not the feet. |
| `linear_velocity` `[vx,vy,vz]` | Frame 0: path tangent × \(v_0\), \(v_y=0\). Later: **finite difference** \((\mathrm{eye}_k-\mathrm{eye}_{k-1})/\Delta t\). Includes heave/sway. | Instantaneous camera velocity used as \(\mathbf{v}_c\) in TTC. Not a filtered IMU. |
| `rotation_euler_deg` | **Cervical / gait offsets only**, degrees, intrinsic yaw→pitch→roll on top of path heading. `pitch` = nod, `yaw` = extra look-left/right, `roll` = head tilt. Clipped to explore limits (6° / 18° / 4°). | How much the head is fidgeting. **This is not the full world attitude.** Street heading \(\mathrm{atan2}(T_x,T_z)\) is *added* in the rig and lives only in `view`. You **cannot** rebuild the view matrix from Euler alone. |
| `intrinsics.fx, fy` | \(f_x = W / (2\tan(\mathrm{fov}_x/2))\), \(f_y=f_x\) (square pixels) | Pinhole \(K\). Default 90° HFOV → \(f_x = W/2\). |
| `intrinsics.cx, cy` | \(W/2\), \(H/2\) | Principal point, pixel units, top-left origin. |
| `intrinsics.resolution` | `[W, H]` | Must match the arrays. |
| `intrinsics.fov_deg` | Horizontal FOV in **degrees** (default 90). Not vertical. | |

Near / far (0.1 m / 100 m) are **not** in JSON; they are in `config.CameraIntrinsicsConfig`.

#### 9.7.4 `environment`

Constant for the whole episode (drawn once in `sample_environment`).

| Field | How measured | What it tells you |
|---|---|---|
| `sun_azimuth_deg` | Uniform [0, 360). Azimuth 0 = sun in **+Z** (forward), 90° = +X. | Shadow direction in the images. |
| `sun_elevation_deg` | Uniform [5, 85]. 90° = zenith. | Low sun → long shadows, warm Kelvin more often. |
| `road_condition` | `"wet_specular"` if `wetness > 0.45`, else `"dry"` | Whether asphalt/sidewalk get puddle highlights. Wetness scalar itself is **not** stored. |
| `ambient_light_intensity` | Uniform [0.1, 0.6] | Hemisphere fill. Low = moodier, noisier-looking shade. |

Fog density and Kelvin are **not** written. They affect RGB only.

#### 9.7.5 `episode` (copied onto **every** frame)

| Field | How measured | What it tells you |
|---|---|---|
| `scenario` | Director draw, e.g. `jaywalker`, `head_overhang` | Which scripted beat was attempted. Ambient street life is always present in addition. |
| `target_class` | Multinomial over `class_ratios` **before** spawn | What the director *aimed* to realize. Use for debugging the sampler, not as a train label. |
| `realized_class` | **Worst threat among objects listed on THIS frame** (`episode_realized_class`). Priority: `CRITICAL_THREAT` > `NEAR_MISS` > `SAFE`. | Instantaneous clip label *for this frame only*. See §9.10. |
| `k_star` | Uniform integer in \([0.40 F,\; 0.80 F)\) | Frame index the director aimed the beat at (8–16 s on a 20 s clip). The worst label often occurs in a *window* around \(k^\*\), not only at \(k^\*\). |

**`realized_class` in JSON is per-frame.** The process stdout `worst=` and `main.counts` are the max over the **whole episode**. If you need an episode-level train label, reduce:

```text
CRITICAL_THREAT if any frame has realized_class == CRITICAL_THREAT
else NEAR_MISS if any frame has NEAR_MISS
else SAFE
```

`target_class` and that reduction can disagree (spawn failed, short `--frames`, or ambient life stole the class).

#### 9.7.6 `objects[]` — who is listed

Each entry is one **`Actor`** that passed the gates in `annotate_frame_full`.

**Included (typical):** `pedestrian`, `vehicle`, `cyclist`, `scooter`, `dustbin`, `bollard`, `tree`, `pothole`, `head_obstacle`, and any other actor class the director spawned.

**Never included:** terrain and buildings (`SKIP_ANNOTATE`), all `StaticProp` (lamps, benches, hydrants, planters, markings, building meshes).

**Dropped even if they are actors:**

- `box.behind` **and** `euclidean_distance > 12` m (entirely behind the near plane, far);
- `not box.visible` **and** `euclidean_distance > 18` m (projected box fully off-screen or failed projection, far).

Near, behind-the-head, or off-screen-but-close actors **are** kept (haptic relevance: something 2 m behind you still matters less, but a 3 m off-screen car about to enter the FOV is listed). Their 2D box may be degenerate or clamped.

Objects are **not** sorted. There is no track ID other than `instance_id` (stable for the episode). No confidence field (this is GT).

---

### 9.8 Per-object fields — measurement and meaning

#### 9.8.1 Identity

| Field | Measurement | Tells you |
|---|---|---|
| `instance_id` | `Actor.instance_id`, same uint written to the ID buffer | Join key to the segmentation PNG. Unique per episode, not globally. |
| `class_name` | String set at `make_actor` / spawn | Semantic category. **Imperfect ontology** — see §9.12. |
| `threat_classification` | `classify_threat` on that frame’s TTC/CPA | Instantaneous haptic class for **this object**. Four strings: `CRITICAL_THREAT`, `NEAR_MISS`, `SAFE_DYNAMIC`, `SAFE_STATIC`. |

Episode-level `realized_class` collapses the two SAFE_* strings to `SAFE`.

#### 9.8.2 Kinematics (world metres, metres / second)

All vectors are **world frame**, +X right, +Y up, +Z forward.

Let \(\mathbf{p}_i\) = actor OBB centre, \(\mathbf{v}_i\) = actor velocity, \(\mathbf{p}_c\) = camera eye, \(\mathbf{v}_c\) = camera finite-difference velocity.

| Field | Formula | Tells you |
|---|---|---|
| `world_position` | \(\mathbf{p}_i\) | Where the **box centre** is, not the feet. For a 1.7 m pedestrian this is ~0.85 m above the walk. Trees are an exception: centre is near the roots (`y≈0`) while the canopy is high. |
| `world_velocity` | \(\mathbf{v}_i\) | Frenet: \(\dot s\,\mathbf{T}\) (Y = 0). Crossing: lateral speed along \(\mathbf{N}\). Parked / furniture: ≈ 0. Sudden-brake: \(\dot s\) decreasing. |
| `relative_position` | \(\mathbf{P}=\mathbf{p}_i-\mathbf{p}_c\) | Vector **from camera to object**. Negative Z means the object is behind the wearer if the street is aligned with +Z; on a curved block interpret in world, not “camera forward.” |
| `relative_velocity` | \(\mathbf{V}=\mathbf{v}_i-\mathbf{v}_c\) | How the gap is changing. Camera heave makes a small \(V_y\) even for a level walker. |
| `euclidean_distance` | \(\lVert\mathbf{P}\rVert\) (**always 3-D**) | Slant range camera-eye → box centre. A 1 m tall bin 2 m ahead is > 2 m because of the height difference. **Not** the quantity used for locomotion TTC. |
| `time_to_collision_sec` | See §9.9. `null` if not converging (code uses \(+\infty\) internally). | Seconds until closest approach **if** both keep current velocity. Not “seconds until first mesh contact.” |
| `closest_point_of_approach_dist` | \(d_\mathrm{CPA}=\lVert\mathbf{P}+\mathbf{V}\,t_\mathrm{CPA}\rVert\) in the same plane as TTC | Miss distance at that future time. 0 = centres coincide. **Centre-to-centre**, not surface-to-surface (a 0.4 m CPA between two 0.3 m-radius bodies is already a graze). |

`converging` is computed (`\(\mathbf{P}\cdot\mathbf{V}<0\)`) but **not stored**. You can recover it: TTC is `null` ⇔ not converging (or relative speed ≈ 0).

#### 9.8.3 How TTC / CPA are measured (the actual code)

`collision_engine.ttc_cpa` / `kinematics_vs_camera`:

1. Form \(\mathbf{P},\mathbf{V}\) in 3-D.
2. **Locomotion classes** (everything except `head_obstacle`, `sign`, `awning`): zero the Y components → horizontal plane \((X,Z)\). Reason: a 1.8 m pedestrian’s centre is above the camera; 3-D CPA would look like a “miss” even when they share the sidewalk.
3. **Head-height classes:** keep full 3-D so a bar at \(y=1.62\) m in front of the forehead collapses TTC.
4. Closing scalar \(c=\mathbf{P}\cdot\mathbf{V}\). Converging iff \(c<0\) and \(\lVert\mathbf{V}\rVert^2>10^{-10}\).
5. If converging: \(t_\mathrm{TTC}=t_\mathrm{CPA}=-c/\lVert\mathbf{V}\rVert^2\) (can be **larger than the remaining clip**). If not: TTC \(=+\infty\) → JSON `null`, \(t_\mathrm{CPA}=0\), \(d_\mathrm{CPA}=\lVert\mathbf{P}\rVert\) (distance **now**).
6. If \(\lVert\mathbf{V}\rVert\approx 0\): same as not converging; \(d_\mathrm{CPA}\) is current centre distance (horizontal or 3-D per class).

Assumptions baked in:

- Constant velocity (no future braking except what already happened to \(\mathbf{v}_i\) this frame).
- Point centres, **not** swept OBBs. A wide van can have \(d_\mathrm{CPA}=0.6\) m (NEAR_MISS / SAFE) while a corner still hits the wearer.
- Camera velocity includes gait jitter, so TTC flickers by a few percent frame to frame.

#### 9.8.4 How `threat_classification` is decided

`classify_threat(speed, ttc, d_cpa, converging, thr)` with `ThreatThresholds` defaults:

| Priority | Label | Gate |
|---|---|---|
| 1 | `CRITICAL_THREAT` | converging **and** \(t<2.0\) s **and** \(d<0.45\) m |
| 2 | `NEAR_MISS` | converging **and** \(t\in[1.8,3.5]\) **and** \(d\in[0.4,1.2]\) |
| 3a | `CRITICAL_THREAT` | \(\lVert v_i\rVert<0.15\) (static) **and** converging **and** \(d<0.45\) **and** \(t<3.5\) |
| 3b | `NEAR_MISS` | static **and** converging **and** \(d\le 1.2\) **and** \(t\le 4\) |
| 3c | `SAFE_STATIC` | other static |
| 4 | `SAFE_DYNAMIC` | not converging, **or** \(t>4\) **and** \(d>1.8\) |
| 5 | `NEAR_MISS` | leftover: converging **and** \(t<3.5\) **and** \(d<1.5\) (catches “close and soon” that missed the tight band) |
| 6 | `SAFE_DYNAMIC` | else |

`speed` is \(\lVert\mathbf{v}_i\rVert\), **not** relative speed. A parked car the wearer is walking toward is “static” and can still be CRITICAL if the camera closes within 0.45 m.

**What the labels are for.** They are the **haptic policy target**: buzz now / soon / ignore. They are **not** “did a mesh intersection occur this frame.” At the exact CPA instant, \(\mathbf{P}\cdot\mathbf{V}\approx 0\), so the object often flips to SAFE even if it just shaved the shoulder — that is why episode class is the **worst label over time**, not the label at \(k^\*\).

Worked intuition:

- Oncoming pedestrian, 2.5 s out, will pass 0.8 m to the side → `NEAR_MISS`.
- Same person, 1.2 s out, 0.2 m CPA → `CRITICAL_THREAT`.
- Receding person behind you → TTC `null`, `SAFE_DYNAMIC`.
- Dustbin 0.3 m left of the walk centre, you will pass it in 2 s → static + converging + small \(d\) → `CRITICAL_THREAT` or `NEAR_MISS`.
- Car in the far lane, 15 m, parallel → large \(d_\mathrm{CPA}\) → `SAFE_DYNAMIC`.

#### 9.8.5 `bounding_box_3d`

| Field | Measurement | Tells you |
|---|---|---|
| `center` | Same as `kinematics.world_position` (OBB centre) | Redundant with kinematics; convenient for 3-D detectors. |
| `extents` | **Full** width, height, depth in metres along the actor’s local axes (not half-extents) | Pedestrian ≈ `(0.55, 1.7, 0.45)`-ish; cars larger. The mesh is *inside or near* this box, not a tight convex hull. |
| `orientation_quaternion` | `[x, y, z, w]` from `glm.quat` (`actors.quat_to_xyzw`) | Yaw-from-velocity for movers (`face_velocity` or Frenet heading). Identity for many statics. |

World-space corners used for 2-D projection:

\[
\mathbf{c} + R\,\mathrm{diag}(\pm e_x/2,\pm e_y/2,\pm e_z/2)
\]

with \(R\) from the quaternion. **There is no 9-DoF / 24-DoF KITTI box** — only centre, extents, quat.

Tree caveat: extents cover the canopy in XZ/Y poorly; the centre sits at the trunk base, so the 3-D box is **wrong for the leaves** (it is a known limitation). Head-obstacle / pothole boxes are the authored extents around a local-space mesh.

#### 9.8.6 `bounding_box_2d`

Axis-aligned box in **pixel coordinates**, same origin as the PNG (top-left).

**How it is measured** (`project_obb_to_2d`):

1. Transform 8 OBB corners to view space.
2. If **all** corners have \(z_\mathrm{view} \ge -\mathrm{near}\), the box is `behind`; JSON still stores a clamped box (often `0,0,0,0`).
3. Else clip the 12 cube **edges** against the near plane \(z=-\mathrm{near}\) (Sutherland–Hodgman on each edge).
4. Project surviving points with the GL projection matrix → NDC → pixels.
5. AABB = min/max \(u,v\) of those pixels (can extend **outside** `[0,W)×[0,H)` before clamp).
6. JSON stores the **clamped** integer box: `floor(xmin), floor(ymin), ceil(xmax), ceil(ymax)` after `clamp` to `[0, W-1]` × `[0, H-1]`.

This is a **2-D AABB around the projected 3-D OBB**, not a tight silhouette and not a semantic mask. Recessed windows, a person’s legs, a car’s mirrors can stick out of or sit inside the box. Compare to the instance mask for a tight region.

`visible=False` (off-screen or behind) still may emit a clamped box if the object was not distance-gated out.

**What it tells you.** Detection GT (YOLO/COCO-style `xyxy`). For training, drop boxes with `truncation_ratio` ~ 1 or `xmax<=xmin`. There is no `difficult` flag; use occlusion/truncation instead.

#### 9.8.7 `visibility`

| Field | How measured | Tells you |
|---|---|---|
| `truncation_ratio` | \(1 - A_\mathrm{clamped}/A_\mathrm{raw}\) of the 2-D AABB. 0 = fully inside the image, 1 = nothing left after clamp (or not visible). | How much of the **box** (not the mesh) is cut by the image border. A person half out of frame ≈ 0.5. |
| `is_truncated` | `truncation_ratio > 0.02` | Boolean convenience for filtering. |
| `occlusion_ratio` | 1 − (visible samples / \(m^2\)), \(m=\)`occlusion_grid` (default **7** → 49 samples) | Fraction of a coarse grid over the **clamped 2-D box** that does not see this instance. 0 = fully visible, 1 = fully hidden or invalid. |
| `is_occluded` | `occlusion_ratio ≥ 0.25` | Flag for “mostly hidden.” A 0.24 occlusion is still `false`. |

**Occlusion algorithm** (`compute_occlusion_and_truncation`), for each grid pixel \((u,v)\):

1. If `instance_mask[v,u] == instance_id` → **visible** (this is the reliable path).
2. Else cast a pinhole ray, intersect the **local AABB**, get Euclidean hit distance \(t\).
3. Compare to `depth[v,u]` (view-space \(Z\)) with an **8 cm** slack: visible if `depth >= t - 0.08` or depth ≈ 0.
4. Samples that miss the AABB are ignored (do not increment visible; they still increment `total`).

So occlusion is **not** “percent of instance pixels hidden.” It is “percent of a 7×7 grid on the *box* that failed the test.” A thin pole in a huge tree box looks heavily occluded. Prefer the instance mask for exact visibility.

Mismatch: depth is \(Z\), \(t\) is ray length. Off-axis, \(Z < t\), so the fallback depth test is slightly pessimistic. When the ID buffer matches, this does not matter.

If the box is behind / invisible: `occlusion_ratio = 1`, `is_occluded = true`.

---

### 9.9 What is *not* in the annotation (so you do not look for it)

| Missing | Where it actually is / why |
|---|---|
| Semantic class IDs | `CLASS_NAME_IDS` in `config.py` only; map from `class_name` yourself |
| Terrain / building / lamp rows | Segmentation PNG only |
| `converging` flag | Infer from TTC `null` |
| Fog, Kelvin, wetness scalar | RGB only; JSON has `road_condition` + ambient |
| Full camera quaternion / view matrix | Reconstruct from the rig if you add logging; Euler is cervical-only |
| Track IDs across episodes | `instance_id` is per-episode |
| 2-D segmentation RLE / polygons | Use the PNG |
| Calibration YAML | Intrinsics are inside every JSON |
| Audio, events, haptic waveform | Labels are the haptic proxy |
| Ground-truth mesh contact / collision boolean | TTC/CPA only |
| `target_class` realized over the whole episode | Reduce `realized_class` across frames, or use the run log |

---

### 9.10 Frame label vs episode label vs object label

Three different questions:

1. **This object, now** → `objects[i].threat_classification`. Train a per-box haptic head on this.
2. **This frame, any object** → `episode.realized_class` in *that* JSON. “Should the belt buzz at this timestamp?”
3. **This 20 s clip** → max of (2) over \(k=0\ldots F-1\). That is what `main.py` prints as `worst=` and counts at the end of a run. Use this for episode-level class balance.

Director `target_class` is (3) **intended**, not measured.

Because TTC is instantaneous, a CRITICAL jaywalker is CRITICAL for ~1–2 s and SAFE the rest of the clip. A clip-level CRITICAL dataset is therefore **sparse in time**. If you train frame-wise, most frames of a CRITICAL episode are still SAFE.

---

### 9.11 Suggested uses (and wrong uses)

| You want to… | Use | Do not |
|---|---|---|
| Detect people/cars in egocentric RGB | PNG + `bounding_box_2d` + `class_name` | Video frames; `CLASS_NAME_IDS` as pixel values |
| Instance masks | Seg PNG ⨝ `instance_id` | 2-D AABB as a mask |
| Semantic terrain | Hard-code IDs 1–5 | Expect buildings in JSON |
| Monocular depth | `depth/*.npy` as \(Z\) | Treat 0 as 0 m of ground; it is sky |
| Haptic / TTC network | `kinematics.*` + `threat_classification` | Assume TTC is mesh-contact time |
| Occupancy / BEV | Lift depth + pose (add view logging) | Use Euler as world R |
| Imitate the director | `scenario` + `k_star` | Treat `target_class` as GT |
| Class-balance a run | Histogram episode-max `realized_class` | Histogram `target_class` or a single frame |

Minimal loader:

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

ep = Path("dataset_output/episode_00000")
k = 120
rgb = np.array(Image.open(ep / "rgb" / f"frame_{k:04d}.png"))
depth = np.load(ep / "depth" / f"frame_{k:04d}.npy")
seg = np.array(Image.open(ep / "segmentation" / f"frame_{k:04d}.png"))
ann = json.loads((ep / "annotations" / f"frame_{k:04d}.json").read_text())

for obj in ann["objects"]:
    mask = seg == obj["instance_id"]
    ttc = obj["kinematics"]["time_to_collision_sec"]  # None if diverging
    label = obj["threat_classification"]
```

---

### 9.12 Class-name strings (actors / furniture)

Written in JSON when the object is an **Actor**:

`vehicle`, `pedestrian`, `cyclist`, `scooter`, `dustbin`, `bollard`, `pothole`, `head_obstacle`, `sign`, `awning`, `tree`.

Reuse / pollution (do not treat as a clean ontology):

| String | Also used for |
|---|---|
| `bollard` | Hydrants (`StaticProp`, not in JSON) and true bollards (actors) |
| `dustbin` | Benches, planters (props) and bins (actors) |
| `sign` | Street lamps (props) and any sign/head geometry |
| `road` | Asphalt (ID 2) **and** lane-paint mesh (other ID, prop) |

`CLASS_NAME_IDS` adds `background, road, sidewalk, curb, building` for a future semantic map; those IDs are **not** the instance-map values (except by coincidence for terrain 1–3 vs that dict).

---

### 9.13 Example JSON (abbreviated)

```json
{
  "dataset_metadata": {
    "generator": "EgocentricAccessibilitySim-Scratch",
    "version": "2.3.0",
    "coordinate_system": "Right-Handed (+X Right, +Y Up, +Z Forward)"
  },
  "frame_id": "ep00003_f0120",
  "timestamp": 4.0,
  "camera_rig": {
    "world_position": [x, y, z],
    "linear_velocity": [vx, vy, vz],
    "rotation_euler_deg": {"pitch": 1.2, "yaw": -4.0, "roll": 0.3},
    "intrinsics": {"fx": 640.0, "fy": 640.0, "cx": 640.0, "cy": 360.0, "resolution": [1280, 720], "fov_deg": 90}
  },
  "environment": {
    "sun_azimuth_deg": 210.4,
    "sun_elevation_deg": 42.1,
    "road_condition": "dry",
    "ambient_light_intensity": 0.33
  },
  "objects": [
    {
      "instance_id": 142,
      "class_name": "pedestrian",
      "threat_classification": "SAFE_DYNAMIC",
      "kinematics": {
        "world_position": [x, y, z],
        "world_velocity": [vx, vy, vz],
        "relative_position": [dx, dy, dz],
        "relative_velocity": [dvx, dvy, dvz],
        "euclidean_distance": 8.41,
        "time_to_collision_sec": null,
        "closest_point_of_approach_dist": 8.41
      },
      "bounding_box_3d": {
        "center": [x, y, z],
        "extents": [0.55, 1.72, 0.48],
        "orientation_quaternion": [0.0, 0.12, 0.0, 0.99]
      },
      "bounding_box_2d": {"xmin": 610, "ymin": 280, "xmax": 690, "ymax": 520},
      "visibility": {
        "occlusion_ratio": 0.08,
        "truncation_ratio": 0.0,
        "is_occluded": false,
        "is_truncated": false
      }
    }
  ],
  "episode": {
    "scenario": "far_crosswalk",
    "target_class": "SAFE",
    "realized_class": "SAFE",
    "k_star": 312
  }
}
```

(`fx=640` is the 1280×90° default; at 960×540 it would be 480.)

---


## 10. Stack and environment

### 10.1 System (Arch)

- NVIDIA GPU + proprietary or mesa EGL (`libEGL`, `libglvnd`)
- Python 3.12+ (developed on 3.14 in `.venv`)
- Optional: **ffmpeg** on PATH for H.264 MP4 (preferred over OpenCV’s `mp4v`)

`setup_arch.sh` installs `mesa`, `libglvnd`, system numpy/scipy/pillow, then a venv with `--system-site-packages` and pip: moderngl, glcontext, pyglm.

### 10.2 Python packages (`requirements.txt`)

| Package | Use |
|---|---|
| `moderngl>=5.10` | GL 3.3 core, FBO, programs |
| `glcontext>=2.5` | EGL backend |
| `pyglm>=2.7` | `glm.mat4` / `vec3` / `quat` |
| `numpy` | meshes, TTC, Perlin, readback |
| `scipy` | listed; not heavily used in hot paths |
| `Pillow` | PNG write |
| `opencv-python` | MP4 fallback |

EGL needs access to the GPU. Sandboxed / headless-CI without `/dev/dri` will fail. On this machine it was verified with an RTX 4060, `id_dtype=u2`.

### 10.3 Venv

```text
/storage/BTP/synth_sim/.venv/bin/python
```

Always use that interpreter (or `source .venv/bin/activate`). System `python` will miss `glm` / `moderngl`.

---

## 11. Run commands

```bash
cd /storage/BTP/synth_sim
source .venv/bin/activate

# Full dataset-style run: 20 s clips, frames + H.264
python main.py --episodes 8 --seconds 20 --output-mode both \
  --output /storage/BTP/synth_sim/dataset_output --seed 20260910

# Resume numbering
python main.py --episodes 8 --start-episode 8 --seconds 20 --output-mode both \
  --output /storage/BTP/synth_sim/dataset_output --seed 20260910

# Video only (still writes JSON)
python main.py --episodes 4 --output-mode video --output ./dataset_output

# Frames only
python main.py --episodes 4 --output-mode frames --output ./dataset_output

# Debug (short). --frames bypasses the 15 s minimum.
python main.py --episodes 1 --frames 12 --output-mode frames --output /tmp/synth_dbg --seed 42

# Resolution / IO threads
python main.py --width 1280 --height 720 --workers 4 --episodes 2
```

`--seconds 12` becomes **15 s** (450 frames). `--seconds 20` is 600 frames. Default if you omit both `--frames` and `--seconds` is 20 s.

First-time Arch:

```bash
bash /storage/BTP/synth_sim/setup_arch.sh
```

Ignore the 5-second examples printed by that script; they are stale.

### 11.1 Expected performance

Rough, RTX 4060, 960×540, populated 160 m street: **~8–16 frames/s** (shadow + MRT + more meshes). A 20 s / 600-frame episode is on the order of **40–80 s** wall time, plus encode. 1280×720 is slower. Building meshes dominate setup; shadows dominate per-frame cost.

### 11.2 CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 4 | How many clips this process |
| `--start-episode` | 0 | First index (`episode_XXXXX`) |
| `--output` | `./dataset_output` | Root folder |
| `--seed` | 20260910 | Master seed |
| `--frames` | unset | Exact frame count; **overrides** duration clamp |
| `--seconds` | unset | Duration; min 15 unless `--frames` |
| `--output-mode` | `both` | `frames` \| `video` \| `both` |
| `--width` `--height` | 1280 720 | Raster size (also video size) |
| `--workers` | 4 | PNG/NPY/JSON thread pool |

There is no `--fps` flag; fps is `SimulationConfig.fps = 30`.

---

## 12. Tunable parameters (defaults)

### Camera

`1280×720`, FOV_x 90°, near 0.1, far 100.

### Gait

`v0 ∈ [1.1, 1.5]`, `alpha_v=0.08`, `f_step=1.8`, heave 0.045 m, sway 0.035 m, pitch/yaw/roll gait 2.5/4.0/1.5°, explore clips 6/18/4°, fBM β 3/8/1.5.

### Street

Road width U(7,12), sidewalk U(2,4), curb 0.25 × 0.15 m, length **160 m**, 8–12 control points, wander 3.5 m, sample 0.45 m.

### Threat gates

See §6.4. Critical offset U(0, 0.35) m; near-miss offset U(0.6, 1.2) m.

### Speed envelopes (inversion clamp)

Ped 0.8–2.8, vehicle 5–18, cyclist/scooter 3–8 m/s.

### Domain

Sun el 5–85°, az 0–360°, Kelvin 2500–7500, ambient 0.1–0.6, wetness 0.05–0.85 (wet_specular if >0.45), fog 0.0012–0.0045.

### Population

See `PopulationConfig` in `config.py` (§5.1 table in §7.3).

### IO

`writer_workers=4`, `shadow_resolution=2048`, occlusion grid 7, occluded flag if ratio ≥ 0.25.

---

## 13. Actor extras (runtime attributes)

Not on the dataclass; hung on the object:

| Attr | Meaning |
|---|---|
| `_follow_spline` | Frenet lock |
| `_path_s`, `_path_lat`, `_path_ds`, `_path_y` | arclength, lateral, \(\dot s\), center Y |
| `_crossing`, `_cross_s`, `_cross_lat0/1`, `_cross_dur`, `_cross_t` | road cross |
| `_brake_frame`, `_brake_accel` | sudden-brake (default −3 m/s²) |
| `scripted` | marks director threats; spline agents also skip social force |

`advance_actor` in `collision_engine.py` is the **only** integrator `main.step_actors` calls.

---

## 14. Bugs already fixed (do not reintroduce)

1. **Road invisible** — back-face cull of ground quads. Cull stays off.
2. **Mirrored image** — `glm.lookAt` +Z. Use `view_matrix_z_forward`.
3. **Flat / wrong depth** — GLM projection indexing `P[2,3]` vs `P[3,2]`.
4. **Head hazard / pothole in the sky** — world-baked mesh + model matrix. Keep hazards **local**.
5. **`dataclass` import dropped** in `collision_engine.py` → `NameError`. Keep the import.
6. **Floating people/cars** — ground-relative mesh + `position.y = extents.y/2` without centering. Dynamic meshes now `_center_y`.
7. **Milky pastel frames** — fog 0.006–0.018 + huge storefront plaster slabs + ACES. Fog lowered; glass is sentinel UV; lighting floor + contrast added.
8. **Hydrants in the walking lane** — moved to gutter lateral.
9. **SAFE episodes realizing NEAR_MISS** — oncoming peds on the camera sidewalk at ~0.6 m CPA. SAFE crowd rules in §7.3.
10. **Sky VAO** — sky shader has only `in_position`; bind `3f 20x` like shadows.
11. **`u_shininess` missing** — PBR shader dropped it; `core_gl.draw` uses `_set_if`.
12. **Long clips drift off the sidewalk** — constant world velocity on a curved spline. Everyone important is Frenet-locked.

---

## 15. Known limitations / honest gaps

- Still a **procedural rasterizer**, not photogrammetry. Cars from the rear are simple lofts; people are capsules; trees are noisy spheres on trunks.
- Building windows use metric UV + world Y; after yaw, brick rows follow the facade but are not unique textured assets.
- Shadow `lookAt` can X-flip relative to the camera; it is self-consistent for lighting.
- Semantic labels for some props are reused (`hydrant` → `bollard`, some benches → `dustbin`).
- Tree OBB is centered at the roots (`position.y=0`) while the canopy is metres up — 2D boxes for trees are loose.
- No walk-cycle bone animation (stride is a static posed mesh).
- No audio, no semantics for traffic lights / crosswalk paint (only lane dashes).
- `setup_arch.sh` help text is stale (5 s / 150 frames).
- Old `dataset_output/` may still have a **flat** v2.0 layout.
- Instance ID space is `100 + episode*1000`; a huge furniture+crowd episode should still fit.
- Video `mp4v` from OpenCV may not play in all browsers; prefer ffmpeg on PATH.
- `target_class` and `realized_class` can disagree (especially short `--frames` debug runs, because \(k^\*\) is a fraction of F).

---

## 16. How to extend (practical)

**New furniture mesh:** add `generate_*` in `geometry.py`, scatter in `_spawn_street_furniture` or `_scatter_street_infra`, pick a `style` and albedo.

**New scenario:** append the name to `SAFE_SCENARIOS` / `NEAR_SCENARIOS` / `CRITICAL_SCENARIOS`, branch in `finalize_with_camera`, spawn with `bind_spline_motion` / `bind_crossing` or inversion. Keep SAFE agents off `world.sidewalk_lateral`.

**New material:** add a `u_style` integer in `shaders.py` `shade_albedo`, set it on the `StaticProp` / `Actor`.

**Different duration:** `--seconds 20` or change `default_episode_seconds`. If you go beyond ~25 s, lengthen `street.street_length` so oncoming cars still exist at meeting time \(s_\mathrm{meet}+v_\mathrm{car}t\).

**Class balance:** `class_ratios` is the *target* draw; `realized` is what TTC says. After a big run, histogram `episode.realized_class` from frame 0 JSON (or the log `worst=`).

---

## 17. Version history (this repo)

| Ver | What |
|---|---|
| 2.0.0 | First modular scratch spec: EGL MRT, gait, TTC inversion, flat `rgb/` output, boxy scene |
| 2.1.0 | Per-episode folders, `--output-mode`, shadows, sky, richer meshes (partial) |
| 2.2.0 | Lofted vehicles, glass buildings, grass/camber, PBR bump, actor Y-centering, lighting/fog fix |
| 2.3.0 | 20 s default, 160 m street, Frenet crowd + crossers + traffic, expanded scenarios, hydrant/corridor SAFE rules |

---

## 18. Mental model (one paragraph)

A Catmull–Rom street is extruded into asphalt/curb/sidewalk. A forehead camera walks the **right** sidewalk with a closed-form oscillating speed and Perlin-noised head. Other agents are points in the Frenet chart \((s,L)\) so they stay on lanes and crossings for the whole 20 s. At a chosen time \(k^\*\) the director inverts a constant-velocity threat onto the camera pose. Each frame a ModernGL MRT writes RGB, linear depth, and instance IDs; CPU code projects OBBs, measures occlusion, and labels TTC/CPA. Disk layout is `episode_XXXXX/{rgb,depth,segmentation,annotations,video.mp4}`.

If you remember only three rules: **+Z forward / custom view matrix**, **local meshes + one model matrix**, **Frenet for anyone who must still be on the street 15 seconds later**.
