# blender_sim — full implementation notes

This file is the future-context dump for `/storage/BTP/blender_sim`. It records **why the pipeline exists**, **how every module fits together**, **the maths**, **every CLI / config knob**, **Blender 5.x pitfalls we already paid for**, and **how to run it**. Prefer this over archaeology in chat logs.

Machine this was built on: Arch Linux, Blender **5.2.1 LTS** at `/usr/bin/blender`. Workspace root: `/storage/BTP/blender_sim`.

---

## 1. The task and why this exists

### 1.1 Problem

Train / evaluate a **low-latency tactile warning model** for a pedestrian wearing a head-mounted RGB camera (eye height 1.6 m). The model must decide, from egocentric video, whether something in front of the walker is:

| Label | Meaning for the vest / belt |
| --- | --- |
| `SAFE_STATIC` | Clutter that will not hit you |
| `SAFE_DYNAMIC` | Moving, but will miss |
| `NEAR_MISS` | Closing; you should feel a graded warning |
| `CRITICAL_THREAT` | Closing *and* on a collision course — fire the strong cue |

This is **not** an autonomous-vehicle dataset. The ego agent is a walking human on a sidewalk, with gait bounce and head jitter. Threats include other pedestrians, cars and cyclists, potholes on the gait line, hanging signs / AC units at head height, and generic `threat_cube` primitives (so the model cannot overfit to “car-shaped” threats).

### 1.2 Why synthetic (Blender), not real video

- Real labels for TTC / CPA need ground-truth 3-D kinematics. A wearable does not have that.
- Domain randomization (lighting, weather, albedo, path shape) is cheaper than filming 30 cities.
- Forced intercepts (`jaywalker`, `swerve_vehicle`, …) can be solved in closed form so the 40 / 30 / 30 class mix actually happens.
- The detector must not overfit to a cube-person silhouette, so later passes replaced boxes with anthropometric meshes and procedural PBR.

### 1.3 What “done” looks like per episode

For each episode the pipeline writes:

1. Per-frame **JSON** (always) — camera pose, environment, every visible actor with threat label + 2-D box.
2. Optional **PNG** sequence (`--media frames` or `both`).
3. Optional **H.264** `preview.mp4` (`--media video` or `both`).
4. `episode.json` manifest + a dataset-level `dataset_summary.json`.

Internal sim is Blender **Z-up**. JSON vectors are the spec’s **Y-up** frame \((X,\,Y_{\mathrm{up}},\,Z_{\mathrm{forward}})\).

---

## 2. Stack and requirements

| Piece | Role |
| --- | --- |
| **Blender 4.2+ or 5.x** | Host. This box: 5.2.1 LTS. Engine id is `BLENDER_EEVEE` on 5.x and `BLENDER_EEVEE_NEXT` on 4.2–4.5. The picker in `configure_eevee()` tries the requested id, then those two, then Cycles / Workbench. |
| **EEVEE** | Real-time raster + optional screen-space raytracing / Fast GI. Not Cycles (too slow for 150 frames × N episodes). |
| **Python bundled with Blender** | Only interpreter that can `import bpy`. System Python can run `threat_math.py` self-tests. |
| **mathutils** | Vectors / matrices / Euler / quaternion (ships with Blender). |
| **ffmpeg** (optional) | Mux `%06d.png` → H.264. If missing, Blender’s VSE + bundled encoder is the fallback. |
| **No pip packages** | Deliberate. No numpy, no OpenCV, no addon assets. Every mesh is generated in Python. |
| **OpenGL / EGL** | EEVEE needs a GPU context. On a desktop session this is automatic. On a true headless box use `xvfb-run -a …` or `--no-render` (JSON only). |
| **GPU** | EEVEE is **not** CPU path-traced. It rasterizes on whichever GL/Vulkan device Blender opened at process start. This laptop is hybrid: AMD Radeon 780M (default display / default EEVEE) + RTX 4060 Max-Q (idle unless PRIME-offloaded). `./run.sh` defaults to `BTP_GPU=auto` → NVIDIA when `nvidia-smi` works. Raw `blender` invocations stay on the iGPU unless you export the PRIME vars (section 10). Cycles CUDA/OptiX see the 4060 but the pipeline does not use Cycles. |

Install on Arch: `sudo pacman -S blender` (and `ffmpeg` if you want system mux).

Blender’s embedded interpreter **does not put the script directory on `sys.path`**. `main.py` inserts `_ROOT` before any local import. Always launch as:

```text
blender --background --python /storage/BTP/blender_sim/main.py -- <argparse flags>
```

Everything after the bare `--` is ours. Without `--`, Blender eats the flags.

---

## 3. Repository layout

```text
/storage/BTP/blender_sim/
  main.py                 orchestrator (CLI, episode loop, JSON, video)
  config.py               single source of hyperparameters (deep-copied per run)
  camera_kinematics.py    sidewalk Frenet + gait + Perlin head jitter
  world_generator.py      street, actors, lighting, EEVEE, scenario injectors
  scenario_compose.py     compound CLI parse, aliases, Frenet occupancy
  humanoid.py             articulated pedestrian meshes + Winter walk cycle
  materials.py            procedural PBR (Object-space metres, EEVEE-safe)
  threat_math.py          TTC / CPA / taxonomy (no bpy; self-testable)
  spatial_threat.py       k×k screen threat matrix (no bpy; CLI --threat-grid)
  spatial_overlay.py      blue→red overlay video (CLI --spatial-overlay)
  projection.py           world AABB → pixel box + raycast occlusion
  run.sh                  thin wrapper: $BLENDER --background --python main.py -- "$@"
  gen_dataset.py          mixed pack builder → datasets/pack_* (seeded, balanced)
  README.md               short user-facing card
  IMPLEMENTATION.md       this file
  .gitignore              output/, __pycache__, *.blend
  output/                 default write root (gitignored)
```

There is no `requirements.txt`. There are no `.blend` assets. Collections created at runtime: `WORLD`, `HAZARDS`, `ACTORS`, `LIGHTS`.

---

## 4. Architecture and per-episode flow

```text
main.main()
  └─ for each episode:
       pick_scenarios(rng, cfg, --scenario)   # one name, or a compound list
       apply_camera_fov(...)           # lens_mm / hfov_deg for this episode
       WorldGenerator.prepare_scenario(names)  # union of gaps / sparse street
       WorldGenerator.build()
         reset scene, configure EEVEE
         choose_environment()          # dawn/noon/dusk/night/harsh_glare/overcast + weather + chaos
         PathSpline.generate()         # centreline (biome folds widths into cfg['world'])
         ribbons: road / sidewalk / curb / grass verge
         buildings + roofs + streetlamps
         apply_domain_randomization()  # sun, fill, bounce, sky, AgX
         Poisson scatter + background peds/cars
         reveal_view_layer(); re-hide daytime lamps
       create_camera() + choose_ego_profile() + CameraRig(...)
       inject_scenarios(names, rig); place_ego_props(rig)  # bench if seated
       # Phase A — CPU: kinematics + JSON (no EEVEE)
       for frame i = 0 .. N-1:
         CameraRig.update + WorldGenerator.update
         view_layer.update()  # required: actors write location, not matrix_world
         snapshot actor/camera poses
         build_frame_record(); collect object frames + k×k grid
       write annotations/annotations.json   # unless --no-annotations
       write spatial_annotations/spatial_annotations.json
       # Phase B — GPU: one animation render, poses replayed in frame_change_pre
       bpy.ops.render.render(animation=True)   # GPU context stays warm
       encode preview.mp4 (h264_nvenc or libx264 veryfast)
       write episode.json
  write dataset_summary.json
```

### 4.1 RNG

- `--seed` seeds a **master** `random.Random`.
- `--start-episode N` advances the master N times (`randrange`) so shard `N` is stable. If omitted, the next free `episode_*` index under `--output` is used so a new launch does not overwrite `episode_0000`.
- Each episode draws `ep_rng = Random(master.randrange(1, 2**31))` and uses **only** that stream for Poisson, colours, Perlin seeds, body types, path shape. Two episodes never share a Perlin / Poisson stream.

### 4.2 Frame budget

`N = min(--frames or config.frames_per_episode, floor(rig.max_time() * fps))`.

`max_time()` is the remaining sidewalk arc after `sidewalk_s0 = 3.0 m`, minus 2 m of tail. A 48–78 m path at 1.0–1.4 m/s easily covers the default 150 frames (5.0 s).

### 4.3 Media policy

| `--media` | PNG kept | `preview.mp4` | JSON |
| --- | --- | --- | --- |
| `frames` | yes | no | always |
| `video` | deleted after mux | yes | always |
| `both` | yes | yes | always |
| `both` + `--no-rgb` | deleted after mux | yes | always |
| `--no-render` | no | no | always |

`--spatial-overlay` is independent of `--media`: it muxes `spatial_overlay.mp4` from the PNG sequence whenever RGB was rendered. It is skipped under `--no-render`. `--no-annotations` skips only `annotations/annotations.json`; spatial matrices still write.

If video mux fails, PNGs are **kept** even when `--media video` or `--no-rgb`.

---

## 5. Coordinate frames (read this before touching kinematics)

| Frame | Axes | Where used |
| --- | --- | --- |
| **Blender world** | X right, **Y forward along a +Y path**, **Z up** | All `bpy` locations, splines, lighting, TTC/CPA **computation** |
| **Spec / JSON** | X right, **Y up**, **Z forward** | Written `world_position`, `velocity`, `relative_velocity` |
| **Camera local** | +X right, +Y up, **looks down −Z** | `CameraRig._compose_matrix` |
| **Humanoid local** | +Y face, +Z up, limbs hang **−Z** | Hip/knee/shoulder rotate about local +X (sagittal) |
| **Car / box local** | +Y is “forward”; `look_along` yaws so local +Y = horizontal heading. \(\theta=\mathrm{atan2}(-d_x,d_y)\) because \(R_z(\theta)(0,1,0)=(-\sin\theta,\cos\theta)\). **Not** `atan2(d_x,d_y)`. | Vehicles, buildings, people |


Conversion (in `threat_math.py`):

```text
blender_zup_to_yup (x, y, z) = (x, z, y)
yup_to_blender_zup (x, y, z) = (x, z, y)   # same permutation
```

A pedestrian’s **feet** sit on the walking surface (Z = 0 on asphalt, Z = 0.12 m on the sidewalk); the camera sits near Z = 1.72 m on the curb (1.6 m eye height + curb). If you feed raw pelvis vs camera origins into CPA you get a ~0.6 m vertical miss and a true hit is labelled SAFE. That is why `threat_point()` exists (section 9).

---

## 6. File-by-file

### 6.1 `config.py`

`CONFIG` is a nested dict. `get_config()` returns `copy.deepcopy(CONFIG)` so an episode cannot leak mutations.

Every numeric default that is not a “magic number inside a formula” lives here. See **section 14** for the full table.

### 6.2 `threat_math.py` (no `bpy`)

Pure Python 3. Vector helpers (`vadd`, `vsub`, `vdot`, `vnorm`, …) avoid numpy.

**Constant-velocity point model.** Let \(\vec{P}_{\mathrm{rel}}=\vec{P}_{\mathrm{obj}}-\vec{P}_{\mathrm{cam}}\), \(\vec{V}_{\mathrm{rel}}=\vec{V}_{\mathrm{obj}}-\vec{V}_{\mathrm{cam}}\).

Range-rate proxy: \(\vec{P}_{\mathrm{rel}}\cdot\vec{V}_{\mathrm{rel}}\). Negative ⇒ converging.

Minimise \(f(t)=\|\vec{P}_{\mathrm{rel}}+t\vec{V}_{\mathrm{rel}}\|^2\):

\[
t^{\star}=-\frac{\vec{P}_{\mathrm{rel}}\cdot\vec{V}_{\mathrm{rel}}}{\|\vec{V}_{\mathrm{rel}}\|^2}
\quad\text{iff}\quad
\vec{P}_{\mathrm{rel}}\cdot\vec{V}_{\mathrm{rel}}<0
\;\text{and}\;
\|\vec{V}_{\mathrm{rel}}\|\ge\varepsilon
\]

\[
D_{\mathrm{cpa}}=\|\vec{P}_{\mathrm{rel}}+t^{\star}\vec{V}_{\mathrm{rel}}\|
\]

If not converging or \(\|\vec{V}_{\mathrm{rel}}\|<\varepsilon\), TTC is \(+\infty\) and CPA falls back to current range. JSON writes undefined TTC as **`9999.0`** (must stay a number).

**Taxonomy** (`classify_threat`), priority so a real hit can never be SAFE:

1. `CRITICAL_THREAT` if converging and \(TTC<2.5\) s and \(D_{\mathrm{cpa}}<0.5\) m
2. `NEAR_MISS` if converging and \(TTC<4.0\) s and \(0.5\le D_{\mathrm{cpa}}\le 1.5\) m
3. `SAFE_STATIC` if \(\|V_{\mathrm{obj}}\|\le 0.05\) m/s and (range \(>5\) m **or** CPA \(>1.5\) m **or** not converging). A static object the walker is about to strike (\(V_{\mathrm{obj}}=0\Rightarrow V_{\mathrm{rel}}=-V_{\mathrm{cam}}\)) still converges and can be CRITICAL.
4. else `SAFE_DYNAMIC`

**Intercept solver** (used by injectors):

\[
\vec{V}_{\mathrm{obj}}=\vec{V}_{\mathrm{cam}}+\frac{\vec{P}_{\mathrm{cam}}-\vec{P}_{\mathrm{obj}}+\vec{o}}{\tau}
\]

\(\vec{o}=\vec{0}\) ⇒ \(D_{\mathrm{cpa}}=0\), \(TTC=\tau\). \(\|\vec{o}\|=c\) ⇒ controlled near-miss of \(c\) metres under the constant-velocity assumption.

**Stationary ego.** Nothing above divides by \(\|V_{\mathrm{cam}}\|\). \(V_{\mathrm{cam}}=0\) (bench / hesitation) makes \(V_{\mathrm{rel}}=V_{\mathrm{obj}}\). A parked object then has \(V_{\mathrm{rel}}=0\) ⇒ TTC \(=+\infty\) ⇒ `SAFE_STATIC`; a ball thrown at a seated walker still solves. Every solver guards \(\|V_{\mathrm{rel}}\|^2 < \varepsilon^2\) so there is no `ZeroDivisionError` and no `NaN`.

**Planar mode.** `relative_kinematics(..., planar=True)` zeroes Blender Z on \(P_{\mathrm{rel}}\) and \(V_{\mathrm{rel}}\) before solving. Used in `main.py` for `park` and `plaza` biomes, where a 1.6 m eye-height offset is not clearance from a trunk or bollard. `threat_point` already clamps Z; planar is belt-and-braces. Non-finite inputs are zeroed (`_finite3`) so a NaN cannot silently classify as SAFE (`NaN < 2.5` is `False`).

Self-test (system Python, no Blender):

```bash
python /storage/BTP/blender_sim/threat_math.py
```

### 6.3 `camera_kinematics.py`

`CameraRig` owns the Blender camera and writes its 4×4 every frame.

**Path.** For `walk`, arc-length \(s(t)=s_0+v t\) with \(s_0=3.0\) m along the **road** centreline (same \(s\) as every actor). Origin is \(\vec{p}(s)+\hat{r}\,L(t)\) with \(L\) the sidewalk lateral (constant in `walk`/`seated`, ramped in `diagonal_cross`, fBm in `erratic`). Gaze is the *road* tangent. \(\hat{r}=\hat{t}\times\hat{z}\), \(\hat{u}=\hat{r}\times\hat{t}\). Eye height is curb + \(h+A\sin(2\pi f t)\) with \(h=1.6\) m standing or \(\sim 0.95\)–\(1.28\) m seated. Do **not** retarget the camera onto `offset_spline(L)` — that spline is resampled by its own arc length, so \(s_{\mathrm{sidewalk}}\) drifts from actor \(s_{\mathrm{road}}\) on a curve.

**Ego modes** (`cfg['ego']`, CLI `--ego-mode` / `--ego`):

| Mode | Weight | What changes |
| --- | --- | --- |
| `walk` | 0.52 | Original constant-speed sidewalk traverse. |
| `diagonal_cross` | 0.14 | \(L(t)\) smoothsteps toward the opposite kerb. Injectors read `lateral_at(t)` so intercepts land on the crossing, not the start kerb. |
| `erratic` | 0.22 | fBm sidestep + speed wobble; optional full stop. \(s(t)\) is a trapezoid table, not \(s_0+vt\). |
| `seated` | 0.12 | \(v=0\), lower eye height, bench spawned behind the HMD. `max_time()` does not divide by speed. Gait bounce is scaled to 0. |

**Gait (spec Part 2), applied on Blender +Z, scaled by ground speed so a seated / halted head does not bob:**

\[
Z(t)=1.6+A\sin(2\pi f t),\qquad A=0.04\,\mathrm{m},\; f=1.8\,\mathrm{Hz}
\]

Optional pitch bob in quadrature: \(\theta_{\mathrm{bob}}=0.015\cos(2\pi f t)\) rad (head dips at mid-stance).

**Perlin micro-saccades.** Independent 1-D improved Perlin (Ken Perlin 2002 fade \(6t^5-15t^4+10t^3\)) + fBm, **not** `mathutils.noise`, so a seed is bit-identical across Blender versions.

\[
n_{\mathrm{fBm}}(x)=\frac{1}{\sum_k p^k}\sum_{k=0}^{O-1}p^k\;\mathrm{noise}(x\,L^k)
\]

Each axis: \(\theta = A_{\mathrm{deg}}\cdot n_{\mathrm{fBm}}(t\cdot f_{\mathrm{Hz}}+\phi)\). Defaults: yaw \(15^\circ\) @ 0.22 Hz (4 oct), pitch \(5^\circ\) @ 0.55 Hz (3 oct), roll \(2^\circ\) @ 1.9 Hz (2 oct, heel-strike).

**Camera matrix.** Columns of the base rotation are \((\hat{r},\,\hat{u},\,-\hat{t})\) so local \(-\mathrm{Z}=+\hat{t}\). Jitter is Euler XYZ **in camera space**, then `rot_base @ rot_jitter`.

**Blender 5.x apply.** Assigning `matrix_world` alone is ignored when `rotation_mode` is Euler. The rig decomposes the matrix and writes `rotation_mode='QUATERNION'`, `location`, `rotation_quaternion`, then `matrix_world`.

**Velocity.** Frame 0: central difference of `predict_position` (correct for diagonal / halt; not `v * tangent`). Later frames: finite difference of eye position. Seated ⇒ \(\vec{v}\approx 0\) (residual is gait, which is also 0).

`predict_position(t)` / `predict_velocity(t)` are the **no-jitter** eye point; injectors use them so intercepts land on the gait line, not on a Perlin-wobble.

### 6.4 `projection.py`

Pipeline (spec Part 5.1), every frame, after `view_layer.update()`:

1. Union of world AABB corners of the object **and every MESH child**. `LocalBoundCache` stores object-space `bound_box` once and transforms by `matrix_world` (no `evaluated_get`; we have no mesh modifiers). A pelvis-only box would under-cover an articulated person / wheeled car.
2. View \(= M_{\mathrm{cam}}^{-1}\). Camera-space \(Z\ge 0\) is behind the lens (Blender looks down local \(-\mathrm{Z}\)) — those corners are dropped.
3. Projection: **one** `Object.calc_matrix_camera(...)` per frame, reused for every actor; fall back to the analytic OpenGL frustum from lens / sensor / aspect.
4. NDC \(\to\) pixels, origin **top-left**:

\[
u=(n_x+1)\,W/2,\qquad v=(1-n_y)\,H/2
\]

5. Box = min/max of surviving pixels, clamped to the image. `truncated` if any corner was off-screen or fewer than 8 corners were in front.
6. All-behind or all-off-screen ⇒ `None` (object omitted from that frame’s JSON).

**Threat point** (not the mesh origin):

- `footprint` (potholes): XY of AABB centre, Z = camera height (occupies the walker’s column).
- `volume` (people, cars, branches): XY of AABB centre, Z clamped into the object’s slab, preferring camera height. A 1.5 m car roof vs a 1.6 m camera leaves a 0.1 m vertical residual.

**Occlusion.** `scene.ray_cast` from camera origin toward the AABB centroid. Self / children do not count. Ground ribbons (`road_surface`, `sidewalk_*`, `curb_*`, `centerline`, `lamp_pole`) are stepped through so a pothole is not “occluded by the sidewalk it sits on”. Up to 10 steps. Hit within `occlusion_epsilon` (0.08 m) of the target is treated as self.

### 6.5 `world_generator.py`

The largest module. Responsibilities:

**PathSpline.** Control polygon → dense sample → uniform-\(ds\) resample (\(ds=0.40\) m). `evaluate(s)`, `tangent(s)`, `right(s)=\hat{t}\times\hat{z}`, `offset_point(s, lateral, z)`, `offset_spline(lateral)`. Factories:

- `straight` — +Y
- `gentle_curve` — cubic Bézier, bend \(\sim U(8,20)\) m
- `s_curve` — cubic Bézier, opposite lobes
- `corner_90` — two straights + circular fillet, radius \(\sim U(5.5,8)\) m

**Poisson-disk on a strip** (Bridson 2007) in \((s,\mathrm{lateral})\) with Euclidean parameter distance ≈ world distance on a gentle curve. Used for ground clutter and head-height hazards.

**Ribbons.** Quad strips along the centreline. UVs \((s/4,\,0\text{ or }1)\) (materials actually sample **Object** metres; UVs are a fallback). Layers: asphalt road, both sidewalks at curb height 0.12 m, curbs, grass verge 4.5 m outside the sidewalk. Centreline paint is **3 m dash / 3 m gap**, 0.12 m wide, built as a Frenet quad strip (`_centerline_dash`) so a corner does not chord a 3 m box across the lane.

**Buildings.** Extruded boxes starting at \(s=14\) m (so a 24 mm lens at \(s_0=3\) m is not filled by one wall). Facade setback = `road_half + sidewalk_w + 1.6`. Each building gets `make_facade(..., night=)` and a slightly larger roof slab (`make_roof`). `look_along` yaws local +Y to the path tangent.

**Camera lateral.** Sidewalk sign is random \(\pm 1\). Walker sits at `sign * (road_half + sidewalk_w * 0.38)` — closer to the curb than the facade. A 24 mm lens 1.3 m from a wall reads as “empty street / all wall”.

**Actors (`Actor` dataclass).** Python twin of a Blender object. Dynamic actors live in Frenet \((s, \mathrm{lateral})\) on the **road** spline: \(s \leftarrow s + \mathrm{speed}\,\Delta t\), and lateral drift (`lat_target` / `lat_speed`) is how jaywalkers, L→R cars, and cut-ins move. After every step, `StreetCorridor.confine` clamps \(|\mathrm{lateral}|\) to the pavement (`road_half + sidewalk_w - margin - pad`), so a curve cannot chord an actor through a facade. World-space `hold_velocity` is a last-resort fallback and is reprojected onto the ribbon if it is ever used. Also: `swerve_t` (start drifting after this time), `stop_t` (sudden stop), `gait` (`WalkRig.apply`), `wheels` (roll \(\Delta\theta=-\Delta s/r\) about local +X so the contact patch moves \(-\mathrm{Y}\) for \(+\mathrm{Y}\) travel). `look_along` runs **before** `_tick_visuals`, so the walk rig can add pelvic list/pitch/bob on top of heading. Heading is the XY velocity \(\hat{t}\,\mathrm{d}s+\hat{r}\,\mathrm{d}l\), not the path tangent, so a jaywalker faces across the street.

**Background traffic.** Cars in the **opposite** lane (`−sign(sidewalk_lateral) * lane_offset`) so background CPA stays \(>1.5\) m. Half travel against the path parameter. Peds on the opposite sidewalk or offset on the same one.

**Streetlamps.** 6 poles, 5.6 m, arm toward the carriageway, emissive bulb, **SPOT** aimed down (local \(-\mathrm{Z}\), identity rotation), 80° cone, energy 900 W at night / 0 by day. Only every other lamp casts shadows (EEVEE shadow pool caps at 2048 pages; 8 point lights + 2 suns overflowed). `reveal_view_layer()` un-hides everything, so `apply_streetlamp_state()` is called **again** after reveal.

**Lighting (`apply_domain_randomization`).** Six states, not four. Weights in `domain_randomization.lighting_weights`.

- **harsh_glare:** sun 1.5–7.5° above the horizon, energy 18–42, aimed *down the gait* (`glare_azimuth_deg`). AgX exposure pulled to −0.45 so highlights roll off into sunset-blindness rather than clipping to white.
- **overcast:** turbidity ≥ 9, sun shadows **off**, flat grey horizon, near-shadowless.
- **night:** moon-energy sun, near-zero world strength, sparse tinted streetlamps (per-lamp dead/gain). Pitch black except the cones.
- **dawn / noon / dusk:** the original four-state set.
- **Dappled:** with probability `dappled_prob` (0.26) an overhead alpha-hashed canopy gobo (`make_canopy_gobo`) punches leaf-shaped shadows onto the whole street.
- Key **SUN**, **SkyFill**, **GroundBounce** as before. World shader still never takes Volume Scatter.
- AgX. Exposure: night 0.30, dawn 0.10, dusk 0.08, noon 0.00, harsh_glare −0.45, overcast ~0.05.

`BTP_LIGHTING=dawn|noon|dusk|night|harsh_glare|overcast` overrides the weighted draw.

**Biomes** (`prepare_biome`, CLI `--biome`). Same Frenet corridor, different widths / ground / props:

| Biome | Weight | Visual |
| --- | --- | --- |
| `street` | 0.42 | Historical defaults: asphalt, kerb, buildings both sides. |
| `avenue` | 0.18 | 13 m carriageway, taller facades, more traffic. |
| `park` | 0.24 | 3 m gravel path on open grass, **no kerb / buildings / lane paint**, ego walks *on the path*, 10–26 instanced trees, dense grass clumps. TTC uses `planar=True`. |
| `plaza` | 0.16 | Wide paved open space, buildings on one side, jittered paving slabs, few cars. TTC planar. |

**MeshLibrary.** Trees, grass clumps, and the seated bench share mesh datablocks (`bpy.data.objects.new(name, shared_mesh)`). Per-instance variety is scale + yaw. `lib.stats()` is printed per episode. Pedestrians stay unique meshes (articulated gait); crowds are 3–11 people, not thousands.

**Ground hazards.** Poisson mix of pothole (capped cylinder well), crater (procedural bowl), broken_slab (tilted paver), debris (multi-box pile), puddle (`make_water`, **not annotated** — a dark patch is not a hole). On-path injectors (`pothole_on_path` / `near` / `offset`) draw from that trip-hazard set.

**EEVEE (`configure_eevee`).** TAA **16** + reprojection, shadows on, volumetric shadows **off**, raytracing on, Fast GI on (4 rays, quality 0.30, 6 steps), `shadow_pool_size='2048'`, cascade max distance 48 m, `indirect_light_intensity=1.35`. All `setattr` calls are `hasattr`-guarded.

**Black-frame bug (Blender 5.x).** Default `scene.render.use_sequencer=True` and `use_compositing=True`. The default VSE has **zero strips**, so every still is the empty sequencer (pitch black) even when the 3-D view is fine. `prepare_still_render()` forces both flags **False** and is called from `configure_eevee` **and** immediately before every PNG. Video mux may turn the sequencer back on; stills must turn it off again.

**Engine picker.** On this 5.2 build `enum_items` for `render.engine` may list only `BLENDER_EEVEE`. Setting `BLENDER_EEVEE_NEXT` raises `TypeError`. Never hard-code Next.

**Cleanup.** `release_episode()` unlinks every object, nulls Actor Python refs, then `bpy.data.orphans_purge(do_recursive=True)` in a loop. Without this a 2000-episode pack leaks every mesh and material.

**Scenario injectors** (after the camera rig exists, so `predict_*` is valid). `prepare_scenario` runs *before* `build()` so cross-street episodes open a building gap at \(s\in[4,20]\) m (compounds with two or more crossers use \([4,28]\) m) and `empty_street` drops background traffic. `empty_street` in a compound still sparsifies the street; the other names still inject.

Crossers stay in Frenet \((s,\mathrm{lateral})\) on the pavement. A through-jaywalker enters one FOV edge and walks at \(1.0\)–\(1.35\) m/s all the way out the other. A turn-jaywalker blends heading with a 0.9 s smoothstep and then walks along the sidewalk toward or away from the camera. Background pedestrians get a Perlin lateral wander; vehicles a lighter one. `cyclist_weaving` / `car_erratic_swerve` use a phased sine, not noise, so the swing is guaranteed to reach the gait.

**Black-frame bug (Blender 5.x).** Default `scene.render.use_sequencer=True` and `use_compositing=True`. The default VSE has **zero strips**, so every still is the empty sequencer (pitch black) even when the 3-D view is fine. `prepare_still_render()` forces both flags **False** and is called from `configure_eevee` **and** immediately before every PNG. Video mux may turn the sequencer back on; stills must turn it off again.

**Engine picker.** On this 5.2 build `enum_items` for `render.engine` may list only `BLENDER_EEVEE`. Setting `BLENDER_EEVEE_NEXT` raises `TypeError`. Never hard-code Next.

**Scenario injectors** (after the camera rig exists, so `predict_*` is valid). `prepare_scenario` runs *before* `build()` so cross-street episodes open a building gap at \(s\in[4,20]\) m (compounds with two or more crossers use \([4,28]\) m) and `empty_street` drops background traffic. `empty_street` in a compound still sparsifies the street; the other names still inject.

Crossers stay in Frenet \((s,\mathrm{lateral})\) on the pavement. A through-jaywalker enters one FOV edge and walks at \(1.0\)–\(1.35\) m/s all the way out the other. A turn-jaywalker blends heading with a 0.9 s smoothstep and then walks along the sidewalk toward or away from the camera.

| Name | Bucket | Mechanism |
| --- | --- | --- |
| `safe_walk` | 40 % | Background only. |
| `empty_street` | 40 % | No background peds/cars; almost no clutter. |
| `oncoming_pedestrian` | 40 % | Opposite sidewalk, walking speed, CPA \(\approx 2\|L\|\). |
| `parallel_pedestrian` | 40 % | Same direction, ~0.8 m toward the curb. |
| `cyclist_same_way` | 40 % | Bicycle in the far lane, same way. |
| `car_pass_far` / `car_approaching` | 40 % | Oncoming car, far vs near lane (CPA stays large). |
| `distant_jaywalk` | 40 % | Person crossing ~16 m ahead. |
| `pothole_offset` / `parked_car_opposite` | 40 % | Static miss. |
| `near_miss_pass` | 30 % | Oncoming ped on the sidewalk, \(D_{\mathrm{cpa}}\approx 1.0\) m, walking speed. |
| `jaywalker` | 30 % critical | Walks in from one FOV edge and **all the way out the other** at \(1.0\)–\(1.35\) m/s. |
| `jaywalker_offset`, `jaywalker_from_left/right` | 30 % | Same through-cut, timed as a graze. |
| `jaywalker_turn_toward` | critical | Side entry, then a ~0.9 s smoothstep turn onto the gait **toward** the camera. |
| `jaywalker_turn_away` | 30 % | Same entry, then turn onto the sidewalk **with** the ego (ahead). |
| `cyclist_near_miss` / `car_near_miss_lane` / `cube_near_miss` | 30 % | Oncoming, CPA 1.0 m. |
| `car_cross_front` | 30 % | Car rolls left↔right across the ribbon in front of you, CPA 1.0 m. |
| `pothole_near` | 30 % | Pothole ~0.85 m off the gait line. |
| `sudden_stop` | critical | Ped 3 m ahead, same speed, `stop_t=1.6` s. |
| `swerve_vehicle` / `car_cut_in` | critical | Oncoming near-lane car, then Frenet drift onto the sidewalk. |
| `pothole_on_path` | critical | Pothole on the gait line. |
| `cube_head_on`, `cube_from_left/right` | critical | Generic cube, along-track or lateral. |
| `cyclist_head_on` | critical | Bicycle on the gait line, toward camera. |
| `head_level_projectile` | critical | Eye-height `threat_cube` along the street (~3.6 m/s, \(\tau=1.8\) s). |
| `car_cross_critical` | critical | L→R car, CPA 0.12 m. |
| `cyclist_weaving` | 30 % | Oncoming bicycle, phased sine weave across the gait. |
| `car_erratic_swerve` | critical | Oncoming car, violent sine weave onto the pavement. |
| `car_runs_off_road` | critical | Vehicle leaves the carriageway and mounts the walker's side. |
| `child_darting` | critical | Child anthropometry (~1.1–1.3 m, larger head fraction), 1.9–3.1 m/s lateral dart. |

`pick_scenarios`: `auto` draws 40 / 30 / 30 then a name from that pool. A comma- or plus-separated list (or repeated `--scenario`) resolves each token and injects **all** of them into one episode. Aliases: `safe`→`safe_walk`, `near_miss`→`near_miss_pass`, `critical`→random from the critical pool, `jaywalk`/`jaywalker`→`jaywalker`, `turn_toward` / `turn_away`, `empty`→`empty_street`, `projectile`→`head_level_projectile`, plus compound shorts `car`→`car_approaching`, `pothole`→`pothole_on_path`, `person`→`oncoming_pedestrian`, `cyclist`→`cyclist_same_way`, `cube`→`cube_near_miss`, `parked`→`parked_car_opposite`, `cut_in`→`car_cut_in`. See §6.5.1.

Horizontal FOV is drawn per episode from `camera.hfov_deg_range` (50°–90°) unless `--hfov` / `--lens-mm` / `--no-random-fov` locks it. Written to `episode.json` and every frame’s `camera_data`.

### 6.5.1 Compound scenarios

One episode can run several injectors at once, e.g. a jaywalker, an oncoming car, and a pothole sharing the same street.

**CLI** (all equivalent):

```bash
./run.sh --episodes 1 --scenario jaywalker,car,pothole --media both
./run.sh --episodes 1 --scenarios jaywalker+car_approaching+pothole_on_path
./run.sh --episodes 1 --scenario jaywalker --scenario car --scenario pothole
```

`--scenario` is repeatable (`action=append`). Tokens split on `,`, `+`, or whitespace. `--scenarios` is joined with `--scenario` if both are set. `--scenario auto` is unchanged (one random injector). A lone `auto` inside a list is itself a random draw (`pothole,auto` = pothole plus one extra event).

**Architecture.** `pick_scenarios` returns the canonical name list (user order). `prepare_scenario(names)` unions geometry flags. `inject_scenarios` sorts a **copy** for spawn order (static hazards → along-track movers → lateral crossers) but the folder slug and `episode.json` keep user order: `jaywalker__car_approaching__pothole_on_path`.

`ComposeSession` (`scenario_compose.py`) is the occupancy board. It is created once per episode, after the camera rig exists and **before** the two-phase sim/render loop. It does not add per-frame work.

**Spatial math.** Every actor is a Frenet capsule — an axis-aligned rectangle in \((s,\mathrm{lateral})\) with class half-extents (person \(0.70\times 0.42\) m, vehicle \(2.45\times 1.05\) m, pothole \(0.90\times 0.55\) m, …). The capsule moves with the same piecewise rates as `Actor.update`: constant \(\mathrm{d}s/\mathrm{d}t\), lateral approach to `lat_target`, optional `swerve_t` delay, optional turn onto `post_speed`. Overlap is sampled at \(\Delta t=0.12\) s over a 5 s horizon (about 42 poses; a few dozen capsules; microseconds).

Placement, in order:

1. Seed capsules from background traffic already in the camera band \([s_0-2, s_{\mathrm{hi}}]\), so an injected car is not born inside a Poisson vehicle. Background **and ground hazards** only block the **spawn cell** (\(t=0\)): a far-lane cruiser must not shove a jaywalker 15 m down the road, and walking past a pothole is allowed. Injected *dynamic* actors still test full 5 s tubes against each other.
2. Stagger groups: extra crossers \(+4\) m of depth and alternate `from_left`; extra along-track actors \(+3.2\) m (or the equivalent \(\tau\)); extra potholes \(+2.4\) m.
3. Lane policy: a generic `car_approaching` in a scene that also has a through-crosser is sent to the **far** lane. Cut-in / swerve / `car_near_miss_lane` keep the near lane (their point) and rely on occupancy.
4. `ComposeSession.reserve` tries, at inject time: requested \((s,\mathrm{lat})\), then a heading or lane flip at that \(s\), then \(+2.6\) m along-track nudges, then a short \(-s\) search. The first conflict-free tube is committed. `_bind` (and `_inject_pothole`) call this after corridor confine; world \(Z\) is still `origin_z + corridor.ground_z(lateral)`.
5. Crossers are clamped so depth \(\le 16\) m, which stays inside the building gap.

There is **no** per-frame steering. That would fight the gait integrator and the pose-replay render. Separation is a setup-time reservation so predicted tubes do not share a cell; the existing corridor clamp still prevents leaving the street.

**Why this avoids clipping.** A through-jaywalker occupies every lane at one \(s\). An oncoming car will pass that \(s\). The only safe degree of freedom is *when* they meet (nudge \(s\) / \(\tau\)) and *where laterally* the jaywalker is at that instant (flip entry side, or put the car in the far lane so the meeting happens while the person is still on the near sidewalk). Static potholes claim the gait cell first so a `sudden_stop` ped is pushed further along the walk. Two jaywalkers get opposite `from_left` and \(4\) m of depth so they do not share a crossing line.

Folder name uses `__` between components (filesystem-safe). Manifest:

```json
{
  "scenario": "jaywalker__car_approaching__pothole_on_path",
  "scenarios": ["jaywalker", "car_approaching", "pothole_on_path"],
  "scenario_requested": "jaywalker,car,pothole"
}
```

Knobs live under `config["scenarios"]["compose"]` (`cross_stride_m`, `nudge_s_m`, `dt_sample`, `look_ahead_m`, …). `python scenario_compose.py` runs the occupancy self-test (no Blender).

### 6.6 `humanoid.py`

Replaces the original box-person. Character faces **+Y**, +Z up.

**Anthropometry** (Drillis & Contini / NASA-STD-3000) as fractions of stature \(H\sim U(1.58, 1.84)\):

| Segment | Fraction of \(H\) |
| --- | --- |
| Head (radius) | 0.068 |
| Neck | 0.048 |
| Torso height | 0.300 |
| Pelvis height | 0.100 |
| Thigh / shank | 0.245 / 0.246 |
| Foot length | 0.152 |
| Upper arm / forearm | 0.186 / 0.146 |
| Biacromial / bi-hip | 0.259 / 0.191 |

Body-type scales: width \(U(0.88,1.16)\), depth \(U(0.90,1.12)\), hip \(U(0.92,1.18)\).

**Meshes**

- **Barr superquadric** (head, pelvis, hands), \(u\in[0,2\pi]\), \(v\in[-\pi/2,\pi/2]\):

\[
x=a\,|\cos v|^{e_1}|\cos u|^{e_2}\mathrm{sgn},\quad
y=b\,|\cos v|^{e_1}|\sin u|^{e_2}\mathrm{sgn},\quad
z=c\,|\sin v|^{e_1}\mathrm{sgn}
\]

  \((e_1,e_2)=(1,1)\) is a sphere; \(\sim(0.55,0.65)\) is a rounded box.

- **Torso loft.** Elliptical rings, half-width / half-depth from smoothstepped stations (waist → chest → axilla → neck). Sagittal offset \(H(0.014\sin(\pi u)-0.009\sin(2\pi u))\) mimics lumbar lordosis / thoracic kyphosis. Back half of the ellipse is slightly flattened.
- **Capsule limbs** along \(-\mathrm{Z}\), origin at the proximal joint. Radius \(\mathrm{lerp}(r_0,r_1,t)\cdot(1+b\sin(\pi t))\), distal hemisphere. Belly \(b\sim 0.06\)–\(0.12\).
- **Neck** is a short +Z column (a \(-\mathrm{Z}\) capsule would grow into the chest).
- **Foot** is a tapered shoe loft, heel behind the ankle, toes +Y.
- Deltoid spheres, ears, hair cap. All parts `use_smooth`.

**Walk cycle (Winter 1991, reduced).** Step length \(\ell\approx 0.41 H\) (Grieve & Gear). \(f=|v|/\ell\), \(\varphi=2\pi f t+\varphi_0\).

\[
\begin{aligned}
\theta_{\mathrm{hip},L/R}&=A_{\mathrm{hip}}\sin(\varphi+\{0,\pi\})\\
\theta_{\mathrm{knee},L/R}&=-A_{\mathrm{knee}}[\max(0,\sin(\varphi+\alpha+\{0,\pi\}))]^{p}\\
\theta_{\mathrm{sh},L/R}&=-A_{\mathrm{arm}}\sin(\varphi+\{0,\pi\})\\
\theta_{\mathrm{ank},L/R}&=A_{\mathrm{ank}}\sin(\varphi+\beta+\{0,\pi\})
\end{aligned}
\]

Knees never hyperextend (half-wave). Arms antiphase with the ipsilateral hip. After `look_along` (which owns heading Z):

- pelvic pitch \(A\sin 2\varphi\), list \(A\sin\varphi\), yaw added on Z, bob \(A_{\mathrm{bob}}|\sin\varphi|\) on location Z
- thorax yaw \(=-0.65\times\) pelvic yaw

Idle / stopped: zeros the pose. `Actor.update` calls `WalkRig.apply` every frame, including frame 0.

Root object is the **pelvis**. 2-D boxes union all mesh children.

### 6.7 `materials.py`

Procedural PBR, **no image textures**. Sample **Object** coordinates (local metres) so a UV-less building cube still gets storeys.

**Blender 4+/5 Mix node trap.** `ShaderNodeMix` has stacked sockets that share a name: `A` is a float *and* a colour. `node.inputs["A"]` is the **float**. `ShaderNodeTexNoise` outputs **`Factor`**, not `Fac`. The first realism pass wrote every layered mix to the wrong socket, so asphalt / cloth / tiles collapsed to a constant albedo. Helpers `_input` / `_output` / `_link` / `_link_to` pick by `(name, type, enabled)`.

Math nodes have two sockets named `Value`. Night-window occupancy and roughness use explicit index linking (`_math_in(node, 1)`).

| Factory | Idea |
| --- | --- |
| `make_asphalt` | Voronoi stones + 1/f noise. Roughness anticorrelated with albedo (oil-dark cells are glossier). Bump from noise. |
| `make_concrete_tiles` | Brick lattice, period 0.40 m, dark grout, invert-Fac bump. |
| `make_grass` | Two-tone noise + strong micro-bump. |
| `make_facade` | Rectangular window pulse train on YZ (±X faces) and XZ (±Y faces), blended by \(\lvert N_x\rvert/(\lvert N_x\rvert+\lvert N_y\rvert)\) in **object** normals (world normals break after `look_along`). Night occupancy is a **per-cell hash** \(\mathrm{fract}(\sin(i\cdot 12.9898+j\cdot 78.233+s)\cdot 43758.5453)\), **not** smooth Noise (smooth noise painted leopard-print emission). Emission goes into Principled `Emission Color/Strength` (no Mix Shader). |
| `make_car_paint` | Metallic + clear coat + slight flake noise. |
| `make_glass` | Transmission 0.82, alpha 0.62, blend method HASH/BLEND if present. |
| `make_skin` | SSS (radius 1.0, 0.35, 0.18) + fine noise + tiny bump. |
| `make_cloth` | 1/f weave + sheen. |
| `make_patterned_cloth` | Chaos print: stripe / checker / camo / dots / magic / noise. Two independent family colours. |
| `make_chaos_car_paint` | Factory metallic, matte wrap, chrome, or neon + flake noise. |
| `make_water` | Near-mirror roughness ramp + mild transmission. EEVEE SSR puddles. |
| `make_foliage` / `make_bark` / `make_canopy_gobo` | Tree canopy / trunk / overhead dappled-shadow sheet. |
| `make_emissive` / `make_roof` / `make_rubber` / `make_simple` | As named. |

**Appearance chaos.** `chaos ∈ [0,1]` interpolates between the original tame palette and full anarchy. Families: natural, saturated, neon, chrome, matte, pastel, dark. `apply_surface_chaos` randomises metallic, roughness, coat, sheen, IOR, and (neon) emission. CLI `--chaos` locks the dial; otherwise each episode draws from `material_chaos` (0.20–1.00).

### 6.8 `main.py`

CLI (after `--`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--episodes` | 4 | How many episodes this process |
| `--start-episode` | *next free* | First episode index. Omitted ⇒ one past the highest `episode_*` already on disk. |
| `--scenario` | `auto` | Injector name, alias, or compound (`jaywalker,car,pothole`). Repeatable. `--list-scenarios` prints the 36 names plus shorts. |
| `--scenarios` | *unset* | Same as `--scenario`; comma or plus separated. Joined with `--scenario` if both are set. |
| `--hfov` | *unset* | Lock horizontal FOV (degrees) for every episode. |
| `--lens-mm` | *unset* | Lock focal length; ignored if `--hfov` is set. |
| `--no-random-fov` | off | Use config `lens_mm` instead of drawing from `hfov_deg_range`. |
| `--output` | `config.output.root` (`./output`) | Dataset root |
| `--seed` | 42 | Master RNG |
| `--frames` | 0 | Override `frames_per_episode` (0 = use config) |
| `--media` | config (`both`) | `frames` \| `video` \| `both` |
| `--no-rgb` | off | Delete `rgb/` after `preview.mp4` (and overlay) succeed. Same keep-policy as `--media video`. |
| `--no-render` | off | JSON only |
| `--no-annotations` | off | Skip `annotations/annotations.json`. Spatial matrices still written. |
| `--no-occlusion` | off | Skip raycast flag |
| `--threat-grid` | 3 | `k` for the episode `k×k` spatial matrices (`spatial_annotations/`). Always written. CLI only. |
| `--spatial-overlay` | off | Write `spatial_overlay.mp4`: hazy RGB + k×k heat (blue=0, red=1). Needs a render. |
| `--plan` | *unset* | JSON from `gen_dataset.py`: per-episode `scenario` list. Overrides `--scenario`. |
| `--biome` | `auto` | `street` \| `avenue` \| `park` \| `plaza` \| `auto`. |
| `--ego-mode` / `--ego` | `auto` | `walk` \| `diagonal_cross` \| `erratic` \| `seated` \| `auto`. |
| `--chaos` | *unset* | Lock appearance chaos in \([0,1]\). Unset ⇒ per-episode draw from `material_chaos`. |
| `--no-trees` | off | Skip procedural trees and grass clumps. |

Per-frame JSON schema (Part 6.1):

```json
{
  "frame_id": "000000",
  "timestamp": 0.0,
  "camera_data": {
    "world_position": [x, y_up, z_fwd],
    "velocity": [..],
    "pitch_yaw_roll": [pitch, yaw, roll],
    "lens_mm": 24.0,
    "hfov_deg": 73.74,
    "sensor_width_mm": 36.0
  },
  "environment": { "lighting": "dusk", "weather": "clear" },
  "objects": [
    {
      "instance_id": "person_000",
      "class_name": "person",
      "threat_label": "SAFE_DYNAMIC",
      "kinematics": {
        "world_position": [..],
        "velocity": [..],
        "relative_velocity": [..],
        "distance": 3.11,
        "ttc": 9999.0,
        "cpa": 3.11
      },
      "bounding_box_2d": { "xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0 },
      "flags": { "truncated": false, "occluded": false }
    }
  ]
}
```

`class_name` values: `person`, `vehicle`, `bicycle`, `threat_cube`, `pothole`, `trash_can`, `scooter`, `barricade`, `tree_branch`, `ac_unit`, `sign`, `truck_door`, `projectile`.

Annotation skips objects with range \(>40\) m or no on-screen box. Velocity prefers the integrator (`actor.velocity`) over finite difference, except when we need FD for a hold that just started.

Video: `ffmpeg -framerate FPS -i %06d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart`. VSE fallback uses `sequences` (4.x) or `strips` (5.x).

A failed episode is logged and the process continues; exit code is 2 if any episode failed.

### 6.9 `run.sh`

```bash
BLENDER="${BLENDER:-blender}"
exec "$BLENDER" --background --python "$ROOT/main.py" -- "$@"
```

---

## 7. Scene contents (what you actually see)

| Element | Construction |
| --- | --- |
| Road | Ribbon, asphalt PBR, Object-metre Voronoi |
| Sidewalks | Ribbon at z = 0.12 m, paver brick |
| Curb | Narrow ribbon, simple concrete |
| Grass verge | 4.5 m, two-tone grass |
| Lane dashes | 3 m on / 3 m off |
| Buildings | Depth × width × height boxes, window lattice, roof cap |
| Streetlamps | Pole + arm + emissive bulb + downward spot |
| Vehicles | Superquadric hull + cabin glass + bumpers + 4 cylinders (axis +X) |
| Pedestrians | Full humanoid + WalkRig |
| Ground hazards | Still mostly boxes: pothole, trash can, scooter, barricade |
| Head hazards | Boxes: branch, AC unit, sign, truck door, at 1.2–1.8 m AGL |
| Projectile | 0.28 m cube |

Hazards were left boxy on purpose (the user ask for realism targeted **people, light, and street materials**). They are still annotatable and participate in TTC/CPA.

---

## 8. Maths pocketbook (everything used in anger)

### 8.1 Camera

\[
Z(t)=h+A\sin(2\pi ft),\qquad
v_Z(t)=A\,2\pi f\cos(2\pi ft)
\]

Perlin fade \(6t^5-15t^4+10t^3\). fBm geometric normalisation so amplitude stays ~\([-1,1]\).

### 8.2 TTC / CPA / intercept

See section 6.2. Intercept with offset \(\vec{o}\) is the unique constant \(\vec{V}_{\mathrm{obj}}\) that makes \(\vec{P}_{\mathrm{obj}}(t^{\star})-\vec{P}_{\mathrm{cam}}(t^{\star})=\vec{o}\).

### 8.3 Projection

OpenGL symmetric frustum, `sensor_fit=HORIZONTAL`. Default 24 mm on a 36 mm sensor ⇒ HFOV \(\approx 2\arctan(18/24)\approx 73.7^\circ\). Per episode the pipeline draws HFOV from \(U(50^\circ,90^\circ)\) and sets `cam.lens = (sensor/2)/\tan(\mathrm{HFOV}/2)`, unless CLI locks it. Both `lens_mm` and `hfov_deg` are written on every frame.

### 8.4 Human gait

\(f=|v|/(0.41 H)\), half-wave knee, antiphase arms, pelvic 2-harmonic, trunk counter-rotation \(-0.65\).

### 8.5 Superquadric / loft

Barr 1981; torso stations are cubic Hermite / smoothstep between anthropometric keys.

### 8.6 Window lattice

\[
\mathrm{win}_Y=\mathbf{1}\{0.22<\{Y/1.70\}<0.80\},\quad
\mathrm{win}_Z=\mathbf{1}\{0.30<\{Z/3.20\}<0.82\}
\]

Front \(=\mathrm{win}_Y\cdot\mathrm{win}_Z\), side \(=\mathrm{win}_X\cdot\mathrm{win}_Z\), blend by object-space \(|N_x|\).

Night hash: classical `fract(sin(dot(cell, vec2(12.9898, 78.233))+seed)*43758.5453)`.

### 8.7 Sky gradient

\(\mu=\mathrm{Incoming}\cdot(0,0,1)\). ColorRamp on \(\mu\) is a discrete Hosek limb \(L_{\mathrm{zenith}}+(L_{\mathrm{horizon}}-L_{\mathrm{zenith}})(1-\mu)^p\). Mixed with `TexSky` (factor 0.55 night, 0.22 dawn/dusk, 0.12 noon).

### 8.8 Poisson / Bézier

Bridson cell size \(r/\sqrt{2}\). Cubic Bézier \(\sum\binom{3}{k}(1-u)^{3-k}u^k P_k\).

---

## 9. Output on disk

```text
<output>/
  dataset_summary.json
  episode_0000_safe_walk/
    episode.json
    preview.mp4
    rgb/000000.png …
    annotations/annotations.json
    spatial_annotations/spatial_annotations.json
  episode_0001_sudden_stop/
    …
```

Folder name is `episode_{id:04d}_{scenario}`. A later `--episodes 4 --scenario auto` run continues at the next free id. `dataset_summary.json` is rebuilt by scanning every `episode_*/episode.json`.

Pixel format: 1920×1080 RGB8 PNG, AgX, film **not** transparent.

The rest of this section is the field-by-field contract: what is written, how it is measured, and what a downstream tactile model is supposed to do with it.

### 9.1 What an episode is

One episode is one continuous sidewalk walk: a fixed path, a fixed lighting/weather draw, a fixed set of background actors, plus **one or more injected** scenarios (`jaywalker`, `sudden_stop`, or a compound such as `jaywalker+car_approaching+pothole_on_path`). Default length is **150 frames at 30 Hz = 5.0 s**, unless `--frames` or the remaining spline is shorter.

There are four products:

| Product | When | Role |
| --- | --- | --- |
| `annotations/annotations.json` | default (skip with `--no-annotations`) | One file: every frame’s object boxes / TTC / labels. |
| `spatial_annotations/spatial_annotations.json` | **always** | One file: every frame’s `k×k` float matrix in `[0,1]`. `k` from `--threat-grid` (default 3). |
| `spatial_overlay.mp4` | `--spatial-overlay` | Hazy RGB with the k×k grid tinted blue→red. |
| `rgb/XXXXXX.png` | `--media frames` or `both` | Egocentric RGB, 1:1 with that JSON. Deleted after mux if `--media video` or `--no-rgb`. |
| `preview.mp4` | `--media video` or `both` | Same PNG sequence muxed at `fps` (CRF 18, yuv420p). Convenience only — do not train on the video clock; use the JSON `timestamp`. |
| `episode.json` | always | Manifest for this walk (scenario, speed, label counts). |

JSON `frame_id` `"000007"` is the same instant as `rgb/000007.png`. If video mux fails, PNGs are kept even when `--media video` or `--no-rgb`.

---

### 9.2 Coordinate convention in every JSON vector

Internal simulation is Blender **Z-up** `(X, Y_forward, Z_up)`. Every 3-D vector written to disk is converted with `blender_zup_to_yup`:

```text
JSON (X, Y, Z)  =  (Blender_X, Blender_Z, Blender_Y)
```

So in the files:

| JSON axis | Physical meaning |
| --- | --- |
| **X** | right (same as Blender X) |
| **Y** | **up / height above the pavement** |
| **Z** | **forward along a +Y Blender path** |

Units are metres and metres/second. Angles are **radians**. Time is seconds. Pixel boxes use the image convention: origin **top-left**, `x` right, `y` down, on a `1920×1080` canvas.

A walker standing at sidewalk \(s=3\) m, eye height 1.6 m above a 0.12 m curb, therefore starts near `world_position ≈ [lateral, 1.72, 3.0]` on a straight \(+\mathrm{Y}\) road.

---

### 9.3 Object annotations (`annotations/annotations.json`)

Written by `main.build_frame_record` after `CameraRig.update` + `WorldGenerator.update` + a depsgraph refresh. **One file per episode** (same idea as the spatial matrix). `--no-annotations` skips this file; spatial matrices are still written. Objects that are too far or fully off-screen are omitted from that frame (they are not listed with empty boxes).

```json
{
  "frames": [
    {"frame_id": "000000", "timestamp": 0.0, "camera_data": {}, "environment": {}, "objects": []},
    {"frame_id": "000001", "timestamp": 0.0333, "camera_data": {}, "environment": {}, "objects": []}
  ]
}
```

#### Top-level fields

| Field | Type | How measured | What it tells |
| --- | --- | --- | --- |
| `frame_id` | string | `f"{i:06d}"` | Join key to `rgb/{frame_id}.png`. |
| `timestamp` | float | \(t = i / \mathrm{fps}\) | Seconds from the start of **this** episode. Not a wall-clock. |
| `camera_data` | object | `CameraRig` snapshot | Ego pose / velocity used for TTC (see 9.4). |
| `environment` | object | Domain-randomization draw at `build()` | Lighting and weather **for the whole episode** (copied onto every frame). |
| `objects` | array | One entry per annotatable actor that passed the filters | The supervision for that frame. |

`environment.lighting` ∈ `{dawn, noon, dusk, night, harsh_glare, overcast}`.  
`environment.weather` ∈ `{clear, light_fog, heavy_smog}`.  
`environment.biome` ∈ `{street, avenue, park, plaza}`.  
`environment.dappled` is the canopy-gobo flag. `environment.chaos` is the appearance dial.  
These are **not** inferred from the image; they are the knobs that built the sky / sun. Use them for domain-shift analysis, not as a network target unless you want an auxiliary weather head.

#### Filters (why an actor may be missing)

An actor is **dropped from this frame** (not written) if:

1. `annotatable` is false (puddles, the seated bench — visual distractors, not collisions).
2. Range from camera to the **threat point** (9.5) is \(> 40\) m (`annotation.max_distance`).
3. `project_object` returns `None`: every AABB corner is behind the camera, or the projected box misses the image entirely.

So `objects` is “what the vest could reasonably see right now”, not the full world census. Background cars 50 m ahead will vanish from the JSON even though they still exist in the sim.

---

### 9.4 `camera_data` — the ego walker

| Field | Units | How measured | What it tells |
| --- | --- | --- | --- |
| `world_position` | m, Y-up | Eye point: road Frenet origin at \(s(t)\) from the arc table, offset by \(L(t)\), then \(Y =\) curb \(+ h + A\sin(2\pi 1.8 t)\). \(h=1.6\) standing, \(\sim 1.1\) seated. **No** Perlin translation. | Where the camera is in the spec frame. \(Y\) is height; it bobs only while walking. |
| `velocity` | m/s, Y-up | Finite difference of the eye point (frame 0: central difference of `predict_position`). | Instantaneous ego velocity **including gait \(dY/dt\)**. Typical walk is \(\sim 1.0\)–\(1.4\) m/s along \(+Z_{\mathrm{JSON}}\). Seated / halt ⇒ \(\approx 0\). |
| `pitch_yaw_roll` | rad | Perlin fBm on each axis **plus** the tiny gait pitch bob. These are **camera-local jitter angles**, not the world heading of the sidewalk. | How much the head is nodding / scanning / rolling **relative to the Frenet gaze**. |
| `lens_mm` / `hfov_deg` / `sensor_width_mm` | mm / deg / mm | Episode FOV draw or CLI lock. `lens = (sensor/2) / tan(HFOV/2)`. | Which frustum this PNG was rendered with. |
| `ego_mode` | string | Profile drawn for the episode. | `walk` / `diagonal_cross` / `erratic` / `seated`. |
| `ego_speed` | m/s | `CameraRig.speed_at(t)`. | Instantaneous ground speed. **0** while seated or hesitating. |

`pitch_yaw_roll` is **not** a global IMU in the JSON Y-up frame. If you need a 4×4 camera matrix, rebuild it from `world_position` + the sidewalk tangent (not stored) + these angles — or treat the PNG as the only appearance signal and use `world_position` / `velocity` only for kinematics losses.

---

### 9.5 How object kinematics are measured (read this before trusting `distance`)

For each actor we keep **two different 3-D points**:

1. **Root origin** — `obj.matrix_world.translation`. For a person this is the **pelvis**; for a car the body centre; for a pothole the box centre. This is what is written as `kinematics.world_position`.
2. **Threat point** — `projection.threat_point`. Used for `distance`, `ttc`, `cpa` only.

Threat point, in Blender Z-up:

- `threat_mode = "volume"` (people, cars, branches, projectile): XY of the world AABB centre (union of the root **and all mesh children**), Z clamped into that AABB, preferring the camera height. A 1.5 m car roof vs a 1.6 m camera leaves a 0.1 m vertical residual instead of a fake 1.6 m miss.
- `threat_mode = "footprint"` (potholes): XY of the AABB centre, **Z = camera height**, so the hole occupies the walker’s vertical column. A pothole you will step in is a collision even though the mesh sits on the pavement.

Then, in the **same Blender frame** (conversion to Y-up happens only when writing vectors):

\[
\vec{P}_{\mathrm{rel}} = \vec{P}_{\mathrm{threat}} - \vec{P}_{\mathrm{cam}},\qquad
\vec{V}_{\mathrm{rel}} = \vec{V}_{\mathrm{obj}} - \vec{V}_{\mathrm{cam}}
\]

\(\vec{V}_{\mathrm{obj}}\) is the integrator velocity (`Actor.velocity`) when the actor is moving, static, or has just stopped; otherwise a one-frame finite difference. \(\vec{V}_{\mathrm{cam}}\) is the ego velocity from 9.4.

**Constant-velocity point model** (`threat_math.py`):

- Converging iff \(\vec{P}_{\mathrm{rel}}\cdot\vec{V}_{\mathrm{rel}} < 0\) (range is shrinking).
- If \(\|\vec{V}_{\mathrm{rel}}\| < 10^{-4}\) m/s or not converging: TTC is undefined.

\[
\mathrm{TTC}
= -\frac{\vec{P}_{\mathrm{rel}}\cdot\vec{V}_{\mathrm{rel}}}{\|\vec{V}_{\mathrm{rel}}\|^2}
\qquad
D_{\mathrm{cpa}}
= \|\vec{P}_{\mathrm{rel}} + \mathrm{TTC}\,\vec{V}_{\mathrm{rel}}\|
\]

Undefined TTC is written as **`9999.0`** (must remain a JSON number). Undefined CPA falls back to current range.

**Implication:** `distance` is \(\|\vec{P}_{\mathrm{threat}}-\vec{P}_{\mathrm{cam}}\|\), **not** \(\|\texttt{world\_position}-\texttt{camera\_data.world\_position}\|\). For a person those can differ by ~0.6–0.9 m (pelvis vs chest-height threat point). Do not recompute TTC from the written `world_position` and expect the same label.

#### `objects[i]` fields

| Field | How measured | What it tells |
| --- | --- | --- |
| `instance_id` | `{class}_{nnn}` from a per-episode counter (`person_000`, `vehicle_001`, …) | Stable id **inside this episode**. The same physical walker keeps the id across frames. Ids restart every episode. |
| `class_name` | Spawn type | Semantic class. See table below. This is **what it is**, not how dangerous it is. |
| `threat_label` | `classify_threat(TTC, CPA, speed, range)` | **What the vest should do.** The primary training target. |
| `kinematics.world_position` | Root origin → Y-up | Where the actor’s root sits. Useful for 3-D debugging / secondary losses. |
| `kinematics.velocity` | \(\vec{V}_{\mathrm{obj}}\) → Y-up | Absolute velocity in the world. Zero ⇒ static (or just stopped). |
| `kinematics.relative_velocity` | \(\vec{V}_{\mathrm{obj}}-\vec{V}_{\mathrm{cam}}\) → Y-up | How the object is moving **as seen from the walker**. Closing along \(-\,Z_{\mathrm{JSON}}\) is “coming at you”. |
| `kinematics.distance` | \(\|\vec{P}_{\mathrm{rel}}\|\) at the threat point | Current range that entered the TTC formula. |
| `kinematics.ttc` | seconds, or `9999.0` | Time until closest approach **if both keep this velocity**. Not “time until impact” unless CPA ≈ 0. |
| `kinematics.cpa` | metres | Miss distance at that future instant. 0 = the two threat-points coincide. 6 m = they pass a lane apart. |
| `bounding_box_2d` | Projected AABB (9.7) | Pixel supervision for a detector head. |
| `flags.truncated` | Box / corners vs the image | The object is cut by the frame edge (or some AABB corners are behind the camera). |
| `flags.occluded` | Raycast (9.8) | Something closer sits on the line of sight to the centroid. |

#### `class_name` values

| `class_name` | Typical threat_mode | What it is |
| --- | --- | --- |
| `person` | volume | Articulated pedestrian (background or injected). |
| `vehicle` | volume | Superquadric car. |
| `pothole` | footprint | Hole on the gait line. |
| `trash_can` | volume | Sidewalk clutter. |
| `scooter` | volume | Sidewalk clutter. |
| `barricade` | volume | Sidewalk clutter. |
| `tree_branch` | volume | Head-height (1.2–1.8 m). |
| `ac_unit` | volume | Head-height. |
| `sign` | volume | Head-height. |
| `truck_door` | volume | Head-height swing. |
| `projectile` | volume | Silent eye-height object (`head_level_projectile`). |

---

### 9.6 `threat_label` — the tactile taxonomy

Computed **after** TTC/CPA, **not** from the pixels. The image can lie (a car behind a wall still has a TTC); the label is kinematic truth for a warning device.

Priority (a real hit can never fall through to SAFE):

| Label | Rule (defaults) | What it tells the vest |
| --- | --- | --- |
| `CRITICAL_THREAT` | converging **and** \(TTC < 2.5\) s **and** \(D_{\mathrm{cpa}} < 0.5\) m | On a collision course and close in time. Strong / immediate cue. |
| `NEAR_MISS` | converging **and** \(TTC < 4.0\) s **and** \(0.5 \le D_{\mathrm{cpa}} \le 1.5\) m | Will pass within a shoulder-width. Graded / early cue. |
| `SAFE_STATIC` | \(\|V_{\mathrm{obj}}\| \le 0.05\) m/s **and** (range \(> 5\) m **or** CPA \(> 1.5\) m **or** not converging) | Furniture / a far pole. No cue. **Exception:** a static object you are walking straight into (pothole, low branch) still converges with \(V_{\mathrm{rel}}=-V_{\mathrm{cam}}\) and can be CRITICAL / NEAR_MISS. |
| `SAFE_DYNAMIC` | moving, and not in the two threat bins | Parallel traffic, receding walkers, a car that will miss by \(> 1.5\) m. No cue, even if it is visually large. |

Worked readings from a real frame (`output/episode_0000/annotations/000000.json`):

- `vehicle_000`: TTC \(1.93\) s but CPA \(6.31\) m → **SAFE_DYNAMIC**. Closing fast, but a full lane away — do not buzz.
- `vehicle_001`: TTC `9999.0`, CPA = distance \(11.6\) m → **SAFE_DYNAMIC**. Relative \(+Z\) is large: they are pulling away / crossing out of the cone of approach.
- `truck_door_000`: static, TTC \(12.4\) s, CPA \(6.14\) m, range \(15.4\) m → **SAFE_STATIC**. Head-height clutter you will walk past, not into.
- `person_001`: TTC \(4.60\) s, CPA \(2.73\) m → **SAFE_DYNAMIC**. Closing, but late and wide of the 1.5 m near-miss band.

A `jaywalker` injector places the person already on the carriageway so they cover ~3 m of lateral travel at \(1.0\)–\(1.35\) m/s (\(TTC\approx 3.2\) s, \(D_{\mathrm{cpa}}\approx 0.12\) m) without leaving the street ribbon. `sudden_stop` is SAFE_DYNAMIC while the lead walker matches your speed, then TTC collapses when they freeze.

`label_histogram` in `episode.json` counts **(object × frame)** labels, not unique instances. 150 frames × 2 people can produce 300 `SAFE_DYNAMIC` ticks.

---

### 9.7 `bounding_box_2d` — how the box is measured

Not a tight silhouette and not a semantic mask. Pipeline (`projection.py`):

1. Take the 8 AABB corners of every **MESH** in the actor hierarchy (pelvis + torso + limbs + head, or body + cabin + wheels). Parenting is evaluated on the depsgraph, so a mid-stride leg is included.
2. Transform to camera space. Drop corners with \(Z_{\mathrm{cam}} \ge 0\) (behind the lens). If none survive → object omitted.
3. Project with `camera.calc_matrix_camera` (the matrix EEVEE itself uses), fallback to the analytic 24 mm / 36 mm frustum.
4. NDC → pixels, origin top-left:

\[
u = (n_x + 1)\,W/2,\qquad
v = (1 - n_y)\,H/2
\]

5. Box = axis-aligned min/max of surviving corners, clamped to \([0,1920]\times[0,1080]\). Integers: `xmin/ymin` floored, `xmax/ymax` ceiled.

So the box is a **conservative 3-D AABB projection**. It is slightly loose around a walking figure (axis-aligned world corners, not a fitted 2-D hull). That is intentional: cheap, stable, and it covers articulated children.

`truncated = true` if any of: `xmin<0`, `ymin<0`, `xmax>W`, `ymax>H`, or fewer than 8 corners were in front of the camera. A car half out of the right edge (`xmax=1920`) is truncated.

If you train a detector, this is the box target. If you train **only** a threat classifier on a crop, still honour `truncated` / `occluded` — a truncated CRITICAL crop is a different visual.

---

### 9.8 `flags.occluded` — how occlusion is measured

A physics ray (`scene.ray_cast` on the evaluated depsgraph) from the **camera origin** toward the actor’s AABB **centroid**:

- Hit the actor (or a child) first → not occluded.
- Hit something else whose distance is at least `occlusion_epsilon` (0.08 m) closer than the centroid → `occluded: true`.
- Ground ribbons (`road_surface`, `sidewalk_*`, `curb_*`, `centerline`, `lamp_pole`) are skipped and the ray is stepped, so a pothole is not “occluded by the sidewalk it sits on”.
- Up to 10 steps. `--no-occlusion` forces `false`.

This is a **centre-ray**, not a pixel-coverage test. A person 90 % hidden behind a pole can still be `occluded: false` if the centroid ray threads the gap, and a person barely covered can be `true`. Treat it as a coarse flag, not as a visibility fraction.

---

### 9.9 `episode.json` — per-walk manifest

| Field | Meaning |
| --- | --- |
| `episode_id` | Numeric id used in the folder name. |
| `dir` | Folder name (`episode_0002_jaywalker`). Present on new writes. |
| `scenario` | Injector that actually ran, or `__`-joined compound slug. |
| `scenarios` | List of canonical injector names (length 1 for a single event). |
| `scenario_requested` | CLI value (`auto`, `jaywalker,car,pothole`, …). |
| `frames` / `fps` | Length of this walk. |
| `walk_speed` | Ego arc-speed drawn for the episode (m/s). **0** when `ego.mode` is `seated`. |
| `sidewalk_lateral` | Signed offset from the road centreline (m). Negative = left side of a +Y path. |
| `biome` | `street` / `avenue` / `park` / `plaza`. |
| `ego` | `{mode, eye_height_m, stationary, sidestep_amp_m, halt_s}`. |
| `camera` | `{lens_mm, hfov_deg, sensor_width_mm}` for this episode. |
| `environment` | Lighting / weather / biome / chaos / dappled for the whole walk. |
| `label_histogram` | Sum of per-frame `threat_label`s. Use it to see whether the injector fired (CRITICAL/NEAR_MISS counts). |
| `render` / `media` | Whether RGB/video were requested. |
| `rgb_dir` | `"rgb"` if PNGs were kept, else `null`. |
| `video` / `video_encoder` | `preview.mp4` and `ffmpeg (...)` or `blender-vse` if mux succeeded. |
| `annotations` | `"annotations/annotations.json"` or `null` if `--no-annotations`. |
| `spatial_annotations_k` / `spatial_annotations` | Grid resolution and path (`spatial_annotations/spatial_annotations.json`). |
| `spatial_overlay` / `spatial_overlay_encoder` | `spatial_overlay.mp4` when `--spatial-overlay` succeeded. |

This file is the index card. It does **not** replace `annotations/annotations.json`.

---

### 9.9a Spatial threat matrix (`spatial_annotations/spatial_annotations.json`)

Always written by `./run.sh` / `main.py` (including `--no-render`). One JSON per episode, not one file per frame. `k` comes **only** from `--threat-grid` (default 3; not a `config.py` key).

```json
{
  "k": 3,
  "frames": [
    {"frame_id": "000000", "matrix": [[0.0, 0.12, 0.0], [0.81, 0.97, 0.41], [0.0, 0.72, 0.0]]},
    {"frame_id": "000001", "matrix": [[0.0, 0.18, 0.0], [0.70, 0.91, 0.33], [0.0, 0.64, 0.0]]}
  ]
}
```

Row 0 is the **top** of the image. Each entry is a float in \([0,1]\). Math lives in `spatial_threat.py`. Ego motion is the **sidewalk tangent × walk speed** (not the jittered eye velocity), so a pothole on the gait does not flicker.

Frenet on the road spline (same \((s,\mathrm{lateral})\) as the injectors): \(s=s_{\mathrm{obj}}-s_{\mathrm{cam}}\), \(\ell=\ell_{\mathrm{obj}}-\ell_{\mathrm{cam}}\), \(R=r_{\mathrm{ego}}+r_{\mathrm{obj}}\). World-XY is the fallback. Then \(W=\max(W_{\mathrm{stop}},W_{\mathrm{path}},W_{\mathrm{cross}},W_{\mathrm{cpa}})\):

- **Stopping volume** \(W_{\mathrm{stop}}\): object disk overlaps the forward rectangle of length \(v_{\mathrm{ego}}T_{\mathrm{react}}+d_{\mathrm{buf}}\) (~2.2 s of walking). A jaywalker 2 m ahead is a hit even if a point-mass CPA says they will have stepped aside.
- **Guaranteed path hit** \(W_{\mathrm{path}}\): they occupy the gait tube now and \(t_{\mathrm{leave}}>t_{\mathrm{arrive}}\) (a static pothole has \(t_{\mathrm{leave}}=\infty\)).
- **Crossing intercept** \(W_{\mathrm{cross}}\): they enter the tube at \(t_{\mathrm{in}}\) and are still at the walker's \(s\).
- **Body CPA** \(W_{\mathrm{cpa}}=\exp(-(d_{\mathrm{clear}}/0.42)^2)\) with \(d_{\mathrm{clear}}=\max(0,D_{\mathrm{cpa}}-R)\).
- \(S=W\,(0.58+0.42\,U)\) plus a mid-band adjacent-lane term. Overlaps: per-cell \(\max\).
- Spatial: any grid cell the 2-D box overlaps gets the full \(S\); neighbours get a Gaussian bleed.

`python spatial_threat.py` is the unit test (no Blender). `--spatial-overlay` writes `spatial_overlay.mp4` next to `preview.mp4` (hazy RGB, cell colour from the matrix, score printed in the cell).

---

### 9.10 `dataset_summary.json` — whole output directory

Rebuilt after every launch by reading **every** `episode_*/episode.json` already on disk (including older `episode_0000` folders without a scenario suffix).

| Field | Meaning |
| --- | --- |
| `episodes_on_disk` | How many manifests were found. |
| `episodes_this_run` | `--episodes` of the process that last wrote the file. |
| `label_histogram` | Sum over all episodes on disk (object×frame counts). |
| `episodes[]` | `{episode_id, dir, scenario, labels}` for each folder. |

Use this to check class balance before training. A healthy `auto` mix should show a non-trivial `NEAR_MISS` + `CRITICAL_THREAT` tail; an all-`safe_walk` pack will be dominated by SAFE_*.

---

### 9.11 RGB and video

- PNG: 1920×1080, 8-bit RGB, AgX view transform, opaque film. Filename index = `frame_id`.
- Video: constant frame rate `fps`, no audio, `+faststart`. It is a preview of the PNG sequence. The **labels live in JSON**, not in a subtitle track.

Do not assume ffmpeg’s frame \(n\) equals `timestamp = n/fps` if you re-encode with a different rate. Always key off `frame_id` / `timestamp` in the JSON.

---

### 9.12 How to consume this for the tactile model

Typical heads:

1. **Threat classifier** on the full frame or on each `bounding_box_2d` crop → `threat_label` (4-way). This is the vest.
2. **Detector** → `bounding_box_2d` + `class_name` (optional).
3. **Optional regression** → `ttc` and `cpa` (mask out `ttc == 9999.0`). A model that only sees pixels cannot be blamed for a label that was computed from 3-D; the regression head is how you check it learned “closing vs missing”.

Recommended filters when building a batch:

- Drop or down-weight `occluded` + `truncated` if you crop tightly.
- Do not treat `SAFE_DYNAMIC` with TTC 1.9 s / CPA 6 m as a collision — the taxonomy already decided it is safe.
- Instance ids are **per episode**. Across episodes, `person_000` is a different human.

---

### 9.13 Common misreads

| Mistake | Reality |
| --- | --- |
| Recompute TTC from `world_position` | Labels used the **threat point**, not the root. |
| `ttc == 9999` means “very far” | It means **not converging** (or relative speed ≈ 0). Current range is `distance`. |
| Small TTC ⇒ CRITICAL | Also need CPA \(< 0.5\) m. A 2 s TTC with CPA 6 m is SAFE_DYNAMIC. |
| `pitch_yaw_roll` is world IMU | Camera-local Perlin jitter only. |
| Histogram “214 SAFE_STATIC” = 214 objects | It is object×frame counts over the walk. |
| `occluded` = pixel IoU | Centre-ray flag. |
| Box = silhouette | Projected 3-D AABB, slightly loose. |

---

## 10. Run commands

Always `cd` is optional if you pass absolute paths.

```bash
# Default useful run: 1 episode, frames + video + JSON
blender --background --python /storage/BTP/blender_sim/main.py -- \
  --episodes 1 --scenario safe_walk --media both --output /storage/BTP/blender_sim/output

# Wrapper
/storage/BTP/blender_sim/run.sh --episodes 1 --scenario safe_walk --media both --output ./output

# Balanced dataset
blender --background --python /storage/BTP/blender_sim/main.py -- \
  --episodes 30 --scenario auto --media both --seed 42 --output ./output

# Shard 10 episodes starting at index 20
blender --background --python /storage/BTP/blender_sim/main.py -- \
  --episodes 10 --start-episode 20 --scenario auto --seed 42 --output ./output

# Annotations only (no GPU / no EGL)
blender --background --python /storage/BTP/blender_sim/main.py -- \
  --episodes 1 --scenario safe_walk --no-render --output ./output

# True headless box
xvfb-run -a blender --background --python /storage/BTP/blender_sim/main.py -- \
  --episodes 1 --scenario auto --media frames --output ./output

# Force time of day (QA only; not a CLI flag)
BTP_LIGHTING=dusk blender --background --python /storage/BTP/blender_sim/main.py -- \
  --episodes 1 --scenario sudden_stop --frames 30 --media both --output ./output

# Math unit test (system Python)
python /storage/BTP/blender_sim/threat_math.py

# Force EEVEE onto the RTX 4060 (hybrid AMD+NVIDIA laptop)
./run.sh --episodes 1 --scenario safe_walk --media frames --output ./output
# equivalent raw blender:
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  blender --background --python /storage/BTP/blender_sim/main.py -- \
    --episodes 1 --scenario safe_walk --media frames --output ./output

# Stay on the AMD iGPU
BTP_GPU=amd ./run.sh --episodes 1 --scenario safe_walk --media frames --output ./output
```

`--scenario` values: `auto`, any name from the 36-injector table in §6.5, plus aliases `safe`, `near_miss`, `critical`, `jaywalk`, `empty`, `projectile`, `car`, `pothole`, `weave`, `erratic`, `child`. Compounds: `--scenario jaywalker,car,pothole` (see §6.5.1). `--list-scenarios` prints the pools and shorts.

```bash
# Compound: jaywalker + oncoming car + pothole on the gait (annotations only)
./run.sh --episodes 1 --scenario jaywalker,car,pothole --no-render --frames 30 --output ./output
```

A 150-frame EEVEE episode is typically well under a minute on an RTX-class GPU after the performance pass (TAA 16, animation batch, NVENC). `--frames 8` is enough to see gait; `--frames 1` is enough to see lighting.

---

## 11. Parameters (authoritative defaults)

Copied from `config.py`. Change them **there**, not by scattering literals.

### Render / camera

| Key | Default |
| --- | --- |
| engine | `BLENDER_EEVEE` (auto-picked) |
| resolution | 1920 × 1080 |
| fps | 30 |
| frames_per_episode | 150 (5.0 s) |
| taa_render_samples | 16 (reprojection on; 32 was oversampling this lighting) |
| use_raytracing | True (`SCREEN` method, half-res traces) |
| fast_gi_ray_count / step / quality | 4 / 6 / 0.30 |
| png_compression | 1 (zlib; 0 is uncompressed) |
| lens / sensor | 24 mm / 36 mm default; HFOV randomized \(U(50^\circ, 90^\circ)\) per episode |
| clip | 0.05 – 120 m |
| eye_height_m | 1.6 |

### Gait / jitter

| Key | Default |
| --- | --- |
| walk_speed | U(1.0, 1.4) m/s |
| amplitude_m | 0.04 |
| frequency_hz | 1.8 |
| pitch_bob_amp_rad | 0.015 |
| yaw / pitch / roll | 15° @ 0.22 Hz / 5° @ 0.55 Hz / 2° @ 1.9 Hz |

### World

| Key | Default |
| --- | --- |
| path length | U(48, 78) m |
| path types | straight, gentle_curve, s_curve, corner_90 |
| road_width / lane_offset | 7.0 / 1.75 m |
| sidewalk_width / curb | 2.4 / 0.12 m |
| sample_ds | 0.40 m |
| building depth / height / gap | U(4,10) / U(6,18) / U(0.4,2.2) m |
| n_streetlamps / height / energy | 6 / 5.6 m / 900 W night |
| Poisson static r / n | 3.2 m / U(6,12) |
| Poisson head r / n | 7.5 m / U(3,7) |
| background peds / cars | U(3,6) / U(2,5) |
| vehicle speed | U(5.0, 9.0) m/s (urban; stays in frame) |
| ped speed | U(0.90, 1.45) m/s |
| bicycle / cube / cross-car speed | U(3.2, 5.5) / U(1.15, 2.20) / U(3.2, 4.8) m/s |
| head hazard height | U(1.2, 1.8) m |

### Threat / scenarios

| Key | Default |
| --- | --- |
| critical | TTC < 2.5 s and CPA < 0.5 m |
| near miss | TTC < 4.0 s and CPA in [0.5, 1.5] m |
| safe static distance | 5.0 m |
| safe dynamic CPA | 1.5 m |
| static_speed_eps | 0.05 m/s |
| auto mix | 0.40 / 0.30 / 0.30 |
| swerve_trigger_s / cut_in_trigger_s | 1.8 / 1.2 |
| sudden_stop lead / trigger | 3.0 m / 1.6 s |
| jaywalker_ttc / cross_person_speed | 3.2 s / U(1.00, 1.35) m/s |
| projectile_ttc / speed | 1.8 s / 3.6 m/s |
| CPA targets | near-miss 1.0 m, critical 0.12 m |
| compose (compounds) | cross/along/static stride 4.0 / 3.2 / 2.4 m; nudge 2.6 m; sample 0.12 s |
| annotation max_distance | 40 m |
| occlusion_epsilon | 0.08 m |

### Domain randomization

Lighting weights: dawn 0.22, noon 0.38, dusk 0.22, night 0.18.  
Weather: clear 0.50, light_fog 0.30, heavy_smog 0.20.  
Sun elevation: dawn 4–18°, noon 55–85°, dusk 3–16°, night −12–−2°.  
`volume_density` is **kept in config but not applied as a world volume** (black-frame / EEVEE world-volume lesson). Fog is faked via sky `air_density` / `aerosol_density` / `turbidity`.

Video: `preview.mp4`, CRF/QP 18, `ffmpeg_bin=ffmpeg`. Encoder is `h264_nvenc` (preset `p4`) when the NVIDIA encoder actually runs, else `libx264` `-preset veryfast`. Override with `output.video_encoder` = `nvenc` | `libx264` | `auto`.

---

## 12. Pitfalls we already hit (do not re-learn)

1. **Black PNGs, valid JSON.** Blender 5 defaults `use_sequencer=True` with an empty VSE. Always `prepare_still_render()` before `write_still`. After VSE mux, turn it off again.
2. **`BLENDER_EEVEE_NEXT` TypeError** on 5.2. Use `BLENDER_EEVEE`. Sky type is `MULTIPLE_SCATTERING`, not `NISHITA`.
3. **World Volume Scatter** in EEVEE = full-frame black. Fog via sky turbidity only.
4. **Euler `matrix_world` ignored.** Camera must be quaternion + decompose.
5. **Collections start excluded** from the view layer. `reveal_view_layer()` walks `layer_collection` and clears `exclude` / `holdout` / `indirect_only`. Then re-apply lamp hide/energy.
6. **24 mm vs facade.** Buildings start at \(s=14\), facade setback +1.6 m, walker at 38 % of sidewalk width from the curb.
7. **Mix / Noise sockets.** Never `node.inputs["A"]` or `noise.outputs["Fac"]`. Use typed helpers.
8. **Smooth Noise as window occupancy** = leopard print. Use a per-cell hash.
9. **Night exposure 0.85 + fill SUN 14 + sky 0.55** = whiteout. Current night: exposure 0.30, fill 1.8, sky strength 0.16, window emit 4.5, spots 900 W.
10. **Shadow buffer full (2053/2048).** Cap casters: key sun only + every other lamp. Fill and bounce: `use_shadow=False`. `shadow_pool_size='2048'`.
11. **Pelvis-only AABB.** `world_aabb_corners` must recurse MESH children or 2-D boxes cover only the hips.
12. **Feet origin vs 1.6 m camera** inflates CPA. Always `threat_point` / planar intercept at camera Z.
13. **Undefined TTC** must be `9999.0`, not `null` / `Infinity`.
14. **Old `output/episode_*` RGB** may still be the black set or the pre-realism boxes. Re-render.
15. **Do not enable world volume** when iterating “atmosphere”.
16. **`look_along` zeroes Euler X/Y** every frame. Walk pelvic list/pitch must be applied *after* it (`_tick_visuals` order is correct; do not invert). Yaw is \(\mathrm{atan2}(-d_x,d_y)\). The old `atan2(d_x,d_y)` moonwalks people, parks cars across the lane, and sends dashes through the asphalt on any heading other than \(+\mathrm{Y}\).
17. **Neck along −Z** grows into the chest. Neck is a +Z column.
18. **Object vs world normals on facades.** After `look_along`, world \(\hat{x}\) is not the street face. Window blend uses TexCoord **Normal** (object space).
19. **One Frenet \(s\).** Camera, dashes, cars, and people all use road \(s\) plus a lateral. Never mix a resampled sidewalk spline’s arc length with road \(s\).
20. **Curb Z.** Sidewalk mesh is at 0.12 m. People / camera / clutter on the sidewalk add that height. Asphalt stays at 0.
21. **Per-frame `bpy.ops.render.render(write_still=True)`.** Restarts EEVEE every still. Simulate on the CPU, then one `animation=True` batch with poses replayed in `frame_change_pre`.
22. **`ray_tracing_method = 'SCREEN_TRACE'`.** That identifier does not exist on Blender 5.2 (enum is `SCREEN`); the setattr was a silent no-op.

---

## 13. Design constraints that are load-bearing

- Asset-free: a stock Blender 4/5 install must run `blender --background --python main.py`.
- Labels are kinematics-first, pixels-second. A beautiful frame with a wrong TTC is a failed episode.
- Domain randomization is the regulariser. Do not “fix” lighting to one pretty dusk for the whole dataset.
- Humans must not be rectangles. The detector will key on that silhouette.
- Still realism is **stylized low-poly PBR**, not photogrammetry. No downloaded textures, no Mixamo, no MakeHuman.

---

## 14. Suggested future work (not implemented)

- Hazard meshes (trash, scooter, signs) are still boxes.
- No facial features / fingers; hands are superquadrics.
- Weather is sky-parameter only; no particles, no wet-road puddles beyond asphalt roughness.
- `volume_density` in config is unused (intentionally, after the world-volume black frames).
- No CLI `--lighting` flag; only `BTP_LIGHTING` env.
- Cycles path is not wired (engine picker can land on it if EEVEE is missing; materials are EEVEE-tuned).
- No train/val split helper; `--start-episode` + `--seed` is the sharding story. Default is append-to-output (next free id).

---

## 15. Performance (how the episode loop was sped up)

The old loop called `bpy.ops.render.render(write_still=True)` 150 times. Each call tears down and rebuilds the EEVEE GPU context, re-uploads shadows / GI, writes a zlib-15 PNG, then Python walked every actor with `evaluated_get` + `calc_matrix_camera` + a fresh AABB crawl + a raycast. That is why a 5 s clip felt slow even on an RTX 4060: the GPU was never allowed to stay warm, and the CPU work between frames was redundant.

Nothing below changes the Frenet math, TTC/CPA, box placement, or occlusion rule. The pixels stay 1920×1080, 8-bit, AgX, raytraced EEVEE. Annotations keep the same keys; JSON is compact (no indent) which is still valid Part-6.1 JSON.

### 15.1 Two-phase episode (biggest wall-clock win)

**Phase A — CPU simulation** (`main.run_episode`). For each frame: integrate camera + actors, `view_layer.update()` (actors write `location` / Euler; `matrix_world` is stale until the depsgraph runs), snapshot local `location` / Euler-or-quat of the camera and every actor descendant (pelvis, limbs, wheels), project boxes. After the loop: write `annotations/annotations.json` (unless `--no-annotations`) and `spatial_annotations/spatial_annotations.json`. No EEVEE.

**Phase B — GPU animation.** Register `frame_change_pre` that restores snapshot `i` when Blender seeks to frame `i`, set `filepath` to `rgb/######`, `use_sequencer=False`, then **one** `bpy.ops.render.render(animation=True)`. EEVEE keeps the shadow pool, TAA history, and ray-trace buffers resident. If the batch does not produce `%06d.png`, we fall back to stills from the same snapshots (never lose the episode).

Why snapshot instead of integrating twice: walk-cycle pelvic bob and wheel `euler[0]` are incremental. Replaying from a pose cache is exact and does not require resetting `Actor` / `WalkRig` state.

Measured on this box (RTX 4060 Max-Q, Blender 5.2.1, 1920×1080, `jaywalker_turn_away`): **30 frames = 0.32 s sim + 12.2 s EEVEE + NVENC mux**. Previously ~0.9 s/frame stills (~27 s GPU plus JSON). Full 150-frame episodes should land around **one minute**, not several.

### 15.2 EEVEE knobs (`configure_eevee`, `config["render"]`)

| Change | Why | Visual |
| --- | --- | --- |
| TAA 32 → **16** + `use_taa_reprojection` | Sample count is linear in GPU time. Reprojection reuses the previous frame on this slow-moving walk. | Indistinguishable on low-poly PBR; 32 was oversampling. |
| Fast GI rays 8 → **4**, steps 8 → **6**, quality 0.45 → **0.30** | We had cranked GI far past Blender’s defaults (2 rays). | Bounce on facades stays; noise is in the TAA. |
| `ray_tracing_method = 'SCREEN'` | Previous `'SCREEN_TRACE'` never applied. Half-res traces (`resolution_scale='2'`) + denoise kept. | Same SSR on paint/glass as a working SCREEN trace. |
| Volumetric tile `'2'` → **`'8'`**, samples 32 → **16**, volumetric shadows **off** | There is no world volume. Tile size 2 was paying for empty froxels. | No change (no volume shader). |
| PNG compression **1**, `color_mode=RGB` | Default zlib 15 is CPU-heavy; RGBA wrote a useless alpha. | Bit-identical RGB. |
| `use_persistent_data`, `use_lock_interface` | Keep GPU resources across the animation; skip UI sync in background. | None. |
| Sequencer/compositor **off** once | Same black-frame guard as before, not per still. | None. |

Raytracing stays **on**. Turning it off would be faster but would change car paint and windows.

### 15.3 Annotation / projection (`projection.py`, `build_frame_record`)

- **`LocalBoundCache`**: limb `bound_box` is constant in object space. Capture once per object pointer; each frame only does `matrix_world @ corner`. Cleared with `clear_bound_caches()` at episode start (pointers are reused after `reset_blender_scene`).
- **Static actors** cache *world* corners for the whole episode (potholes, trash, parked cars).
- **One view matrix + one `calc_matrix_camera` per frame**, passed into every `project_object`. Previously both were rebuilt per actor.
- **No `evaluated_get`**. None of our meshes have modifiers; parenting is already in `matrix_world` after `view_layer.update()`.
- AABB / threat point / raycast centroid share the **same corner list** (was three walks).
- Coarse **distance** and **behind-camera** culls (`max_distance+5 m`, \(>6\) m behind the tangent) skip projection and raycasts for actors that cannot appear in the JSON anyway.
- Integrator velocity is used directly; finite-difference only if the actor is moving but `velocity` is still zero.

Box math, truncation, and `scene.ray_cast` occlusion (including ground-ribbon skip) are unchanged.

### 15.4 Video encode (`_encode_with_ffmpeg`)

PNG decode stays on the CPU (there is no useful CUDA PNG decoder). Encode:

1. Probe `h264_nvenc` with a 64×64 lavfi frame (the encoder can be *listed* without a working driver).
2. If the probe succeeds: `-c:v h264_nvenc -preset p4 -tune hq -rc constqp -qp 18` (QP 18 ≈ x264 CRF 18).
3. Else: `libx264 -preset veryfast -crf 18 -threads 0` (old path used the slow default medium preset at the same CRF).

VSE fallback is unchanged if ffmpeg is missing.

### 15.5 I/O

JSON is `separators=(",", ":")` with no indent. Schema and numeric rounding are the same; parsers do not care about whitespace. Compact dumps are CPU-cheap next to EEVEE.

### 15.6 What we did not do (and why)

- **JPEG / 16-bit / float16 colour.** Would change pixels or blow disk; the display buffer is already 8-bit PNG.
- **`bpy.ops.render.opengl`.** Viewport shading is not EEVEE-Next raytrace/GI.
- **Cycles.** Slower, not the look.
- **Dropping raytracing or Fast GI.** Faster, but not perceptually identical to the current lighting.
- **Numpy in the Blender interpreter.** No extra pip; the hot path is EEVEE, not Python loops.
- **Per-frame N-body steering for compounds.** Occupancy is reserved once at inject time (`ComposeSession`); extra actors only add a few objects to the existing pose snapshot / animation replay.

---

## 16. Quick “where is X?” index

| I want to change… | File / symbol |
| --- | --- |
| Resolution, FPS, episode length | `config.py` → `render` |
| Lens / eye height | `config.py` → `camera` |
| Gait bounce / Perlin | `config.py` → `gait`, `jitter`; logic in `CameraRig` |
| TTC thresholds / class mix | `config.py` → `threat`, `scenarios` |
| Street width, Poisson, lamp watts | `config.py` → `world` |
| Sun / sky / AgX | `apply_domain_randomization`, `_setup_world_shader` |
| Window grid / asphalt | `materials.py` |
| Body proportions / walk | `humanoid.py` → `spawn_humanoid`, `WalkRig` |
| Forced collisions | `WorldGenerator.inject_scenarios` and `_inject_*` |
| Compound occupancy / CLI aliases | `scenario_compose.py` |
| JSON schema | `main.build_frame_record` |
| Spatial threat matrix | `spatial_threat.py`; CLI `--threat-grid` |
| Spatial overlay video | `spatial_overlay.py`; CLI `--spatial-overlay` |
| Mixed dataset pack | `gen_dataset.py`; `main.py --plan` |
| 2-D boxes / occlusion | `projection.py` |
| TAA / GI / PNG / encoder | `config.py` → `render`, `output`; `configure_eevee` |
| Animation-batch render | `main._render_animation_sequence` |
| AABB cache | `projection.LocalBoundCache` |
| Black frames | `prepare_still_render` |
| Camera not rotating | `CameraRig._apply_camera_matrix` |

---

*Last updated for the performance pass (animation-batch EEVEE, TAA 16, NVENC, AABB cache) on Blender 5.2. If the code and this file disagree, the code wins — then fix this file.*
