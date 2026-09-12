# blender_sim — internals

This file is the complete technical specification of the repository. It exists so a person or another model can reconstruct **what every module does, why it exists, and how data moves**, without opening the Python. How to **run** the pipeline and what folders it writes is in [`README.md`](README.md). If this file and the code disagree, the code wins — then this file is stale.

Developed on Arch Linux, Blender **5.2.1** at `/usr/bin/blender`. Workspace: `/storage/BTP/blender_sim`.

---

## 0. How to use this document

Read §1–§4 first. Those four sections are the whole architecture. Everything after them is a zoom-in: one file, one data type, one injector, one JSON field.

If you are implementing a change:

1. Find the concern in §19 (“Where is X?”).
2. Read the matching file section here. That section names the functions, the invariants, and the reason they exist.
3. Only then open the `.py`. The comments in the code are short reminders of decisions already explained here.

If you are an AI that has been given only this file:

- Treat the formulas, field tables, injector tables, and output schema as the contract.
- Do not invent a second Frenet parameter, a second threat point, or a second TTC formula.
- Do not assume `numpy`, pip packages, `.blend` assets, or a world-space N-body solver. None of those exist.
- When two numbers conflict (for example “distance” in JSON vs Euclidean distance of two `world_position` vectors), the **threat-point** kinematics win. That is deliberate.

---

## 1. What this program is

A headless Blender process that, for each **episode** (one 5-second clip by default), builds a procedural street / avenue / park / plaza, walks a head-mounted camera along it, and writes:

1. **Kinematic truth** — time-to-collision (TTC), closest-point-of-approach (CPA), and a four-class `threat_label` — computed from 3-D motion, not from pixels.
2. Optional **RGB** (`rgb/*.png`, `preview.mp4`).
3. A **K×K spatial threat matrix** for every frame (default 3×3).

It is **not** an autonomous-vehicle dataset. The ego is a pedestrian. Labels exist so a tactile vest can learn “buzz / graded buzz / silent”.

Why synthetic:

- A wearable has no ground-truth 3-D. You cannot film a city and later recover exact TTC.
- Forced intercepts (`jaywalker`, `swerve_vehicle`, `cube_head_on`, …) can be aimed in closed form. Real footage cannot guarantee a 0.12 m miss at 2.4 s.
- Domain randomization (lighting, weather, FOV, body type, chaos, path shape) is cheaper than filming 30 cities, and it is the only thing that stops a detector from overfitting to the low-poly primitives.

What is deliberately **absent**:

- No pip packages. No `numpy`. No `.blend` assets. No downloaded textures.
- Meshes and PBR materials are generated in Python.
- System Python can import the modules that do not need `bpy`: `threat_math`, `spatial_threat`, `spatial_overlay`, `scenario_compose`, `config`, `gen_dataset`.
- There is no per-frame collision solver. Occupancy is reserved **once** at inject time. During the 150-frame loop, actors only integrate Frenet `(s, lateral)` and get clamped to the street corridor.

The detector this dataset trains must learn:

- A moving canopy is **not** a threat (trunk is static; foliage Euler is visual only).
- A hole in the pavement **is** a threat (footprint threat point at eye height).
- A sphere, pyramid, cube, car, cyclist, and child are all obstacles if they occupy the gait tube.
- A seated walker facing a parked bollard is SAFE (nothing is closing). A seated walker with a car driving at them is not.

---

## 2. Design constraints that shape every file

These are the reasons the code looks the way it does. Almost every “why” in later sections traces back here.

**C1. One Frenet parameter.** Every actor and the camera share the **road centreline** arc-length `s` plus a signed `lateral` (positive = road-right, \(\hat{r}=\hat{t}\times\hat{z}\)). The sidewalk is an offset of that spline, not a second path. Mixing a resampled sidewalk spline’s arc length with road `s` would desynchronise injectors from the annotator.

**C2. Constant-velocity point TTC.** The label is the spec’s point-mass TTC / CPA of a **threat point**, not the mesh origin and not a swept volume. The spatial matrix *is* body-aware. Those two products answer different questions and must not be collapsed.

**C3. Stationary ego is a first-class state.** Seated / hesitate have `walk_speed = 0`. Every solver must accept \(V_{\mathrm{cam}}=0\) without dividing by it. Intercepts spawn at \(\max(v_{\mathrm{ego}}+v_{\mathrm{obj}},\,v_{\mathrm{obj}})\tau\), never on top of the HMD.

**C4. Two-phase sim / render.** `bpy.ops.render.render(write_still=True)` once per frame tears down EEVEE every still. Phase A integrates on the CPU and snapshots poses. Phase B restores those poses and renders an animation. Walk-cycle bob and wheel roll are incremental, so they cannot be re-integrated during Phase B.

**C5. Asset-free, instanced geometry.** A thousand unique grass meshes exhausts GPU memory. `MeshLibrary` uploads one vertex buffer per shape key and instances it. Pedestrians stay unique because they are articulated.

**C6. No N-body during the episode.** A per-frame collision resolver would fight the gait, the two-phase render, and the closed-form intercepts. `ComposeSession` reserves Frenet capsules once. `StreetCorridor.confine` is the only runtime spatial constraint.

**C7. Blender 5.x is hostile in specific ways.** Assigning `matrix_world` alone is ignored when rotation mode is Euler. Empty VSE + `use_sequencer=True` renders pitch black. `BLENDER_EEVEE_NEXT` is not always in the enum. World Volume Scatter is full-frame black in EEVEE. All of these have dedicated guards.

**C8. Leak across episodes is fatal.** A 2000-episode pack that does not purge orphans accumulates every mesh and node group it ever built. `release_episode` + recursive `orphans_purge` is not optional cleanup; it is the memory budget.

**C9. Labels from kinematics, boxes from pixels.** A tree’s 2-D box covers the canopy. Its threat point and Frenet extents come from the trunk. A pothole’s render mesh drops below the pavement; its box is a thin mouth slab; its threat point is at camera Z. Consumers who recompute TTC from `world_position` will get the wrong answer, and that is documented in the JSON contract.

**C10. Determinism is per-episode, not per-process.** `--seed` plus `--start-episode N` advances a master RNG N times, then each episode draws its own `ep_rng`. Two episodes never share a Perlin / Poisson / colour stream. Re-running shard N with the same seed reproduces that shard.

---

## 3. How a process starts

```text
./run.sh [flags]
  └─ export PRIME vars if BTP_GPU=auto|nvidia and nvidia-smi works
  └─ exec $BLENDER --background --python $ROOT/main.py -- "$@"
       └─ main.py inserts _ROOT on sys.path  (Blender does not)
       └─ parse_args() reads only argv after the bare "--"
       └─ main() → run_episode() × N
```

### 3.1 `run.sh`

The shell script is not decorative. EEVEE rasterizes on whichever OpenGL device Blender opened **at process start**. On a hybrid AMD+NVIDIA laptop, a raw `blender …` usually hits the iGPU and is 5–10× slower, or fails to create a GPU context.

What the script does, in order:

1. `ROOT="$(cd "$(dirname "$0")" && pwd)"` — absolute path, so it works from any cwd.
2. `BLENDER="${BLENDER:-blender}"` — override with `BLENDER=/path/to/blender ./run.sh …`.
3. `BTP_GPU` is `auto` (default), `nvidia`, or `amd`. Anything else exits 2.
4. If `auto` or `nvidia`, and `nvidia-smi` succeeds, it exports:
   - `__NV_PRIME_RENDER_OFFLOAD=1`
   - `__GLX_VENDOR_LIBRARY_NAME=nvidia`
   - `__VK_LAYER_NV_optimus=NVIDIA_only`
5. `exec "$BLENDER" --background --python "$ROOT/main.py" -- "$@"`

`--background` is headless. The bare `--` is required: without it Blender eats `--episodes` and friends as its own flags. Missing `blender` → the shell’s 127. `BTP_GPU=amd` skips the PRIME exports and stays on the iGPU (useful when the NVIDIA context is broken).

There is no `source venv`. There is no `pip install`. Blender’s bundled Python is the runtime.

### 3.2 `main.py` argument split

`_argv_after_double_dash(argv)` returns everything after `--`. If you run `python main.py` (no Blender; useful only for `--help` / `--list-scenarios` on a machine that can import `bpy`, which system Python usually cannot), it drops the script name and treats the rest as argparse.

Every flag `parse_args` accepts:

| Flag | Default | Role |
| --- | --- | --- |
| `--episodes N` | 4 | How many episodes this process writes. Ignored if `--plan` is set (plan length wins). |
| `--start-episode N` | one past highest `episode_*` on disk | First numeric id. `0` starts a fresh pack. Omitted + `--plan` starts at 0. |
| `--scenario NAME` | `auto` | Repeatable. Alias, canonical name, or `a,b,c` / `a+b+c`. |
| `--scenarios LIST` | none | Joined with `--scenario` if both are set. |
| `--list-scenarios` | | Print pools, aliases, compound examples, exit 0. |
| `--output DIR` | `<repo>/output` | Root for `episode_*` folders. |
| `--seed INT` | 42 | Master RNG. |
| `--frames N` | 0 = config 150 | Override length. Still clamped by remaining road. |
| `--media frames\|video\|both` | config `both` | What visual product to keep. |
| `--no-rgb` | | Delete `rgb/` after mux. Ignored if no video is being written. PNGs kept if mux fails. |
| `--no-render` | | Skip EEVEE. JSON only. Overlay skipped. |
| `--no-annotations` | | Skip `annotations/annotations.json`. Spatial JSON still written. |
| `--no-occlusion` | | Skip the centre-ray occlusion flag (faster; boxes still computed). |
| `--biome street\|avenue\|park\|plaza\|alley\|residential\|market\|auto` | auto | Weighted draw if auto. |
| `--ego-mode walk\|diagonal_cross\|crosswalk\|erratic\|seated\|auto` | auto | Locks `choose_ego_profile` weights to one mode. Crossing injectors force `crosswalk` when this is `auto`. |
| `--ego-height short\|typical\|tall\|auto\|metres` | auto | Standing eye-height band or explicit metres. |
| `--chaos FLOAT` | unset | Locks `material_chaos` to `[c, c]`. `0` is the pre-chaos pipeline. |
| `--no-trees` | | Zeroes `n_trees`, `n_grass_clumps`, `n_street_trees`, `n_median_trees`, `n_path_trees` on world **and** every biome override. |
| `--wind calm\|breeze\|windy\|auto` | auto | Locks `wind_weights` to one label. `auto` draws from config (calm 0.32 / breeze 0.48 / windy 0.20). Also overridable by `BTP_WIND`. |
| `--hfov DEG` / `--lens-mm MM` / `--no-random-fov` | | See `apply_camera_fov`. The lock flag is `--no-random-fov` (not `--lock-fov`). |
| `--threat-grid K` | 3 | Spatial matrix size. Must be ≥ 1. |
| `--spatial-overlay` | | Write `spatial_overlay.mp4`. Requires render. |
| `--plan PATH` | | JSON from `gen_dataset.py`. Per-episode `scenario` overrides the CLI request. |

### 3.3 `main()` control flow

1. `get_config()` — deep copy of `CONFIG`. An episode mutates `cfg['world']` (biome fold-in, sparse overrides); the next episode must not inherit that.
2. `--list-scenarios` prints the three pools, aliases, and compound examples, then returns 0.
3. `scenario_request_from_tokens` joins `--scenario` / `--scenarios` into one request string.
4. Optional `--plan` JSON: must exist, must parse, must have non-empty `episodes[]`. Episode count becomes `len(episodes)`.
5. Resolve `--output`, create it.
6. Resolve start id. Without `--start-episode` and without a plan, `next_free_episode_id` scans existing `episode_NNNN_*` folders so a second run never overwrites.
7. `master = random.Random(seed)`. Advance `master.randrange(1, 2**31)` exactly `start_id` times. That is what makes `--start-episode 100 --seed 42` a stable shard: episodes 0–99 of a full run used those 100 draws.
8. For each episode `k`: `episode_id = start_id + k`. If a plan is loaded, the request is `plan.episodes[k].scenario`. Call `run_episode`. A failure is logged to stderr; the process **continues**. Exit 2 if any episode failed.
9. `rebuild_dataset_summary` scans **every** `episode_*/episode.json` already on disk (including older folders in the same root) and writes `dataset_summary.json`.

RNG contract: each episode does `ep_rng = random.Random(master.randrange(1, 2**31))`. That object is the **only** RNG passed into `WorldGenerator`, `choose_ego_profile`, `apply_camera_fov`, and `CameraRig`. Two episodes never share a noise stream. Re-seeding Blender’s own `mathutils.noise` is not used anywhere, because it is not bit-stable across Blender versions.

---

## 4. Per-episode flow (`run_episode`)

This is the heart of the program. One call = one folder on disk.

```text
cfg = deepcopy(cfg)                     # isolate biome / chaos mutations
pick_scenarios(ep_rng, cfg, request)    # scenario_compose
apply_camera_fov(...)                   # lens from HFOV
if --chaos: lock material_chaos
if --no-trees: zero all tree/grass counts on world + biomes
WorldGenerator(cfg, ep_rng)
  prepare_biome(biome)                  # fold widths into cfg['world']
  prepare_scenario(names)               # building gap / empty_street
clear_bound_caches()
WorldGenerator.build()                  # meshes, lights, background actors
create_camera() + choose_ego_profile() + CameraRig(...)
inject_scenarios(names, rig)            # forced events, Frenet occupancy
place_ego_props(rig)                    # bench if seated
freeze()                                # partition movers vs static

# Phase A — CPU. No EEVEE.
for i in 0 .. N-1:
    CameraRig.update(t, dt)
    WorldGenerator.update(t, dt)
    view_layer.update()                 # required: actors write location/Euler
    snapshot poses (cam + every actor descendant)
    build_frame_record(...)             # boxes, TTC, K×K splat
write annotations/annotations.json      # unless --no-annotations
write spatial_annotations/...json

# Phase B — GPU. One animation render.
frame_change_pre restores snapshot i
bpy.ops.render.render(animation=True)   # rgb/######.png
mux preview.mp4; optional overlay; maybe delete rgb/
release_episode()                       # unlink + orphans_purge
write episode.json
```

### 4.1 Order that must not flip

**Biome before scenario.** `prepare_biome` writes road width, tree counts, `buildings=False` for park, etc. into `cfg['world']`. `prepare_scenario` then zeroes background traffic if `empty_street` is in the mix. If you reversed them, a sparse scenario’s `(0,0)` counts would be overwritten by the biome’s `(3,6)`.

**`build()` before the rig.** The camera needs the road spline, sidewalk lateral, walk speed, and corridor. Those do not exist until `build()` returns a `WorldState`.

**Rig before inject.** Injectors call `rig.arc_length_at(t)`, `rig.lateral_at(t)`, and `rig.predict_position(t)` to place intercepts on the **future gait line**. That line includes diagonal-cross smoothstep and erratic halt. Placing against a constant `s0 + v t` would miss a hesitating walker.

**`freeze()` after inject.** `freeze` partitions `_movers` and `_annotatable`. Injecting after freeze would leave new actors out of the 150-frame loop (or force a linear scan of every bench every frame).

**`view_layer.update()` every sim frame.** Actors write `location` and Euler, not `matrix_world`. The depsgraph keeps the previous pose until this call. Skipping it makes every AABB project empty (`objects=0` in the log). This is the most common “I deleted a line and labels died” failure.

### 4.2 Frame count

\(N = \min(\texttt{--frames or config},\; \lfloor \mathrm{rig.max\_time()}\cdot\mathrm{fps}\rfloor)\), then at least 2.

Default: 150 frames at 30 Hz = 5.0 s.

`max_time()` is remaining road arc after `s0 = 3.0 m`, minus a 2 m tail, converted to time via the **arc table** (not `remaining / walk_speed`). Seated / halt therefore do not divide by zero and do not get an episode truncated to nothing. A seated walker returns a huge sentinel (`1e6`); the frame count from config wins.

`choose_ego_profile` is given `episode_seconds = n_frames / fps` **before** the clamp, so a hesitation window is drawn inside the intended episode. After the clamp, a very short road can still cut the last frames; the halt may then sit near the end. That is accepted.

### 4.3 Why two phases

`bpy.ops.render.render(write_still=True)` once per frame:

- tears down and rebuilds the EEVEE pipeline every still,
- pays shader compile / TAA warmup 150 times,
- is the historical “one episode takes minutes” path.

Phase A integrates kinematics and writes JSON with **no** GPU raster. Phase B registers a `frame_change_pre` handler that restores snapshot `i` when `scene.frame_current == i`, then calls `bpy.ops.render.render(animation=True)` once. The GPU stays warm. TAA temporal history is meaningful.

Poses are **snapshotted**, not re-integrated, because:

- `WalkRig` writes incremental pelvic bob on top of `look_along`.
- Wheel roll is \(\Delta\theta = -\Delta s / r\) per frame.
- Foliage Euler is a function of `t` but is applied as an offset from a stored rest pose; re-running Phase A’s integrator from `t=0` during render would be possible for foliage, but gait and wheels are not closed-form in the same way once `look_along` has zeroed Euler X/Y.

`_collect_pose_objects` walks the camera plus every actor root **and every descendant**. Missing a child (a shin, a wheel, a leaf) means Phase B renders that part at the bind pose.

If the animation batch misses files (`_ensure_six_digit_pngs` fails), `_render_stills_fallback` uses the **same** pose list and writes stills. Labels stay consistent with pixels either way.

### 4.4 Media decisions

| Condition | PNG sequence | `preview.mp4` | overlay | `rgb/` kept |
| --- | --- | --- | --- | --- |
| `--no-render` | no | no | no | n/a |
| `--media frames` | yes | no | if requested | yes |
| `--media video` | yes (for mux) | yes | if requested | no, unless mux fails or `--no-rgb` off |
| `--media both` | yes | yes | if requested | yes, unless `--no-rgb` |
| `--no-rgb` + no video | ignored, warning printed | | | |

Video mux: probe `h264_nvenc`; if the encoder actually encodes a tiny test, use it (`preset p4`). Else `libx264 -preset veryfast -crf 18`. If `ffmpeg` is missing, Blender’s VSE fallback muxes the PNG sequence (`sequences` on 4.x, `strips` on 5.x). After VSE mux, `prepare_still_render()` is called **again** so `use_sequencer` does not stay True for the next episode.

### 4.5 Cleanup

`release_episode(state)`:

1. Nulls every Python cycle: `Actor.obj`, `gait`, `wheels`, `follow_spline`, `threat_obj`, `foliage`, `wind`, `_bounds`, wander/speed noise.
2. Clears `state.actors`, `collections`, `materials`.
3. Unlinks every remaining object.
4. Unlinks child collections; `scene.world = None`.
5. `purge_orphans()` up to 8 recursive passes (`bpy.data.orphans_purge(do_recursive=True)`).

Then `clear_bound_caches()` because object pointers are reused after the next `reset_blender_scene`.

The episode log prints `cleanup: purged N datablock(s) [M unique mesh datablocks, K linked instances]`. A healthy 2000-episode pack shows a large N every time and a flat RSS. A shrinking N plus climbing RSS means a new datablock type is leaking (usually a node group hanging off a material).

---

## 5. Coordinate frames

All computation inside Blender is **Z-up**. JSON is **Y-up**. Mixing them is the fastest way to invent a 90° error in every velocity.

| Frame | Axes | Used for |
| --- | --- | --- |
| Blender world | X right, **Y forward** on a +Y path, **Z up** | All `bpy` locations, splines, lighting, TTC **computation** |
| JSON / spec | X right, **Y up**, **Z forward** | Written `world_position`, `velocity`, `relative_velocity` |
| Camera local | +X right, +Y up, looks down **−Z** | `CameraRig._compose_matrix` |
| Humanoid local | +Y face, +Z up, limbs hang **−Z** | Hip / knee / shoulder about local +X |
| Car / box local | +Y forward | `look_along` yaws so local +Y = horizontal heading |
| Frenet | \(s\) along road centreline, `lateral` along \(\hat{r}=\hat{t}\times\hat{z}\) | Camera, actors, occupancy, spatial score |
| Image | origin **top-left**, +u right, +v down, pixels | `bounding_box_2d`, spatial matrix row 0 = top |

Conversion (the permutation is an involution):

```text
blender_zup_to_yup (x, y, z) = (x, z, y)
yup_to_blender_zup (x, y, z) = (x, z, y)
```

A camera at Blender `(0, 5, 1.6)` looking +Y writes JSON position `(0, 1.6, 5)`. A velocity of `1.2 m/s` along +Y writes `(0, 0, 1.2)`.

### 5.1 `look_along`

Yaw is \(\theta=\mathrm{atan2}(-d_x,\,d_y)\) because \(R_z(\theta)\,(0,1,0)=(-\sin\theta,\,\cos\theta)\). **Not** `atan2(d_x, d_y)` — that moonwalks people and parks cars across the lane.

`look_along` writes Euler Z and **zeroes Euler X and Y**. That is why `WalkRig.apply` must run **after** `look_along` in the same frame: pelvic pitch/list live on Euler X/Y. Foliage sway is on child empties, so it is safe; the trunk’s lean is set once at spawn and `look_along` is not called on a static tree.

### 5.2 Ground heights

| Surface | Blender Z |
| --- | --- |
| Asphalt / park path | 0.00 |
| Sidewalk slab | `curb_height` (0.12 m street, 0.00 park, 0.02 plaza) |
| Camera eye | curb + eye height + gait sine (standing 1.6 m; seated \(U(0.95,1.28)\)) |
| Vehicle hull origin | ~0.4–0.6 m (axle / body centre) |
| Person root | **pelvis**, not feet (~0.9 m for an adult) |
| Pothole mesh | mouth at ground, well drops ~0.4 m |
| Tree trunk origin | ground at the planting Z |

Feeding pelvis vs camera into CPA invents a ~0.6–1.6 m vertical miss. That is why `threat_point()` exists. In park / plaza, even that is not enough (no kerb step; a seated walker vs a bollard differs by ~1 m in Z), so `relative_kinematics(..., planar=True)` zeroes Blender Z before the solve.

### 5.3 One Frenet `s`

```text
world_point(s, lateral, z) = evaluate(s) + right(s) * lateral, with .z = z
```

`StreetCorridor.ground_z(lateral)` is curb if `|lateral| ≥ road_half - 0.08`, else 0. Actors add `origin_z` on top of that (a person’s pelvis sits above the slab; a cube’s centre sits at `size/2`).

Never:

- take `s` from `PathSpline.offset_spline(sidewalk_lateral)` and add it to a road `s`,
- treat world-Y as arc length on a curve,
- convert a world-space intercept velocity into Frenet by ignoring curvature.

`PathSpline.project(p)` is the inverse: nearest polyline foot, then signed lateral along `right(s)`.

---

## 6. File map

| File | Needs `bpy` | Role |
| --- | --- | --- |
| `run.sh` | no | PRIME offload + exec Blender |
| `config.py` | no | Nested dict of every tunable. `get_config()` deep-copies. |
| `threat_math.py` | no | Vectors, TTC, CPA, four-class taxonomy, intercept velocity |
| `camera_kinematics.py` | mathutils | Perlin, ego profile, `CameraRig` |
| `projection.py` | yes | AABB → 2-D box, threat point, occlusion ray |
| `world_generator.py` | yes | Scene, actors, lighting, EEVEE, injectors |
| `scenario_compose.py` | no | CLI tokens, aliases, Frenet occupancy |
| `humanoid.py` | yes | Articulated pedestrian + Winter walk cycle |
| `materials.py` | yes | Procedural PBR, chaos, Blender 4/5 Mix-node guards |
| `spatial_threat.py` | no | Body-aware score + K×K splat |
| `spatial_overlay.py` | no | PPM heat + ffmpeg overlay graph |
| `gen_dataset.py` | no | Balanced pack planner; execs `run.sh --plan` |
| `main.py` | yes | CLI, episode loop, annotation JSON, render, mux |

`python threat_math.py`, `python spatial_threat.py`, `python spatial_overlay.py`, `python scenario_compose.py`, and `python gen_dataset.py --self-test` are the unit tests. They run on system Python.

---

## 7. `config.py`

`CONFIG` is one nested dict. There is no YAML, no CLI-to-config merge except the flags listed in §3.2. `get_config()` returns `copy.deepcopy(CONFIG)` so `prepare_biome` / `prepare_scenario` / `--chaos` / `--no-trees` cannot leak into the next episode.

### 7.1 `apply_camera_fov`

Writes `camera.lens_mm` and `camera.hfov_deg` for this episode.

Pinhole (full-frame, horizontal fit):

\[
\mathrm{lens} = \frac{w/2}{\tan(\mathrm{HFOV}/2)}, \qquad
\mathrm{HFOV} = 2\arctan\frac{w/2}{\mathrm{lens}}
\]

with \(w=36\) mm. Default 24 mm ⇒ ≈ 73.74°.

Priority:

1. Explicit `--hfov DEG`.
2. Else explicit `--lens MM`.
3. Else, if not `--no-random-fov` and `randomize_hfov` and an RNG is provided: uniform draw from `hfov_deg_range` `(50, 90)`.
4. Else the config default lens.

`--no-random-fov` without `--hfov` / `--lens-mm` freezes the config default. That is how a pack can disable FOV randomization without pinning a number. Internally this is the `lock=` argument of `apply_camera_fov`.

### 7.2 Sections (what lives where)

**`render`.** Engine name (advisory — `configure_eevee` picks what the binary exposes), 1920×1080, 30 fps, 150 frames, PNG 8-bit, TAA 16, raytracing on, Fast GI 4/6/0.30, volumetric knobs (present but world volume is unused), `png_compression=1`.

**`camera`.** Name, lens, sensor 36 mm, `HORIZONTAL` fit, clip 0.05–120 m, eye 1.6 m, HFOV range.

**`gait`.** Walk speed bands stroll \(U(0.70,1.00)\) / walk \(U(1.00,1.50)\) / hurry \(U(1.50,2.05)\) m/s, bounce 0.04 m @ 1.8 Hz, optional pitch bob 0.015 rad. Seated is exactly 0.

**`ego`.** Mode weights (walk 0.44, erratic 0.22, seated 0.14, crosswalk 0.12, diagonal 0.08), seated eye \(U(0.95,1.28)\), sidestep / wobble / hesitate windows, diagonal / crosswalk target fraction and span. Standing eye height is drawn from short / typical / tall bands unless `--ego-height` locks it.

**`jitter`.** Per-axis amplitude, frequency, octaves, persistence, lacunarity for camera-local Perlin.

**`world`.** Path length \(U(48,78)\) m, path types, road 7.0 m, sidewalk 2.4 m, curb 0.12 m, `sample_ds=0.40`, building depth/width/height/gap, **`building_setback=3.6`** (planting strip; trees sit here only; canopies must not reach the wall), `median_width` (0 except avenue), `cobble_prob`, lamps, Poisson clutter, background counts, speed ranges (`vehicle`, `pedestrian`, `bicycle`, `cube`, `shape`, `cross_car`), `corridor_margin=0.22`, biome table, tree / grass knobs.

Biome overrides **replace** matching keys on `cfg['world']` for that episode. They do not deep-merge nested dicts except by overwriting the key. Park sets `ground="grass"`, `buildings=False`, `curb_height=0`, `lane_paint=False`, `n_background_vehicles=(0,0)`, wide verge, path types without `corner_90`. Avenue widens the road to 13 m, plants a grass median (`median_width≈2.6`), and raises the setback to 4.2 m. Plaza paints a wide paved room with buildings on **one** side only (`building_sides=(1.0,)`). Alley / residential / market change widths, setback, cobble chance, and crowd counts. `prepare_biome` then jitters widths from `domain_randomization.layout_jitter`.

**`threat`.** Taxonomy thresholds (see §8). `user_hitbox_radius=0.30` is the injector / spatial ego half-width, not a visual mesh.

**`scenarios`.** 40/30/30 auto mix, three named pools, timing constants (`swerve_trigger_s=1.8`, `cut_in_trigger_s=1.2`, pothole leads, `jaywalker_ttc=3.2`, `projectile_ttc=1.8`, CPA targets 1.0 / 0.12, compose strides).

**`domain_randomization`.** Six lighting states with energy / elevation, azimuth range, lighting weights, weather weights, dappled probability, `material_chaos=(0.20,1.00)`, colour boxes, palettes.

**`output`.** Folder names, media, video CRF / encoder / presets.

**`annotation`.** `max_distance=40` m, `occlusion_epsilon=0.08` m.

Change numbers **here**, not by scattering literals into injectors. Injectors that still have literals (`tau=2.4`, cube size ranges) are documented in §15; prefer promoting a new one to config if you touch it twice.

---

## 8. `threat_math.py` (no `bpy`)

Pure Python 3-tuples. No numpy. Convention-agnostic as long as camera and object share a frame. The pipeline evaluates in Blender Z-up, then converts vectors at write time.

### 8.1 Vector helpers

`vadd`, `vsub`, `vscale`, `vdot`, `vnorm`, `vdist`, `vnormalize`, `as_vec3`. `_finite3` replaces any non-finite component with 0. That exists because `NaN < 2.5` is `False`, so a poisoned TTC would silently classify as SAFE.

`vflat(a)` zeroes Z (Blender ground plane).

`blender_zup_to_yup` / `yup_to_blender_zup` — same permutation, see §5.

`vec_to_list` rounds to 4 decimals for JSON.

### 8.2 Constant-velocity model

\[
P_{\mathrm{rel}}=P_{\mathrm{obj}}-P_{\mathrm{cam}},\qquad
V_{\mathrm{rel}}=V_{\mathrm{obj}}-V_{\mathrm{cam}}
\]

Converging iff \(P_{\mathrm{rel}}\cdot V_{\mathrm{rel}}<0\) (range rate negative).

Critical point of \(f(t)=\|P_{\mathrm{rel}}+t V_{\mathrm{rel}}\|^2\):

\[
\mathrm{TTC}=-\frac{P_{\mathrm{rel}}\cdot V_{\mathrm{rel}}}{\|V_{\mathrm{rel}}\|^2}
\quad\text{if }\|V_{\mathrm{rel}}\|\ge\varepsilon\text{ and converging}
\]

\[
D_{\mathrm{cpa}}=\|P_{\mathrm{rel}}+\mathrm{TTC}\,V_{\mathrm{rel}}\|
\]

Otherwise TTC is \(+\infty\) (JSON `9999.0`) and CPA falls back to current range.

`rel_speed_eps` default `1e-4`. Compared in **squared** form so there is no `sqrt` and no divide-by-zero. `converging_eps` default `0`. A pair with range rate in \([-\varepsilon,0)\) is treated as not converging if you raise it; the shipped config does not.

### 8.3 `relative_kinematics`

Returns a frozen `RelativeKinematics`: `p_rel`, `v_rel`, `distance`, `converging`, `ttc`, `cpa`. All in the frame that was passed in.

`planar=True` applies `vflat` to both \(P_{\mathrm{rel}}\) and \(V_{\mathrm{rel}}\) **before** the solve. `main.py` sets this for `park` and `plaza`. A 1.6 m eye-height offset is then not “clearance” from a trunk.

`v_cam = (0,0,0)` is supported. Then \(V_{\mathrm{rel}}=V_{\mathrm{obj}}\). A parked object has TTC \(=\infty\). Something thrown at a seated walker still solves.

`as_json_dict(to_yup=True)` is available but `build_frame_record` writes fields itself so it can also attach `world_position` of the **root**, which is a different point.

### 8.4 `classify_threat`

Priority order — a real hit can never fall through to SAFE:

1. **`CRITICAL_THREAT`** — converging, finite TTC/CPA, TTC < 2.5 s, CPA < 0.5 m.
2. **`NEAR_MISS`** — converging, TTC < 4.0 s, CPA in [0.5, 1.5] m.
3. **`SAFE_STATIC`** — \(\|V_{\mathrm{obj}}\|\le 0.05\) m/s **and** (range > 5 m **or** CPA > 1.5 m **or** not converging). A static object you are walking into still converges (\(V_{\mathrm{rel}}=-V_{\mathrm{cam}}\)) and can be CRITICAL. A static object with an in-between CPA (e.g. 0.3 m at TTC = 3.2 s) is promoted to CRITICAL rather than left in a hole between the buckets.
4. else **`SAFE_DYNAMIC`**.

A **stationary ego** inverts the static-on-gait case: \(V_{\mathrm{rel}}=0\), never converges, SAFE_STATIC at any range. Correct: neither body is moving. Objects that move toward a seated walker keep normal TTC / CPA.

Non-finite TTC/CPA cannot satisfy a `<` threshold. Speed NaN is treated as 0.

### 8.5 `intercept_velocity`

Unique constant \(V_{\mathrm{obj}}\) that meets the camera at time \(\tau\) with world-space miss `offset`:

\[
V_{\mathrm{obj}}=V_{\mathrm{cam}}+\frac{P_{\mathrm{cam}}-P_{\mathrm{obj}}}{\tau}+\frac{\mathrm{offset}}{\tau}
\]

CPA equals \(\|\mathrm{offset}\|\) under the constant-velocity assumption. Injectors **do not** usually call this in world space. They place the actor in Frenet so that at time \(\tau\) it occupies \(s_{\mathrm{cam}}(\tau)\) and \(L_{\mathrm{cam}}+\mathrm{cpa}\), which is the same idea expressed in the ribbon. The helper exists for tests and for any future world-space projectile.

### 8.6 Worked example

Walker at \((0,0,1.6)\), \(V_{\mathrm{cam}}=(0,1.2,0)\). Cube at \((0,6,0.3)\), \(V_{\mathrm{obj}}=(0,-1.2,0)\). Threat points brought to a common Z (or planar):

\(P_{\mathrm{rel}}\approx(0,6,0)\), \(V_{\mathrm{rel}}\approx(0,-2.4,0)\). Range rate \(= -14.4 < 0\). TTC \(= 14.4 / 5.76 = 2.5\) s. CPA \(= 0\). Label: **CRITICAL_THREAT**.

Same cube offset to \(x=1.0\): CPA = 1.0 m, TTC still 2.5 s → **NEAR_MISS** if TTC < 4.

Parked cube, walker as above: \(V_{\mathrm{rel}}=(0,-1.2,0)\), TTC = 5 s if 6 m ahead — not critical (TTC ≥ 2.5). At 2.4 m ahead, TTC = 2.0 s, CPA = 0 → CRITICAL. That is why a pothole on the gait works with \(V_{\mathrm{obj}}=0\).

### 8.7 Self-test

`python threat_math.py` exercises parallel, diverging, head-on, seated, planar, and taxonomy edge cases. Run it after changing any threshold or the planar branch.

---

## 9. `camera_kinematics.py`

Owns Perlin, the ego-mode draw, and the Blender camera’s 4×4 every frame.

### 9.1 `Perlin1D`

Ken Perlin 2002 fade \(6t^5-15t^4+10t^3\). Table size must be a power of two (default 256). Gradients are 1-D, uniform \([-1,1]\), duplicated so `(i+1) & mask` needs no wrap logic.

**Not** `mathutils.noise`. Blender’s hash is not guaranteed bit-identical across versions; injectors and the sim loop must see the same \(s(t)\) that was baked into the arc table.

`fbm` sums octaves with persistence and lacunarity, then divides by the geometric amplitude sum so the result stays in roughly \([-1,1]\) regardless of octave count.

Independent instances (different seeds) are used for yaw, pitch, roll, gait sidestep, speed wobble, actor wander, actor speed, and the grove wind. A shared seed would correlate head-scan with sidestep, which looks like a broken gimbal.

### 9.2 `CameraState`

Snapshot consumed by the annotator and written into `camera_data`:

| Field | Meaning |
| --- | --- |
| `t` | Episode time (s) |
| `position` | Eye, Blender world (Frenet + curb + gait). **No** Perlin translation |
| `velocity` | Finite difference of that eye (includes gait \(dZ/dt\)) |
| `tangent`, `right`, `up` | Road Frenet at current `s` |
| `pitch`, `yaw`, `roll` | Camera-**local** jitter + gait pitch bob (radians) |
| `matrix_world` | The 4×4 actually written to the object |
| `walk_speed` | Instantaneous **ground** speed; 0 seated / halt |
| `arc_length`, `lateral` | Frenet of the ego this frame |
| `mode` | `walk` / `diagonal_cross` / `erratic` / `seated` |

`pitch_yaw_roll` in JSON is these local angles, **not** a world IMU.

### 9.3 `EgoProfile` and `choose_ego_profile`

Drawn once per episode from `cfg['ego']`. Separated from `CameraRig` so it can be unit-tested without a camera object.

**`walk`.** Constant lateral, constant speed, gait bob on.

**`seated`.** `stationary=True` ⇒ `speed_at=0`, `travelled=0`, `max_time=1e6`, gait gain 0. Eye height drawn from `seated_eye_height_m`. Lateral is later nudged 0.35 m toward the building line (a bench sits back from the kerb).

**`diagonal_cross`.** `diag_frac` \(U(0.45,1.00)\) of the way to the opposite kerb. Window is `diagonal_span` as a fraction of episode length. Lateral ramps with smoothstep \(u^2(3-2u)\). Span is \(\max(2|L|, 2.5)\) so a centreline start still produces a real crossing. Stored as a **fraction**, not metres, because the profile is drawn before the rig knows this biome’s corridor width.

**`erratic`.** Sidestep amplitude \(U(0.22,0.80)\) m at \(U(0.10,0.38)\) Hz (fBm). Speed wobble \(U(0.15,0.55)\). 55% chance of a halt inside `hesitate_window_s` for `hesitate_duration_s`, clamped so it lands before the last 0.6 s.

`--ego-mode walk` rebuilds the profile with `mode_weights={walk: 1}`. Unknown names raise.

### 9.4 `CameraRig`

Constructor arguments: Blender camera object, road `PathSpline` (duck-typed: `evaluate` / `tangent` / `length`), `cfg`, `ep_rng`, nominal `walk_speed` (0 if seated), starting `lateral`, `EgoProfile`, `lateral_limit` from `corridor.lateral_limit(0.30, True)`.

`__post_init__`:

- Six Perlin seeds (`base+17, +101, +233, +331, +457`) and three phase offsets \(U(0,64)\) so two episodes with the same walk speed still differ.
- Writes lens / sensor / clip onto `cam_obj.data`.
- Reads `curb_height` from world cfg.
- Seated: lateral += `copysign(0.35, lateral)`.
- Builds the arc table if needed.

**`speed_at(t)`** is the single source of truth for “is the ego moving”. Arc table, gait amplitude, and exported `walk_speed` all read it. Seated → 0. Halt window → 0. Else `walk_speed * (1 + wobble * fBm(t*0.45))`, never negative.

**`_build_arc_table`** — cumulative trapezoid of `speed_at` at `_arc_dt=1/120` s over 90 s (~10k floats). Skipped entirely for `walk` and `seated` (closed form \(s=vt\)). Only modulated modes pay for it.

**`travelled(t)`** — distance walked since t=0. Linear interpolate the table, or `v*t`, or 0.

**`arc_length_at(t)`** — `sidewalk_s0 + travelled(t)`, clamped to `(0.05, length-0.05)`. `sidewalk_s0` is **3.0 m**. That keeps the first frame from sitting inside a building that starts at s=0, and it is the origin injectors use as “now”.

**`lateral_at(t)`** — walk/seated: constant (then clamp). Diagonal: smoothstep toward the far kerb. Erratic: base + `sidestep_amp * fBm(t * rate)`. Then `_clamp_lateral`.

**`max_time()`** — remaining = `length - s0 - 2`. Seated → `1e6`. No table → `remaining / max(v, 1e-3)`. With table → first sample whose travelled ≥ remaining. Never divides by a halted speed.

**`sample_angles(t)`** — three independent fBm streams scaled by `jitter[axis]`. Time is `t * frequency_hz + phase` so “slow yaw” really is slow.

**`_gait_gain(t)`** — 0 if seated; else `speed_at(t) / walk_speed` clamped to [0,1]. A bobbing camera with zero ground velocity is a strong, wrong cue: it looks like motion the labels deny.

**`gait_height`** — `eye + A * gain * sin(2π f t)`.

**`gait_pitch_bob`** — `amp * gain * cos(2π f t)` (quadrature: head dips at mid-stance).

**`predict_position(t)`** — eye **without** look-jitter. Injectors aim at this, not at a Perlin wobble. Includes gait Z.

**`predict_velocity(t)`** — central difference of `predict_position`.

**`update(t, dt)`** — the sim-loop call:

1. Frenet origin at `(arc_length_at, lateral_at)`. Origin Z is **curb**, not asphalt, even when a diagonal crossing is on the road — a known simplification (the walker does not step down 12 cm in the integrator). Park curb is 0, so it is exact there.
2. Re-orthogonalize `up = right × tangent`.
3. Add gait height.
4. Sample jitter + pitch bob.
5. `_compose_matrix`: columns `(right, up, −tangent)` so local −Z = +tangent. Jitter is Euler XYZ **in camera space**, then `rot_base @ rot_jitter`.
6. `_apply_camera_matrix`: decompose to loc/quat, set `rotation_mode='QUATERNION'`, write location, quaternion, scale 1, **then** `matrix_world`. Blender 5 ignores `matrix_world` alone when the object is still in Euler mode.
7. Velocity: frame 0 (or `dt~0`) is a central difference of `predict_position` (correct under diagonal / halt). Later frames: `(position - prev) / dt` of the **jittered** eye. That includes a tiny look-jitter translation of zero (jitter is rotation about the eye) and the gait \(dZ/dt\).

### 9.5 What is not in the camera

- No translation jitter. A translating HMD would desynchronise intercepts (injectors use the no-jitter eye).
- No roll from the road (the spline is planar).
- No collision with actors. The camera is a ghost. Occupancy keeps *other* bodies off its spawn cell; during the episode they can still enter the gait tube — that is the point of a jaywalker.

---

## 10. `projection.py`

Every frame, after `view_layer.update()`, this module turns world AABBs into pixel boxes and a threat point.

### 10.1 Bound cache

`LocalBoundCache` stores object-space `bound_box` once per `as_pointer()`. Each frame only does `matrix_world @ corner`. No `evaluated_get` — there are no mesh modifiers, and evaluated meshes allocate.

`bound_cache_for(obj)` is a dict lookup. `Actor._bounds` holds the handle for dynamic actors so the annotation loop does not re-hash every frame.

`clear_bound_caches()` at episode start **and** after `release_episode`. Pointers are reused; a stale cache would project a previous episode’s cube as this episode’s car.

Union of world AABB corners of the object **and every MESH child**. Pelvis-only would under-cover a person (arms, head) and a wheeled car.

### 10.2 Projection

Per frame, once:

- `view = cam.matrix_world.inverted()`
- `proj = Object.calc_matrix_camera(...)` if available, else the analytic OpenGL frustum from lens / sensor / aspect (`construct_projection_matrix`)

For each corner:

1. Camera space. **Z ≥ 0 is behind the lens** (Blender camera looks down −Z). Those corners drop.
2. Clip = `proj @ cam`. Discard `w ≤ 1e-8`.
3. NDC → pixels, origin **top-left**: \(u=(n_x+1)W/2\), \(v=(1-n_y)H/2\).

Box = integer min/max of surviving pixels, clamped to the image. `truncated` if any corner was off-screen or fewer than 8 were in front. All-behind or all-off-screen ⇒ `None` ⇒ object omitted from that frame (not listed with a dummy box).

This is a **3-D AABB**, slightly loose (a diagonal limb sticks out of the AABB). It is not a tight silhouette and not a segmentation mask.

### 10.3 Threat point

`threat_point_from_corners(corners, cam_z, mode)`:

- **`volume`** (people, cars, trees, shapes): XY of AABB centre, Z clamped into `[zmin, zmax]`, preferring `cam_z`. A car roof at 1.5 m vs a 1.6 m camera has a 0.1 m residual, not 1.6 m.
- **`footprint`** (pothole / crater / puddle / broken_slab): XY of centre, **Z = camera height** so the hole occupies the walker’s vertical column.

Trees: the 2-D box uses the full hierarchy (canopy included). The threat point and Frenet extents use `actor.threat_obj` (the trunk). Taking the canopy AABB as the threat point would put the hazard 1.5 m off the path and a swaying crown would jitter TTC.

Sunken classes (`pothole`, `crater`, `broken_slab`): `footprint_mouth_corners` builds a thin world slab (±3 cm) at the pavement mouth from the local XY radius. The render mesh’s buried well is ignored for the box. Without this, the box inflates downward and can miss the screen when only the mouth is in frame.

Static actors cache `_world_corners` once. Dynamics recompute from `matrix_world` every frame.

### 10.4 Occlusion

`raycast_occluded`: `scene.ray_cast` from camera toward the AABB centroid.

- Self and children do not count.
- Ground ribbons (`road_surface`, `sidewalk_*`, `curb_*`, `centerline` / `dash_*`, `lamp_pole`) are stepped through (up to 10 hits, 0.08 m epsilon) so a pothole is not “occluded by the sidewalk it sits on”.
- This is a **centre-ray**, not pixel coverage. A person 90% hidden by a trunk but with centroid visible is `occluded=false`. A person with centroid behind a pole but limbs visible is `occluded=true`. That is acceptable for a tactile-warning dataset; it is not a matting label.

`--no-occlusion` skips the ray. Boxes still compute.

### 10.5 `BoundingBox2D`

`xmin, ymin, xmax, ymax` integers; `truncated`, `occluded` bools; `as_dict()` for JSON.

---

## 11. `world_generator.py`

Largest module (~5k lines). Scene, actors, lighting, EEVEE, injectors. Coordinate frame: Blender Z-up. The sidewalk is a lateral offset, not a second arc-length.

### 11.1 Wind and foliage (visual only)

**`WindField`.** One Perlin per episode, constructed in `build()` **after** `choose_environment` so strength and direction match the drawn label. `state(t)` returns `(envelope, gust, dir_x, dir_y)`. Envelope = `strength * max(0, 0.52 + 0.48 * gust)`. Cached on `t` so every tree in a frame sees the same gust.

Label is drawn from `domain_randomization.wind_weights` (calm 0.32 / breeze 0.48 / windy 0.20) unless `--wind` or `BTP_WIND` locks it. Strength is then \(U\) of `wind_strength[label]`: calm `(0.02, 0.08)`, breeze `(0.28, 0.55)`, windy `(0.70, 1.00)`. Direction is a uniform world-XY yaw (`wind_dir_deg`). The grove shares this one field; per-tree phase / flutter add modal rustle.

**`FoliagePart`.** A branch empty or a single triangle leaf: object, rest Euler, phase, flutter, kind, flex.

**`foliage_wind_euler`.** GPU Gems 3 sum of sines. Branch joints: slow cantilever into the wind (`amp = 0.14 * env * flex`). Leaves: that plus 2–5 Hz rustle (`amp = 0.48 * env * flex`). Parenting is hierarchical, so a mid-limb bend moves the distal crown. Returns a local Euler **offset**. `_tick_visuals` does `rest + offset`.

**Invariant:** trunk pose and `Actor.velocity` stay 0. Moving leaves never enter TTC. Phase B already walks children, so sway is in the video.

### 11.2 `PathSpline`

Control polygon → cumulative length → uniform-\(s\) resample at `ds` (world default 0.40 m). `evaluate(s)` / `tangent(s)` are binary searches plus lerp. `right(s) = tangent × world_up`, renormalized. `frame(s)` returns `(p, tan, right)`.

`offset_point(s, lateral, z)` — Frenet point with explicit Z (does not use `ground_z`; callers pass curb or 0).

`offset_spline(lateral)` — a **new** spline through offset samples. Used only as a visual / legacy sidewalk handle. **Do not** take arc length from it.

`project(p)` — closest point is the foot of the perpendicular on each polyline **segment**, not the nearest sample vertex. Lateral is the planar offset along `right(s)`.

Factories (`PathSpline.generate`):

| Type | Construction |
| --- | --- |
| `straight` | `(0,0,0)` → `(0,L,0)` |
| `gentle_curve` | Cubic Bézier, bend \(U(8,20)\) m left or right |
| `s_curve` | Two opposing bends |
| `corner_90` | Fillet radius \(U(5.5,8)\) m, then a +X or −X finish |

Park biomes drop `corner_90` from `path_types` so a gravel path does not make a city block.

`_cubic_bezier` samples 80 points before resample. Density is enough for a 78 m path at ds=0.40.

### 11.3 `offset_folds`

On the inside of a tight corner, a large lateral offset lands back on the pavement (the offset curve cusps). Sampled at three stations × three laterals via `spline.project`. If the projected lateral collapses toward the road, the station is a fold. Buildings and planting trees **skip** that station. Without this, a facade or a trunk phases through the sidewalk on `corner_90`.

### 11.4 `StreetCorridor`

`confine(s, lat, pad, allow_sidewalk)` clamps `s` to `(0.05, length-0.05)` and `|lat|` to `lateral_limit`. Limit is `max_abs_lateral - pad` if sidewalk is allowed, else `road_half - 0.05 - pad`, floored at 0.20 m.

`max_abs_lateral` is set at build time to `road_half + sidewalk_w - corridor_margin` (margin 0.22 m). Facades sit further out at `road_half + sidewalk_w + building_setback`.

`ground_z(lat)` — curb on the sidewalk band, 0 on asphalt.

`world_to_sl` / `world` — project / offset convenience.

This is the **only** runtime spatial constraint on movers. It keeps a jaywalker on the ribbon. It does not keep two jaywalkers apart (that is `ComposeSession`).

### 11.5 `poisson_disk_strip`

Bridson 2007 in the `(s, lateral)` plane. Distance is Euclidean in parameter space, which is a close approximation to world metres on a gentle curve and an approximation on a tight corner (accepted). Cell size `radius/√2`, `k_candidates=20`. Used for sidewalk furniture and (historically) head hazards. Trees use a simpler reject loop (`_tree_free`) because they have role-specific along-track spacing.

### 11.6 Mesh helpers

`_link` — `collection.objects.link`. `ensure_collection` — get-or-create a child of the scene master.

`make_principled` / `assign_mat` — simple one-colour materials for poles and leftover primitives. Most surfaces go through `materials.py`.

`create_mesh` — unique datablock (people, buildings, one-off hazards). `create_box` — 8 vertices, box-projected UVs. `create_z_cylinder` — lamp-adjacent / unused much now.

`_box_project_uv` — object-space box UVs so a UV-less cube still gets storeys / tiles when a shader uses UV. Facade shaders actually use **Object** coordinates (see §13); the UVs are belt-and-braces.

`shade_smooth` — per-polygon `use_smooth`.

`yaw_from_xy` / `heading_from_tangent` / `look_along` — see §5.1.

### 11.7 `MeshLibrary`

Get-or-build mesh/material datablocks. `instance()` is `bpy.data.objects.new(name, shared_mesh)` plus per-object scale / yaw.

Why: 1000 grass clumps × unique meshes = 1000 vertex buffers. The library uploads one clump mesh (or a handful of sprig variants) and instances it.

Materials cached the same way. A shared mesh carries its material on the **mesh** slot. Per-instance colour (threat shapes) uses `_bind_object_material` so linked meshes do not all turn red.

`lib.stats()` prints `"{built} unique mesh datablocks, {reused} linked instances"` on cleanup.

A **fresh** library per `build()`. The previous episode’s datablocks were purged; holding them is a use-after-free.

Pedestrians are **not** instanced. They are unique articulated graphs.

### 11.8 `purge_orphans` / `reset_blender_scene` / `release_episode`

`purge_orphans(max_passes=8)` — recursive `orphans_purge` until a pass frees nothing or 8 passes. Node groups hang off materials and need the second pass.

`reset_blender_scene` — object-mode if possible, remove every object, sweep unused meshes/lights/cameras/materials/curves/worlds, unlink child collections, remove unused collections, purge. Returns the scene. Called at the start of every `build()`.

`release_episode` — see §4.5. Called at the **end** of `run_episode`, after pixels and JSON, so peak RSS of episode n+1 does not include episode n.

### 11.9 Environment draw

`choose_environment` — weighted lighting, weather, and wind; sun elev/azim/energy from the matching ranges; dappled coin-flip; chaos \(U(c_{lo},c_{hi})\); wind strength / direction. `BTP_LIGHTING` / `BTP_WIND`, if they name a known state, override the corresponding draw (debug). `--wind` locks `wind_weights` to one key before this function runs.

Six lighting states and **why they exist**:

| State | What it breaks in a detector |
| --- | --- |
| `dawn` / `dusk` | Warm, low, long shadows |
| `noon` | High key, short shadows, the “easy” domain |
| `night` | Moon-energy sun, sparse tinted spots, some lamps dead |
| `harsh_glare` | Horizon sun, 18–42 energy, AgX exposure −0.45. Sun azimuth is forced down the gait (`glare_azimuth_deg`) so the disc is in the walker’s eyes, not behind them |
| `overcast` | Turbidity ≥ 9, sun angle 60°, sun shadows **off**. Contrast dies |

`glare_azimuth_deg(tangent)`: a Blender sun with `rotation_euler = (90°−e, 0, a)` emits along \(d=(-\sin a\cos e,\;\cos a\cos e,\;-\sin e)\). The disc sits at \(-d\). To put it on the walker’s line: \(a=\mathrm{atan2}(T_x,-T_y)\).

Weather (`clear` / `light_fog` / `heavy_smog`) only changes sky turbidity / aerosol. **No world Volume Scatter** — that is full-frame black in EEVEE. `volume_density` in older comments is unused on purpose.

### 11.10 Lighting graph (`apply_domain_randomization`)

Three SUNs + sky + view transform:

1. **Key sun** — energy / colour / angular diameter per state. Night: cool, faint, `energy = max(3e, 0.25)`. Glare: warm, huge energy, 0.4° disc. Overcast: white, 60° disc, `use_shadow=False`. Day: energy ∝ `sin(elev)`, colour lerps warm→white.
2. **SkyFill** — opposite azimuth, 58° elevation, `use_shadow=False`. Shadow page cap: fill must not consume cascade pages.
3. **GroundBounce** — 165° elevation (from below), brown/grey, no shadows. Stops the shady facade from being a black slab.

`apply_streetlamp_state` — night only, honouring `btp_lamp_dead` / `btp_lamp_gain` stored on the object at spawn. Every other live lamp casts (`i % 2 == 0`) so the shadow pool (2048) survives. This function runs **twice**: once inside lighting, once after `reveal_view_layer` un-hides everything. The two calls must agree; that is why dead/gain are on the object, not re-rolled.

`_setup_world_shader` — Background + Sky Texture. Blender 5 sky type is `MULTIPLE_SCATTERING`, not `NISHITA` (Nishita was 2.9–4.x). Sun elevation/rotation driven from the same elev/azim. Turbidity / dust from weather. No volume node.

`_apply_view_transform` — AgX. Exposure: glare −0.45, overcast +0.12, dawn +0.10, dusk +0.08, noon 0, night left at 0 (lamps carry the image). Gamma 1 if the attribute exists.

### 11.11 EEVEE (`configure_eevee`)

`available_render_engines` reads the scene’s enum. `pick_render_engine` prefers the requested name if present, else `BLENDER_EEVEE`, else `BLENDER_EEVEE_NEXT`, else whatever is first. Hard-coding Next on 5.2.1 raises `TypeError`.

TAA 16 + reprojection. Raytracing method **`SCREEN`** (not `'SCREEN_TRACE'`, a silent no-op on 5.2). Fast GI 4 rays / 6 steps / quality 0.30. Shadow pool 2048, cascade 48 m. All `setattr` `hasattr`-guarded so 4.2 and 5.2 both run.

`reveal_view_layer` — new collections start excluded in 4/5. Walks the layer collection tree and sets `exclude=False`, `holdout=False`, `indirect_only=False`. Then daytime lamps are put back to sleep.

`prepare_still_render` — `use_sequencer=False`, `use_compositing=False`, PNG compression, filepath. Blender 5 defaults both sequencer and compositing **True**. An empty VSE renders **pitch black** with valid JSON. This is the #1 “black PNGs” cause.

`_gpu_backend_report` — prints device string once so a hybrid-laptop log shows whether PRIME worked.

### 11.12 Ribbons and buildings

`_ribbon_mesh(name, spline, lat_a, lat_b, z, col, mat, z_amp=, rng=)` — a strip of quads along the spline between two laterals. Plaza sidewalks get a tiny `z_amp` (0.012) so the pavers are not a perfect plane (catches light). Names `road_surface`, `sidewalk_L/R`, `curb_L/R`, `verge_L/R` are in the occlusion ignore list (except verge).

`_centerline_dash` — 3 m mark, 3 m gap, 0.12 m wide, Frenet quads. World-axis boxes would chord across a corner.

`_extrude_buildings` — from `s=14` m (a 24 mm lens at `s0=3` m must not be filled by one wall) to near the end. Facade lateral = `sign * (road_half + sw + building_setback)` plus per-lot setback jitter. Depth / width / height from config ranges. Gap \(U(0.4,2.2)\) plus explicit `gaps` from `prepare_scenario` (crossers open `[4,20]` or `[4,28]`). Massing is cheap extra boxes: box, stepped, L-plan, arcade recess, sloped roof. `offset_folds` skip. Night: `make_facade(..., night=True)` so windows emit.

### 11.13 Trees (`spawn_tree` + `_scatter_trees`)

Weber–Penn / Honda recursive tree (1995 / 1971). `actor.obj` is an **unscaled root empty** (lean + heading). The trunk mesh is a child with **no children of its own** and is `threat_obj`, so TTC uses the bole. Limbs are unscaled empties + instanced tapered tubes. Children spawn along the parent (monopodial, golden-angle 137.5°) and the tip splits in two or three (dichotomous). Leaves are instanced twig sprays (`_leaf_spray_geom`): many small kite triangles hung along the twig, not a ball at the origin. The 2-D box is the wood + sprays. Do not add a dummy crown hull — an unshaded blob still draws in EEVEE even with `hide_render`.

Shapes: `round` (spherical ShapeRatio), `conical` (fir: central leader + whorls), `columnar`, `spreading` (hemispherical), `bare` (wood only). Small crowns (`max_canopy < 1.15`) use fewer stems / shallower depth.

Lean: planting / curb trees lean **toward the road** on the **root** (`rotation_euler Y = -copysign(lean, lat_sign)`). Heading is the root yaw. The trunk mesh stays identity in local space.

`Actor` for a tree: `category='static'`, `class_name='tree'`, `obj=root`, `threat_obj=trunk`, `threat_mode='volume'`, `foliage=[...]`, `wind=WindField`, `annotatable=True`. Velocity stays 0.

Roles in `_scatter_trees`:

| Role | Config key | Where | Crown cap | Along-track spacing |
| --- | --- | --- | --- | --- |
| `plant` | `n_trees` | Planting strip (facade − crown − 0.40), or park field | 2.8 / computed | 7.5 m |
| `curb` | `n_street_trees` | Planting strip, sidewalk-adjacent (`road_half + sw + U(0.35,1.15)`). Never on asphalt. Default counts are `(0,0)`. | 1.35 m | 8.0 m |
| `median` | `n_median_trees` | Only if `median_width ≥ 1.4` (avenue grass strip). Rejected if the trunk would sit on a driving lane. | ≤ half-median | 10.0 m |
| `path` | `n_path_trees` | Park **verge**, off the gravel gait (`|lat| ≥ road_half + 0.70`) | 1.55 m | 6.0 m |

A hard reject (`_on_drive_or_walk`) drops any candidate whose `|lat|` is on the carriageway or the walking slab.

`_tree_free(s, lat, along, xy=3.8)` — along-track spacing in the **same strip** (`|Δlat|` small) uses `along`; Euclidean XY uses 3.8 m. A planting tree at lat 7.5 must **not** ban a median trunk at lat 0. Early versions used one Euclidean radius and rejected every median tree.

`offset_folds` applied to plant role so a corner does not put a trunk on the pavement.

`--no-trees` zeroes all four count keys on world and strips them from biome overrides so a biome cannot put them back.

### 11.14 Grass, gobo, lamps

`scatter_grass` — instanced tufts. `exclude_abs_lat` keeps them off asphalt and walking slabs. Park: field of 60–160 clumps. Street: 8–28 on the verge.

`_spawn_canopy_gobo` — one alpha-hashed ribbon at 6.5–10.5 m. Hidden from camera / diffuse / glossy if those flags exist; `visible_shadow=True`. Models “avenue of plane trees” as a moving shadow, not 10k leaves.

`_spawn_streetlamps` — pole + arm toward the carriageway + emissive bulb + downward SPOT. Alternating sides. `_site_free` vs trunks. Dead/gain custom props. Daytime: energy 0, hidden.

### 11.15 Furniture and head hazards

`FURNITURE_CLASSES` = trash can, scooter, barricade, puddle (visual). `_scatter_ground` Poisson-samples the **shop-front half** of the sidewalk (outer band), not the curb walking line. `_site_free` vs trees/lamps. Puddles are `annotatable=False` (a sheen is not a trip in this taxonomy). Head-height boxes (`tree_branch`, `ac_unit`, `sign`, `truck_door`) only if `n_head_hazards > 0` (default `(0,0)`). They read as junk on the gait. `head_level_projectile` is a **scenario**, not this scatter.

### 11.16 Background traffic

**Vehicles** — opposite lane so CPA stays > 1.5 m. If median trees exist, lane is pushed outward (`_carriage_free` rejects a spawn whose XY hits a trunk). Speed \(U(5,9)\) street. Light Perlin wander. `allow_sidewalk=False`.

**Pedestrians** — other sidewalk, or offset on the same one. Speed \(U(0.90,1.45)\). `_add_wander` attaches `Perlin1D` + amplitude. `allow_sidewalk=True`.

`_add_wander` is also used on some injected extras so a compound street does not look like a train set.

### 11.17 Spawn factories

| Function | What it builds | Root | Threat |
| --- | --- | --- | --- |
| `spawn_humanoid` (via `spawn_pedestrian`) | Articulated person | pelvis | volume, whole body |
| `spawn_vehicle` | Superquadric hull + glass + 4 +X wheels | body | volume |
| `spawn_bicycle` | Frame + wheels + rider-less | frame | volume |
| `spawn_tree` | Recursive forks + triangle leaves | root empty | `threat_obj=trunk` (no foliage children) |
| `spawn_ground_hazard` | Pothole well / crater bowl / tilted slab / debris | mouth | `footprint` (debris volume) |
| `spawn_threat_shape` | Unit cube/sphere/cylinder/pyramid/cone/capsule/lump | centre | volume; class `threat_<kind>` |
| `spawn_threat_cube` | Thin wrapper → `kind='cube'` | | |
| `spawn_head_hazard` | Floating box | centre | volume |
| `spawn_bench` | Seated-ego prop | | `annotatable=False` |
| `spawn_projectile` | Eye-height cube | centre | volume, class `projectile` |

`SHAPE_KINDS` = `cube, sphere, cylinder, pyramid, cone, capsule, lump`. `kind in {shape, random, any, ''}` draws uniformly. Aspect jittered by chaos. Colour from a fixed saturated palette so the detector cannot key on “orange cube = threat”. `_bind_object_material` so instanced unit meshes do not share albedo.

`_Counters.next_id(class_name)` → `person_000`, `threat_sphere_001`, …

`_tag` writes `instance_id` and `class_name` as custom properties (debug in a `.blend` dump; JSON uses the Actor fields).

### 11.18 `Actor` motion model

See the field table in §11.19. `update(t, dt, corridor)`:

1. If `static` or `stopped`: velocity = 0, `_tick_visuals`, return. Foliage still sways. Gait goes to idle.
2. If `t >= stop_t`: latch `stopped`, same as (1).
3. If `follow_spline` is set (all street actors):
   - `_commanded_rates`: `(ds, dlat)` from `speed` / `lat_speed`→`lat_target`, with a smoothstep blend after `turn_t` into `post_speed` / `post_lat_*`. `swerve_t` holds lateral rate at 0 until that time (cut-in / swerve).
   - Multiply `ds` by `_speed_scale` (erratic fBm, never negative).
   - Integrate `s += ds*dt`. Integrate lateral toward target without overshoot. Then add `_wander_rate*dt` (differenced displacement; first sample returns 0 so frame 0 does not teleport).
   - `corridor.confine`.
   - `location = p + right*lat`, Z = `origin_z + ground_z(lat)`.
   - `velocity = tan*ds + right*(vlat + v_wander)` — wander is in the **reported** velocity so TTC sees the true instantaneous motion.
   - `look_along` on the horizontal heading (or `±tan` if stopped in s).
   - **Then** `_tick_visuals` (gait / wheels / foliage).
4. Else `hold_velocity` world step (legacy; street actors should not hit this), or raw `location += velocity*dt`.

`finite_velocity(dt)` — fallback if `velocity` is still ~0 on a mover that is not marked stopped. Used by `build_frame_record` so a first-frame actor still gets a TTC.

### 11.19 `Actor` fields

| Field | Role |
| --- | --- |
| `obj` | Blender root |
| `instance_id`, `class_name` | JSON identity |
| `category` | `static` / `dynamic` — static skips integration |
| `velocity` | World m/s, Z-up, written every update |
| `speed` | Commanded \(\mathrm{d}s/\mathrm{d}t\) (signed; negative = oncoming) |
| `follow_spline`, `s`, `lateral` | Frenet state |
| `origin_z` | Added to corridor ground (pelvis, cube centre) |
| `behavior` | Tag for occupancy / debug (`through`, `oncoming`, `cut_in`, …) |
| `swerve_t` | No lateral rate until this time |
| `stop_t`, `stopped` | Sudden stop |
| `hold_velocity` | World-space fallback |
| `annotatable` | False → omitted from JSON (puddle, bench) |
| `threat_mode` | `volume` / `footprint` |
| `threat_obj` | Sub-object for threat point (tree trunk) |
| `gait` | `WalkRig` |
| `wheels` | `(obj, radius)` |
| `lat_speed`, `lat_target` | Jaywalk / cut-in |
| `corridor_pad`, `allow_sidewalk` | Confine |
| `wander_*`, `weave_*` | fBm drift vs deterministic sine |
| `speed_noise`, `speed_amp` | Erratic \(\mathrm{d}s/\mathrm{d}t\) |
| `_bounds`, `_world_corners` | Projection caches |
| `turn_t`, `turn_dt`, `post_*` | Smooth heading blend (jaywalk-turn, run-off-road) |
| `foliage`, `wind` | Visual only |

### 11.20 `WorldGenerator` lifecycle methods

**`__init__`.** Holds cfg, rng, empty lib, `_sites`, `_wind`, compose session, mover caches.

**`prepare_biome`.** Weighted or requested name. Copies biome dict onto `cfg['world']`. One substrate: a park “lane” is a lateral band of a 3 m gravel path. Injectors do not branch on biome except where `ground_z` / planar TTC already handle it.

**`prepare_scenario`.** Records slug, opens building gaps if any name is in `CROSS_GAP_SCENARIOS`, sparsifies if `empty_street` is in the mix, sets `force_ego_mode="crosswalk"` for `CROSS_EGO_SCENARIOS`.

**`build`.** Reset scene, configure EEVEE, new `MeshLibrary`, `_wind = None`. Draw path and sidewalk lateral. `choose_environment`, then construct `WindField` from that draw (not before — an early field would ignore `--wind`). Ribbons, buildings, trees (which receive `_wind`), lamps, grass, lighting, furniture, background traffic. Fold `wind` / `wind_strength` onto the environment dict stored in `WorldState`. Returns `WorldState`.

**`create_camera`.** Empty camera object, linked, clip/lens later overwritten by the rig.

**`place_ego_props`.** If seated, spawn a bench at the (nudged) lateral, not annotatable.

**`freeze`.** `_movers` = actors that are dynamic, or have gait, wheels, foliage, or `stop_t`. `_annotatable` = `annotatable=True`. Invalidated on `_append`.

**`update`.** `mover.update` for each mover with the corridor.

**`annotatable`.** Frozen list, or a live filter if freeze was skipped (should not happen).

**Site occupancy** (`_occupy`, `_site_free`, `_carriage_free`, `_tree_free`, `_count_range`) — cheap XY / along-track reject for trees, lamps, furniture, background cars. Independent of `ComposeSession` (which is for **injected** dynamics vs each other and vs a seed of nearby background).

### 11.21 Injector plumbing

`_scenario_handlers` — canonical name → callable. Built once per `inject_scenarios`. Closures capture `near_miss_cpa_target` (1.0) and `critical_cpa_target` (0.12).

`inject_scenarios`:

1. Resolve aliases.
2. Unknown names raise with the known list.
3. `ComposeSession.from_cfg` with `s_lo = s0+3.2`, `s_hi = min(length-5, s0+look_ahead)`.
4. `seed_from_actors` — background in the camera-relevant band become capsules tagged `"background"`.
5. `sort_for_inject` — static → along-track → lateral (stable, user order preserved inside a bucket). Folder slug keeps **user** order (`compose_slug`).
6. For each name: `session.begin`, skip noops, call handler.
7. Clear `_compose`. Return slug.

Shared helpers:

- `_spawn_kind(kind, s, lat, heading_sign, cube_size, cube_z)` — person / vehicle / bicycle / shape.
- `_bind(...)` — confine, `reserve` if composing (may flip lat / lane / nudge s), write speed / lat / turn / stop, snap location + `look_along`, `_append`.
- `_cam_s` / `_cam_sl` — prefer `rig.arc_length_at` so hesitation is in the intercept.
- `_ego_speed` — may be 0. Never floored to 0.25.
- `_gait_lat` — `state.sidewalk_lateral`.
- `_near_lane` / `_far_lane` — `±lane_offset`, sign relative to walker. `prefer_far_lane` can push a generic oncoming car out of a jaywalker’s ribbon.
- `_resolve_lat(dlat, L, pad)` — `dlat` may be `"opposite"`, a float offset, or an absolute.
- `_kind_pad` — vehicle 1.05, bicycle 0.40, shapes ~0.35, person 0.35.
- `_frustum_half_width(depth, frac)` — `depth * tan(0.5*hfov*frac)`, min 1.20 m. Through-crossers spawn on a FOV edge, not a building face.
- `_scale_person` — uniform scale for `child_darting` if the child mesh path is not used; the real child path uses `spawn_humanoid(child=True)`.
- `_inject_oncoming` — spawn at \(s_0+\max(v_{\mathrm{ego}}+v_{\mathrm{obj}},v_{\mathrm{obj}})\tau\), lateral = gait + `dlat` (or opposite sidewalk). Speed is **negative** (toward the camera) for oncoming kinds. Seated still gets \(v_{\mathrm{obj}}\tau\) ahead.
- `_inject_through_cross` — enter one FOV/corridor edge, `lat_target` the other edge, `speed=0` (pure lateral), `behavior='through'`. Depth 3.8–7 m plus compose stride. CPA is a small lateral graze of the gait line, not a teleport.
- `_inject_jaywalk_turn` — side entry, then `turn_t` blends into along-track toward or away from the camera.
- `_inject_static_shapes` — `n=1` one cube; `n=0` draws 2–4 mixed shapes. On gait, 2.55 m apart, lead 5.4 m + static stride.
- `_inject_pothole` — lead from config, `dlat` 0 / 0.85 / 1.80. On-gait: pothole/crater/broken_slab. Offset may also be debris.
- `_inject_car_lane` / `_inject_parallel_person` / `_inject_cyclist_same_way` / `_inject_distant_jaywalk` / `_inject_parked_car` / `_inject_sudden_stop` / `_inject_cut_in` / `_inject_weaving` / `_inject_run_off_road` / `_inject_child_dart` / `_inject_head_cube` — see §15.

There is **no** per-frame N-body. If two injectors still overlap after 14 nudges, `reserve` commits the last candidate (best-effort). The corridor clamp still keeps everyone on the street.

---

## 12. `scenario_compose.py` (no `bpy`)

CLI parse + Frenet occupancy. Injector **bodies** stay in `world_generator.py` so this file can be tested with system Python.

### 12.1 Catalogs

**`CROSS_GAP_SCENARIOS`.** Lateral travellers that need a building gap so they are not born in a facade. Includes all jaywalk variants, crossing cars, cube/shape from left/right, `child_darting`, `distant_jaywalk`, `group_crossing`, `crossing_car_side`, `scooter_from_sidewalk`. `prepare_scenario` opens `[4,20]` m, or `[4,28]` if two or more.

**`CROSS_EGO_SCENARIOS`.** `crossing_street`, `crossing_car_side`, `group_crossing`, `crossing_head_on`. `prepare_scenario` sets `force_ego_mode="crosswalk"` so the walker turns onto the ribbon in Frenet `(s, lateral)` (heading follows the motion vector).

**`THROUGH_CROSSERS`.** Occupies every lane at a fixed `s` over a few seconds. Used to send a generic `car_approaching` to the far lane (`prefer_far_lane`).

**`SPARSE_SCENARIOS`.** `{empty_street}` — zeroes background peds/cars and most clutter.

**`NEAR_LANE_LOCKED`.** Cut-in / graze / swerve / weave / run-off **must** keep the near lane. Occupancy staggers `s` instead of flipping the lane.

**`SCENARIO_ALIASES`.** Short names for compounds: `car`→`car_approaching`, `pothole`→`pothole_on_path`, `cube`→`cube_near_miss`, `cubes`→`cube_on_path`, `shape`→`shape_near_miss`, `shapes`→`shapes_on_path`, `sphere`/`pyramid`→`shape_head_on`, `child`→`child_darting`, etc. `--list-scenarios` prints the full map.

**`_STATIC_CLASSES` / `_is_static_class`.** Potholes, trees, all `threat_*`. Static capsules only block the **spawn cell** (`t=0`), not the 5 s tube. Walking past a hole is realistic; treating the tube as solid shoved jaywalkers ~10 m down the road.

**`_INJECT_PRIORITY`.** Lower runs first. 0 = static holes/shapes, 1 = parked, 2 = noops, 10 = along-track, 20 = lateral. `sort_for_inject` is a stable sort on this. User order is preserved inside a bucket and in the folder slug.

**`_EXTENT_S` / `_EXTENT_LAT`.** Half-extents for occupancy. Unknown / `threat_*` map to `threat_cube` (0.48 × 0.36). `extents_for(cls, pad)` adds pad into the returned pair.

### 12.2 Parsing

`split_scenario_tokens` — commas, plus signs, whitespace; lowercased; `-` → `_`.

`resolve_scenario_name` — alias, or canonical, or error.

`pick_one_auto` — 40/30/30 over the three pools.

`pick_scenarios` — `auto` / `random` / empty → one auto draw. Else split, resolve, drop empties.

`pick_scenario` — first of `pick_scenarios` (legacy single-name API).

`compose_slug` — user order, `_` joined, truncated to 72 chars for folder names.

`compose_display` — same for the log line.

`is_noop` — `safe_walk` / `empty_street` (empty still sparsifies in `prepare_scenario`; the injector is a no-op).

`scenario_request_from_tokens` — joins repeated `--scenario` and `--scenarios`.

### 12.3 `FrenetCapsule`

Axis-aligned rectangle in `(s, lateral)` that moves with the **same piecewise rates** as `Actor.update`. If the occupancy predictor disagrees with the integrator, compounds that looked free at inject time interpenetrate on camera. The closed form is therefore a contract:

1. Along-track: \(s(t)=s_0 + (\mathrm{d}s/\mathrm{d}t)\,t\) until `turn_t`. After `turn_t`: \(s = s_0 + (\mathrm{d}s/\mathrm{d}t)\,t_{\mathrm{turn}} + v_{\mathrm{post}}(t-t_{\mathrm{turn}})\). The occupancy helper does **not** model the 0.85 s smoothstep — it snaps the rate at `turn_t`. That is slightly conservative (the real actor is still blending) and is accepted.
2. Lateral: `_lat_after(lat0, target, speed, t)` integrates toward the target without overshoot, same as `Actor.update`. If `swerve_t` is set, lateral time is 0 until that instant, then `t - swerve_t`.
3. After `turn_t`, lateral target becomes `post_lat_target` (else the original target) at `post_lat_speed`.

`Placement` is the triple `(s, lat, lat_target)` returned to `_bind`. `_bind` may then overwrite the actor’s `lat_target` with the reserved one when occupancy flipped a crossing.

### 12.4 `ComposeSession`

Per-episode reservation board. Cost is \(O(n_{\mathrm{actors}}\times n_{\mathrm{samples}}\times n_{\mathrm{nudges}})\) **once**, not per frame. Defaults come from `cfg['scenarios']['compose']`.

**Seeding.** `seed_from_actors` copies every background actor in \([s_0-2,\,s_{\mathrm{hi}}+4]\) as a capsule tagged `"background"`. Those capsules only block the **spawn cell** (`t=0`). A far-lane cruiser must not shove every jaywalker 10 m down the road.

**Stagger, not N-body.** Before a handler even calls `reserve`:

- `take_group_offset("cross"|"along"|"static")` adds \(n\times\) stride to the next member of that family (4.0 / 3.2 / 2.4 m). First member gets 0.
- `take_cross_layout(from_left)` returns `(n * cross_stride, side)` and **alternates entry side** after the first crosser so two jaywalkers do not occupy the same ribbon cell from the same kerb.
- `prefer_far_lane` sends a generic `car_approaching` / `car_pass_far` to the far lane when any through-crosser is in the mix. `NEAR_LANE_LOCKED` scenarios refuse that flip; they stagger `s` instead.

**`reserve` search order** (keeps the intended depth band as long as possible):

1. Requested `(s, lat)`.
2. Heading flip: swap start/target if `allow_flip_lat` (through-crossers).
3. Lane flip: `-lat` if `allow_lane_flip` (generic oncoming cars).
4. Walk `+s` in `nudge_s` (2.6 m) steps, up to `max_nudges` (14).
5. Short `−s` search (3 steps) so a crowded street can still place slightly closer.
6. If nothing is free, **commit the original anyway**. Occupancy is best-effort. The corridor clamp still keeps everyone on the street. A hard failure here would drop a requested critical event, which is worse than a rare overlap.

**Conflict test.** Two capsules conflict if at any sampled time their expanded rectangles overlap: `|Δs| < half_s1+half_s2+0.20` and `|Δlat| < half_lat1+half_lat2+0.18`. Sample times are `{0}` when either capsule is background or a static class; otherwise 5.0 s at 0.12 s (about 42 samples).

There is **no** per-frame N-body. If you add a new motion mode to `Actor.update` (a second weave, a teleport, a speed ramp that is not `post_speed`), you must teach `FrenetCapsule.pose_at` the same thing or compounds will lie.

`python scenario_compose.py` asserts aliases, extents for `threat_*`, inject priority, and a few reserve / flip cases.

---

## 13. `humanoid.py`

Articulated pedestrian. Faces +Y, +Z up. Root = **pelvis** (that is `Actor.obj`). A rectangular torso has a constant silhouette under yaw — the detector then keys on that rectangle. This module exists so “person” is an articulated biped, not a 1.7 m box.

### 13.1 Why these surfaces

Drillis & Contini / NASA-STD-3000 fractions of stature \(H\). Barr superquadrics for head / pelvis / hands: \((e_1,e_2)=(1,1)\) is a sphere; `(0.5, 0.6)` is a rounded hip without a hard edge. Torso is a loft of elliptical stations. Limbs are tapered capsules along **local −Z**. Neck is a **+Z** column — a −Z capsule grows into the chest (pitfall #14). Shoes are a lofted last.

`spawn_humanoid(..., child=False)` draws adult \(H\sim U(1.58,1.84)\) at chaos 0 and widens toward 1.15–2.05 m as chaos → 1, plus width/depth/hip scales. Proportions stay internally consistent (limbs remain fractions of \(H\)), so the Winter step length \(0.41H\) is still valid at every size.

`child=True` (`child_darting`) switches to a ~7-year-old station set: \(H\sim U(1.10,1.32)\), head fraction 0.090 (adult 0.068), shorter legs. It does **not** uniformly scale an adult — that would keep adult limb ratios and look like a doll.

Skin / hair tones are small palettes. Cloth goes through `make_patterned_cloth` so two pedestrians in one frame are not the same albedo.

Returns `(pelvis_root, WalkRig, hip_height)`. `origin_z` on the Actor is the hip height so `ground_z + origin_z` puts feet on the slab.

### 13.2 `WalkRig` (Winter 1991, reduced)

Step length \(\ell\approx 0.41H\), \(f=|v|/\ell\), floored at 0.8 Hz. Phase \(\phi=2\pi f t+\phi_0\).

| Joint | Law | Why |
| --- | --- | --- |
| Hip | \(A_{\mathrm{hip}}\sin(\phi+\{0,\pi\})\) | Opposite legs |
| Knee | \(-A_{\mathrm{knee}}[\max(0,\sin(\phi+\alpha+\{0,\pi\}))]^p\) | Half-wave: no hyperextension |
| Shoulder | \(-A_{\mathrm{arm}}\sin(\phi+\{0,\pi\})\) | Antiphase with **ipsilateral** hip (contralateral swing) |
| Elbow | \(-A_{\mathrm{elb}}(0.40+0.60\,\mathrm{sw}_{\mathrm{opposite}})\) | Flexes with the opposite knee |
| Ankle | \(A_{\mathrm{ank}}\sin(\phi+\beta)\) | Small; reads as push-off |
| Pelvis list / pitch / yaw | after `look_along` | Heading is Euler Z; list/pitch are X/Y |
| Thorax yaw | \(-0.65\times\) pelvic yaw | Reciprocal trunk rotation |
| Bob | \(A_{\mathrm{bob}}|\sin\phi|\) added to pelvis Z | Two peaks per stride, always up at double support |

Amplitudes (radians / metres): hip 0.40, knee 0.88, arm 0.44, elbow 0.38, ankle 0.22, list 0.055, pitch 0.035, yaw 0.070, bob 0.018. Knee phase 0.35, power 1.15.

Idle / `stopped` / `|speed|<0.08`: all joint angles 0, pelvic X/Y 0, thorax yaw 0. Called **every** frame including 0, and **after** `look_along` (which zeroes Euler X/Y). Calling it before `look_along` silently discards list/pitch.

Wheels on cars/bikes are not in this file. They live on `Actor.wheels` and roll \(\Delta\theta=-\Delta s/r\) about local +X. The bottom of a +X-axis wheel must move −Y (local) for +Y travel; the sign is easy to get backwards (the car then moonwalks).

---

## 14. `materials.py`

Procedural PBR. **Object** metres (UV-less cubes still get storeys). No image textures. Domain randomization lives here: the low-poly meshes are what they are; colour, roughness, window occupancy, and cloth patterns are what stop a detector from memorising “grey box = building”.

### 14.1 Blender 4/5 socket traps

These wasted real days and will waste yours:

- `ShaderNodeMix` has stacked sockets that **share a name**. `inputs["A"]` is the **float**, not the colour. A facade that silently mixes two greys is this bug.
- Noise Texture outputs **`Factor`**, not `Fac` (the 2.7 name).
- Math nodes have two sockets named `Value`. Night-window strength uses `_math_in(node, 1)` (the second operand).
- Helpers `_input` / `_output` / `_link` / `_link_to` pick by `(name, type, enabled)` and skip disabled sockets. **Always** use them. Never `node.inputs["A"]` or `noise.outputs["Fac"]`.

`_new_mat` builds a material with nodes, returns `(mat, tree, output, bsdf)`. `_object_coords` is a Texture Coordinate → Mapping chain in object metres. `_mix_rgba` is a Mix node forced to RGBA. `_bump_from` wires a height into the BSDF normal.

### 14.2 Factories and why each exists

| Factory | Used on | Notable law |
| --- | --- | --- |
| `make_asphalt` | Street carriageway | Voronoi + noise; not a flat grey |
| `make_concrete_tiles` | Sidewalk, park path | Object-space brick; `tile_m` randomised |
| `make_grass` | Verge ribbons | Green noise, high roughness |
| `make_facade` | Buildings | Object-normal window lattice + **per-cell occupancy hash**, not smooth noise. After `look_along`, world \(\hat{x}\) is not the street face — object normals are. Night: emissive cells. Smooth noise made whole floors glow as one smear |
| `make_car_paint` / `make_chaos_car_paint` | Vehicles | Clearcoat; chaos flakes / hue |
| `make_glass` | Car windows | Transmission, slight tint |
| `make_skin` | Humanoid | SSS-ish principled, not plastic |
| `make_cloth` / `make_patterned_cloth` | Garments | Chaos picks stripes / checks / noise |
| `make_water` | Puddles | Transmission + ripple normal |
| `make_foliage` / `make_bark` | Trees | Two-sided-ish leaf, rough bark |
| `make_canopy_gobo` | Overhead dapple sheet | Alpha-hashed; shadow only |
| `make_emissive` | Lamp bulbs, night windows | Strength in W-ish EEVEE units |
| `make_roof` / `make_rubber` / `make_simple` / `make_metal_paint` | Roofs, tyres, poles | One-layer principled |
| `make_chaos_surface` | Threat shapes | Roughness + hue wander so “orange cube” is not a class cue |

### 14.3 Chaos dial

`chaos ∈ [0,1]` interpolates tame → anarchy. CLI `--chaos` locks `material_chaos` to `(c,c)` for every episode; else each episode draws \(U(0.20,1.00)\).

`pick_family` — weighted cloth/paint family as chaos rises (more patterns, more hue).

`chaos_albedo(rng, base, amount, family)` — HSV jitter. `amount` is often `chaos * 0.45` for roads (a magenta street is allowed at 1.0 but not forced at 0.2).

`apply_surface_chaos` — extra noise / roughness on an existing BSDF.

`--chaos 0` is the pre-chaos pipeline: useful as an ablation (“did the detector need anarchy, or was the geometry enough?”).

---

## 15. Injector catalog

`prepare_scenario` runs **before** `build()` so gaps exist in the mesh. Injection runs **after** the rig exists so `predict_*` / `_cam_s` match the episode that will actually play.

CPA targets: near-miss **1.0 m**, critical **0.12 m** (`near_miss_cpa_target` / `critical_cpa_target`). Those are lateral offsets of the gait line, not the point-mass CPA the annotator will later compute — but under constant Frenet rates they agree to centimetres.

### 15.1 Shared placement laws

**Oncoming** (`_inject_oncoming`, also cars via `_inject_car_lane`):

\[
s = s_0 + \max(v_{\mathrm{ego}}+v_{\mathrm{obj}},\,v_{\mathrm{obj}})\,\tau
\]

Speed on the actor is **negative** (toward the camera). `heading_sign=-1`. Seated: spawn at \(v_{\mathrm{obj}}\tau\), not on the HMD. Compose adds `along_stride / v_close` to \(\tau\) for extra along-track members.

**Through-cross** (`_inject_through_cross`):

- Depth \(d\) from a small FOV/speed formula, clamped to [3.8, 7] m, plus 1.1× extra for larger CPA, plus compose cross-stride (cap 16 m).
- `lat_start` / `lat_end` are the FOV edges at that depth, clamped to the corridor. The path is **edge-to-edge**, not a 2 m shuffle that dies in the middle of the road.
- `speed=0`, `lat_speed=walk`, `lat_target=lat_end`. CPA is the graze of the gait line as they pass \(L\).

**Turn** (`_inject_jaywalk_turn`): side entry, merge onto \(L\pm\mathrm{cpa}\), then `turn_t ≈ t_{\mathrm{arrive}}-0.40` blends into `post_speed` (negative = toward camera, positive = same way and pulling away).

**Static on gait** (`_inject_static_shapes`, `_inject_pothole` with `dlat=0`): lead 5.4–7.8 m so a 50–90° HFOV still sees the object in the lower third. 4.2 m sat below a typical walking VFOV at 1.6 m eye height — that is why the pothole leads live in config.

**Cut-in / swerve** (`_inject_cut_in`): near-lane car, `swerve_t` = 1.2 s (cut-in) or 1.8 s (swerve), then `lat_target` = gait ± cpa. Occupancy is `NEAR_LANE_LOCKED`.

**Weave** (`_inject_weaving`): `weave_amp` / `weave_hz` on the Actor. Deterministic sine, **not** fBm — the injector aims the swing at the walker’s line. Amplitude is exactly `weave_amp`.

### 15.2 One row per named scenario

| Name | Bucket | Mechanism |
| --- | --- | --- |
| `safe_walk` | safe | Background only. Injector is a no-op. |
| `empty_street` | safe | `prepare_scenario` zeroes background peds/cars and most clutter. Injector no-op. |
| `oncoming_pedestrian` | safe | Opposite sidewalk (`dlat="opposite"`), 0.95–1.30 m/s, \(\tau=3.0\). CPA stays large. |
| `parallel_pedestrian` | safe | Same way, +0.80 m toward the curb, 4 m ahead, ego speed. |
| `cyclist_same_way` | safe | Far lane, same way, bike speed, 7 m ahead. |
| `car_pass_far` | safe | Oncoming, far lane, 5.5–8.0 m/s, \(\tau=3.2\). |
| `car_approaching` | safe | Oncoming, near lane unless a through-crosser is present (then far). 5.0–7.0 m/s, \(\tau=3.0\). CPA still > 1.5 m because the lane offset is ~1.75 m. |
| `distant_jaywalk` | safe | Crossing ~16 m ahead, not a hit. |
| `pothole_offset` | safe | Hole / crater / slab / debris at `dlat=1.80`, lead 8.0 m. |
| `parked_car_opposite` | safe | Static vehicle, far lane, 9 m ahead. |
| `near_miss_pass` | near | Oncoming ped, `dlat=1.0`, \(\tau=2.8\). |
| `jaywalker_offset` | near | Through-cross, CPA 1.0, from left. |
| `jaywalker_from_left` / `_right` | near | Through-cross, CPA 1.0, forced side. |
| `jaywalker_turn_away` | near | Side entry, merge off the gait, then walk the same way. |
| `cyclist_near_miss` | near | Oncoming bike, `dlat=1.0`, \(\tau=2.5`. |
| `car_near_miss_lane` | near | Oncoming car, `dlat=1.0`, \(\tau=2.8\), pad 1.05, **near lane locked**. |
| `car_cross_front` | near | Through-cross vehicle, CPA 1.0, random side, 3.2–4.8 m/s. |
| `cube_near_miss` / `shape_near_miss` | near | Oncoming primitive, `dlat=1.0`, \(\tau=2.6\), size ~0.35–0.65 m. |
| `pothole_near` | near | Hole at `dlat=0.85`, lead 7.2 m. Shin-graze, not a trip. |
| `cyclist_weaving` | near | Near-lane bike, sine 0.9–1.9 m @ 0.28–0.62 Hz. |
| `jaywalker` | critical | Through-cross person, CPA 0.12, random side. |
| `jaywalker_turn_toward` | critical | Side entry, then oncoming on the gait. |
| `sudden_stop` | critical | Ped 3 m ahead (`sudden_stop_lead_m`), `stop_t=1.6` s. Walker closes on a now-static body. |
| `swerve_vehicle` | critical | Near-lane car, `swerve_t=1.8`, then drift onto gait, CPA 0.12, \(v=5.5\), \(t_{\mathrm{hit}}=3.4\). |
| `car_cut_in` | critical | Same family, `swerve_t=1.2`, \(v=6.2\), \(t_{\mathrm{hit}}=2.9\). |
| `pothole_on_path` | critical | On-gait hole, lead 7.8 m, `dlat=0`. |
| `cube_on_path` | critical | One static cube on the gait. |
| `shapes_on_path` | critical | 2–4 mixed static primitives on the gait, 2.55 m apart. |
| `cube_head_on` / `shape_head_on` | critical | Oncoming primitive, `dlat=0.12`, \(\tau=2.4\). |
| `cube_from_left` / `_right` / `shape_from_*` | critical | Through-cross primitive, CPA 0.12. |
| `cyclist_head_on` | critical | Oncoming bike on the gait, \(\tau=2.3\). |
| `head_level_projectile` | critical | Eye-height cube, 3.6 m/s, \(\tau=1.8\) (`_inject_head_cube`). |
| `car_cross_critical` | critical | Through-cross vehicle, CPA 0.12. |
| `car_erratic_swerve` | critical | Near-lane car, weave 1.0–2.2 m @ 0.18–0.40 Hz. |
| `car_runs_off_road` | critical | Car in the near lane, then `turn_t` mounts the walker’s sidewalk (`_inject_run_off_road`). |
| `child_darting` | critical | `spawn_humanoid(child=True)`, 1.9–3.1 m/s lateral dart through the gait. |
| `crossing_street` | safe | No extra injector. Forces ego `crosswalk` (Frenet lateral + heading). |
| `cyclist_overtake` | safe | Bike same way, ~0.85 m toward the road, faster than ego, lead ~2 m. |
| `group_crossing` | near | 2–3 through-cross pedestrians; forces ego `crosswalk`. |
| `scooter_from_sidewalk` | near | Through-cross bicycle from a FOV edge. |
| `parked_car_door` | near | Parked car in the near gutter + a static door-height box on the gait. |
| `crossing_car_side` | critical | Through-cross vehicle + ego `crosswalk`. |
| `crossing_head_on` | critical | Oncoming car in the far lane while the ego crosses. |
| `backing_vehicle` | critical | Car ahead, `look_flip`, slow reverse (`ds < 0`) toward the walker. |

Aliases that resolve into this table are in §12.1. `gen_dataset.py` families (at most one member per compound) are: jaywalk, pothole, car, cyclist, cube, shape, ped, erratic_car, crossing, sidewalk_dyn.

### 15.3 Worked inject: `jaywalker` + `car_approaching` + `pothole`

1. `prepare_scenario` sees a `CROSS_GAP` name → building gap `[4,20]` (only one crosser).
2. `sort_for_inject` → `pothole_on_path` (pri 0), `car_approaching` (10), `jaywalker` (20).
3. Pothole: lead 7.8 m on the gait, reserved as static (blocks spawn cell only).
4. Car: `prefer_far_lane` is True because a through-crosser is in the mix → far lane, \(\tau=3.0\). `reserve` may nudge +2.6 m if a background car sits there.
5. Jaywalker: `take_cross_layout` (first crosser, no extra depth), through-cross at ~5 m, CPA 0.12. Static hole does not push them down the road. Far-lane car’s 5 s tube is not tested against them (background-style? No — the **injected** car is tagged with `current_key`, not `"background"`, so the full tube **is** tested). If the tubes overlap, the jaywalker is nudged +2.6 m or flipped.

That last point is why injection order is static → along → lateral: the expensive lateral actor searches against already-committed tubes.

---

## 16. `spatial_threat.py` (no `bpy`)

A cell is hot when the walker would **collide** with (or step into) the object: body-aware path occupancy, not a point-mass CPA. Ego motion for this score is **sidewalk tangent × walk speed**, not the jittered eye, so a pothole on the gait does not flicker as the head bobs.

`k` is **only** `--threat-grid`. It is not a config key. Changing it does not require a rebuild of anything else.

### 16.1 Score knobs (not CLI)

| Symbol | Value | Role |
| --- | --- | --- |
| `EGO_HALF_M` | 0.30 m | Shoulder half-width + sway |
| `REACT_S` | 2.2 s | Time the walker still occupies if they keep going |
| `REACT_BUF_M` | 0.55 m | Extra length of the stopping rectangle |
| `HIT_FLOOR` | 0.58 | Definite collision maps to at least this before urgency |
| `LAMBDA_V` | 0.22 | Urgency vs closing speed |
| `LAMBDA_T` | 1.25 | Urgency vs time-to-hit |
| `ADJ_PEAK` | 0.50 | Cap of the adjacent-lane term |
| `ADJ_LANE_M` | 1.70 m | Preferred neighbour offset |
| `CPA_SOFT_M` | 0.42 m | Softness of the planar CPA bump |
| `CLOSE_GATE_MPS` | 0.30 | Closing rate at which the swept-volume term reaches full weight |
| `BLEED_CELLS` | 0.55 | Gaussian extra width in grid-cell units |
| `BLEED_WEIGHT_MIN` | 0.02 | Discard bleed below this |

### 16.2 Frenet of one object

\(\hat{s}\) = horizontal heading (camera tangent). \(\hat{r}=\hat{s}\times\hat{z}\) (via `heading_frame`).

\[
s=(P_{\mathrm{obj}}-P_{\mathrm{cam}})_{xy}\cdot\hat{s},\qquad
\ell=(P_{\mathrm{obj}}-P_{\mathrm{cam}})_{xy}\cdot\hat{r}
\]

If the actor is on the same spline as the camera (or is static), `build_frame_record` passes `path_s = actor.s - cam.arc_length` and `path_lat = actor.lateral - cam.lateral` instead. That is the **ribbon** distance, not the chord, and it is what you want on a `corner_90`.

Object half-extents come from `horizontal_extents(threat_corners, heading)` — projected AABB onto \(\hat{s},\hat{r}\), floored at 0.12 m. Trees use trunk corners here.

\(v_s,v_\ell\) = object velocity in that frame. \(v_{\mathrm{close}}=v_{\mathrm{ego}}-v_s\) (>0 if the gap in \(s\) is shrinking). \(R=r_{\mathrm{ego}}+r_{\mathrm{obj,lat}}\).

Already behind and not closing (`s < -(r_long+0.6)` and `v_close ≤ 0.05`) → score 0. Cheap reject for the crowd behind the camera.

### 16.3 Four ways to hit, then a neighbour term

\[
W=\max(W_{\mathrm{stop}},W_{\mathrm{path}},W_{\mathrm{cross}},W_{\mathrm{cpa}})
\]

1. **Stopping volume.** Disk of the object vs the forward rectangle of length \(v_{\mathrm{ego}}T_{\mathrm{react}}+d_{\mathrm{buf}}\) and half-width \(R\). Soft edges (`_soft_unit`). Gated by \(v_{\mathrm{close}}/0.30\). A jaywalker filling the frame at 2 m is inside this box even if a point-mass CPA says they will have stepped aside. A bollard in front of a **seated** walker is not — the walker sweeps nothing.

2. **Guaranteed path hit.** They occupy the gait tube now and \(t_{\mathrm{leave}}>t_{\mathrm{arrive}}\). Static hole: \(t_{\mathrm{leave}}=\infty\). Independent of camera bob. **`inf - inf` is NaN** — if `t_arr` is non-finite, `w_path=0` (we never get there); if `t_leave` is inf and `t_arr` is finite, `w_path=1`. Never subtract two infinities.

3. **Crossing intercept.** They *enter* the tube at `t_in` (`_time_to_enter`) and are still at the walker’s \(s\) then. Gaussian on the along-track gap, exponential decay in `t_in`. Oncoming car, late cut-in.

4. **Body-aware planar CPA.** Relative velocity in the ground plane, \(t^*=-(p\cdot v)/\|v\|^2\), clearance = miss − \(R\), \(W_{\mathrm{cpa}}=\exp(-(d_{\mathrm{clear}}/0.42)^2)\). `\|v\|~0` → no approach to solve; proximity alone is not a CPA.

Urgency:

\[
U=1-\exp(-\lambda_v\max(v_{\mathrm{close}},0)-\lambda_t/\max(t_{\mathrm{hit}},0.08))
\]

\[
S_{\mathrm{hit}}=W\,(0.58+0.42\,U)
\]

**Adjacent high speed** (not a hit): mid-band, peak 0.50, only if `lat_gap>0.05`. A fast neighbour that will miss.

Return \(\mathrm{clamp}_{01}(S_{\mathrm{hit}}+(1-W)S_{\mathrm{adj}})\). `_clamp01` maps NaN → 0 so a bad frame cannot write the JSON token `NaN`.

Stationary ego: arrival times divide by **closing** rate \(v_{\mathrm{ego}}-v_s\), never by \(v_{\mathrm{ego}}\). A car driving at a seated walker still produces a time-to-hit. Every division is floored.

### 16.4 Splat

`splat_object` scores, then `splat_bbox` paints the 2-D box’s cells. Occupied cells (pixel overlap > 1.5 px in both axes) keep the full score. Neighbours get a size-aware Gaussian (`σ = half_box + 0.55·cell`). Per-cell **max** — two hazards in one cell do not add to 1.8.

Row 0 = **top** of the image (same as `bounding_box_2d`).

`empty_grid` / `finalize_grid` / `frame_spatial_entry` / `episode_spatial_payload` are the JSON helpers. Spatial JSON is **always** written, even with `--no-annotations`.

`python spatial_threat.py` self-tests seated, static-on-gait, adjacent miss, NaN guards, and splat monotonicity.

---

## 17. `spatial_overlay.py` (no `bpy`)

Debug movie, not a training product. `write_heat_sequence` writes one PPM per frame (blue → red via `threat_to_rgb`). `overlay_filter_complex` is an ffmpeg graph: hazy RGB + the heat cells + per-cell score text. Cell size is a constant (`CELL_W`/`CELL_H`) so a 3×3 and a 5×5 stay readable.

`main.encode_spatial_overlay_video` writes PPMs under `spatial_annotations/_heat/`, muxes `spatial_overlay.mp4`, then deletes `_heat`. Requires a render (`--no-render` prints a skip). Mux failure keeps `rgb/` so the episode is still inspectable.

`python spatial_overlay.py` writes a tiny PPM and checks the filter string contains `geq` / overlay nodes.

---

## 18. `gen_dataset.py` (system Python)

Does **not** import `bpy`. Builds a balanced scenario list, writes `datasets/pack_*/pack.json` + `plan.json`, then `exec`s `./run.sh --plan …` with the flags you passed through (`--media`, `--spatial-overlay`, `--no-render`, …).

### 18.1 Why a planner at all

Each episode already randomizes lighting, weather, path, FOV, body, colours, clutter. That is `ep_rng`. This script only chooses **which scenario name(s)** go in each episode so a 200-clip pack is not 80% `safe_walk`.

### 18.2 `build_plan`

1. Quotas: leftover after compounds is split **40 / 30 / 30** like `--scenario auto`.
2. Compounds: `_COMPOUND_FRAC=0.22` of \(N\). `draw_compound` picks **at most one name per family** (jaywalk, pothole, car, cyclist, cube, shape, ped, erratic_car) so occupancy stays sane. `jaywalker+jaywalker_from_left` is refused by construction.
3. Singles: cycle-draw from each pool so a small \(N\) still sees several names, not 8 copies of the first.

`--dry-run` writes the JSON and prints the mix, does not launch Blender. `--self-test` checks determinism (same seed → same plan) and family uniqueness inside compounds.

`pack_dirname` is `pack_YYYYMMDD_HHMMSS_s{seed}_n{N}` plus an optional name. `unique_pack_dir` appends `_2` if the folder exists.

`launch_blender` is `os.execv` of `./run.sh` — it **replaces** the Python process. Flags after `--` include `--plan`, `--output` (the pack dir), and whatever media/overlay/render flags were on the planner CLI.

---

## 19. `main.py` (the rest)

§3–§4 covered startup and the episode skeleton. This section is the annotation record, episode folders, and mux.

### 19.1 `build_frame_record`

Called once per sim frame after `view_layer.update()`. Returns `(record, threat_grid)`.

For each `world.annotatable()` actor:

1. **Range cull.** Skip if \(\|P-P_{\mathrm{cam}}\|^2 > (40+5)^2\), or if the object is more than 6 m **behind** the camera tangent (`(P-Pcam)·t < -6`). The +5 m pad and the −6 m back-face keep a wide box that is about to enter from the side.
2. **Corners.** Sunken → cached mouth slab. Static → cached world AABB. Dynamic → `actor._bounds.world_corners()`.
3. **Threat corners.** `threat_obj` if set and distinct (tree trunk), else the silhouette corners.
4. **Threat point.** `threat_point_from_corners(..., cam.position.z, actor.threat_mode)`.
5. **Velocity.** `actor.velocity`, or `finite_velocity(1/fps)` if that vector is ~0 and the actor is not static/stopped (first-frame movers).
6. **Kinematics.** `relative_kinematics(..., planar=(biome in {park,plaza}))`. Skip if `kin.distance > 40`.
7. **Box.** `project_object` with the shared view/proj matrices. `None` → omit.
8. **Label.** `classify_threat(kin, \|v_obj\|, cfg['threat'])`.
9. **Extents.** `horizontal_extents(threat_corners, tangent)` for the spatial score.
10. **Path Frenet.** If the actor has `follow_spline` or is static: `path_s=actor.s-cam.arc_length`, `path_lat=actor.lateral-cam.lateral`. Background-only objects without a spline fall back to world-XY inside `object_threat_score`.
11. **Splat.** `splat_object(...)`.
12. **JSON object.** Root origin (not threat point) as `world_position`. Threat-point kinematics for distance / ttc / cpa / relative_velocity. All vectors Y-up. TTC non-finite → `9999.0`.

Camera block: eye position (no Perlin translation), finite-difference velocity (includes gait \(dY/dt\) in Y-up), local pitch/yaw/roll, lens, HFOV, sensor, `ego_mode`, `ego_speed` (0 seated / halt — a consumer can tell “no optical flow” from “sensor dropout”).

Environment block: copied from `world.state.environment` every frame (drawn once at `build()`). Includes `biome`, `dappled`, `chaos`.

Dropped from a frame (not listed): `annotatable=False`, range cull, `project_object is None`. A tree behind the camera does not appear. A puddle never appears.

### 19.2 Episode directories

Pattern: `episode_(\d{4,})(?:_|$)` — `episode_0007`, `episode_0007_jaywalker`, `episode_12` all count. `existing_episode_ids` / `next_free_episode_id` scan the output root.

`allocate_episode_dir(out_root, id, slug)` — if `episode_{id:04d}_{slug}` exists it bumps the id (a crashed re-run must not overwrite). Returns the chosen id and path.

`rebuild_dataset_summary` walks every `episode_*/episode.json`, aggregates counts / histograms / media, writes `dataset_summary.json`. Older folders in the same root are included on purpose: a second `--episodes 4` appends, and the summary is the whole root.

JSON is written with `separators=(',', ':')` (no spaces) plus a trailing newline. Compact on purpose — 150 frames × dozens of objects is already large.

### 19.3 Pose snapshot / restore

`_collect_pose_objects` — camera + every actor root and **every descendant** (gait bones, wheels, branch joints, triangle leaves). Missing a child means Phase B renders that part at the bind pose: a frozen walk cycle, a wheel that does not roll, a crown that does not rustle.

`_snapshot_poses` stores `(location, rotation_mode, rotation_euler or quaternion, scale)` per object.

`_restore_poses` writes them back. The `frame_change_pre` handler restores snapshot `scene.frame_current`. Blender may call the handler more than once per frame (depsgraph); restoring is idempotent.

`_render_animation_sequence` sets `filepath` to `rgb/######.png`, `frame_start/end`, registers the handler, `bpy.ops.render.render(animation=True)`, unregisters. `_ensure_six_digit_pngs` checks `000000.png` … exist. Failure → `_render_stills_fallback` (same poses, `write_still=True` per frame). Labels are already written; pixels must match the snapshotted poses, not a re-integration.

### 19.4 Video

`_ffmpeg_has_encoder` / `_nvenc_usable` — actually encode a 2-frame test, do not trust the encoder list (a stub NVENC appears on some AMD boxes).

`_encode_with_ffmpeg` — image2, 30 fps, yuv420p, `h264_nvenc` preset `p4` or `libx264` `veryfast` CRF 18.

`_encode_with_blender_vse` — `sequences` on 4.x, `strips` on 5.x. After mux, `prepare_still_render()` again (sequencer must not stay on).

`encode_episode_video` tries ffmpeg then VSE. Failure sets `keep_frames=True`.

### 19.5 `--wind`

`run_episode(..., wind=)` accepts `calm|breeze|windy|auto`. A concrete label replaces `cfg['domain_randomization']['wind_weights']` with `{label: 1.0}` **before** `prepare_biome` / `build()`, so `choose_environment` always draws that label. Strength is still a uniform draw inside that label’s range (a locked `breeze` is not a single number). `auto` keeps the config weights. `BTP_WIND` can still override the draw if it names a key that remains in the weights dict.

The field is then constructed as `WindField(seed, strength, direction, label)` and stored on every tree Actor. Trunk pose and `Actor.velocity` stay 0. JSON `environment.wind` / `environment.wind_strength` come from that draw (copied onto every frame). The model (shared gust, per-joint phases) is §11.1.

---

## 20. Scene contents (what exists in the viewport)

| Element | How it is built | Annotated? |
| --- | --- | --- |
| Road / path | Frenet ribbon. Asphalt Voronoi, or gravel tiles in park | no (ignore list) |
| Sidewalks | Ribbon at curb Z, paver brick. Plaza: slight Z jitter | no |
| Curb | Narrow ribbon. Park: none | no |
| Grass verge | Outside the sidewalk (or path). Instanced tufts, never on asphalt | no |
| Lane dashes | 3 m on / 3 m off, Frenet quads | no |
| Buildings | Depth × width × height + roof. Start `s=14` m. Setback 3.6–4.2 m | no |
| Streetlamps | Pole + arm toward the road + emissive bulb + downward SPOT | no (pole is ignore) |
| Trees | Recursive forks + triangle leaves. Plant / curb / median / path | **yes**, class `tree`, threat = trunk |
| Vehicles | Superquadric hull + glass + 4 +X-axis wheels | yes, `vehicle` |
| Pedestrians | `humanoid.spawn_humanoid` + `WalkRig` | yes, `person` |
| Bicycles | Frame + wheels | yes, `bicycle` |
| Furniture | Trash / scooter / barricade / puddle on the **shop-front** sidewalk | trash/scooter/barricade yes; puddle **no** |
| Trip holes | Injectors only: pothole well, crater bowl, tilted slab, debris | yes; footprint except debris |
| Shape obstacles | Injectors: instanced unit cube/sphere/cylinder/pyramid/cone/capsule/lump | yes, `threat_<kind>` |
| Head boxes | Off by default (`n_head_hazards=(0,0)`). Scenario `head_level_projectile` is separate | if spawned, yes |
| Bench | Seated ego only | **no** |
| Canopy gobo | Overhead alpha sheet | no (hidden from camera) |

Collections: `WORLD`, `HAZARDS`, `ACTORS`, `LIGHTS`. New collections start excluded — `reveal_view_layer()` is mandatory.

---

## 21. Output contract

Folder: `episode_{id:04d}_{slug}`. `dataset_summary.json` is rebuilt from every `episode_*/episode.json` already in that output root.

Pixels: 1920×1080 RGB8 PNG, AgX, opaque film (`film_transparent=False` — no checkerboard alpha).

### 21.1 `annotations/annotations.json`

```json
{
  "frames": [
    {
      "frame_id": "000000",
      "timestamp": 0.0,
      "camera_data": {
        "world_position": [x, y_up, z_fwd],
        "velocity": [..],
        "pitch_yaw_roll": [pitch, yaw, roll],
        "lens_mm": 24.0,
        "hfov_deg": 73.74,
        "sensor_width_mm": 36.0,
        "ego_mode": "walk",
        "ego_speed": 1.12
      },
      "environment": {
        "lighting": "dusk",
        "weather": "clear",
        "biome": "street",
        "dappled": false,
        "chaos": 0.65,
        "wind": "breeze",
        "wind_strength": 0.41
      },
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
          "bounding_box_2d": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0},
          "flags": {"truncated": false, "occluded": false}
        }
      ]
    }
  ]
}
```

| Field | How it is measured | Do not confuse with |
| --- | --- | --- |
| `timestamp` | \(i/\mathrm{fps}\) from the start of **this** episode | Wall clock |
| `camera_data.world_position` | Eye (Frenet + curb + gait bounce). **No** Perlin translation | IMU / pelvis |
| `camera_data.velocity` | Finite difference of the eye (includes gait \(dY/dt\)) | `ego_speed` (ground, no vertical) |
| `camera_data.pitch_yaw_roll` | Camera-**local** Perlin + pitch bob | World Euler |
| `camera_data.ego_speed` | Ground speed; **0** seated / halt | Optical-flow magnitude |
| `environment.*` | Drawn once at `build()`, copied onto every frame. Includes `wind` / `wind_strength` | Per-frame weather or a swaying canopy as a kinematic threat |
| `kinematics.world_position` | **Root** origin (pelvis / body centre) → Y-up | Threat point |
| `kinematics.distance` / `ttc` / `cpa` | **Threat point**, not the root | `‖world_position − camera.world_position‖` |
| `bounding_box_2d` | Projected 3-D AABB (slightly loose), integers, top-left origin | Tight silhouette |
| `flags.truncated` | Box hits the image edge, or < 8 corners in front | Occlusion |
| `flags.occluded` | Centre-ray hit something else first | Pixel coverage |

`class_name` values you will actually see: `person`, `vehicle`, `bicycle`, `tree`, `threat_cube`, `threat_sphere`, `threat_cylinder`, `threat_pyramid`, `threat_cone`, `threat_capsule`, `threat_lump`, `pothole`, `crater`, `broken_slab`, `debris`, `trash_can`, `scooter`, `barricade`, `tree_branch`, `ac_unit`, `sign`, `truck_door`, `projectile`. Puddle and bench exist in the viewport and are **not** listed.

`label_histogram` in `episode.json` counts **(object × frame)**, not unique instances. A 5 s jaywalker is ~150 CRITICAL counts, not 1.

### 21.2 `spatial_annotations/spatial_annotations.json`

Always written.

```json
{"k": 3, "frames": [{"frame_id": "000000", "matrix": [[0.0, 0.12, 0.0], [0.81, 0.97, 0.41], [0.0, 0.72, 0.0]]}]}
```

Row 0 = top of the image. Entries in `[0,1]`. A 3×3 is a tactile-vest resolution, not a segmentation.

### 21.3 `episode.json`

`episode_id`, `dir`, `scenario` (slug), `scenarios` (resolved list), `scenario_requested`, `frames`, `fps`, `walk_speed` (0 if seated), `sidewalk_lateral`, `biome`, `ego` (mode, eye height, stationary, sidestep, halt window), `camera` (lens / HFOV / sensor), `environment`, `label_histogram`, `render` / `media`, `rgb_dir` (null if deleted), `video` / `video_encoder`, `annotations` path or null, `spatial_annotations_k` / `spatial_annotations`, overlay paths.

### 21.4 Worked label: seated vs walking into a trunk

Walking, 1.2 m/s, trunk 2.0 m ahead on the gait, planar off (street). \(V_{\mathrm{rel}}=-V_{\mathrm{cam}}\), TTC ≈ 1.67 s, CPA ≈ 0 → **CRITICAL_THREAT**. Spatial \(W_{\mathrm{path}}=1\).

Same trunk, seated. \(V_{\mathrm{rel}}=0\), TTC = 9999, **SAFE_STATIC**. Spatial \(W_{\mathrm{stop}}=0\) (not closing), \(W_{\mathrm{path}}=0\) (`t_arr` infinite). Leaves may still sway in RGB. That pair is the point of the foliage design.

---

## 22. Load-bearing pitfalls

1. **Black PNGs, valid JSON.** Empty VSE with `use_sequencer=True`. Always `prepare_still_render()` before writing pixels; turn it off again after VSE mux.
2. **`BLENDER_EEVEE_NEXT` TypeError** on 5.2. Use whatever `enum_items` lists. Sky type is `MULTIPLE_SCATTERING`, not `NISHITA`.
3. **World Volume Scatter** in EEVEE = full-frame black. Fog via sky turbidity only. Do not “add atmosphere” with a volume shader.
4. **Euler `matrix_world` ignored.** Camera must be quaternion + decompose (`_apply_camera_matrix`).
5. **Collections start excluded.** `reveal_view_layer()` then re-apply daytime lamp hide. A “empty viewport, labels exist” episode missed the reveal.
6. **`view_layer.update()` every sim frame** or AABBs are empty (`objects=0`).
7. **One Frenet `s`.** Never mix sidewalk-spline arc length with road `s`. Never treat world-Y as arc length on a curve.
8. **`look_along` zeroes Euler X/Y.** Walk pelvic list/pitch **after** it. Yaw = `atan2(-dx, dy)`, not `atan2(dx, dy)`.
9. **Threat point ≠ root.** Labels used the threat point. `distance` is not `‖world_position − camera.world_position‖`. Trees: box = canopy, threat = trunk.
10. **Undefined TTC** is `9999.0`, not `null` / `Infinity`. Small TTC + large CPA is SAFE_DYNAMIC. `NaN < 2.5` is False — `_finite3` exists so that cannot silently classify as SAFE.
11. **Mix / Noise sockets.** Never `node.inputs["A"]` or `noise.outputs["Fac"]`. Use `_input` / `_output`.
12. **Facade windows** use **object** normals (after `look_along`, world \(\hat{x}\) is not the street face). Occupancy is a per-cell hash, not smooth noise.
13. **Shadow pool 2048.** Key sun + every other lamp. Fill/bounce: `use_shadow=False`. Adding a dozen shadowed area lights pages out the key.
14. **Neck along −Z** grows into the chest. Neck is +Z.
15. **Seated `walk_speed=0`.** Intercepts use `max(v_ego+v_obj, v_obj)*tau`. Nothing floors ego speed to 0.25. Spatial arrival divides by **closing** rate, not \(v_{\mathrm{ego}}\).
16. **Tree threat is the trunk.** Canopy sway must not change `Actor.velocity` or `threat_obj` pose.
17. **`ray_tracing_method='SCREEN'`** (not `'SCREEN_TRACE'`, a silent no-op on 5.2).
18. **`inf - inf` in `w_path`.** Resolve infinite `t_arr` / `t_leave` **before** subtracting or the matrix gets NaN.
19. **Phase B descendants.** Snapshot every child. A missing sprig is a frozen canopy in the video and a swaying one in your head.
20. **`release_episode` is the memory budget.** Null Python cycles first, then unlink, then recursive purge. A new datablock type (node group, image, curve) that you allocate without a users-drop will leak across 2000 episodes.
21. **`MeshLibrary` is per-episode.** Do not cache it on the generator across `build()` calls. The datablocks were purged.
22. **Occupancy predictor must match `Actor.update`.** New motion modes need a `FrenetCapsule.pose_at` twin.
23. **Planar TTC in park/plaza only.** Turning it on for streets would collapse a 1.6 m overpass / sign into a ground hit. Turning it off in a park labels a seated walker vs a bollard SAFE on a vertical residual.
24. **Head hazards default off.** Re-enable via `n_head_hazards` if a pack specifically wants floating boxes. Do not “put them back” in `_scatter_ground` — that is how they ended up on the gait as junk.

---

## 23. Defaults that matter (`config.py`)

| Key | Default | Why it is that number |
| --- | --- | --- |
| Resolution / fps / length | 1920×1080 / 30 / 150 frames | Vest training clip; 5.0 s |
| HFOV | \(U(50^\circ,90^\circ)\) unless CLI locks | 35 mm → 18 mm full-frame |
| Walk speed / bounce | stroll / walk / hurry \(U(0.70,2.05)\), 0.04 m @ 1.8 Hz | Adult sidewalk gait; seated is 0 |
| Path | \(U(48,78)\) m; straight / gentle / S / 90° | 5 s at 1.4 m/s is 7 m; the rest is look-ahead + buildings |
| `sidewalk_s0` | 3.0 m (code, not config) | First frame is not inside a facade |
| Street ribbon | road 7.0 m, sidewalk 2.4 m, curb 0.12 m, setback 3.6 m | Two lanes + planting strip |
| Trees | plant \(U(6,12)\); curb/median counts 0 unless avenue planted strip | No trunks on asphalt |
| Furniture Poisson | \(U(3,7)\), radius 3.2 m, shop-front band | Walking line stays clear |
| Head hazards | `(0, 0)` | Floating boxes read as junk |
| Background peds / cars | \(U(3,6)\) / \(U(2,5)\) | Occupancy still has room for injectors |
| Auto mix | 0.40 / 0.30 / 0.30 | Spec Part 4.3 |
| Critical / near-miss | TTC 2.5 s + CPA 0.5 m / TTC 4.0 s + CPA [0.5, 1.5] | Spec Part 4.2 |
| Inject CPA targets | 1.0 m / 0.12 m | Graze vs hit under constant rates |
| Jaywalk / projectile | \(\tau=3.2\) s, \(U(1.00,1.35)\) m/s / \(\tau=1.8\) s, 3.6 m/s | Walking crosser, not a sprint; cube at eye |
| Pothole leads | 7.8 / 7.2 / 8.0 m | In the lower third at 50–90° HFOV |
| Compose strides | cross 4.0 / along 3.2 / static 2.4 m; nudge 2.6 m | One body + clearance |
| Lighting weights | noon 0.24, dawn/dusk/night 0.16, glare/overcast 0.14 | Easy domain is not the majority |
| Weather | clear 0.50, light_fog 0.30, heavy_smog 0.20 | Turbidity only |
| Chaos | \(U(0.20,1.00)\) unless `--chaos` | Never fully tame unless asked |
| Wind | calm 0.32 / breeze 0.48 / windy 0.20; strengths `(0.02,0.08)` / `(0.28,0.55)` / `(0.70,1.00)` | Leaves move; trunk TTC does not |
| Annotation range | 40 m | Vest horizon |
| TAA / Fast GI | 16 + reprojection; 4 rays, 6 steps, quality 0.30 | Matches 32 samples on this lighting |
| Shadow pool | 2048 | Key + half the night lamps |
| PNG compression | 1 | Almost as fast as 0, much smaller |
| Video | CRF 18, `veryfast` / NVENC `p4` | Preview, not archival |

Change numbers **in `config.py`**, not by scattering literals. Injector literals that remain (`tau=2.4` on `cube_head_on`, weave Hz ranges) are listed in §15.2; promote them if you touch them twice.

---

## 24. Where is X?

| Change… | File / symbol |
| --- | --- |
| Resolution, FPS, episode length, TAA/GI | `config.py` → `render`; `configure_eevee` |
| Lens / eye height / HFOV range | `config.py` → `camera`; `apply_camera_fov` |
| Gait / Perlin / ego weights | `config.py` → `gait`, `jitter`, `ego`; `CameraRig`, `choose_ego_profile` |
| TTC thresholds / class mix | `config.py` → `threat`, `scenarios.ratios` |
| Street widths, setback, tree counts, Poisson | `config.py` → `world` / `world.biomes` |
| Sun / sky / AgX | `apply_domain_randomization`, `_setup_world_shader`, `_apply_view_transform` |
| Asphalt / windows / foliage / chaos | `materials.py` |
| Body / walk cycle | `humanoid.py` → `spawn_humanoid`, `WalkRig` |
| Forced collisions | `WorldGenerator._inject_*`, `_scenario_handlers` |
| CLI aliases / occupancy / inject order | `scenario_compose.py` |
| Per-frame object JSON | `main.build_frame_record` |
| Spatial matrix / overlay | `spatial_threat.py` / `spatial_overlay.py` |
| Mixed pack | `gen_dataset.py` → `main.py --plan` |
| 2-D boxes / occlusion / threat point | `projection.py` |
| Animation-batch render | `main._render_animation_sequence` |
| Black frames | `prepare_still_render` |
| Camera not rotating | `CameraRig._apply_camera_matrix` |
| GPU device | `run.sh` / `BTP_GPU` |
| Leak across episodes | `release_episode`, `purge_orphans` |
| Leaf wind / `--wind` / `BTP_WIND` | `choose_environment`, `WindField`, `foliage_wind_euler`; weights in `config.py` → `domain_randomization` |
| Tree / building overlap | `building_setback`, `_scatter_trees`, `offset_folds` |
| Median trees rejected | `_tree_free(..., along=)` — do not use one Euclidean radius |
| Moving leaves labelled CRITICAL | Foliage must not write `Actor.velocity`; threat is the trunk |
| `objects=0` in the sim log | Missing `view_layer.update()`, or `reveal_view_layer`, or bound cache not cleared |
| Compound actors inside each other | `ComposeSession.reserve` / `FrenetCapsule.pose_at` mismatch with `Actor.update` |
| Park bollard labelled SAFE | `planar=True` not applied; or threat point not at camera Z |
| New primitive shape | `SHAPE_KINDS`, `_shape_unit_geom`, `spawn_threat_shape`, aliases, `_STATIC_CLASSES`, `gen_dataset` family `shape` |

---

## 25. How to add something (without breaking C1–C10)

### 25.1 A new named scenario

1. Pick a bucket and add the canonical name to `safe_pool` / `near_miss_pool` / `critical_pool` in `config.py`.
2. Add a handler in `_scenario_handlers` that only uses Frenet (`_spawn_kind` + `_bind` or `_inject_oncoming` / `_inject_through_cross` / `_inject_static_shapes`).
3. If it crosses the ribbon, add the name to `CROSS_GAP_SCENARIOS` and `THROUGH_CROSSERS` as appropriate.
4. Set `_INJECT_PRIORITY` (0 static, 10 along, 20 lateral).
5. Optional alias in `SCENARIO_ALIASES`.
6. If it is a new class of motion, teach `FrenetCapsule.pose_at`.
7. If it is a new `class_name`, add extents and decide `_is_static_class`.
8. If packs should compound it, put it in exactly one `_FAMILIES` entry in `gen_dataset.py`.
9. Add a row to §15.2 of this file.
10. `python scenario_compose.py` and a `--no-render` episode.

Do **not** aim a world-space `hold_velocity` through a `corner_90`. Do **not** divide by `walk_speed` to place an intercept.

### 25.2 A new biome

Add a key under `world.biomes` and a weight under `biome_weights`. You may override widths, `ground`, `buildings`, `building_sides`, `lane_paint`, tree counts, traffic counts, `path_types`. You may **not** introduce a second spline parameter. Park works because a “lane” is still a lateral band. If you need a river, it is a ribbon at a new lateral, not a new `s`.

### 25.3 A new annotatable class

Spawn an `Actor` with `class_name`, `threat_mode`, `annotatable=True`, Frenet `s`/`lateral` if it lives on the street. If the silhouette ≠ the hazard (tree), set `threat_obj`. Add extents. If it is sunk into the pavement, add the class to `SUNKEN_CLASSES` in `main.py` and give it a mouth-slab path. Mention it in §21.1.

### 25.4 A new lighting state

Add energy + elevation ranges and a weight. Teach `apply_domain_randomization` colour / angle / shadows and `_apply_view_transform` exposure. Do not add a world volume. Do not add a dozen shadowed lamps.

---

## 26. Self-tests and what they prove

| Command | What must stay true |
| --- | --- |
| `python threat_math.py` | Parallel / diverging / head-on / seated / planar / taxonomy priority / no NaN / no ZeroDivision |
| `python spatial_threat.py` | Seated bollard ~0, static-on-gait high, adjacent miss mid, `inf-inf` guarded, splat max not sum |
| `python spatial_overlay.py` | PPM header, filter graph parses |
| `python scenario_compose.py` | Aliases, `threat_*` extents, inject order, reserve flip / nudge, static spawn-only |
| `python gen_dataset.py --self-test` | Same seed → same plan; one name per family in compounds |
| `./run.sh --episodes 1 --scenario safe_walk --no-render --output /tmp/btp` | `objects>0` if the street has people; `cleanup: purged` is large |
| `./run.sh --episodes 1 --scenario jaywalker --no-render` | Some `CRITICAL_THREAT` in `label_histogram` for a walking ego |
| `./run.sh --episodes 1 --ego-mode seated --scenario pothole_on_path --no-render` | Hole is `SAFE_STATIC`; TTC 9999 |
| `./run.sh --list-scenarios` | Pools + aliases, exit 0 |

A render test (`--media frames --frames 4`) is the check for black PNGs and for Phase B descendants (look at a tree: leaves should not be at rest if wind > 0).

---

## 27. Data-flow recap (one page)

```text
run.sh → blender --python main.py -- <flags>
           │
           ├─ get_config() deepcopy
           ├─ pick_scenarios / plan JSON
           └─ run_episode
                ├─ apply_camera_fov, chaos lock, no-trees
                ├─ prepare_biome → cfg['world'] widths / props
                ├─ prepare_scenario → gaps / sparse
                ├─ build()
                │    reset scene, EEVEE, MeshLibrary, WindField
                │    PathSpline, ribbons, buildings, trees, lamps, grass
                │    lighting, furniture, background cars/peds
                │    WorldState {road, corridor, actors, walk_speed, …}
                ├─ CameraRig on road spline (arc table if erratic)
                ├─ inject_scenarios
                │    ComposeSession.seed background
                │    sort_for_inject → _inject_* → _bind → reserve
                ├─ freeze() → movers / annotatable
                ├─ Phase A: for each t
                │    rig.update, world.update, view_layer.update
                │    snapshot poses
                │    build_frame_record
                │       threat_point → relative_kinematics → classify_threat
                │       project_object → splat_object
                ├─ write annotations.json + spatial_annotations.json
                ├─ Phase B: restore poses, render animation, mux
                ├─ episode.json
                └─ release_episode + clear_bound_caches
```

Everything a downstream model needs is in those two JSON files plus optional RGB. Everything it must not recompute from the wrong point is in §21.1.

---

## 28. Glossary

| Term | Meaning in this repo |
| --- | --- |
| Episode | One clip: one street, one ego, one (compound) scenario, one folder |
| Frenet `(s, lateral)` | Arc length along the **road** centreline + signed offset along \(\hat{t}\times\hat{z}\) |
| Threat point | The 3-D point TTC/CPA use. Not the mesh origin. Not the 2-D box centre |
| Threat label | One of four strings from `classify_threat` |
| Spatial matrix | K×K body-aware heat, separate product from the four-class label |
| Injector | Closed-form placement of a forced event after the rig exists |
| Compose | Occupancy reservation so several injectors share one street |
| Chaos | Appearance dial, not motion. `--chaos 0` is tame materials |
| Biome | Widths / ground / props on the **same** Frenet corridor |
| Mover | Actor that `freeze()` put in the per-frame integration list |
| Phase A / B | CPU integrate+label / GPU animation render |
| PRIME | Laptop GPU offload; `run.sh` + `BTP_GPU` |

If a word is not in this table, it is not a hidden API — it is ordinary English.