# blender_sim

Headless Blender pipeline that builds an **egocentric pedestrian** dataset for a tactile warning model. The camera is a walking (or seated) human at ~1.6 m eye height, not a car. Every frame gets 2-D boxes plus a four-class threat label computed from time-to-collision (TTC) and closest-point-of-approach (CPA), not from pixels.

Internals, file-by-file: [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## What you need

- Blender **4.2+** or **5.x** (this repo is developed on 5.2.1 at `/usr/bin/blender`)
- No pip packages. Blender’s bundled Python runs the sim. System Python is enough for the math self-tests and `gen_dataset.py`.
- A GPU OpenGL/EGL context for EEVEE. On a desktop session that is automatic. On a true headless box use `xvfb-run -a …` or `--no-render` (JSON only).
- Optional: system `ffmpeg` to mux `preview.mp4`. If it is missing, Blender’s bundled H.264 encoder is the fallback.

Arch: `sudo pacman -S blender ffmpeg`

Prefer `./run.sh` over raw `blender`. On hybrid AMD+NVIDIA laptops it PRIME-offloads EEVEE onto the NVIDIA GPU when `nvidia-smi` works (`BTP_GPU=auto|nvidia|amd`).

Everything after the bare `--` is ours. Without `--`, Blender eats the flags.

```bash
cd /storage/BTP/blender_sim
./run.sh --episodes 1 --scenario safe_walk --no-render --output ./output
```

## Quick start

```bash
cd /storage/BTP/blender_sim

# Annotations only (no GPU render). Good for checking TTC / labels.
./run.sh --episodes 1 --scenario safe_walk --no-render --output ./output

# One rendered episode: PNG sequence + preview.mp4 + JSON
./run.sh --episodes 1 --scenario jaywalker --media both --output ./output

# Mixed 40 / 30 / 30 threat buckets (default 4 episodes)
./run.sh --episodes 8 --scenario auto --media both --seed 42 --output ./output

# Video only: keep the mp4, delete rgb/ after mux
./run.sh --episodes 1 --scenario swerve_vehicle --media video --no-rgb --output ./output

# PNG sequence only
./run.sh --episodes 1 --scenario safe_walk --media frames --output ./output

# Compound street: jaywalker + oncoming car + pothole
./run.sh --episodes 1 --scenario jaywalker,car,pothole --media both --output ./output

# Park path, seated ego, locked appearance chaos, JSON only
./run.sh --episodes 1 --biome park --ego seated --chaos 1 --no-render --output ./output

# Ego turns onto a crosswalk; a car comes from the side
./run.sh --episodes 1 --scenario crossing_car_side --biome street --no-render --output ./output

# Short walker on a cobbled alley
./run.sh --episodes 1 --biome alley --ego-height short --scenario safe_walk --no-render --output ./output

# Print every injector name and alias, then exit
./run.sh --list-scenarios
```

Raw Blender (no PRIME helper):

```bash
blender --background --python main.py -- --episodes 1 --scenario auto --output ./output
```

True headless:

```bash
xvfb-run -a ./run.sh --episodes 1 --scenario auto --media frames --output ./output
```

Force time of day (QA; not a CLI flag): `BTP_LIGHTING=dusk ./run.sh …`

## CLI (`./run.sh` / `main.py`)

| Flag | Default | What it does |
| --- | --- | --- |
| `--episodes N` | `4` | How many episodes this process writes |
| `--start-episode N` | next free id | First folder index. Omitted ⇒ one past the highest `episode_*` already in `--output` (never overwrites). Pass `0` to start at `episode_0000`. |
| `--scenario NAME` | `auto` | Injector, alias, or compound (`jaywalker,car,pothole`). Repeatable. |
| `--scenarios LIST` | unset | Same tokens, comma or `+` separated. Joined with `--scenario` if both are set. |
| `--list-scenarios` | | Print pools and aliases, exit |
| `--output DIR` | `./output` | Dataset root |
| `--seed N` | `42` | Master RNG. Each episode then draws its own stream. |
| `--frames N` | config `150` | Frames per episode (`0` = config). 30 fps ⇒ 150 frames = 5.0 s |
| `--media` | `both` | `frames` (keep `rgb/`), `video` (mux then delete PNGs), `both` |
| `--no-rgb` | off | Delete `rgb/` after `preview.mp4` (and overlay) succeed. PNGs are kept if mux fails. Ignored with `--media frames`. |
| `--no-render` | off | Skip EEVEE. Still writes JSON. |
| `--no-annotations` | off | Skip `annotations/annotations.json`. Spatial matrices still write. |
| `--no-occlusion` | off | Skip the centre-ray occlusion flag |
| `--threat-grid K` | `3` | `K×K` spatial threat matrix, always written |
| `--spatial-overlay` | off | Also write `spatial_overlay.mp4` (needs a render) |
| `--biome` | `auto` | `street` \| `avenue` \| `park` \| `plaza` \| `alley` \| `residential` \| `market` \| `auto` |
| `--ego-mode` / `--ego` | `auto` | `walk` \| `diagonal_cross` \| `crosswalk` \| `erratic` \| `seated` \| `auto` |
| `--ego-height` | `auto` | `short` (1.35–1.50 m) \| `typical` \| `tall` (1.75–1.90 m) \| metres |
| `--chaos X` | per-episode | Lock appearance chaos in `[0,1]` |
| `--wind` | `auto` | `calm` \| `breeze` \| `windy` \| `auto` — leaf rustle; trunk stays still |
| `--no-trees` | off | No procedural trees or grass |
| `--hfov DEG` | random 50–90° | Lock horizontal FOV for every episode |
| `--lens-mm MM` | | Lock focal length (ignored if `--hfov` is set) |
| `--no-random-fov` | off | Use config `lens_mm` |
| `--plan FILE` | | `plan.json` from `gen_dataset.py`. Overrides `--scenario`. |

`--seed` + `--start-episode` is how you shard a large pack without collisions. A later `./run.sh --episodes 4 --output ./output` appends after whatever is already on disk.

## What gets written

```
output/
  dataset_summary.json
  episode_0000_safe_walk/
    episode.json
    preview.mp4                         # --media video or both
    spatial_overlay.mp4                 # --spatial-overlay
    rgb/000000.png …                    # --media frames or both; deleted by --no-rgb / --media video
    annotations/annotations.json        # object boxes + TTC + labels (skip with --no-annotations)
    spatial_annotations/spatial_annotations.json   # always; every frame’s K×K matrix
  episode_0001_jaywalker__car_approaching__pothole_on_path/
    …
```

| Product | When | Use it for |
| --- | --- | --- |
| `annotations/annotations.json` | default | Per-frame objects: `threat_label`, TTC, CPA, pixel box |
| `spatial_annotations/spatial_annotations.json` | always | `K×K` float heat in `[0,1]`, row 0 = top of image |
| `rgb/XXXXXX.png` | `frames` / `both` | 1920×1080 RGB, AgX. `000007.png` is `frame_id` `"000007"` |
| `preview.mp4` | `video` / `both` | Same sequence muxed at `fps`. Train on JSON timestamps, not the video clock. |
| `spatial_overlay.mp4` | `--spatial-overlay` | Hazy RGB + blue→red grid |
| `episode.json` | always | Manifest: scenario list, biome, ego, label counts |
| `dataset_summary.json` | always | Rebuild of every `episode_*/episode.json` already in this folder |

`--media video` and `--no-rgb` delete `rgb/` only after mux succeeds.

JSON 3-D vectors are **Y-up** `(X right, Y height, Z forward)`. The sim itself is Blender Z-up. When TTC is undefined (not converging, or relative speed ≈ 0) the field is the number `9999.0`, never `null`.

## Scenarios

`--scenario auto` draws **40 % safe / 30 % near-miss / 30 % critical**, then one name from that pool. There are **50** named injectors. `--list-scenarios` is the live catalog.

| Bucket | Names |
| --- | --- |
| Safe | `safe_walk`, `empty_street`, `oncoming_pedestrian`, `parallel_pedestrian`, `cyclist_same_way`, `car_pass_far`, `car_approaching`, `distant_jaywalk`, `pothole_offset`, `parked_car_opposite`, `crossing_street`, `cyclist_overtake` |
| Near miss | `near_miss_pass`, `jaywalker_offset`, `jaywalker_from_left`, `jaywalker_from_right`, `jaywalker_turn_away`, `cyclist_near_miss`, `car_near_miss_lane`, `car_cross_front`, `cube_near_miss`, `shape_near_miss`, `pothole_near`, `cyclist_weaving`, `group_crossing`, `scooter_from_sidewalk`, `parked_car_door` |
| Critical | `jaywalker`, `jaywalker_turn_toward`, `sudden_stop`, `swerve_vehicle`, `pothole_on_path`, `cube_head_on`, `cube_from_left`, `cube_from_right`, `cube_on_path`, `shape_head_on`, `shape_from_left`, `shape_from_right`, `shapes_on_path`, `car_cut_in`, `cyclist_head_on`, `head_level_projectile`, `car_cross_critical`, `car_erratic_swerve`, `car_runs_off_road`, `child_darting`, `crossing_car_side`, `crossing_head_on`, `backing_vehicle` |

**Compounds.** Several injectors in one street. Tokens split on `,`, `+`, or spaces. A lone `auto` inside a list is itself a random draw (`pothole,auto` = pothole plus one extra event).

```bash
./run.sh --scenario jaywalker,car,pothole
./run.sh --scenarios jaywalker+car_approaching+pothole_on_path
./run.sh --scenario jaywalker --scenario car --scenario pothole
```

Occupancy is reserved once in Frenet `(s, lateral)` at inject time so actors are not born inside each other. Folder slug joins names with `__`.

**Short aliases** (usable in compounds):

| Token | Canonical |
| --- | --- |
| `safe` | `safe_walk` |
| `empty` | `empty_street` |
| `near_miss` | `near_miss_pass` |
| `jaywalk` | `jaywalker` |
| `turn_toward` / `turn_away` | `jaywalker_turn_toward` / `jaywalker_turn_away` |
| `car` / `cars` / `vehicle` | `car_approaching` |
| `pothole` / `hole` | `pothole_on_path` |
| `person` / `ped` | `oncoming_pedestrian` |
| `cyclist` / `bike` | `cyclist_same_way` |
| `cube` | `cube_near_miss` |
| `cubes` | `cube_on_path` |
| `shape` | `shape_near_miss` |
| `shapes` | `shapes_on_path` |
| `sphere` / `pyramid` | `shape_head_on` |
| `parked` | `parked_car_opposite` |
| `cut_in` | `car_cut_in` |
| `weave` | `cyclist_weaving` |
| `erratic` | `car_erratic_swerve` |
| `runoff` | `car_runs_off_road` |
| `child` / `kid` | `child_darting` |
| `projectile` | `head_level_projectile` |
| `swerve` | `swerve_vehicle` |
| `cross` / `crossing` / `crosswalk` | `crossing_street` |
| `side_car` | `crossing_car_side` |
| `group` | `group_crossing` |
| `scooter` | `scooter_from_sidewalk` |
| `overtake` | `cyclist_overtake` |
| `door` | `parked_car_door` |
| `backing` / `reverse` | `backing_vehicle` |

**What the interesting ones do**

- `jaywalker` / `jaywalker_from_left` / `jaywalker_from_right` — person enters one FOV edge and walks out the other on the street ribbon (~1.0–1.35 m/s).
- `jaywalker_turn_toward` — comes in from the side, then turns onto the sidewalk **toward** you.
- `jaywalker_turn_away` — turns onto the road and walks **with** you, ahead.
- `empty_street` — almost no background traffic or clutter. In a compound it still sparsifies the street; the other names still inject.
- `cube_*` / `shape_*` — generic primitives (cube, sphere, cylinder, pyramid, cone, capsule, lump) so the model cannot overfit to cars. `*_on_path` is stationary on the gait; `*_head_on` comes at you; `*_from_left/right` crosses.
- Street trees are **not** a named injector. They sit in the **planting strip** (and, on an avenue, a planted grass median that driving lanes do not use). They are never spawned on asphalt or the walking slab. Each tree is a recursive fork (trunk → limbs that split again) with individual triangle leaves. The **trunk** is the obstacle; rustling leaves are not. `--wind calm` almost still; `--wind windy` a real gust.
- `crossing_street` / `crossing_car_side` / `group_crossing` / `crossing_head_on` force the ego into `crosswalk` mode: Frenet lateral change plus heading that turns onto the crossing. A side car or other pedestrians on that path are ordinary injectors.

## World and ego (not injectors)

| `--biome` | What you see |
| --- | --- |
| `street` | Asphalt, kerb, buildings both sides, planting-strip trees only |
| `avenue` | Wider carriageway, planted grass median, taller facades, more traffic |
| `park` | 3 m gravel path on grass, no buildings / kerb / lane paint. Ego walks **on the path**. TTC is planar. Trees stay off the gravel. |
| `plaza` | Wide paving, buildings on one side. TTC planar. |
| `alley` | Narrow cobbled lane, close facades, little or no planting |
| `residential` | Quieter street, deeper front gardens, lower houses |
| `market` | Wide sidewalks, shop-front clutter, more people than cars |
| `auto` | Weighted draw per episode (street / park / residential / avenue / plaza / alley / market) |

| `--ego` | What the camera does |
| --- | --- |
| `walk` | Constant-speed sidewalk traverse (stroll / walk / hurry pace) |
| `diagonal_cross` | Cuts from one kerb toward the other |
| `crosswalk` | Full kerb-to-kerb turn; gaze follows Frenet motion |
| `erratic` | Sidesteps + speed wobble; may stop |
| `seated` | `walk_speed = 0`, lower eye, bench behind the HMD |
| `auto` | Weighted draw (`walk` 0.44, `erratic` 0.22, `seated` 0.14, `crosswalk` 0.12, `diagonal_cross` 0.08) |

Default sidewalk clutter is shop-front furniture (trash, scooter, barricade, puddle), not floating boxes. Potholes and debris come from scenario injectors.

## Threat labels

Evaluated at a **threat point** on the object (camera-height clamp), not the mesh origin. A pedestrian’s feet-to-camera gap must not inflate CPA.

| Label | Rule | Vest |
| --- | --- | --- |
| `CRITICAL_THREAT` | converging, TTC `< 2.5` s, CPA `< 0.5` m | Collision course, close in time |
| `NEAR_MISS` | converging, TTC `< 4.0` s, CPA in `[0.5, 1.5]` m | Will pass within a shoulder-width |
| `SAFE_STATIC` | object speed ≈ 0 and (far, or will miss, or not converging) | Furniture / a far trunk. A hole you will step in still converges and can be CRITICAL. |
| `SAFE_DYNAMIC` | moving, not in the two threat bins | Parallel traffic, a car a lane away |

`ttc == 9999.0` means **not converging**, not “very far”. Current range is `distance`. Small TTC with CPA 6 m is `SAFE_DYNAMIC`.

`class_name` is what the object is (`person`, `vehicle`, `bicycle`, `tree`, `threat_sphere`, `pothole`, …). `threat_label` is what the vest should do.

## Mixed pack (`gen_dataset.py`)

Writes a **new** folder under `datasets/` and fills it with N balanced episodes (singles from each bucket plus ~22 % compounds). Lighting, path, FOV, people, and colours still randomize per episode from `--seed`.

```bash
./gen_dataset.py --n 24 --seed 7
./gen_dataset.py --n 8 --seed 1 --dry-run
./gen_dataset.py --n 16 --seed 3 --spatial-overlay --media both --no-rgb
./gen_dataset.py --n 4 --seed 9 --no-render --name smoke
./gen_dataset.py --self-test
```

```
datasets/pack_YYYYMMDD_HHMMSS_s7_n24/
  pack.json
  plan.json                 # consumed by main.py --plan
  dataset_summary.json
  episode_0000_jaywalker/
  …
```

`gen_dataset.py` flags that pass through to Blender: `--media`, `--no-rgb`, `--frames`, `--threat-grid`, `--spatial-overlay`, `--biome`, `--ego-mode`, `--ego-height`, `--chaos`, `--wind`, `--no-trees`, `--no-render`, `--no-annotations`.

## Checks (no Blender)

```bash
python threat_math.py
python spatial_threat.py
python spatial_overlay.py
python scenario_compose.py
python gen_dataset.py --self-test
```
