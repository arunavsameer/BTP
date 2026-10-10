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

# Known-broken injector (gaze often misses the car) — JSON-only inspect
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
| `--frames N` | config `450` | Frames per episode (`0` = config). 30 fps ⇒ 450 frames = 15.0 s |
| `--duration SEC` | config 15.0 s | `frames = round(SEC * fps)`. `--frames` wins if both are set. |
| `--media` | `both` | `frames` (keep `rgb/`), `video` (mux then delete PNGs), `both` |
| `--no-rgb` | off | Delete `rgb/` after `preview.mp4` (and overlay) succeed. PNGs are kept if mux fails. Ignored with `--media frames`. |
| `--no-render` | off | Skip EEVEE. Still writes JSON. |
| `--no-annotations` | off | Skip `annotations/annotations.json`. Spatial matrices still write. |
| `--no-occlusion` | off | Skip the centre-ray occlusion flag |
| `--threat-grid K` | `3` | `K×K` spatial threat matrix, always written |
| `--spatial-overlay` | off | Also write `spatial_overlay.mp4` (needs a render) |
| `--biome` | `auto` | `street` \| `avenue` \| `park` \| `plaza` \| `alley` \| `residential` \| `market` \| `auto` |
| `--ego-mode` / `--ego` | `auto` | `walk` \| `diagonal_cross` \| `crosswalk` \| `erratic` \| `hasty` \| `seated` \| `auto` |
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
| `spatial_annotations/spatial_annotations.json` | always | `K×K` float heat in `[0,1]`, row 0 = top of image. **Not** the four-class label. |
| `rgb/XXXXXX.png` | `frames` / `both` | 1920×1080 RGB, AgX. `000007.png` is `frame_id` `"000007"` |
| `preview.mp4` | `video` / `both` | Same sequence muxed at `fps`. Train on JSON timestamps, not the video clock. |
| `spatial_overlay.mp4` | `--spatial-overlay` | Hazy RGB + blue→red grid |
| `episode.json` | always | Manifest: scenario list, biome, ego, label counts |
| `dataset_summary.json` | always | Rebuild of every `episode_*/episode.json` already in this folder |

`--media video` and `--no-rgb` delete `rgb/` only after mux succeeds.

JSON 3-D vectors are **Y-up** `(X right, Y height, Z forward)`. The sim itself is Blender Z-up. When TTC is undefined (not converging, or relative speed ≈ 0) the field is the number `9999.0`, never `null`.

## Scenarios

`--scenario auto` draws **40 % safe / 30 % near-miss / 30 % critical**, then one name from that pool. There are **63** named injectors (55 in the auto mix, plus 8 empty-center / side-threat names). **21** of them are `--groups` forkable (`FORKABLE_SCENARIOS`). `--list-scenarios` is the live catalog.

| Bucket | Names |
| --- | --- |
| Safe | `safe_walk`, `empty_street`, `oncoming_pedestrian`, `parallel_pedestrian`, `cyclist_same_way`, `car_pass_far`, `car_approaching`, `distant_jaywalk`, `pothole_offset`, `parked_car_opposite`, `crossing_street`, `cyclist_overtake`, `hasty_look` |
| Near miss | `near_miss_pass`, `jaywalker_offset`, `jaywalker_from_left`, `jaywalker_from_right`, `jaywalker_turn_away`, `cyclist_near_miss`, `car_near_miss_lane`, `car_cross_front`, `cube_near_miss`, `shape_near_miss`, `pothole_near`, `tree_near`, `lamp_near`, `cyclist_weaving`, `group_crossing`, `scooter_from_sidewalk`, `parked_car_door` |
| Critical | `jaywalker`, `jaywalker_turn_toward`, `sudden_stop`, `swerve_vehicle`, `pothole_on_path`, `tree_on_path`, `lamp_on_path`, `cube_head_on`, `cube_from_left`, `cube_from_right`, `cube_on_path`, `shape_head_on`, `shape_from_left`, `shape_from_right`, `shapes_on_path`, `car_cut_in`, `cyclist_head_on`, `head_level_projectile`, `car_cross_critical`, `car_erratic_swerve`, `car_runs_off_road`, `child_darting`, `crossing_car_side`, `crossing_head_on`, `backing_vehicle` |

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
| `tree` | `tree_on_path` |
| `lamp` / `streetlamp` | `lamp_on_path` |
| `hasty` | `hasty_look` |
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
| `periph` / `clear_center` / `side_empty` | `periph_empty` |
| `side_pass` | `periph_car_side` |
| `side_parked` | `periph_parked` |
| `side_ped` | `periph_ped_side` |
| `side_turn` | `periph_car_turn` |
| `side_cut` | `periph_ped_cut` |
| `side_runoff` | `periph_car_runoff` |
| `side_child` | `periph_child_cut` |

**Every named injector** (what the clip shows). `--groups` is yes only for `FORKABLE_SCENARIOS` in `outcome_groups.py` — moving intercepts plus `cube_on_path` (rolls aside on a miss). `safe_walk` is mixed-pack only. `--list-scenarios` is the live catalog.

| Name | Auto mix | `--groups` | What the clip shows |
| --- | --- | --- | --- |
| `safe_walk` | safe | no | Sidewalk walk, background traffic only. Injector is a no-op. |
| `empty_street` | safe | no | Almost no background people/cars/clutter. In a compound it still sparsifies; the other names still inject. |
| `oncoming_pedestrian` | safe | no | Person on the opposite sidewalk walking toward you (large CPA). |
| `parallel_pedestrian` | safe | no | Person same way, slightly toward the curb, a few metres ahead. |
| `cyclist_same_way` | safe | no | Bike in the far lane, same direction. |
| `car_pass_far` | safe | no | Oncoming car in the far lane (~14.4 s). |
| `car_approaching` | safe | no | Oncoming car in the near lane unless a through-crosser is also in the mix (then far). |
| `distant_jaywalk` | safe | no | Person crossing far ahead, not a hit. |
| `pothole_offset` | safe | no | Hole / crater / slab / debris ~1.8 m off the gait. |
| `parked_car_opposite` | safe | no | Parked vehicle in the far gutter. |
| `crossing_street` | safe | no | You turn onto a kerb-to-kerb crosswalk. No extra threat actor. |
| `cyclist_overtake` | safe | no | Bike same way, slightly toward the road, faster, starts just ahead. |
| `hasty_look` | safe | no | Background street; large high-frequency look jitter (`--ego hasty`). |
| `near_miss_pass` | near | no | Oncoming person who will graze (~1 m CPA). |
| `jaywalker_offset` | near | no | Through-cross from the left, ~1 m CPA. |
| `jaywalker_from_left` | near | yes | Through-cross from the left (HFOV-edge start, walking by t≤6 s). Groups force a collision-course CPA. |
| `jaywalker_from_right` | near | yes | Same from the right. |
| `jaywalker_turn_away` | near | no | Enters from a corridor edge, turns, walks **with** you ahead. |
| `cyclist_near_miss` | near | no | Oncoming bike, ~1 m CPA. |
| `car_near_miss_lane` | near | no | Oncoming car in the near lane, ~1 m CPA. |
| `car_cross_front` | near | no | Vehicle crossing in front, ~1 m CPA. |
| `cube_near_miss` | near | no | Oncoming cube, ~1 m CPA. |
| `shape_near_miss` | near | no | Oncoming primitive (sphere / pyramid / …), ~1 m CPA. |
| `pothole_near` | near | no | Hole ~0.85 m off the gait (shin graze). |
| `tree_near` | near | no | Extra trunk ~0.8 m off the gait, ~16.5 m ahead. |
| `lamp_near` | near | no | Extra pole ~0.8 m off the gait, ~16.5 m ahead. |
| `cyclist_weaving` | near | no | Near-lane bike, sine weave toward the gait. |
| `group_crossing` | near | no | 2–3 people through-cross while you use a crosswalk. |
| `scooter_from_sidewalk` | near | no | Bicycle cutting across from a corridor edge. |
| `parked_car_door` | near | no | Parked car in the near gutter plus a door-height box on the gait. |
| `jaywalker` | crit | yes | Person from an on-screen HFOV edge (not the far kerb); walking by t≤6 s; occupy the gait ~14.3 s. |
| `jaywalker_turn_toward` | crit | yes | Corridor-edge wait, then turns onto the sidewalk **toward** you. |
| `sudden_stop` | crit | no | Person ahead stops at 11.5 s; you close on a now-static body. |
| `swerve_vehicle` | crit | yes | Near-lane car drifts onto the gait (~14.4 s). |
| `pothole_on_path` | crit | no | Hole on the gait ~18 m ahead. |
| `tree_on_path` | crit | no | Trunk on the gait ~17.5 m ahead. Splat / TTC use the bole. |
| `lamp_on_path` | crit | no | Pole on the gait ~17.5 m ahead. Splat / TTC use the pole, not the arm. |
| `cube_head_on` | crit | yes | Cube coming at you on the gait. |
| `cube_from_left` / `cube_from_right` | crit | yes | Cube crossing through the gait. |
| `cube_on_path` | crit | yes | Static cube on the gait. Groups: it rolls aside on a miss. |
| `shape_head_on` | crit | yes | Mixed primitive coming at you on the gait. |
| `shape_from_left` / `shape_from_right` | crit | yes | Primitive crossing through the gait. |
| `shapes_on_path` | crit | no | 2–4 static primitives on the gait. |
| `car_cut_in` | crit | yes | Near-lane car cuts onto the gait (~14.3 s). |
| `cyclist_head_on` | crit | yes | Oncoming bike on the gait. |
| `head_level_projectile` | crit | no | Eye-height cube at 3.6 m/s, τ=4.0 s. |
| `car_cross_critical` | crit | yes | Vehicle crossing through the gait, ~0.12 m CPA. |
| `car_erratic_swerve` | crit | no | Near-lane car with a large weave. |
| `car_runs_off_road` | crit | yes | Near-lane car mounts the sidewalk ~14 s. |
| `child_darting` | crit | yes | Child waits on-screen (faces the crossing), darts at t0∈[7.2, 10.4] s — not a 12 s kerb statue. |
| `crossing_car_side` | crit | yes | **Intended:** near-lane oncoming car while you turn onto a crosswalk. **Still broken:** camera often looks at empty pavement / puddles; no car in frame. Do not train on this name until it is fixed. |
| `crossing_head_on` | crit | no | Far-lane oncoming car while you crosswalk (same injector family as `crossing_car_side`). |
| `backing_vehicle` | crit | no | Car ahead reversing toward you. |
| `periph_empty` | — | no | Empty sidewalk ahead. No extra actor. Not in `--scenario auto`. |
| `periph_car_side` | — | no | Far-lane car stays in lane on a FOV edge. |
| `periph_parked` | — | no | Parked car in a gutter ~18 m ahead. |
| `periph_ped_side` | — | no | Person on the opposite sidewalk only. |
| `periph_car_turn` | — | yes | Empty sidewalk; near-lane car holds until ~11 s, then cuts onto the gait. |
| `periph_car_runoff` | — | no | Same start, then mounts the pavement. |
| `periph_ped_cut` | — | yes | Empty sidewalk, then a person from an HFOV edge cuts the gait. |
| `periph_child_cut` | — | yes | Empty sidewalk, then a child dart through the gait. |

Peripheral names (`periph_*`) are **not** in `--scenario auto`. Use `--scenario periph_ped_cut` or `gen_dataset.py --theme peripheral`. Ego is forced to `walk`; auto-biome skips park / plaza. `crossing_street` / `group_crossing` / `crossing_head_on` / `crossing_car_side` force ego `crosswalk`. `cube_*` / `shape_*` are generic primitives so the model cannot overfit to cars. Street trees in the **planting strip** (and, on an avenue, a planted grass median) are background; the **trunk** is the obstacle; rustling leaves are not. `--wind calm` almost still; `--wind windy` a real gust.

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
| `erratic` | Sidesteps + speed wobble; may ease to a stop (gaze stays down the path) |
| `hasty` | Large high-frequency yaw/pitch/roll (look only; gait path is unchanged) |
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

`class_name` is what the object is (`person`, `vehicle`, `bicycle`, `tree`, `streetlamp`, `threat_sphere`, `pothole`, …). `threat_label` is what the vest should do.

## Spatial heat

`--threat-grid K` (default 3, often 5) writes a `K×K` matrix per frame. A cell is hot when a human walker would **slow or stop** for that object **and** that object currently covers the cell (inscribed mesh parts: cabin, limb, trunk, **pole** — not the outer AABB, not a cobra-arm / canopy hull, no neighbour bleed). Mere distance is not a threat: a lamp beside a seated walker stays 0. Behind the camera, fully off-screen, occluded, or already passed (static / hole with `path_s<0`) cannot paint red.

Each object scores \(S=\max(S_{\mathrm{hit}},S_{\mathrm{path}},S_{\mathrm{stat}},S_{\mathrm{prox}})\); the cell keeps the max over objects that overlap it. Cars, people, holes, and furniture use different envelopes (`kind_for` in `spatial_threat.py`). Both still (seated + a pole) → \(S=0\). Urgency is a class TTC window: a pedestrian ~5 m on a collision course is already amber (time to act); 11 m stays cold. A person filling the gait at ~2 m is red.

- \(S_{\mathrm{hit}}\) — will the hulls collide if both keep their current planar velocity? Hit likelihood × class TTC window. Walking into a hole, a head-on car, a child through the chest.
- \(S_{\mathrm{path}}\) — a **mover** is cutting the space I would walk into, even if the disk centres miss. A car crossing 5–7 m in front is a stop; the same car parallel in the next lane stays cold. A person cutting the gait ~6–8 m ahead is already warm; ~2 m is red.
- \(S_{\mathrm{stat}}\) — I am walking **toward** a world-static body on my gait (parked car, tree, pole). Inverse in gap. An on-gait tree warms from ~5–6 m. A lamp I will pass is not a threat.
- \(S_{\mathrm{prox}}\) — already inside personal space in front and closing.

JSON `threat_label` / TTC / CPA stay the **point-mass** solve above. The overlay can disagree with the label on purpose (a crossing car with a 5 m CPA that you would still halt for).

Formulas and knobs: [`IMPLEMENTATION.md`](IMPLEMENTATION.md) §16.

## Mixed pack (`gen_dataset.py`)

Writes a **new** folder under `datasets/` and fills it with N balanced episodes (singles from each bucket plus ~22 % compounds). Lighting, path, FOV, people, and colours still randomize per episode from `--seed`.

```bash
./gen_dataset.py --n 24 --seed 7
./gen_dataset.py --n 8 --seed 1 --dry-run
./gen_dataset.py --n 16 --seed 3 --spatial-overlay --media both --no-rgb
./gen_dataset.py --n 30 --seed 919 --name sample30_grid5 --threat-grid 5 --spatial-overlay --media video --no-rgb
./gen_dataset.py --n 4 --seed 9 --no-render --name smoke
./gen_dataset.py --n 40 --seed 11 --theme peripheral --name side40 --dry-run
./gen_dataset.py --n 40 --seed 11 --theme peripheral --name side40 --media both --no-rgb --spatial-overlay --ego walk
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

`gen_dataset.py` flags that pass through to Blender: `--media`, `--no-rgb`, `--frames`, `--duration`, `--threat-grid`, `--spatial-overlay`, `--biome`, `--ego-mode`, `--ego-height`, `--chaos`, `--wind`, `--no-trees`, `--no-render`, `--no-annotations`. Planner-only: `--theme`, `--groups`, `--bg-variants`, `--group-scenarios`, `--dry-run`.

`--theme peripheral` (aliases `side`, `clear_center`) builds only the empty-center / side-threat catalog: ~18 % empty street, ~37 % side-stay (car / parked / opposite ped), ~45 % side-cut (car turns in, person or child cuts through). Mixed packs are unchanged. `--groups` ignores `--theme`.

## Outcome groups (`gen_dataset.py --groups`)

`--groups` is **opt-in** on `gen_dataset.py` (not a `main.py` flag). Each **sample** is one threat with **two** endings that share RGB until `diverge_s` (12.15 s of a 15 s / 450-frame clip; intercept τ ~14.3 s; hit at 14.35 s). Actors are **not** sped up to fill the extra seconds.

- `collision/` — always written; the threat continues into the walker
- `miss_left/` **or** `miss_right/` — one miss per sample (50/50 from `--seed`). The threat peels to that side. The unchosen miss folder is omitted.

`--groups` cycles `FORKABLE_SCENARIOS` (see the table above). It **drops `safe_walk`**. `safe_walk` still exists for mixed packs and for `./run.sh --scenario safe_walk`. `--theme` is ignored. Unknown names in `--group-scenarios` are ignored; if none remain, the full forkable catalog is used. `--n` is the number of **samples** (not videos): each sample writes **2 outcomes** × `--bg-variants` clips. `--bg-variants` (default **2**, clamped 1–8) is how many `bg_YY` worlds each sample gets.

A vehicle miss needs ≥ 2.50 m of lateral room after the corridor clip. If that side is a facade (typical right-sidewalk cut-in), the car takes the **far driving lane** instead of hugging the wall and still colliding. People peel at ~1.25–2.15 m/s; cars at ~2.4–5.4 m/s. `StreetCorridor` pad may go **negative** so the peel can use the verge.

Inside each ending, `bg_00`, `bg_01`, … rebuild non-threat scatter (trees, buildings, furniture; `full` / `sparse` / `bare`) **and** a distinct time-of-day slot per `bg_YY`: `dawn` / `noon` / `night` / `dusk` / `overcast` / `harsh_glare`. Lighting-per-bg is automatic with `--groups`; there is **no** `--bg-lighting` CLI flag. Path, ego, clothes, and the threat injector stay fixed. Collision and miss of the **same** `bg_YY` share lighting. Collision is a full render; miss clips **hard-link** (or copy) prefix PNGs, skip the prefix depsgraph / label pass, concat-copy the prefix `preview.mp4`, and only re-render the tail.

Folder names are trainer-indexable: `sample_XXXX_<scenario>/{collision,miss_left|miss_right}/bg_YY/`. `groups_index.json` lists only clips that exist (`y=1` collision, `y=0` miss; `train_glob` is in that file).

```bash
./gen_dataset.py --groups --n 4 --bg-variants 2 --seed 21 --name groups15 \
  --threat-grid 5 --spatial-overlay --media video --no-rgb --biome auto
# Optional: pin names (otherwise the forkable catalog is cycled)
./gen_dataset.py --groups --n 2 --bg-variants 2 --group-scenarios jaywalker,car_cut_in --dry-run
```

```
datasets/groups15_s21_n4/
  pack.json
  plan.json
  groups_index.json          # flat list for a training dataloader
  sample_0000_jaywalker/
    group.json
    collision/bg_00/preview.mp4
    miss_left/bg_00/…        # this sample drew miss_left; others may draw miss_right
```

Python index (`y=1` collision, `y=0` miss):

```python
import json
from pathlib import Path
root = Path("datasets/groups15_s21_n4")
clips = json.loads((root / "groups_index.json").read_text())["clips"]
row = clips[0]
video = root / row["path"] / "preview.mp4"
# sample_0000_<scenario>/{collision,miss_left|miss_right}/bg_YY/
```

Walk speed is not increased for 15 s clips. People spawn on an on-screen HFOV edge (not a far kerb the lens has already cropped), face `lat_target` while they wait, and start the crossing by t≤6 s at a human pace so they occupy the gait around 14 s — the last second is the collision or the miss, not empty road.

`--no-rgb` / `--media video` deletes `rgb/` after mux. That is fine for later overlay remux: `remux_spatial_from_json.py` reads `preview.mp4` + `annotations/annotations.json`, not the PNG folder. `--no-annotations` would block remux. `--no-render` writes JSON only (no `preview.mp4`, no overlay).

## Overlay remux (no Blender)

After a pack exists, recompute `spatial_overlay.mp4` (and rewrite spatial JSON) from current `spatial_threat.py` without an EEVEE re-render:

```bash
python remux_spatial_from_json.py --pack datasets/groups15_s21_n4
python remux_spatial_from_json.py --clip datasets/groups15_s21_n4/sample_0000_jaywalker/miss_left/bg_00
```

Needs `preview.mp4` and `annotations/annotations.json` in each clip dir. Streetlamp / tree paint a pole/trunk strip, not the cobra-arm AABB. JSON TTC / `threat_label` are left alone (point-mass).

## Checks (no Blender)

```bash
python threat_math.py
python spatial_threat.py
python spatial_overlay.py
python scenario_compose.py
python outcome_groups.py
python gen_dataset.py --self-test
```

`python remux_spatial_from_json.py --pack …` is not a unit test; it rewrites overlays on an existing pack.
