# blender_sim

Headless Blender 4.x / EEVEE-Next pipeline that synthesizes an **egocentric pedestrian** dataset for a low-latency tactile warning model (wearable camera at 1.6 m, sidewalk gait, head-level hazards).

This is not an AV dataset. The camera walks the road Frenet frame (centreline arc-length plus a sidewalk lateral) with a biomechanical gait bounce and Perlin micro-saccades. Episodes randomize **biome** (street / avenue / park / plaza), **ego trajectory** (walk / diagonal / erratic / seated), lighting (including sunset glare and night), and materials (chrome cars, patterned clothes). Every frame is labelled with 2D boxes **and** a four-class threat taxonomy derived from time-to-collision and closest-point-of-approach.

## Layout

| File | Role |
| --- | --- |
| `config.py` | Resolution, FPS, gait, Perlin octaves, Poisson radii, scenario mix, domain-randomization bounds |
| `camera_kinematics.py` | `CameraRig`: \(Y(t)=1.6+A\sin(2\pi f t)\) plus 1-D fractal Perlin yaw/pitch/roll |
| `world_generator.py` | Bézier / filleted-corner streets, Poisson-disk clutter, actors, forced-collision injectors |
| `scenario_compose.py` | Compound CLI tokens, aliases, Frenet occupancy for multi-injector episodes |
| `threat_math.py` | TTC, CPA, taxonomy (pure Python, no `bpy`) |
| `projection.py` | World AABB → camera → projection → pixel box, plus raycast occlusion |
| `main.py` | Headless orchestrator (`blender --background --python main.py`) |

## Requirements

- Blender **4.2+** or **5.x** (this machine: 5.2.1). The pipeline picks `BLENDER_EEVEE` or `BLENDER_EEVEE_NEXT` from whatever the binary exposes.
- Arch: `sudo pacman -S blender`
- No pip packages. The pipeline uses only the interpreter bundled with Blender plus the Python modules in this repo.

EEVEE needs an OpenGL/EGL context. On a desktop session this is automatic. On a true headless box:

```bash
xvfb-run -a blender --background --python main.py -- --episodes 1 --no-render
```

`--no-render` still writes the JSON annotations (useful to validate TTC/CPA balance without a GPU).

## Run

```bash
cd /storage/BTP/blender_sim

# One safe episode, annotations only (no GPU required):
blender --background --python main.py -- \
  --episodes 1 --scenario safe_walk --no-render --output ./output

# Several episodes, mixed scenarios, each in its own folder
# (episode_0000_safe_walk, episode_0001_jaywalker, …). A later run
# continues after the highest existing id — it will not overwrite.
blender --background --python main.py -- \
  --episodes 8 --scenario auto --media both --seed 42 --output ./output

# Video only (per-frame JSON is still written; PNGs are deleted after mux):
blender --background --python main.py -- \
  --episodes 1 --scenario swerve_vehicle --media video --output ./output

# PNG sequence only:
blender --background --python main.py -- \
  --episodes 1 --scenario safe_walk --media frames --output ./output

# Lock a 70° horizontal FOV (otherwise each episode draws 50–90°):
blender --background --python main.py -- \
  --episodes 4 --scenario auto --hfov 70 --media frames --output ./output

# Compound street: jaywalker + oncoming car + pothole on the gait
blender --background --python main.py -- \
  --episodes 1 --scenario jaywalker,car,pothole --media both --output ./output
```

# Park biome, seated ego, max material chaos, JSON only:
./run.sh --episodes 1 --biome park --ego seated --chaos 1 --no-render --output ./output

### Mixed dataset pack

`gen_dataset.py` writes a **new folder** under `datasets/` and fills it with N episodes. Scenarios are balanced (safe / near-miss / critical singles plus compounds such as jaywalk+car+pothole). Lighting, weather, path, walk speed, FOV, people, and colours still randomize per episode from `--seed`.

```bash
./gen_dataset.py --n 24 --seed 7
./gen_dataset.py --n 8 --seed 1 --dry-run          # print the mix, no Blender
./gen_dataset.py --n 16 --seed 3 --spatial-overlay --media both --no-rgb
./gen_dataset.py --n 4 --seed 9 --no-render --name smoke
```

```
datasets/pack_20260911_035812_s7_n24/
  pack.json                 # seed, quotas, every recipe
  plan.json                 # consumed by main.py --plan
  dataset_summary.json
  episode_0000_jaywalker/
  episode_0001_safe_walk/
  episode_0002_jaywalker__car_approaching__pothole_on_path/
  …
```

CLI flags: `--episodes`, `--start-episode`, `--scenario` (repeatable; accepts `jaywalker,car,pothole`), `--scenarios`, `--list-scenarios`, `--output`, `--seed`, `--frames` (count), `--media {frames,video,both}`, `--no-rgb` (delete `rgb/` after the video is written), `--hfov`, `--lens-mm`, `--no-random-fov`, `--no-render`, `--no-annotations` (skip object-box JSON; spatial matrices still written), `--no-occlusion`, `--threat-grid K` (default 3; always writes one `spatial_annotations/spatial_annotations.json` with every frame’s `K×K` matrix), `--spatial-overlay` (writes `spatial_overlay.mp4`: hazy RGB + blue→red k×k heat), `--biome {street,avenue,park,plaza,auto}`, `--ego-mode` / `--ego {walk,diagonal_cross,erratic,seated,auto}`, `--chaos` (lock appearance dial in `[0,1]`), `--no-trees`.

## Scenario injectors (Part 4.3)

`--scenario auto` still mixes **40 % safe / 30 % near-miss / 30 % critical**. There are **36** named injectors. Pass several at once to compose a street: `--scenario jaywalker,car,pothole` (short aliases `car`→`car_approaching`, `pothole`→`pothole_on_path`). Occupancy is reserved in Frenet `(s, lateral)` at inject time so the actors do not spawn inside each other. People cross at walking speed (~1.0–1.35 m/s) on the street ribbon; nothing is integrated in a straight world line (that used to clip through buildings on curves).

| Bucket | `--scenario` names |
| --- | --- |
| Safe | `safe_walk`, `empty_street`, `oncoming_pedestrian`, `parallel_pedestrian`, `cyclist_same_way`, `car_pass_far`, `car_approaching`, `distant_jaywalk`, `pothole_offset`, `parked_car_opposite` |
| Near miss | `near_miss_pass`, `jaywalker_offset`, `jaywalker_from_left`, `jaywalker_from_right`, `jaywalker_turn_away`, `cyclist_near_miss`, `car_near_miss_lane`, `car_cross_front`, `cube_near_miss`, `pothole_near`, `cyclist_weaving` |
| Critical | `jaywalker`, `jaywalker_turn_toward`, `sudden_stop`, `swerve_vehicle`, `pothole_on_path`, `cube_head_on`, `cube_from_left`, `cube_from_right`, `car_cut_in`, `cyclist_head_on`, `head_level_projectile`, `car_cross_critical`, `car_erratic_swerve`, `car_runs_off_road`, `child_darting` |

`jaywalker` / `jaywalker_from_left` / `jaywalker_from_right` cut **through** the frame at walking speed (enter one edge, exit the other, stay on the street). `jaywalker_turn_toward` walks in from the side then **turns onto the sidewalk toward you**; `jaywalker_turn_away` turns onto the road and walks **with you** (ahead). Heading blends over ~0.9 s, it does not snap.

Horizontal FOV is stored on every frame (`camera_data.hfov_deg`, `lens_mm`) and in `episode.json`. Default: randomize per episode in \(50^\circ\)–\(90^\circ\) (about 35 mm–18 mm on a 36 mm sensor). Pass `--hfov` or `--lens-mm` to lock it.

## Threat taxonomy

For \(\vec{P}_{rel}=\vec{P}_{obj}-\vec{P}_{cam}\), \(\vec{V}_{rel}=\vec{V}_{obj}-\vec{V}_{cam}\):

\[
TTC=\frac{-(\vec{P}_{rel}\cdot\vec{V}_{rel})}{\|\vec{V}_{rel}\|^{2}}
\quad\text{iff}\quad
\vec{P}_{rel}\cdot\vec{V}_{rel}<0
\]

\[
D_{cpa}=\|\vec{P}_{rel}+\vec{V}_{rel}\cdot TTC\|
\]

| Label | Rule |
| --- | --- |
| `CRITICAL_THREAT` | \(TTC<2.5\) s and \(D_{cpa}<0.5\) m |
| `NEAR_MISS` | \(TTC<4.0\) s and \(0.5\le D_{cpa}\le 1.5\) m |
| `SAFE_STATIC` | \(\|V_{obj}\|\approx 0\) and (range \(>5\) m or will miss) |
| `SAFE_DYNAMIC` | moving, not a labelled threat |

Kinematics are evaluated at a **threat point** on the object's world AABB (camera-height clamp) so a pedestrian's feet-to-camera vertical gap cannot inflate \(D_{cpa}\) past the 0.5 m critical threshold.

## Output

```
output/
  dataset_summary.json       # rebuilt from every episode_* on disk
  episode_0000_safe_walk/
    episode.json
    preview.mp4              # --media video or both
    spatial_overlay.mp4      # --spatial-overlay (hazy RGB + k×k heat)
    rgb/000000.png …         # --media frames or both; deleted if --no-rgb or --media video
    annotations/annotations.json  # object boxes / TTC / labels for every frame (skip with --no-annotations)
    spatial_annotations/spatial_annotations.json  # always, all k×k matrices (default k=3)
  episode_0001_jaywalker/
    …
```

`--scenario auto` draws the 40/30/30 mix so successive episodes differ. Omit `--start-episode` to append; pass `--start-episode 0` only when you intend to start a numbered shard at 0000.

`--media` defaults to `both` (`config.py` → `output.media`). Object boxes live in one `annotations/annotations.json` (`frames[]`, same fields as before). `--no-annotations` skips that file; `spatial_annotations/spatial_annotations.json` is still written. Video is muxed from the PNG sequence with system `ffmpeg` (CRF 18, yuv420p) or, if ffmpeg is missing, Blender's bundled H.264 encoder. `--no-rgb` (or `--media video`) deletes `rgb/` after a successful mux so you keep the ~5 MB mp4 instead of ~900 MB of PNGs. `--no-render` skips both frames and video.

Each frame record matches the Part 6.1 schema (`frame_id`, `camera_data`, `environment`, `objects[]` with `threat_label`, `kinematics`, `bounding_box_2d`, `flags`). When a pair is not converging, `ttc` is written as `9999.0` so the field stays a JSON number.

Internal simulation is Blender **Z-up**. JSON vectors are converted to the spec's **Y-up** frame \((X, Y_{up}, Z_{forward})\).

## Math self-test (no Blender)

```bash
python threat_math.py
python spatial_threat.py
python spatial_overlay.py
python gen_dataset.py --self-test
```
