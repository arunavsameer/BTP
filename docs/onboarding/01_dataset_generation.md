# Dataset generation — how we mint labeled walking videos

This is the first thing to understand after the problem statement.

The vest needs thousands of examples of “you are about to hit something” and “the road ahead is empty.” Filming that on a real visually-impaired walker is slow, unsafe, and almost never captures a true collision. So we **simulate** a person walking a street, we **know the 3-D truth**, and we write labels from physics rather than from pixels.

Two simulators exist. **`blender_sim` is the one the models actually train on.** `synth_sim` is an earlier from-scratch GPU rasterizer with the same scientific job. Read blender first; synth is “we also tried building a renderer without Blender.”

---

## 1. What we are trying to teach

Imagine you are walking on a sidewalk with a camera on your forehead.

Things that should make the vest buzz:

- A person stepping into your stride
- A car cutting toward you
- A pothole you will actually step in
- A low branch at head height
- A cube / sphere / random shape on the path (so the model cannot cheat by only recognizing cars)

Things that should stay quiet:

- A parked car in the far lane
- Someone walking the other sidewalk
- Trees whose **leaves** rustle (the trunk is the obstacle, not the canopy)
- Texture, puddles that are not holes, distant traffic

The product is **not** an autonomous car. The ego is a pedestrian. Labels exist so a tactile vest can learn **buzz / graded buzz / silent**.

We also decided, for the learned models, that we will **not** put YOLO, a tracker, or a time-to-collision formula on the vest. Those live only in the **simulator**, to create ground truth. The wearable should look at cheap RGB frames and output a danger map.

---

## 2. Why synthetic data

| Real footage problem | What the sim gives us |
| --- | --- |
| You cannot recover exact 3-D time-to-hit from a phone video | The sim knows every object’s position and velocity |
| True collisions are rare and unethical to stage | We can *aim* a jaywalker to miss by 12 cm in 2.4 s |
| Almost every frame is “empty sidewalk” | We force a mix of safe / near-miss / critical clips |
| A detector overfits to “cars look like this” | We also spawn cubes, spheres, pyramids, children, scooters |
| Lighting / body height / FOV are fixed | Every episode randomizes those on purpose |

The detector this dataset trains must learn physics-shaped ideas:

- A moving canopy is **not** a threat (trunk is static; foliage is visual only)
- A hole in the pavement **is** a threat
- A seated walker facing a parked bollard is SAFE (nothing is closing)
- A seated walker with a car driving at them is **not** SAFE

---

## 3. The unit of data: an episode

An **episode** is one short clip of one walk.

In `blender_sim` (what we train on):

- **150 frames** at **30 fps** → **5 seconds**
- Camera is a walking (or seated) human at roughly **1.6 m** eye height
- Output folder looks like `episode_0000_jaywalker/`

Each episode writes some mix of:

| File | What it is |
| --- | --- |
| `preview.mp4` | The forehead RGB video (this is what models see) |
| `rgb/000000.png` … | Same frames as PNGs (optional; often deleted after mux) |
| `annotations/annotations.json` | Per-object boxes + TTC + four-class threat label |
| `spatial_annotations/spatial_annotations.json` | The **K×K heatmap** the models train on |
| `episode.json` | Manifest: scenario names, biome, ego mode, label counts |

**Models do not train on the four-class object labels.** They train on the **spatial heatmap**: a tiny grid painted over the image that says “how collision-like is this patch of the world.”

Think of the object JSON as the *physics notebook*. Think of the heatmap as the *exam the student has to pass*.

---

## 4. The camera is a walking head, not a car

The ego is a person. That changes everything.

**Height.** Standing eye height is drawn per episode: short (~1.35–1.50 m), typical (~1.55–1.72 m), tall (~1.75–1.90 m). Seated drops to ~0.95–1.28 m.

**Gait.** The head bobs a few centimeters with each step and fidgets (tiny yaw / pitch / roll). If we froze the camera on a tripod, a network would learn “optical flow always means walking forward,” which is a lie.

**Ego modes** (how the body moves along the street):

| Mode | What the camera does |
| --- | --- |
| `walk` | Constant-speed sidewalk traverse (stroll / walk / hurry) |
| `diagonal_cross` | Cuts from one kerb toward the other |
| `crosswalk` | Turns onto a crossing; gaze follows the motion |
| `erratic` | Sidesteps + speed wobble; may stop |
| `seated` | Speed = 0, lower eye, bench behind the head |

**Seated / stopped is a first-class state.** Every formula must survive `walk_speed = 0` without dividing by zero. A parked car 20 m away from a sitting person is SAFE. A ball thrown at a sitting person is still CRITICAL.

**Field of view.** Horizontal FOV is randomized ~50–90° unless locked. A model that only ever saw 73° would overfit to that crop.

---

## 5. The street is a ribbon, not a video game map

Everything that moves lives in a simple chart called **Frenet coordinates**:

- `s` = how far along the **road centreline** you have walked (metres of arc)
- `lateral` = signed leftover to the side (one sidewalk is a constant offset of that ribbon)

Why this matters in one sentence: a jaywalker, a car, and the camera all share **one** length-along-the-road number. If the camera used a different path length than the car, a “hit me in 2 seconds” spawn would land behind you.

Actors are **not** bouncing off each other every frame. Occupancy is reserved **once** when the scenario is injected (“this capsule of sidewalk is taken”). During the 150 frames they just slide along `(s, lateral)` and stay inside the street corridor. That keeps intercepts mathematically aimed instead of fighting a physics engine.

**Biomes** change what the ribbon *looks* like: street, avenue, park, plaza, alley, residential, market. A park is a gravel path on grass with no buildings. An alley is a narrow cobbled lane. `auto` draws a mix.

---

## 6. Time-to-collision, explained like you asked a friend

Forget neural nets. Two points are moving in 3-D.

You are at position **P_cam** with velocity **V_cam**.
An object is at **P_obj** with velocity **V_obj**.

The **gap vector** is `P = P_obj − P_cam` (where is it relative to me).
The **relative velocity** is `V = V_obj − V_cam` (how is that gap changing).

If `P · V` is **negative**, the gap is shrinking. They are **converging**. If not, they are moving apart or sliding sideways and there is no future collision under “keep going as you are.”

If they keep those velocities, the time when they come **closest** is:

```
TTC = − (P · V) / ||V||²
```

Plug that time back in and you get the miss distance:

```
CPA = || P + TTC · V ||
```

**TTC** = “how many seconds until the closest moment.”
**CPA** = “how many metres apart will we be at that moment.”

Intuition:

- Head-on, same sidewalk, 2 seconds out → small TTC, CPA ≈ 0 → **you will hit**
- They will pass 80 cm to your left in 2.5 s → small TTC, CPA ≈ 0.8 m → **near miss** (shoulder-width)
- A car in the far lane, 15 m, parallel → large CPA → **safe**, even if it looks big in the picture
- Receding person behind you → not converging → TTC is undefined (we write `9999.0`, not “very far”)

Assumptions we happily accept:

- Constant velocity (we do not pretend to know future braking)
- **Point** at a chosen “threat point,” not a full swept mesh. A wide van can graze you even if centre-to-centre CPA is 0.6 m. That is why the heatmap (next section) is **body-aware**, while the four-class label is the simple point model.

**Threat point, not mesh origin.** A pedestrian’s feet-to-camera gap must not inflate the miss distance. For a person we evaluate near camera height. For a **pothole** we lift the threat point up to eye height so “I will step in this hole” is not counted as a 1.6 m vertical miss.

**Planar mode** (park / plaza): ignore vertical separation. A bollard and a 1.6 m eye should not look like they “miss” just because one is on the ground.

---

## 7. Four-class object labels (the physics notebook)

Evaluated every frame, per object, from TTC + CPA + whether the object is moving.

| Label | Rule of thumb | Vest meaning |
| --- | --- | --- |
| `CRITICAL_THREAT` | converging, TTC &lt; 2.5 s, CPA &lt; 0.5 m | Collision course, close in time |
| `NEAR_MISS` | converging, TTC &lt; 4.0 s, CPA in [0.5, 1.5] m | Will pass within a shoulder-width |
| `SAFE_STATIC` | object basically still, and far / will miss / not closing | Furniture, a far trunk |
| `SAFE_DYNAMIC` | moving, but not in the two threat bins | Parallel traffic, a car a lane away |

A pothole you **will** step in still converges (`V_obj = 0`, `V_rel = −V_cam`) and can be CRITICAL. A hole you will walk around is SAFE_STATIC.

`class_name` is *what it is* (`person`, `vehicle`, `pothole`, `tree`, …).
`threat_label` is *what the vest should do about it*.

These labels are useful for balancing a pack (“make 30% of episodes contain a critical event”). They are **not** the tensor the CNN is scored on.

---

## 8. The 5×5 heatmap — the actual training target

Pretend the image in front of you is a tic-tac-toe board, but **5 by 5**.

```
              LEFT    LEFT   CENTER   RIGHT   RIGHT
 far  (row 0)   ·       ·       ·       ·       ·     ← distant
               ·       ·       ·       ·       ·
               ·       ·       ·       ·       ·
               ·       ·       ·       ·       ·
near  (row 4)   ·       ·       ·       ·       ·     ← at your feet
```

Each cell is a number in **[0, 1]**:

- **0** = empty / you will not collide with whatever is there
- **1** = collision-worthy: you would hit or step into this

**Rows:** top of the image = far. Bottom = near / feet. Near rows matter more for the buzz.
**Columns:** left → right. The middle column is the walking corridor.

This matrix is **not** “paint the 2-D box of every object.” It is **path occupancy**:

1. Score the object in metres: would this body occupy the walker’s **gait tube** (shoulder width + a little sway) in the next ~2 seconds?
2. Then **splat** that score onto the image grid using the object’s 2-D box, with a little Gaussian bleed so a person is not a hard pixel.

Ways an object can score high (plain language):

1. **Stopping volume** — it sits in the rectangle you will sweep if you keep walking (gated by closing speed, so a seated person staring at a bollard scores 0)
2. **Still in the tube when you arrive** — a static pothole on the gait never leaves; `t_leave = ∞`
3. **Crossing intercept** — they *enter* the tube at the same `s` you will be at (late cut-in)
4. **Adjacent high speed** — not a hit, but a fast neighbour; mid band, not a collision

Leaves rustling are **not** scored as the tree’s threat. The **trunk** is. That is a deliberate dataset rule: we do not want the vest screaming at foliage.

Older packs used **3×3**. Live packs used by the models are **native 5×5**. If you ever upsample 3×3 → 5×5 with nearest-neighbour you get ugly 2×2 blocks. Prefer native 5×5.

**Corridor prior (this will haunt every model):** even on a “safe” walk, the center column often sits around 0.4–0.7 because the walking lane *looks* occupied by the road itself / distant clutter. Copying “danger now = danger in 1 s” is therefore a shockingly good cheat. We come back to that in the model notes.

---

## 9. Scenarios — how we force interesting events

If we only simulated “a random street,” almost every clip would be SAFE. So each episode **injects** one or more named events.

`--scenario auto` draws roughly **40% safe / 30% near-miss / 30% critical**, then picks a name from that bucket.

Examples you will see in folder names:

| Bucket | Examples |
| --- | --- |
| Safe | `safe_walk`, `empty_street`, `car_pass_far`, `parallel_pedestrian` |
| Near miss | `jaywalker_offset`, `cyclist_near_miss`, `car_near_miss_lane`, `parked_car_door` |
| Critical | `jaywalker`, `pothole_on_path`, `cube_head_on`, `child_darting`, `car_cut_in` |

**Compounds** stack injectors on one street: `jaywalker,car,pothole`. Occupancy is reserved so they are not born inside each other.

**Generic primitives** (`cube_*`, `shape_*`) exist so the network cannot overfit to “threat = car mesh.” `*_on_path` sits still on the gait; `*_head_on` comes at you; `*_from_left/right` crosses.

**Peripheral / clear-center** episodes (`periph_*`) keep the **image centre empty** and put threat on the side, then sometimes cut in. Those are hard negatives for a model that loves to paint the corridor red. They are **not** in the default `auto` mix; they are built with `--theme peripheral`.

How a critical jaywalker is *aimed*: the director knows where the camera will be at time `t*`, then sets the person’s velocity so they arrive there (or miss by a chosen lateral offset). That is **trajectory inversion**. Closed-form, not luck.

---

## 10. Domain randomization — why every episode looks different

Same scenario name, different pictures, on purpose:

- Time of day / lighting (including dusk via env)
- Weather-ish wetness, fog in synth; Blender chaos for materials
- Walk speed, stature, FOV
- Building colours, shop-front clutter
- Path shape (the centreline wanders)
- Wind: leaf rustle; trunk stays still
- Background crowd that must **not** steal the class on SAFE episodes (oncoming people are kept off the wearer’s walk center)

`gen_dataset.py` does **not** re-randomize those. It only chooses **which scenario(s)** go in each episode so a pack is balanced (~22% compounds, leftover 40/30/30). Lighting and gait still randomize from `--seed` inside Blender.

Typical packs the models mention:

- `mixed250_…` — balanced mix
- `side50_…` — peripheral / false-positive hard negatives
- `pack_…_n1000` — large older pack (sometimes mixed into training)

Episode folder names **collide** across packs (`episode_0000_jaywalker` exists twice). Cache keys are therefore `source/episode_dir`, e.g. `mixed/episode_0000_pothole_on_path`.

---

## 11. How an episode is actually produced (`blender_sim`)

Two phases, because rendering every frame as a still would tear down the GPU renderer 150 times.

```
Pick scenario + biome + ego
Build the world (meshes, lights, background people)
Inject the scripted event (reserve Frenet occupancy)
Place the camera rig

Phase A (CPU, no pretty pictures)
  for each frame:
      step gait + actors
      snapshot poses
      compute boxes, TTC/CPA labels, K×K splat

Phase B (GPU, one animation)
  restore snapshots
  render rgb/*.png → mux preview.mp4
  optional spatial_overlay.mp4 (hazy RGB + blue→red grid)

Purge meshes so episode 2000 does not still hold episode 0's trees
```

JSON 3-D vectors are **Y-up** `(X right, Y height, Z forward)`. Blender itself is Z-up. The conversion happens only when writing files.

No pip packages inside Blender. No downloaded `.blend` assets. Meshes are procedural. Run via `./run.sh` so a laptop with AMD+NVIDIA actually uses the NVIDIA GPU for EEVEE.

---

## 12. The other generator: `synth_sim`

Same scientific job, different engine.

It is **not** Blender / Unity / Unreal. It is a **headless rasterizer**: procedural meshes + ModernGL + EGL (offscreen OpenGL). It was built so a GPU box without a display could mint RGB + **depth** + **instance segmentation** + kinematics.

Differences you will care about:

| | `blender_sim` | `synth_sim` |
| --- | --- | --- |
| Engine | Blender EEVEE | Custom ModernGL |
| Default clip | 5 s / 150 frames | 20 s / 600 frames |
| Extra products | heatmap + object JSON | depth `.npy`, instance PNG |
| What models use today | **yes** (`preview.mp4` + 5×5 JSON) | research / richer GT; not the live training cache |

If you open `synth_sim`, the threat gates are slightly different numbers (e.g. critical TTC 2.0 s vs blender 2.5 s) and it writes per-frame JSON instead of one annotations file. Do not mix the two schemas in one loader.

---

## 13. What a training script actually reads

`collision_jepa` and `model_trial_2` do **not** recompute TTC. They:

1. Decode `preview.mp4` to a small square (`128px` or `160px` for the student, `256px` for V-JEPA)
2. Load the 5×5 matrices from `spatial_annotations.json` into `heatmaps.npy` of shape `[150, 5, 5]`
3. Split **by episode**, never by frame (consecutive frames of one 5-second walk are almost the same sample; leaking them into val is cheating)

At time `t` the label for the wearable is **the heatmap 1.0 second later**: frame `t + 30`.

That is the entire contract between dataset and model:

> Given a few recent RGB frames ending at `t`, paint the 5×5 that the simulator will write at `t + 1 s`.

Everything else — V-JEPA, LoRA, JEPA losses — is *how* we try to learn that mapping. The mapping itself is this file.

---

## 14. Where to look in code

| Question | Place |
| --- | --- |
| TTC / CPA / four-class labels | `blender_sim/threat_math.py` |
| 5×5 score + splat | `blender_sim/spatial_threat.py` |
| Gait / seated / crosswalk | `blender_sim/camera_kinematics.py` |
| Scenario mix and aliases | `blender_sim/README.md`, `scenario_compose.py` |
| Balanced pack planner | `blender_sim/gen_dataset.py` |
| Full blender internals | `blender_sim/IMPLEMENTATION.md` |
| Scratch renderer + depth/seg spec | `synth_sim/IMPLEMENTATION.md` |
| Older YOLO + box-growth demo | `impl/collision.py` (different system; reuse only the latch idea) |

---

## 15. Mental model (one paragraph)

We walk a procedural human down a procedural street for five seconds. Because we authored the 3-D world, we can compute “will this body occupy the gait tube soon?” without looking at pixels. That answer becomes a **5×5 heat map** aligned with the forehead image. We force jaywalkers, potholes, and cubes so the pack is not 99% empty sidewalk. The RGB video is the input. The heatmap 1 second ahead is the exam. The next two notes are about students who sit that exam.
