# Collision-JEPA — the first learned vest

This is what the team tried after we had labeled walking videos.

You already know the exam: **a few cheap RGB frames → 5×5 danger map 1 second ahead → LEFT / CENTER / RIGHT + SAFE / CAUTION / STOP.**

This folder (`collision_jepa/`) is the first serious attempt to learn that exam with a **teacher / student** setup. It is not the current best architecture (`model_trial_2` is). Read this so you understand *why* trial 2 looks the way it does — most of those choices are scars from this trial.

---

## 1. What we refused to put on the vest

There is already a demo in `impl/` / `run_collision.py`: detect people/cars with **YOLO**, track boxes, fit a line to “the box is getting bigger,” call that a collision.

That demo is a different system. For the learned vest we locked these rules:

- **No object detector.** We do not name “car.”
- **No tracker.** We do not keep IDs.
- **No optical flow network.**
- **No TTC formula at runtime.** TTC lives only in the simulator, to make labels.
- **No huge video transformer on-device.** The vest has to run in a few milliseconds on CPU.

So the wearable model is allowed to see **three small pictures** (128×128) from the last third of a second, and must spit out a 5×5 map.

---

## 2. The annoying baseline: just copy yesterday

Before any neural net, we ask:

> What if the “prediction” of the map in 1 second is simply **the map right now**?

```
Ĥ(t + 1 s)  :=  H(t)
```

That is the **copy baseline**. It is *privileged* — it cheats by using the **true** current heatmap, which the vest will never have. It is still the number to beat, because:

- 5-second sidewalk clips change slowly
- Only about **a quarter** of cells move by more than 0.15 in one second
- The walking corridor is already a bit “warm” even when safe

On the first 101-episode smoke set, copy scored roughly:

| | HT-recall | STOP-F1 | dir-acc |
| --- | --- | --- | --- |
| copy `H(t+1s)=H(t)` | **0.65** | **0.84** | **0.73** |

If your fancy model cannot beat a photocopier on the frames that **actually change** (a person steps in, a car pulls out), it is not helping a blind walker. It may still look “okay” on average MSE because it painted the average corridor blob.

Keep that in your pocket. It explains almost every later design trick.

---

## 3. JEPA, explained like you asked a friend

Most people meet neural nets as **autoencoders**: smash an image to a vector, rebuild the pixels. That wastes capacity on brick texture and sky color. We do not care about pixels. We care about “is this patch about to hit me.”

**JEPA** = **J**oint-**E**mbedding **P**redictive **A**rchitecture (Meta; I-JEPA, then V-JEPA for video).

The slogan:

> Don’t predict pixels. Predict **notes about** the video, in a language the network invented.

Three characters:

1. **Encoder E** — looks at what you *can* see (a clip, or a masked clip) and writes notes `Z`.
2. **Predictor P** — guesses the notes for a part you *cannot* see (the future, or a masked region).
3. **Target encoder Ē** — writes the “correct” notes for that hidden part. Its weights are a slow copy of E (or a frozen giant). You **stop gradients** through Ē so the student cannot cheat by making both sides output garbage that happens to match.

Nobody ever says “rebuild this JPEG.” The loss is “your guessed notes should look like the target notes.”

**V-JEPA 2** is Meta’s 2025 video version. It was trained on huge amounts of real video. Internally it chops a clip into **tubelets** (a patch of space that also covers a couple of frames of time) and predicts missing tubelets in note-space. The public checkpoints we use are HuggingFace models like:

- `facebook/vjepa2-vitl-fpc64-256` — ViT-Large, ~300 million weights, 256 px, pretrained with 64-frame clips
- `facebook/vjepa2-vitg-fpc64-256` — Giant, ~1 billion (trial 2)

`fpc64` means “frames per clip 64” in pretraining. We still feed **8 frames**. The positional embeddings interpolate. We are borrowing a **motion brain**, not running their exact pretraining recipe.

Why this is attractive for a vest:

The Giant already “understands” looming, walking, cars moving. We cannot fit it on a vest. So we use it as a **teacher** offline, and teach a tiny CNN to write **similar notes** from cheap frames.

That teacher/student split **is** our JEPA. The vest never runs V-JEPA.

---

## 4. Two networks, because one is huge and one must be tiny

```
Offline, on a desktop GPU
─────────────────────────
8 sharp 256 px frames  →  frozen V-JEPA-2  →  tokens
                              ↓ small learned adapter
                         Z  (16 numbers on each of 25 cells)
                              ↓ tiny decoder
                         5×5 heatmap

On the vest (and during student training)
─────────────────────────────────────────
3 cheap 128 px frames  →  TinyCNN  →  Z_t
                              ↓ residual predictor
                         Ẑ+   “notes one second from now”
                              ↓ frozen teacher decoder
                         5×5 heatmap 1 s ahead
```

`Z` is **not** the beep. It is a compact “what is going on in this cell?” description that *can* be turned into a beep. If the student writes notes in the teacher’s handwriting, the frozen decoder will read them correctly.

Shapes we locked for this trial:

| Tensor | Size | Meaning |
| --- | --- | --- |
| `H` | 5×5 | heatmap in (0, 1) |
| `Z` | 16×5×5 | collision notes (400 numbers) |
| Student input | 3 × 128×128 RGB | offsets `t−10, t−5, t` (~167 ms apart) |
| Teacher input | 8 × 256×256 RGB | stride 2, ending at time `e`: `[e−14, …, e]` |
| Horizon τ | 30 frames | 1.0 second at 30 fps |

---

## 5. How a 5×5 becomes a buzz (no extra neural net)

This is shared with later trials. Code: `collision_jepa/warning.py`.

1. Multiply each row by a **near-weight** (far rows count less). Here: `[0.4, 0.6, 0.8, 1.0, 1.0]`.
2. LEFT = max of columns 0–1. CENTER = column 2. RIGHT = columns 3–4.
3. The winning direction’s score becomes severity:
   - &lt; 0.45 → **SAFE** (quiet)
   - ≥ 0.45 → **CAUTION** (beep; counts as nuisance if the road was empty)
   - ≥ 0.75 → **STOP** (must beep; silence is a miss)

**HitLatch** (anti-flicker, stolen from the YOLO demo): wait **3** danger frames in a row to turn on, **5** safe frames in a row to turn off. A single noisy frame must not punch you in the ribs.

---

## 6. The pipeline, in the order we actually ran it

All scripts live under `collision_jepa/scripts/`. Dedicated venv; 4 GB RTX 2050 was the original machine.

### Prepare (`00_prepare_data.py`)

Unzip a balanced subset of simulated episodes, split **by episode**, cache:

- `frames_128.npy` — student RGB
- `frames_256.npy` — teacher RGB (created when needed)
- `heatmaps.npy` — `[150, 5, 5]`

Current config aims at ~500 episodes, **80% native 5×5** packs so smeared 3×3 maps do not dominate. Split file: `cache/split.json`. Never split by frame.

### Copy baseline (`01_eval_copy_baseline.py`)

Mandatory. If this is strong, you know the label is autocorrelated. RGB models must earn their keep on **change**.

### Stage 0 (`02_train_stage0.py`) — no V-JEPA at all

A ~93k-parameter TinyCNN sees three frames, concatenates “now” with two feature differences, and paints `Ĥ+` directly.

**Question:** can cheap RGB + diffs beat copy?

**Answer (smoke run):** no. HT-recall 0.52 vs copy 0.65. STOP-F1 0.66 vs 0.84.

That is not a failure of the idea “use RGB.” It is proof that **copying the current map is a high bar**, and a tiny net will collapse to the **average corridor** unless you give it a better motion prior (the teacher) and force it to explain *change*.

### Stage A (`03_train_teacher.py`) — teach the Giant’s cousin to speak heatmap

Backbone: frozen `vjepa2-vitl-fpc64-256`. We **do not** train its 300M weights (4 GB GPU, and 101–500 short sim clips would wreck the prior).

We train a small **adapter**:

1. V-JEPA tokens → reshape into a spatial grid, mix **last** temporal token (approach) with the **mean** (context)
2. 1×1 reduce channels
3. **Spatial attention pool** down to 5×5 (learned queries; better than avg-pool, which would mix a person with leaves in the same coarse cell)
4. Bottleneck to 16-d `Z`
5. Decoder: `Z → H` (sigmoid)
6. Occupancy head: last 4 channels of `Z` → “is this cell a threat?” (helps ignore leaves)

**Stage A objective:** reconstruct the **current** heatmap `H(t)` from a clip **ending at t**. Not the future. We are asking: *can V-JEPA tokens even be read as our 5×5?*

**Result:** yes. Best teacher around epoch 14: HT-recall ~0.90, STOP-F1 ~0.85, MSE ~0.03 on **current** H. `decoder(Z+)` vs future `H+` was also ~0.03. So **Z is a good future target if the student uses it right.** The teacher did its job. The student is where it hurt.

### Stage B (`04_cache_teacher_z.py`)

Run the trained teacher on every frame of every episode. Save `teacher_z.npz`. The Giant does **not** sit in the student training loop (it would not fit, and it would be slow). If you change the adapter, **re-cache**. Stale Z silently poisons Stage C.

### Stage C (`05_train_student.py`) — the wearable

TinyCNN on 3 frames → motion via a small **temporal 3-tap conv** (order-aware: LEFT→CENTER→CENTER is not the same as LEFT→LEFT→LEFT) → `Z_t` → residual predictor `Ẑ+ = Z_t + P(Z_t, motion)`.

Future heatmap is **copy-residual in heatmap space**:

```
Ĥ_now  = decoder(Z_t)                         # danger now
ΔH     = tanh(head(Ẑ+ − Z_t))                # learned change
Ĥ+     = clamp( stopgrad(Ĥ_now) + ΔH × occupancy )
```

`stopgrad` (`.detach()`) is the important trick: the network is not allowed to ignore “now” and paint the future from scratch. Gradients for the 1 s map go through **Δ**. Copying `H(t) ≈ H(t+1s)` is already a good cheat; we want the encoder to explain **change**.

Occupancy gating: leaves (`occ ≈ 0`) cannot raise STOP via ΔH.

The teacher decoder (and occupancy head) are **copied in and frozen**. If they stayed trainable, a messy student `Z` could still look like a pretty heatmap.

Then there is the JEPA term: `Ẑ+` should match the **cached teacher Z at t+30**, stop-grad. That is “write the notes the teacher would write when the future arrives.”

### Eval / overlay / ONNX (`06`, `08`, `07`)

Latency was never the problem. Exported fp32 ONNX was ~**2.7 ms/frame** on CPU. INT8 was *slower* here (depthwise-conv kernels). `student_fp32.onnx` is the deploy artifact.

---

## 7. Losses — heatmap first, in theory

The current student objective (see `collision_jepa/losses.py` `student_loss`) is:

```
L = L(H+) + 0.5 L(H_now)
  + balanced readout (LEFT/CENTER/RIGHT scores)
  + balanced STOP (did we cross the STOP threshold?)
  + balanced occupancy BCE
  + balanced decoder(Ẑ+) vs H+     # so the predictor is not a dead branch
  + balanced L_JEPA                # match teacher Z+
```

“Balanced” means: scale each extra term to a **fraction of L(H+)** with a cap, so a huge JEPA number cannot drown the exam we actually care about.

Heatmap regression itself is clutter-aware:

- Empty / leaf cells (`H ≈ 0`) are cheap
- Cells that are already safe on both sides can be ignored
- False alarms (predict CAUTION on a truly safe cell) are extra-punished
- Cells that **change** between now and +1 s get more weight

This is all later hardening. The **first** student training was simpler and broke. Next section.

---

## 8. What went wrong (the honest postmortem)

Early Stage C (60 epochs, best = epoch 2):

| | HT-recall | STOP-F1 | dir-acc |
| --- | --- | --- | --- |
| copy | 0.65 | 0.84 | 0.73 |
| student | 0.52 | 0.81 | 0.58 |

The student **did not beat copy**. Later epochs overfit (train loss collapsed, val stuck). Mean prediction was a **fatter corridor blob** (center ~0.85 vs GT ~0.70). MSE vs the *dataset-mean* 5×5 was 0.023; MSE vs the *true* future map was 0.102. Translation: it learned the average walking-lane painting, not motion.

Diagnosed causes (do not ignore these; trial 2 is mostly a response):

1. **Copy / mean-map ceiling.** Most cells do not move. Predicting the average 5×5 already matches copy MSE. You must win on the ~24% of cells that change.

2. **Corridor prior collapse.** Center column is warm even when safe. A net that always paints the corridor “a bit red” looks decent and fails as a vest (nuisance).

3. **L_JEPA drowned L_H.** Teacher Z was unnormalized, range about **[-60, +60]**. With `λ = 0.1`, about **93% of the loss** was matching Z, not the heatmap. The intern was copying handwriting and failing the exam.

4. **Horizontal-flip bug (fixed in code now).** Augmentation flipped RGB + heatmap, then attached **unflipped** teacher Z. On half of train batches the heatmap loss and the JEPA loss **contradicted** each other. Current `ClipAugmentor` flips latents with the image; `StudentDataset._spatial_latents` feeds Z through that path; `tests/test_augment_flip.py` guards it.

5. **Best checkpoint epoch 2.** Training never recovered while losses fought.

6. **Teacher was not the villain.** `decoder(Z+)` vs `H+` was already good. Distillation *can* work if Z is scaled, flipped with the image, and heatmap dominates.

Secondary: 128×128 hides small/far hazards; the first student never predicted residual in H-space (that copy-residual formula came later); near-row error was highest, which is exactly where labels change most.

---

## 9. What the code looks like *now* vs the first smoke run

The Python in `collision_jepa/` has been patched toward those diagnoses. If you only read the short `README.md`, you will miss:

| Fix | Where |
| --- | --- |
| Scale JEPA MSE by teacher per-sample std | `losses.jepa_distance` |
| Balance aux terms to a fraction of L(H+) | `losses.balanced_term` / `student_loss` |
| Flip Z with RGB | `data/augment.py`, `StudentDataset` |
| Occupancy channels + gate on ΔH | `models/decoder.py` `OccupancyHead`, `models/student.py` |
| Temporal 3-tap instead of two subtractions | `TemporalConv3` |
| Copy-residual heatmap + frozen decoder | `Student.forward` |
| Prefer native k=5 episodes | `configs/default.yaml` `k5_fraction: 0.8` |
| Attention pool in the teacher | `SpatialAttentionPool` |

The README smoke table is still the **measured first run**. The architecture in the `.py` files is the **repaired** Collision-JEPA. Trial 2 then rebuilt the teacher (Giant + LoRA, Z=32, no 128 px, no shuffle) rather than only patching this repo.

---

## 10. File map

```
collision_jepa/
  configs/default.yaml
  collision_jepa/
    models/   teacher.py student.py tiny_cnn.py decoder.py baseline.py
    data/     dataset.py augment.py video.py splits.py unzip.py
    losses.py warning.py metrics.py engine.py
  scripts/    00_prepare … 08_render_overlay
  tests/      shapes, warning, flip, student loss, clutter, …
```

| I want to… | Go here |
| --- | --- |
| Understand JEPA distillation | `models/student.py`, `losses.jepa_distance` |
| Change beep thresholds | `configs/default.yaml` `warning:` |
| See why Stage 0 exists | `models/baseline.py`, `scripts/02_train_stage0.py` |
| Teacher adapter | `models/teacher.py` |
| The original failure writeup | `collision_jepa/prompt.txt` (handoff notes; code has moved on) |

---

## 11. What we took from this trial

- Frozen V-JEPA **can** decode to our heatmap. The motion prior is real.
- A tiny RGB student **will** collapse to the corridor unless change, empty-road, and loss scale are treated as first-class.
- Copy baseline is the adult in the room. Report HT-recall / STOP-F1 / wearable miss+nuisance, not just MSE.
- Distill **teacher Z(t+τ)**, not a live Giant in the student loop.
- Freeze the decoder so the student cannot hide a bad Z.
- Flip every spatial map together.
- Do not let latent MSE outrun heatmap MSE.

`model_trial_1` (not a separate onboarding doc) then asked: *what if we shuffle the 5×5 image tiles at train time so the net cannot lean on “danger lives in the middle column”?* Shuffle helped a bit on MSE and stopped the fat-blob collapse, but still did not clear the wearable gates. **Trial 2** dropped shuffle (V-JEPA hates mosaics), grew the teacher to Giant + LoRA, and made copy-residual + wearable miss/nuisance the checkpoint rule.

That is the next note.
