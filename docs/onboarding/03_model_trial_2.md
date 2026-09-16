# model_trial_2 — Collision V-JEPA (the current architecture)

This is the trial you should treat as **current**.

You already know: synthetic 5×5 labels, copy baseline is a cheat-y ceiling, Collision-JEPA proved a frozen V-JEPA teacher can speak heatmap but the student collapsed to the corridor, and a shuffle experiment (`model_trial_1`) fought that prior without putting V-JEPA on the vest.

**Trial 2’s bet:** 600 Blender episodes cannot teach a 2-million-weight CNN a general “things looming at a walker” prior. They **can** teach a small adapter (and a thin LoRA) to read “collision vs empty road” out of a **Giant** video model that already watched the internet. The wearable student then only has to imitate that collision-shaped `Z` from five cheap frames.

No region shuffle. Natural video only. Only the student ships.

---

## 1. The product, one more time

Forehead camera → **five** 160 px frames from the last ~0.7 s → tiny CNN → **5×5 map 1 s ahead** → LEFT / CENTER / RIGHT × SAFE / CAUTION / STOP.

The Giant **never** runs on the vest.

```
Camera (30 fps)
    → 5 frames @ 160 px
    → TinyCNN + loom diffs
    → Z (32 numbers × 25 cells)
    → predictor → Δ
    → clamp(danger_now + Δ)
    → beep
```

---

## 2. How this differs from the other two trials

| | Collision-JEPA | model_trial_1 (shuffle) | **This trial** |
| --- | --- | --- | --- |
| Teacher | Frozen ViT-L + pool | EMA copy of the same TinyCNN | **Giant ViT + adapter, then LoRA last 6 blocks** |
| Teacher’s job | Reconstruct **current** H(t) | (no V-JEPA) | Encode clip@t → Z_t; decode H(t) **and** residual H(t+1s) |
| Shuffle | No | Yes (train only) | **No** — mosaics would wreck V-JEPA tokens |
| Student Z | 16-d, 3 frames @ 128 px | 24-d, 5 frames @ 160 px | **32-d**, 5 frames @ 160 px (must match teacher) |
| JEPA target | Cached teacher Z+ | EMA of the student encoder | Cached teacher **Z(t+τ)** |
| Why it exists | Prove tokens → heatmap | Test shuffle as a motion bias | Give the student a **pretrained motion latent nudged toward threat** |

Corridor bias is fought with **heatmap losses, a false-positive penalty, and oversampling empty/cool-center frames** — not with tile permutation.

---

## 3. V-JEPA Giant and LoRA, friend version

Collision-JEPA froze ViT-L entirely. Tokens were generic video. Collision “red / not red” lives in **our** labels. So trial 2 does two phases:

**Phase 1 — probe.** Freeze the Giant. Train only a small adapter + decoder + predictor. Question: *can we even read a danger map out of these tokens?* If the probe cannot paint threats, LoRA will not save you.

**Phase 2 — polite nudge.** Still do **not** unfreeze 1 billion weights (600 short sim clips would overwrite the prior we wanted). Instead attach **LoRA** to every Linear in the **last 6** encoder blocks.

LoRA in one picture:

```
output = W x  +  (α/r) B A x
```

`W` is the frozen Giant weight. `A` and `B` are tiny matrices (`r = 16`). At the start `B` is all zeros, so the Giant behaves **exactly** as pretrained. Training only learns a thin side path toward “collision.”

If Giant will not fit in 8 GB VRAM, the same code path loads ViT-L (`facebook/vjepa2-vitl-fpc64-256`). Do not drop LoRA as the first OOM fix — drop batch size or switch to Large.

---

## 4. The 5×5 map and the beep (trial-2 numbers)

Same geometry as Collision-JEPA; slightly different cutoffs (tuned on this mix):

```
              LEFT    LEFT   CENTER   RIGHT   RIGHT
 far  ×0.50     ·       ·       ·       ·       ·
      ×0.65     ·       ·       ·       ·       ·
      ×0.80     ·       ·       ·       ·       ·
      ×1.00     ·       ·       ·       ·       ·
near  ×1.00     ·       ·       ·       ·       ·

SAFE < 0.42     CAUTION ≥ 0.42     STOP ≥ 0.70
```

Latch: 3 frames to arm, 5 to release. Code: `cvjepa/warning.py`. Overlay colors match Blender (blue → amber → red). “Red” in conversation = high heatmap / STOP, not a separate class head.

---

## 5. What one training moment looks like

Videos are still 150 frames at 30 fps (5 s). At time `t` we want the map at `t+30`.

**Teacher input** — 8 sharp 256 px frames, every other frame, ending at `t`:

```
[t-14, t-12, t-10, t-8, t-6, t-4, t-2, t]     ≈ 0.47 s of natural video
```

**Student input** — 5 cheaper 160 px frames:

```
[t-20, t-15, t-10, t-5, t]     every 5 frames = 167 ms
```

The dataset also loads:

- `H(t)`, `H(t+15)`, `H(t+30)` — now / half-second / 1 s heatmaps
- After Stage B: teacher notes `Z(t)` and `Z(t+30)`

Valid student `t` is `[20, 119]` (need history and a future). Split **by episode**. Train never sees a val walk.

---

## 6. Teacher forward (the notes factory)

```
clip [B, 8, 3, 256, 256]
        │  normalize, ViT-G (fp16)
        ▼
tokens  (Giant hidden size ≈ 1408, 16×16 spatial grid)
        │  mean over time, 1×1 then 3×3 mix, pool to 5×5, 32-d bottleneck
        │  LayerNorm per cell
        ▼
Z_t  [B, 32, 5, 5]
        ├─ decoder            → Ĥ_now     danger *right now*
        ├─ ResidualPredictor  → Ẑ+ = Z_t + P(Z_t)
        └─ Δ head(Ẑ+ − Z_t)  → Δ in (−1, 1)
                                 Ĥ+ = clamp( Ĥ_now.detach() + Δ )
```

The 3×3 before pooling is deliberate: neighbouring ViT patches should mix before we collapse to LEFT vs CENTER.

**Copy-residual again:** `Ĥ_now.detach()` so the future map is “copy now, then add change.” Decoder uses sigmoid (0–1). Delta uses tanh (−1 to 1) so a cell can **light up or go dark**.

`encode(clip)` returns **only Z_t**. That is what we cache. The teacher’s predictor is **not** cached. The student has its own P. We distill “what the teacher would encode when the future clip arrives” (a **state**), not the teacher’s one-step forecast.

Why the teacher has a predictor at all: if it only decoded `H(t)`, Stage A would never be scored on 1 s anticipation, and wearable eval would be meaningless. The residual head forces `Z_t` to contain enough motion that Δ can move threat.

---

## 7. Student forward (the vest)

The student never loads V-JEPA.

Each of the 5 frames goes through **TinyCNN** independently → 128 channels on 5×5.

Then something very intuitive: **keep “now”, and keep how now differs from each older frame.** Growing blobs in those diffs are looming (things getting bigger because they are coming at you).

Four extra **loom** channels are hand-made from brightness, not learned:

1. Mean luminance now
2. Mean growth (now − oldest)
3. Max growth
4. Fill (max − mean now)

644 channels in (`5×128` features + 4 loom) → 32-d `Z` on the 5×5 grid. Same shape as the teacher.

Then **the teacher’s decoder and Δ head, frozen**. If those stayed trainable, a messy student `Z` could still look like a pretty heatmap. Frozen heads force the intern to write notes in the expert’s handwriting.

Deploy path:

```python
# cvjepa/models/student.py
def predict_future_heatmap(self, frames):
    return self.forward(frames)["h_plus_hat"]
```

No Giant. No cache file. No shuffle.

---

## 8. Three training stages

```
python scripts/00_prepare_data.py --reset-split
python scripts/07_make_test_split.py
python scripts/01_eval_copy_baseline.py

python scripts/02_train_teacher.py      # Stage A
python scripts/03_cache_teacher_z.py    # Stage B
python scripts/04_train_student.py      # Stage C

python scripts/05_eval.py --split both --who both
python scripts/06_overlay.py --split test --limit 8 --latch
```

Caches and checkpoints stay **inside** `model_trial_2/`. Do not write into `collision_jepa/` or `model_trial_1/` caches. Channel counts will not match (`Z=16` vs `24` vs `32`).

### Stage A — train the teacher

Walk each train episode’s 256 px frames, sample `t = 14, 18, …, 119`, micro-batch (1 on 8 GB, 8 on a big GPU).

- Phase 1: adapter + heads, 6 epochs, lr 1e-3
- Phase 2: wrap LoRA, 14 epochs, lr 2e-4, Giant dropout stays off (`eval()`)

Teacher loss (plain language):

| Term | We nag the model about |
| --- | --- |
| L_H | 1 s map should match the label, especially hot cells and **changing** cells |
| L_now | Right now should also look right |
| L_dir | LEFT / CENTER / RIGHT should match |
| L_FP | Empty walking lane must stay dark (anti-false-alarm) |
| L_Δ | The change head should match “future minus now” |

False-positive penalty: on cells with true `H < 0.25`, any predicted redness is squared and punished, **extra hard** in the center columns. This is “don’t cry wolf on empty road.”

### Stage B — save the notes

`teacher_best.pt` (not last) encodes every valid frame of train+val+test into `teacher_z.npz`. Re-run this after any adapter / LoRA change.

### Stage C — train the student

Same heatmap losses, plus JEPA: predicted `Ẑ+` vs cached teacher `Z(t+30)`, stop-grad, **divided by teacher std** so a large Z does not drown L_H (the Collision-JEPA scar). Also a half-way map at 0.5 s so change is learned smoothly.

If you forget Stage B, `L_JEPA = 0` and you silently train heatmap-only.

---

## 9. How we pick a “good” checkpoint (not MSE)

A map that paints the empty corridor red can have okay pixel error and still be a terrible vest.

**Miss** = stayed quiet while the true label was STOP (you walk into something).
**Nuisance** = beeped while the true label was SAFE (you stop trusting it).

CAUTION on a true STOP is **not** a miss. CAUTION on SAFE **is** a nuisance.

Score:

```
score = 1 − 0.5·nuisance − 0.5·miss
eligible  iff  nuisance ≤ 0.20  and  miss ≤ 0.25
```

Any eligible checkpoint beats any ineligible one, then higher score. Wearable metrics use the **latched** severity.

Copy baseline is still run first. It is the privileged ceiling on miss (it has true `H(t)`). RGB models should get **near** that miss without copy’s cheating on static frames, and **must** beat copy on change frames while keeping nuisance ≤ 0.20.

Oversampling during student training pushes the vest off “always beep the corridor”: extra weight on rare episodes, quiet peaks, quiet names (`empty_street`, `safe_walk`, …), and **cool center** (inner 3×3 of the future map is dark).

---

## 10. Design rules that shape every file

If you change code, these are the load-bearing ones:

1. **No shuffle** on any RGB that enters V-JEPA or this student.
2. Teacher `encode` is a function of a clip **ending at e**, not of τ.
3. Student JEPA target is `z_end[t+τ]`, not the teacher predictor’s `Ẑ+`.
4. `teacher.z_channels == student.z_channels == 32` unless you change **both** and recache.
5. `feature_grid == target_grid == 5`.
6. Deploy path is `CollisionStudent.predict_future_heatmap` only.
7. Select on wearable eligibility, not on MSE.
8. Re-cache `teacher_z.npz` after any teacher weight change.
9. Do not fine-tune all ViT blocks on this dataset.
10. GroupNorm in TinyCNN (small batches, mixed empty/hot scenes). Trial 1 needed this because shuffle mixed layouts; it is still the safer default.

---

## 11. Data mix this trial expects

From `blender_sim` packs (see the dataset note):

- all of mixed (~250)
- all of side / peripheral (~50) — false-positive hard negatives
- a stratified slice of a large pack (300 or 500 depending on the YAML you actually run)

Native 5×5 heatmaps. Prepare writes **both** `frames_160.npy` and `frames_256.npy` plus `heatmaps.npy`. Episode keys are `source/episode_dir`.

`07_make_test_split.py` carves test from val and **never touches train**, so you can compare against a run that already started.

---

## 12. File map

```
model_trial_2/
  configs/default.yaml          # live recipe
  cvjepa/
    models/teacher.py lora.py student.py tiny_cnn.py decoder.py
    data/   catalog, video cache, dataset, GPU aug
    losses.py warning.py metrics.py engine.py
    viz/overlay.py
  scripts/00 … 07
  tests/test_shapes.py          # no HF download
  tests/test_wearable.py
  HOW_IT_WORKS.md               # another plain-language walkthrough
  IMPLEMENTATION.md             # tensor-level spec (code wins if they disagree)
```

| Question | File |
| --- | --- |
| V-JEPA load, adapter, copy-residual heads | `cvjepa/models/teacher.py` |
| LoRA wrap | `cvjepa/models/lora.py` |
| TinyCNN, loom, frozen decoder copy | `cvjepa/models/student.py` |
| Losses (focal MSE, FP, JEPA) | `cvjepa/losses.py` |
| LEFT / CENTER / RIGHT + latch | `cvjepa/warning.py` |
| Nuisance / miss / eligible | `cvjepa/metrics.py`, `engine.wearable_better` |
| Stage A / B / C | `scripts/02_train_teacher.py`, `03_cache_teacher_z.py`, `04_train_student.py` |

Friendly walkthrough that mirrors this note: `model_trial_2/HOW_IT_WORKS.md`.
Internals contract: `model_trial_2/IMPLEMENTATION.md`.

---

## 13. How to think about “are we done?”

You are done with a *checkpoint* when it is **eligible** (nuisance ≤ 0.20 and miss ≤ 0.25) and overlays on **held-out episodes** show threats moving, not a stuck corridor blob.

You are not done with the *project* until that holds on walks the net has never seen, including peripheral / empty-center packs (the model that always paints column 2 will look fine on `jaywalker` and fail on `periph_empty`).

Operational loop the team uses:

1. Copy baseline (sanity that labels are predictable at all)
2. Teacher adapter until val high-threat recall is non-zero
3. LoRA: watch **nuisance vs miss**. Giant motion features often drop miss and raise nuisance. Turn **FP weight / spatial centre weights**, not “unfreeze more layers”
4. Cache Z from **best**
5. Student 40 epochs. If L_JEPA dwarfs L_H, lower `jepa.lambda`. If the student ignores Z, raise it slightly
6. Eval val **and** test. Overlay with `--latch`. Report nuisance and miss, not just MSE

---

## 14. Mental model (one paragraph)

A Giant video model already knows motion. We attach a 5×5 adapter and a whisper of LoRA so its notes become *our* collision language. We freeze those notes into a cache. A two-million-weight CNN learns: from five small pictures and looming brightness diffs, write the notes the Giant would have written one second later. A frozen decoder turns notes into a heatmap; a latch turns the heatmap into a buzz. The simulator’s 5×5 is still the exam. The Giant is the tutor. The vest is the intern who has to sit the exam without the tutor in the room.
