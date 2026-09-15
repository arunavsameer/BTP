# model_trial_2 — internals

This file is the complete technical specification of **Collision V-JEPA** as it exists in `/storage/BTP/model_trial_2`. It exists so a person or another model can reconstruct **what every module does, why it exists, how tensors move, and how teacher then student are trained**, without opening the Python. How to **run** the scripts is also here. If this file and the code disagree, the **code wins** — then this file is stale.

[`README.md`](README.md) is the short trial note. Treat **this file + `configs/default.yaml`** as current.

Developed on Arch Linux. Training GPU: NVIDIA GeForce RTX 4060 Laptop (8 GB). Workspace: `/storage/BTP/model_trial_2`. Labels come from [`../blender_sim`](../blender_sim) (see that repo’s `IMPLEMENTATION.md` for how heatmaps are born). The previous frozen-teacher prototype is [`../collision_jepa`](../collision_jepa). The shuffle / EMA-TinyCNN trial is [`../model_trial_1`](../model_trial_1). **Do not write caches or checkpoints into those folders.**

---

## 0. How to use this document

Read §1–§6 first. Those sections are the whole architecture. Everything after them is a zoom-in: one file, one loss term, one YAML key, one metric.

If you are implementing a change:

1. Find the concern in §26 (“Where is X?”).
2. Read the matching file section. That section names the functions, the tensor shapes, the invariants, and the reason they exist.
3. Only then open the `.py`. Comments in the code are short reminders of decisions already explained here.

If you are an AI that has been given only this file:

- Treat the tensor shapes, **no-shuffle** contract, copy-residual formula, wearable gates, teacher→student Z cache, and YAML keys as the **contract**.
- Do not add region shuffle. Do not feed mosaics to V-JEPA 2. Do not run the ViT at deploy time.
- Do not load a trial-1 (`Z=24`, 5-frame TinyCNN with EMA) checkpoint into this student (`Z=32`, frozen teacher decoder). Channel counts will not match.
- Do not train the Giant live in the student loop. Stage B caches `Z`. Stage C reads the cache.
- When two stories conflict (README vs this file vs YAML), **YAML + `teacher.py` / `student.py` win**.

---

## 1. What this program is

A two-stage collision anticipator for a walking pedestrian’s forehead camera.

For each time \(t\), the system predicts a **5×5 threat heatmap 1 second ahead** (\(t+30\) frames at 30 fps). That heatmap is collapsed to:

- a direction: **LEFT / CENTER / RIGHT**
- a severity: **SAFE / CAUTION / STOP**

**Red** (high cell values, especially in near rows) means an obstacle the vest should treat as CAUTION/STOP. **Not red** on empty road means do not beep. The product metric is therefore:

- **miss** \(= P(\text{silent}\mid\text{STOP})\) — obstacle present, vest stayed quiet
- **nuisance** \(= P(\text{beep}\mid\text{SAFE})\) — nothing to avoid, vest buzzed

It is **not** an autonomous-vehicle detector and **not** a pixel autoencoder.

Two networks:

| Role | What it is | When it runs |
|---|---|---|
| **Teacher** | Frozen-then-LoRA **V-JEPA 2 Giant** + spatial adapter + decoder | Offline only (Stage A train, Stage B cache, optional eval) |
| **Student** | ~2M TinyCNN collision encoder + residual predictor + **frozen teacher decoder** | Train Stage C, eval, overlay, vest |

There is **no region shuffle**. V-JEPA 2 was pretrained on natural video. A mosaic would destroy those tokens. Corridor bias is fought with heatmap losses, false-positive penalty, and oversampling — not with tile permutation.

---

## 2. How this differs from the other two trials

| | [`collision_jepa`](../collision_jepa) | [`model_trial_1`](../model_trial_1) | **This trial** |
|---|---|---|---|
| Teacher | Frozen ViT-L + 1×1 pool | EMA copy of TinyCNN | **Giant ViT + adapter, then LoRA last 6 blocks** |
| Teacher target | Reconstruct **current** \(H(t)\) | Predict \(H(t+\tau)\) in student \(Z\) | Encode clip@\(t\) → \(Z_t\); decode \(H(t)\) and residual \(H(t+\tau)\) |
| Shuffle | No | Yes (train only) | **No** |
| Student \(Z\) | 16-d, 3 frames @ 128px | 24-d, 5 frames @ 160px | **32-d**, 5 frames @ 160px (must match teacher) |
| JEPA target | Cached teacher \(Z_+\) | EMA of the same encoder | Cached teacher \(Z(t+\tau)\) |
| Why it exists | Prove V-JEPA tokens can decode to a heatmap | Test shuffle as a motion inductive bias | Give the student a **pretrained motion latent** that has been **nudged toward threat** |

The scientific bet: 600 Blender episodes cannot teach a 1.8M CNN a general looming prior, but they **can** teach a small adapter (and a thin LoRA) to read “collision vs clutter” out of V-JEPA tokens. The student then only has to imitate that \(Z\) from cheap frames.

---

## 3. Design constraints that shape every file

**C1. Tokens of the *task* are heatmap cells, not ViT patches.** The decoder always outputs the same \(5\times 5\) grid the Blender annotator writes. `student.feature_grid` **must** equal `data.target_grid` **and** `teacher` adapter `feature_grid`. There is no bbox head.

**C2. V-JEPA input is natural video.** Never shuffle, never 180° tiles, never photometric mosaic. Teacher clips are unshuffled RGB at 256px.

**C3. Teacher \(Z\) is a *current* collision state.** `encode(clip ending at \(e\))` produces \(Z_e\) that should decode to \(H(e)\). The residual predictor on the teacher turns \(Z_e\) into \(\hat H(e+\tau)\). The student predicts \(Z_{t+\tau}\) (the teacher’s encoding of the *future* clip), not the teacher’s \(\hat Z_+\).

**C4. Copy-residual heatmap.** Predicting \(H(t+\tau)\) from scratch loses to copy \(H(t+\tau)\approx H(t)\). Both teacher and student use

\[
\hat H(t+\tau)=\mathrm{clamp}\bigl(\hat H(t).\mathrm{detach()}+\Delta,\,0,1\bigr),\qquad \Delta=\tanh\bigl(\delta(\hat Z_+-Z_t)\bigr).
\]

The encoder must explain **change**. The decoder is not allowed to ignore \(H(t)\).

**C5. Wearable score is not MSE.** Checkpoint selection uses latched nuisance and miss with hard gates (`nuisance ≤ 0.20`, `miss ≤ 0.25`). A heatmap that lights empty corridor is a failed vest.

**C6. Split by episode, never by frame.** `split.json` keys are `source/episode_dir`. Train never sees a val/test episode.

**C7. Two pixel sizes.** Teacher needs 256px (V-JEPA crop). Student needs 160px. Prepare caches **both** as `frames_256.npy` and `frames_160.npy` under the same episode folder. Heatmaps are shared (`heatmaps.npy`, native 5×5).

**C8. Giant does not sit in the student training loop.** Stage B writes `teacher_z.npz`. Stage C mmapped-reads \(Z_t\) and \(Z_+\). If you “just call V-JEPA in the student forward,” the 8 GB card dies.

**C9. Fine-tune only the last blocks, and only with LoRA.** Full FT of ViT-G on 600 short sim clips overwrites the prior we wanted. Phase 1 = probe. Phase 2 = rank-16 LoRA on last 6 encoder blocks. Base Linear weights stay frozen.

**C10. GroupNorm in TinyCNN, not BatchNorm.** Small batches and mixed empty/hot scenes. (Shuffle is gone, but GroupNorm is still the safer default.)

**C11. Overlay colours match Blender.** Blue → amber → red in `cvjepa/viz/overlay.py`, same as `blender_sim/spatial_overlay.py`. “Red” in conversation = high heatmap / STOP, not a separate class head.

---

## 4. How a process starts

```text
cd /storage/BTP/model_trial_2
source .venv/bin/activate          # create from requirements.txt if needed

python scripts/00_prepare_data.py --config configs/default.yaml --reset-split
python scripts/07_make_test_split.py --config configs/default.yaml
python scripts/01_eval_copy_baseline.py

python scripts/02_train_teacher.py --config configs/default.yaml
python scripts/03_cache_teacher_z.py --config configs/default.yaml
python scripts/04_train_student.py --config configs/default.yaml

python scripts/05_eval.py --config configs/default.yaml --split both --who both
python scripts/06_overlay.py --config configs/default.yaml --split test --limit 8 --latch
```

Every script:

1. Inserts the repo root on `sys.path`.
2. Loads `cvjepa.config.Config` from YAML (default `configs/default.yaml`).
3. Talks to cache / split / checkpoints / HuggingFace cache via **paths in that YAML**.

There is no `run.sh`. There is no Blender in this repo. OpenCV decodes `preview.mp4` at prepare time and draws overlays. `transformers` is required only for teacher load (scripts 02, 03, 05 `--who teacher`).

`--phase adapter|lora|both` on `02_train_teacher.py` (default `both`). `--resume` continues from `checkpoints/teacher_last.pt`.

### 4.1 End-to-end data products

| Path (from `default.yaml`) | Role |
|---|---|
| `/storage/BTP/blender_sim/datasets/{mixed250_s69_n250, side50_s69_n50, pack_…_n1000}` | Source episodes |
| `cache/` | Per-key `frames_160.npy`, `frames_256.npy`, `heatmaps.npy`, later `teacher_z.npz` |
| `cache/catalog.json` | 600 selected episodes |
| `cache/split.json` | `train` / `val` / `test` + family / rare / source maps |
| `hf_cache/` | HuggingFace V-JEPA 2 weights (multi-GB, local to this trial) |
| `checkpoints/teacher_best.pt` | Wearable-selected teacher (adapter + LoRA + heads) |
| `checkpoints/teacher_last.pt` | Last teacher epoch |
| `checkpoints/student_best.pt` / `student_last.pt` | Wearable-selected / last student |
| `overlays/` | GT \| PRED mp4s |
| `logs/` | Optional tee of train stdout |

---

## 5. The teacher forward (tensor diagram)

One training sample is anchored at frame \(t\). Teacher history uses **dense even spacing**, not the student’s five taps.

\[
\text{indices}(t) = \bigl[t-(F-1)S,\; t-(F-2)S,\;\ldots,\; t\bigr]
\]

Default \(F=8\), \(S=2\) → \([t-14, t-12, \ldots, t]\). About **0.47 s** of 256px RGB.

```text
clip [B, 8, 3, 256, 256] in [0, 1]
        │  (x − 0.5) / 0.5   (or HF image_mean / image_std)
        ▼
VJEPA2Backbone  (ViT-G, fp16, no_grad in phase 1; LoRA grads in phase 2)
        │
tokens [B, N, C]     C = hidden_size (Giant ~1408; read from config)
        │  keep first T'·S·S tokens, S = crop/patch = 16
        │  reshape [B, T', 16, 16, C], mean over T'
        ▼
spatial map [B, C, 16, 16]
        │  SpatialAdapter: 1×1 C→128, 3×3, adaptive pool 5×5, 1×1 128→32
        │  LayerNorm over the 32-d vector at each cell
        ▼
Z_t [B, 32, 5, 5]
        ├─ decoder                → Ĥ_now [B, 5, 5] ∈ (0,1)
        ├─ ResidualPredictor P    → Ẑ+ = Z_t + P(Z_t)
        └─ delta_head(Ẑ+ − Z_t)  → Δ ∈ (−1,1)
                                    Ĥ_plus = clamp(Ĥ_now.detach() + Δ, 0, 1)
```

**Targets**

| Tensor | Supervision |
|---|---|
| `Ĥ_now` | \(H(t)\) |
| `Ĥ_plus` | \(H(t+30)\)  ← **the wearable target** |
| `Δ` | \(H(t+30)-H(t)\) clamped to \([-1,1]\) |

`encode(clip)` returns only \(Z_t\). Stage B stores that. The predictor is **not** cached; the student has its own \(P\).

### 5.1 Why the teacher has a predictor at all

If the teacher only decoded \(H(t)\), Stage A would never be scored on 1 s anticipation — and wearable eval would be meaningless. The residual head forces \(Z_t\) to contain **enough motion** that \(\Delta\) can move threat. After Stage A, we still distill **\(Z\) of the future clip**, not \(\hat Z_+\), so the student learns “what the teacher would encode when the future arrives,” which is a state, not a one-step forecast baked into \(P\).

### 5.2 V-JEPA 2 clip length vs 64 pretraining frames

The Giant checkpoint is named `fpc64` (64 frames). This trial feeds **8 frames**. HuggingFace typically interpolates tubelet positional embeddings. `collision_jepa` already used 8×256 on ViT-L. If a future transformers version **requires** 64 frames, pad/repeat in `VJEPA2Backbone.forward` — do not change the student.

Token grid: `spatial = crop_size // patch_size` (256/16 = 16). Temporal tokens \(T' \approx N / 16^2\). Extra CLS/register tokens, if any, are dropped by slicing `usable = T' · 16 · 16`.

---

## 6. The student forward (tensor diagram)

```text
frames [B, 5, 3, 160, 160] in [0, 1]
  offsets {t−20, t−15, t−10, t−5, t}     (every 5 frames = 167 ms)
        │  TinyCNN per frame, independently
        ▼
feats [B, 5, 128, 5, 5]
        │  stack: F(t), F(t)−F(t−5), F(t)−F(t−10), F(t)−F(t−15), F(t)−F(t−20)
        │  + 4 loom channels (luminance mean now, mean growth, max growth, fill)
        ▼
CollisionEncoder 644 → 128 → 64 → 32, LayerNorm per cell
        ▼
Z_t [B, 32, 5, 5]
        ├─ frozen teacher decoder → Ĥ_now
        ├─ student P              → Ẑ+
        └─ frozen teacher Δ       → Ĥ_plus = clamp(Ĥ_now.detach() + Δ, 0, 1)
                                      Ĥ_mid  = clamp(Ĥ_now.detach() + 0.5 Δ, 0, 1)
```

**JEPA (Stage C only)**

\[
L_{\mathrm{JEPA}} = \mathrm{MSE}\Bigl(\frac{\hat Z_+}{\sigma(Z_+)}, \frac{Z_+}{\sigma(Z_+)}\Bigr) + \bigl(1-\cos(\hat Z_+, Z_+)\bigr)
\]

\(Z_+\) is `teacher_z.npz` at index \(t+30\), **stop-grad**. \(\sigma\) is the teacher tensor’s per-sample std (collision_jepa trick: teacher \(Z\) is not LayerNorm-unit like trial 1 EMA \(Z\)).

Deploy: `predict_future_heatmap(frames)` → `Ĥ_plus`. No V-JEPA, no cache, no shuffle.

---

## 7. Layout

```text
model_trial_2/
  configs/default.yaml
  cvjepa/                 # importable package
    config.py
    warning.py metrics.py losses.py engine.py
    data/                 # catalog, split, video cache, datasets, GPU aug
    models/               # teacher, lora, student, tiny_cnn, decoder
    viz/overlay.py
  scripts/00…07
  tests/test_shapes.py    # no HF download
  tests/test_wearable.py
  requirements.txt
```

Python env is **not** required to live inside this folder; `model_trial_1/.venv` can run the unit tests. Teacher training needs `transformers` (listed in `requirements.txt`).

---

## 8. `cvjepa/config.py`

`Config.load(path)` reads YAML. `cfg.get("teacher.clip_frames")` walks dotted keys. `resolve_device("auto")` → `cuda` if `torch.cuda.is_available()` else `cpu`. No interpolation of relative paths — YAML already has absolute `/storage/BTP/...` paths.

---

## 9. `cvjepa/models/teacher.py`

### 9.1 `SpatialAdapter`

`forward(tokens, spatial)`:

1. `t_prime = N // (spatial²)`, `usable = t_prime * spatial²`
2. reshape `[B, T', S, S, C]`, mean \(T'\) → `[B, S, S, C]` → `[B, C, S, S]`
3. `reduce`: Conv 1×1 \(C\to 128\), GELU, Conv 3×3 128, GELU
4. `adaptive_avg_pool2d` → 5×5
5. `bottleneck`: 1×1 128→128 GELU, 1×1 128→32
6. LayerNorm on the 32-d vector at each of the 25 cells

3×3 after the 1×1 is deliberate: neighbouring ViT patches should mix before the collapse to 5×5 (LEFT vs CENTER).

### 9.2 `ResidualPredictor` (teacher)

Same recipe as trial 1: \(Z \to 3Z\) (3×3) → dilated 3×3 (padding 2, dilation 2) → 1×1 back to \(Z\), residual add. Dilation is how threat can jump a cell in 1 s.

### 9.3 `VJEPA2Backbone`

- Lazy `transformers.AutoModel.from_pretrained(hf_model_id, cache_dir=, torch_dtype=, attn_implementation="sdpa")`. SDPA failure falls back without it.
- All parameters `requires_grad_(False)`, `eval()`.
- `_grad_enabled`: False in phase 1 (`torch.no_grad` around the ViT). True after `enable_lora` (LoRA A/B get grad; frozen Linear still has `requires_grad=False`).
- Normalize clip with registered `_mean/_std` (config `image_mean/std` or 0.5).
- Forward tries `pixel_values_videos=x`, then `get_vision_features`, then `model(x)`. Returns `last_hidden_state.float()`.

`enable_lora(last_n, rank, alpha)` calls `apply_lora_to_last_blocks` then `gradient_checkpointing_enable()` if present.

### 9.4 `CollisionTeacher`

Members: `backbone`, `adapter`, `predictor`, `decoder`, `delta_head`.

`trainable_parameters(include_lora=False)` = adapter + P + decoder + Δ, plus LoRA A/B when requested. **Never** unfreezes base ViT weights through this API.

Optional `backbone=` injection is for tests (`FakeBackbone`); production always loads HF.

---

## 10. `cvjepa/models/lora.py`

No `peft` dependency.

`LoRALinear(linear, rank=16, alpha=16)`:

\[
y = W x + \frac{\alpha}{r} B A x,\qquad A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\; B\in\mathbb{R}^{d_{\mathrm{out}}\times r}
\]

\(W\) frozen. \(A\) Kaiming, \(B\) zeros (start as identity of the frozen layer).

`apply_lora_to_last_blocks` finds encoder blocks by probing `encoder.layer` / `encoder.layers` / `encoder.blocks` (and `model.model` wrappers). Replaces every `nn.Linear` in the last `last_n` blocks. If that search fails, Stage A phase 2 raises `RuntimeError` — fix the walker, do not silently skip LoRA.

---

## 11. `cvjepa/models/decoder.py`

**HeatmapDecoder:** 1×1 \(Z\to 48\) ReLU, 3×3 48 ReLU, 1×1 → 1, sigmoid, squeeze. Output `[B,5,5]` in (0,1).

**HeatmapDelta:** same widths, **tanh** so \(\Delta\in(-1,1)\).

Hidden 48 (trial 1 used 32) because \(Z\) is 32-d not 16/24. After Stage A, **both modules are copied into the student and frozen.** If the student decoder stayed trainable, it could hide a bad student \(Z\).

---

## 12. `cvjepa/models/tiny_cnn.py`

Per-frame 160×160 RGB → 128 channels on 5×5.

- Stem: 3×3 stride 2, width 32, GroupNorm+ReLU6
- DW-separable: 32→64 s2, 64→64 s2, 64→128 s2, 128→128 s1
- Bilinear to 10×10, 2×2 avg-pool → 5×5

`feat_channels = width * 4 = 128`.

---

## 13. `cvjepa/models/student.py`

### 13.1 `pixel_loom`

On the 5-frame clip, luminance \(0.299R+0.587G+0.114B\), adaptive avg/max pool to 5×5. Four channels: mean now, mean(now)−mean(first), max(now)−max(first), max(now)−mean(now). First frame is \(t-20\), last is \(t\). Not learned.

### 13.2 `CollisionEncoder`

Input channels = `128 * n_frames + 4` = **644**. 3×3 644→128 GroupNorm ReLU, 3×3 128→64 ReLU, 1×1 64→32, LayerNorm per cell.

### 13.3 `ContextEncoder`

TinyCNN on `B*5` frames, reshape, concat now + four diffs + loom, CollisionEncoder.

### 13.4 `CollisionStudent`

`load_frozen_heads(teacher_state)` strips `decoder.*` and `delta_head.*` from a Stage A `state_dict`. `trainable_parameters()` skips frozen heads.

`n_frames` must equal `len(student.frame_offsets)` (the train script sets this). `z_channels` must equal `teacher.z_channels`.

---

## 14. `cvjepa/data/`

### 14.1 `__init__.py` — catalog and split

Copied from trial 1. Episode keys `source/episode_dir` because names collide across packs.

`build_catalog(sources)`: each source `{name, path, take}` with `take: all` or an int (stratified by family). Family keywords: vehicle / pedestrian / static / empty / other. `has_rare` from `episode.json` label histogram NEAR_MISS or CRITICAL_THREAT.

`make_split`: 80/20 by family (`val_fraction: 0.2`). `test_fraction: 0` at prepare time; script 07 carves test from val.

### 14.2 `video.py`

`cache_episode(..., sizes=[160, 256], target_grid=5)` writes:

- `heatmaps.npy` `[150, 5, 5]` float32 (nearest upsample if a pack is still 3×3)
- `frames_{size}.npy` `[150, size, size, 3]` uint8 RGB for each size

Decode `preview.mp4` with OpenCV INTER_AREA.

### 14.3 `dataset.py`

`teacher_clip_indices(t_end, F, S)` → `[t_end-(F-1)S, …, t_end]`.

`StudentDataset`: index \(t \in [20, 119]\) (because min offset = −20, \(\tau=30\), \(N=150\)). Train stride 2, val stride 3.

`__getitem__` returns uint8 NHWC `frames` plus `h_now`, `h_future`, `h_mid` (\(t+15\)), and if `teacher_z.npz` exists:

- `z_t` = `z_end[t]`
- `z_plus` = `z_end[t+30]`

`sample_weights` (train sampler):

\[
w = (1 + \mathrm{hot}\,H_{\mathrm{fut}}^{\max} + \mathrm{change}\,|H_{\mathrm{fut}}-H_{\mathrm{now}}|_{\max})
\]

then multiply rare episode, quiet peak (`max H_fut < 0.25`), quiet name markers, cool-center (inner 3×3 mean of \(H_{\mathrm{fut}} < 0.28\)). Defaults match YAML (`hot_mult` 1.5, `quiet_mult` 4.0, `center_cool_mult` 5.5, …).

Teacher training does **not** use this Dataset; it iterates whole episodes (see §19) to keep 256×8 clips off the DataLoader.

### 14.4 `gpu_preprocess.py`

Student only. Workers copy uint8. GPU: `/255`, brightness/contrast/noise/blur, 50% horizontal flip of frames **and** heatmaps **and** teacher \(Z\) (flip last spatial dim). Teacher Stage A currently has **no** photometric aug (ViT-G + micro-batch 1 already slow; add later if the adapter overfits Blender lighting).

---

## 15. `cvjepa/losses.py`

All heatmap losses are on **unshuffled** 5×5 maps.

### 15.1 `spatial_heatmap_weights(grid=5, …)`

Chebyshev rings around the centre. Raw: centre 2.2, inner ring 1.5, edge 1.0, corner 0.65. Divide by mean so \(\mathbb{E}[w]=1\). After normalize, centre ≈ 1.91, ring ≈ 1.30, edge ≈ 0.87, corner ≈ 0.56.

### 15.2 `weighted_focal_mse`

\[
\mathrm{SE}=(\hat H-H)^2,\quad
w = 1 + f\,H^{\gamma} + c\,|\Delta H| + f_n\,\mathrm{ReLU}(H-\hat H)\,[H>0.3]
\]

then \(w \leftarrow w \odot w_{\mathrm{spatial}}\). Mean over cells. Defaults \(f=1, \gamma=2, c=2.5, f_n=0.6\).

### 15.3 `false_positive_penalty`

On cells with \(H < 0.25\), penalize \(\hat H^2\). Columns `[1,2,3]` ×3.5. This is “don’t paint the empty walking lane red.”

### 15.4 `warning_alignment_loss`

LEFT/CENTER/RIGHT scores = max of row-weighted cells in each column group. MSE to GT scores + peak under-prediction on real threats + 0.25 CE on argmax direction + over-prediction on quiet frames. Same as trial 1.

### 15.5 `jepa_distance`

See §6. Student Stage C only. Teacher Stage A has **no** JEPA term (nothing to distill from).

### 15.6 Teacher total loss (`scripts/02_train_teacher.py`)

\[
L = L_H + 0.35\,L_{\mathrm{now}} + 0.30\,L_{\mathrm{dir}} + 0.45\,L_{\mathrm{FP}} + 0.50\,L_{\Delta}
\]

\(L_H\) is focal MSE of \(\hat H_+\) vs \(H(t+\tau)\). No mid-horizon on the teacher (no `h_mid_hat` head).

### 15.7 Student total loss (`scripts/04_train_student.py`)

\[
L = L_H + 0.35\,L_{\mathrm{now}} + 0.20\,L_{\mathrm{mid}} + 0.30\,L_{\mathrm{dir}} + 0.45\,L_{\mathrm{FP}} + 0.25\,L_{\mathrm{JEPA}} + 0.50\,L_{\Delta}
\]

If `z_plus` is missing from the batch (forgot Stage B), \(L_{\mathrm{JEPA}}=0\) and training degrades to heatmap-only.

---

## 16. `cvjepa/warning.py` — red vs not-red

Row weights (far → near): `[0.5, 0.65, 0.8, 1.0, 1.0]`.

Columns: LEFT `[0,1]`, CENTER `[2]`, RIGHT `[3,4]`.

Direction score = max of (heatmap ⊙ row weights) in that group. Severity from the **winning** direction:

- `< 0.42` SAFE (not red / no beep)
- `≥ 0.42` CAUTION (amber; counts as a **beep** for nuisance)
- `≥ 0.70` STOP (red)

`HitLatch`: 3 consecutive danger frames to arm, 5 consecutive SAFE to release. Wearable metrics use the **latched** severity (`selection.use_latch: true`).

**Miss** uses STOP only (silent while GT is STOP). **Nuisance** uses any beep (CAUTION or STOP) while GT is SAFE. CAUTION on a true STOP is **not** a miss.

---

## 17. `cvjepa/metrics.py`

`MetricAccumulator` (per heatmap pair): MSE, high-threat P/R/F1 at 0.5, direction accuracy when GT ≥ CAUTION, STOP P/R/F1.

`WearableAccumulator`: episode-reset latch, then

\[
\mathrm{nuisance}=\frac{\#\{\text{beep}\wedge\text{SAFE}\}}{\#\text{SAFE}},\quad
\mathrm{miss}=\frac{\#\{\text{silent}\wedge\text{STOP}\}}{\#\text{STOP}},
\]

\[
\mathrm{score}=1-0.5\,\mathrm{nuisance}-0.5\,\mathrm{miss},
\]

\(\mathrm{eligible} \iff \mathrm{nuisance}\le 0.20 \land \mathrm{miss}\le 0.25\).

`wearable_better(cur, best)`: any eligible beats any ineligible; then higher score.

---

## 18. `cvjepa/engine.py`

- `evaluate_future_heatmap`: student DataLoader, `Ĥ_+` vs `h_future`.
- `evaluate_wearable(..., teacher=False)`: load `frames_160.npy`, `predict_episode_heatmaps` with student offsets, score \(\hat H(t)\) against \(H(t+\tau)\).
- `evaluate_wearable(..., teacher=True)`: load `frames_256.npy`, `predict_teacher_episode` builds 8-frame clips ending at each valid \(t\), micro-batch 1. **Slow** (ViT-G every frame of every val episode). That is the number we select on.
- `copy_baseline=True`: pred \(= H(t)\) (privileged). Script 01.

Teacher heatmap-only eval during Stage A (`heatmap_eval` in script 02) uses **sample_stride 4** for speed. Wearable eval is full-rate.

---

## 19. `scripts/02_train_teacher.py` — Stage A loop

For each train episode, mmap `frames_256.npy`, walk \(t = 14, 18, \ldots, 119\) (`t_lo=(8-1)*2`, `sample_stride=4`), stack clips in RAM, shuffle sample order, micro-batch 1.

Phase 1 (`adapter_epochs: 6`, `adapter_lr: 1e-3`): `set_backbone_grad(False)`, AdamW on adapter heads, AMP.

Phase 2 (`lora_epochs: 14`, `lora_lr: 2e-4`): wrap LoRA, `set_backbone_grad(True)`, AdamW on adapter heads **+ LoRA**. `backbone.model.eval()` so dropout stays off.

Each epoch: print loss every 20 episodes; val heatmap (strided) + val wearable (full); save `teacher_last.pt`; if `wearable_better`, save `teacher_best.pt`. Checkpoint dict: `model`, `cfg`, `epoch`, `phase`, `lora` (bool), `val`.

Resume: if `lora: true` in ckpt, wrap LoRA **before** `load_state_dict`.

`--phase adapter` stops before LoRA. Use that to debug the probe without touching the ViT.

VRAM: micro_batch 1, fp16 weights, empty_cache after eval. If Giant still OOM, YAML `hf_model_id: facebook/vjepa2-vitl-fpc64-256` (300M, same adapter code). Do not drop LoRA as the first OOM fix — drop clip_frames or switch to ViT-L.

---

## 20. `scripts/03_cache_teacher_z.py` — Stage B

Loads **`teacher_best.pt`**. For every train+val+test episode, for every \(e \in [14, 149]\), `encode(clip ending at e)` → `z_end[e]`. Invalid prefix stays zeros; `mask` marks valid rows. `np.savez_compressed(.../teacher_z.npz)`.

Student must not use \(t < 20\) anyway (history offset). \(Z_+\) at \(t+30\) needs `e=t+30 ≥ 14`, which is true.

If you change LoRA / adapter after a cache, **re-run Stage B**. Stale \(Z\) silently poisons Stage C.

---

## 21. `scripts/04_train_student.py` — Stage C

WeightedRandomSampler, batch 32, 2 workers, GPU aug, AMP, cosine LR to 5% of 1e-3, 40 epochs, grad clip 1.0.

Loads frozen decoder/Δ from `teacher_best.pt`. Trains context + predictor.

Val: heatmap loader + wearable on 160px. Selection = wearable_better → `student_best.pt`.

---

## 22. Other scripts

| Script | Role |
|---|---|
| `00_prepare_data.py` | Catalog 600, split 80/20, cache 160+256+heatmaps, `--reset-split` |
| `07_make_test_split.py` | Carve ~half of val into test, stratified family×source×rare; train untouched |
| `01_eval_copy_baseline.py` | Wearable of \(H(t)\) as \(\hat H(t+\tau)\) |
| `05_eval.py` | `--who student\|teacher\|both` `--split val\|test\|both` |
| `06_overlay.py` | Side-by-side GT \| student pred, optional `--latch` |

Prepare workers default 6. `--limit` for smoke catalogs.

---

## 23. `configs/default.yaml` — current recipe

**Data.** mixed 250 + side 50 + pack 300. Native 5×5 (`heatmap_upsample: nearest` is identity). `n_frames: 150`, `fps: 30`.

**Teacher.** Giant, 256px, 8 frames stride 2, \(Z=32\), pool 128, fp16, micro_batch 1, adapter 6 + LoRA 14, rank 16 alpha 16 last 6 blocks, sample_stride 4.

**Student.** 160px, width 32, grid 5, \(Z=32\), loom on, copy_residual on, offsets `[-20,-15,-10,-5,0]`.

**Horizon.** \(\tau=30\), mid 15.

**Loss / train / warning / selection.** See §15–§17. Same wearable gates as trial 1.

---

## 24. Overlay and “red”

`threat_to_rgb`: 0–0.55 blue→amber, 0.55–1 amber→red. A cell that is **STOP** (direction score ≥ 0.70 on a near-weighted peak) is the vest-red the user cares about. Overlay draws **every** cell; wearable uses the max in a column group. Do not equate “one pale-red far cell” with a STOP miss.

---

## 25. Tests

`tests/test_shapes.py` — no HF download. Fake backbone (48-d, 4×4 spatial), adapter, teacher heads, student 5-frame, LoRALinear, decoder, spatial weights.

`tests/test_wearable.py` — latch, gates, spatial-weight shape/mean, FP centre.

```bash
python tests/test_shapes.py && python tests/test_wearable.py
```

---

## 26. Where is X?

| Concern | File / key |
|---|---|
| V-JEPA load / normalize / no_grad | `cvjepa/models/teacher.py` `VJEPA2Backbone` |
| Token → 5×5 \(Z\) | `SpatialAdapter` |
| LoRA wrap | `cvjepa/models/lora.py` |
| Copy-residual formula | `CollisionTeacher._heads`, `CollisionStudent._future_from_now` |
| Frozen decoder copy | `CollisionStudent.load_frozen_heads` |
| Teacher clip indices | `teacher_clip_indices` in `data/dataset.py` |
| Student 5-frame offsets | `student.frame_offsets` |
| Stage A loop / phases | `scripts/02_train_teacher.py` |
| Z cache | `scripts/03_cache_teacher_z.py` |
| JEPA loss | `losses.jepa_distance` |
| Spatial loss map | `losses.spatial_heatmap_weights` |
| Nuisance / miss | `metrics.WearableAccumulator` |
| Checkpoint rule | `engine.wearable_better` |
| Beep thresholds | `warning.caution_threshold` / `stop_threshold` |
| Cool-center sampling | `StudentDataset.sample_weights` |
| 160 vs 256 cache | `data/video.py` `cache_episode(..., sizes=)` |
| Split keys | `data/__init__.py` `build_catalog` / `make_split` |
| OOM fallback | `teacher.hf_model_id` → `facebook/vjepa2-vitl-fpc64-256` |
| Overlay colours | `cvjepa/viz/overlay.py` |

---

## 27. Invariants (do not violate)

1. **No shuffle** on any RGB that enters V-JEPA or the student in this trial.
2. Teacher `encode` is a function of a clip **ending at \(e\)**, not of \(\tau\).
3. Student JEPA target is `z_end[t+τ]`, not the teacher predictor’s \(\hat Z_+\).
4. `teacher.z_channels == student.z_channels == 32` unless you change **both** and recache Z.
5. `feature_grid == target_grid == 5`.
6. Deploy path is `CollisionStudent.predict_future_heatmap` only.
7. Select on wearable eligibility, not on MSE.
8. Re-cache `teacher_z.npz` after any teacher weight change.
9. Do not write into `model_trial_1/cache_grid5` or `collision_jepa/` caches.
10. Do not fine-tune all ViT blocks on this dataset.

---

## 28. Training plan (operational)

1. Prepare 600-ep cache (160+256) and carve test from val.
2. Run copy baseline. That is the privileged ceiling on miss (it has true \(H(t)\)). RGB models should get **near** that miss without copy’s cheating on static frames, and **must** beat copy on change frames while keeping nuisance ≤ 0.20.
3. Stage A adapter: wait until val HT-recall / STOP-F1 are non-zero. If the probe cannot paint threats, LoRA will not save it — check labels, clip indices, decoder LR.
4. Stage A LoRA: watch **nuisance vs miss**. Giant motion features often drop miss and raise nuisance. FP weight and spatial centre weights are the knobs, not “unfreeze more layers.”
5. Stop Stage A when an **eligible** checkpoint appears, or when the ineligible Pareto no longer moves. Save that as `teacher_best.pt`.
6. Stage B from **best**, not last.
7. Stage C 40 epochs. If \(L_{\mathrm{JEPA}}\) dwarfs \(L_H\), lower `jepa.lambda` (0.25 → 0.10). If the student ignores \(Z_+\), raise it slightly.
8. Eval val and test (`--who both`). Overlay `--latch` on test. Report nuisance and miss, not just MSE.

Expected wall time on 4060 (order of magnitude): prepare ~3 min; teacher adapter slower per step than trial 1 (ViT-G); LoRA slower still; wearable val is the expensive part of each teacher epoch; student ~1–2 min/epoch once Z is cached.
