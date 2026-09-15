# model_trial_1 — internals

This file is the complete technical specification of **RS-JEPA** (Region-Shuffle Collision JEPA) as it exists in `/storage/BTP/model_trial_1`. It exists so a person or another model can reconstruct **what every module does, why it exists, how tensors move, and how the current student is trained**, without opening the Python. How to **run** the scripts is also here; if this file and the code disagree, the **code wins** — then this file is stale.

[`README.md`](README.md) is a short trial note and **lags the code** (it still talks about 3 frames, bilinear 3×3→5×5, and an older 800-episode mix). Treat **this file + `configs/default.yaml`** as current.

Developed on Arch Linux. Training GPU: NVIDIA GeForce RTX 4060 Laptop (8 GB). Python env: `/storage/BTP/model_trial_1/.venv` (PyTorch 2.11 + CUDA 12.8). Workspace: `/storage/BTP/model_trial_1`. Labels come from [`../blender_sim`](../blender_sim) (see that repo’s `IMPLEMENTATION.md` for how heatmaps are born).

---

## 0. How to use this document

Read §1–§5 first. Those five sections are the whole architecture. Everything after them is a zoom-in: one file, one loss term, one YAML key, one metric.

If you are implementing a change:

1. Find the concern in §24 (“Where is X?”).
2. Read the matching file section. That section names the functions, the tensor shapes, the invariants, and the reason they exist.
3. Only then open the `.py`. Comments in the code are short reminders of decisions already explained here.

If you are an AI that has been given only this file:

- Treat the tensor shapes, shuffle/deshuffle contract, copy-residual formula, wearable gates, and YAML keys as the **contract**.
- Do not invent a ViT, a pixel reconstructor, a second heatmap grid, or a shuffle at deploy time. None of those exist.
- Do not load a 3×3 / 4-frame checkpoint into the current 5×5 / 5-frame student. Channel counts will not match.
- When two stories conflict (README vs this file vs YAML), **YAML + `student.py` win**.

---

## 1. What this program is

A small wearable student that, for each time \(t\), takes **five RGB frames** of a forehead camera and predicts a **5×5 threat heatmap 1 second ahead** (\(t+30\) frames at 30 fps). That heatmap is then collapsed to:

- a direction: **LEFT / CENTER / RIGHT**
- a severity: **SAFE / CAUTION / STOP**

It is **not** an autonomous-vehicle detector. The ego is a walking pedestrian. The buzz pattern of a tactile vest is the product.

It is a **JEPA**, not a pixel autoencoder:

| V-JEPA-2 (Meta, 2025) | This trial |
|---|---|
| ViT-L tubelet tokens | 5×5 collision cells, aligned with labels |
| Random masking | Region shuffle + 180° on opposite cells |
| EMA encoder \(\bar{E}\) | EMA copy of the TinyCNN collision encoder |
| Predict in \(Z\), not pixels | Residual predictor \(\hat{Z}_{t+\tau} = Z_t + P(Z_t)\) |
| Frozen backbone + probe | Tiny decoder \(Z \to H\) trained jointly |

The frozen 300M `facebook/vjepa2-vitl-fpc64-256` teacher is **not** downloaded. Distilling it is a later stage. Deployment never runs a ViT — only this student (~1.79M parameters with the current 5-frame / \(Z=24\) config).

**Training corruption (shuffle) is the scientific hypothesis.** Cut each 160×160 frame into a 5×5 mosaic, permute the 25 cells with the **same permutation across time and across the future clip**, and 180°-rotate a cell when it lands on the opposite side of the image. Encode that mosaic, **unpermute \(Z\)**, then predict in the **real** layout. If the encoder has learned looming / closing inside a cell rather than “cars live in the walking corridor,” deshuffled predictions stay correct.

**Eval, overlay, ONNX, and the vest never shuffle.**

---

## 2. Design constraints that shape every file

These are the reasons the code looks the way it does. Almost every “why” in later sections traces back here.

**C1. Tokens are heatmap cells, not ViT patches.** The decoder outputs the same \(k \times k\) grid the Blender annotator writes. There is no extra spatial head, no bbox, no segmentation. `student.feature_grid` **must** equal `data.target_grid`.

**C2. Shuffle the RGB, not the loss target.** The encoder sees a scrambled mosaic so it cannot lean on a centre-corridor prior. The predictor and decoder see **unpermuted** \(Z\) so threat can move across neighbouring cells (a jaywalker walking from RIGHT into CENTER). Scoring is always against the original heatmap.

**C3. Same permutation on now-clip and future-clip.** Temporal differences and loom stay **inside a cell**. If you shuffled the two clips independently, motion would be garbage. `apply_shuffle_pair` concatenates on the time axis and shuffles once.

**C4. GroupNorm, never BatchNorm, in the CNN.** A batch mixes natural layouts and shuffled mosaics. BatchNorm would mix those statistics. GroupNorm is layout-agnostic.

**C5. Copy-residual is the current heatmap path.** Predicting \(H(t+\tau)\) from scratch fought a strong copy baseline \(H(t+\tau)\approx H(t)\). The v2 student predicts \(\Delta\) and adds it to a **detached** current heatmap:

\[
\hat{H}(t+\tau)=\mathrm{clamp}\bigl(\hat{H}(t).\mathrm{detach()}+\Delta,\,0,1\bigr)
\]

The encoder still has to explain motion; the decoder is no longer allowed to ignore \(H(t)\).

**C6. Wearable score is not MSE.** Checkpoint selection uses latched **nuisance** \(P(\text{beep}\mid\text{SAFE})\) and **miss** \(P(\text{silent}\mid\text{STOP})\), with hard gates. A pretty heatmap that beeps on empty road is a failed vest.

**C7. Split by episode, never by frame.** Consecutive frames of one clip are almost the same sample. `split.json` keys are `source/episode_dir`. Train never sees a val/test episode.

**C8. Cache is uint8 RGB + native \(k\times k\) heatmaps.** Workers only `mmap` and slice. `/255`, photometric aug, and shuffle run on GPU. That is why training actually uses the 4060 instead of starving it.

**C9. History is uniformly sampled in time.** Current offsets are \(t-20, t-15, t-10, t-5, t\) (every 5 frames = 167 ms at 30 fps). The earliest frame still sets \(t_{\mathrm{lo}}=20\). Adding \(t-15\) does **not** shrink the valid window; it only adds a motion channel.

**C10. 3×3 experiments are frozen.** `cache_grid3/`, `checkpoints_grid3*`, and `configs/grid3*.yaml` are a previous trial. Do not overwrite them. The live recipe is 5×5 native labels under `cache_grid5/` / `checkpoints_grid5/`.

---

## 3. How a process starts

```text
cd /storage/BTP/model_trial_1
source .venv/bin/activate

python scripts/00_prepare_data.py --config configs/default.yaml --reset-split
python scripts/05_make_test_split.py --config configs/default.yaml
python scripts/02_train.py          --config configs/default.yaml
python scripts/03_eval.py           --config configs/default.yaml --split both \
                                    --ckpt checkpoints_grid5/student_best.pt
python scripts/06_overlay.py        --config configs/default.yaml --split test --latch
python scripts/04_export_onnx.py    --config configs/default.yaml
```

Every script:

1. Inserts the repo root on `sys.path`.
2. Loads `rs_jepa.config.Config` from YAML (default `configs/default.yaml`).
3. Talks to cache / split / checkpoints via paths **in that YAML**, not via cwd-relative guesses.

There is no `run.sh`. There is no Blender in this repo. OpenCV is used only to decode `preview.mp4` at prepare time and to draw overlays.

### 3.1 End-to-end data products

| Path (from `default.yaml`) | Role |
|---|---|
| `/storage/BTP/blender_sim/datasets/{mixed250_s69_n250, side50_s69_n50, pack_…_n1000}` | Source episodes (RGB `preview.mp4` + `spatial_annotations.json` + `episode.json`) |
| `cache_grid5/` | 160×160 uint8 frames + 5×5 heatmaps, one folder per catalog key |
| `cache_grid5/catalog.json` | All 600 selected episodes |
| `cache_grid5/split.json` | `train` / `val` / `test` lists + family / rare / source maps |
| `checkpoints_grid5/student_best.pt` | Wearable-selected weights (also copied to `student.pt`) |
| `checkpoints_grid5/student_last.pt` | Every epoch |
| `logs_grid5/train.log` | Epoch lines |
| `overlays_grid5/` | GT \| PRED mp4s |
| `export_grid5/student_fp32.onnx` | Deploy graph (no shuffle) |

### 3.2 Current dataset mix (`configs/default.yaml`)

| Source name | Folder | Take | Native grid |
|---|---|---|---|
| `mixed` | `blender_sim/datasets/mixed250_s69_n250` | all 250 | **5×5** |
| `side` | `blender_sim/datasets/side50_s69_n50` | all 50 | **5×5** |
| `pack` | `blender_sim/datasets/pack_20260911_043022_s67_n1000` | 300, family-stratified | **5×5** |

**600 episodes.** Older 3×3 packs (`mixed250_s67_n250`, `side40_s11_n50`) are **not** in the live config. If a heatmap is already 5×5, `upsample_heatmap` is a no-op.

Episode folder names collide across packs, so the cache key is always `source/episode_dir`, e.g. `mixed/episode_0000_pothole_on_path`.

---

## 4. Forward pass (the actual student)

Deployment path (`predict_future_heatmap`) — **no shuffle**:

```text
frames  [B, 5, 3, 160, 160]   offsets t-20, t-15, t-10, t-5, t
   │
   ├─ TinyCNN per frame ──────────────► feats [B, 5, 128, 5, 5]
   │                                      now = feats[:, -1]
   │                                      diffs = now - feats[:, i]  for i in 0..3
   │                                      stack = cat(now, diffs)     [B, 640, 5, 5]
   │
   ├─ pixel_loom(frames) ─────────────► [B, 4, 5, 5]
   │                                      mean_now, mean_growth, max_growth, fill
   │
   ▼
CollisionEncoder(644 → 128 → 64 → 24) + per-cell LayerNorm
   │
   Z_t  [B, 24, 5, 5]
   │
   ResidualPredictor:  Zhat+ = Z_t + P(Z_t)
   │
   HeatmapDecoder(Z_t)           →  H_now_hat  in (0,1)
   HeatmapDelta(Zhat+ − Z_t)     →  Δ          in (−1,1)
   H_plus_hat = clamp(H_now_hat.detach() + Δ, 0, 1)      # t+1.0s
   H_mid_hat  = clamp(H_now_hat.detach() + 0.5·Δ, 0, 1)  # t+0.5s
```

Training path adds, **before** the encoder:

1. GPU photometric / flip aug on the concatenated now+future clips.
2. With probability `shuffle.prob` (0.15): the same \(k\times k\) permutation + 180° opposite-cell rotation on both clips.
3. After encoding: `unpermute_map(Z, perm)` so predictor/decoder sit in the **real** layout.
4. EMA teacher `target` encodes the **identically shuffled** future clip, then is also unpermuted. Stop-grad. JEPA loss is \(\|\hat{Z}_{t+\tau} - \bar{E}(\text{future})\|\) in that aligned \(Z\).

`n_frames = len(student.frame_offsets)` is the only thing that sizes the collision encoder’s first conv. Changing offsets from 4 to 5 frames is a config change plus a new training run. Old weights will not load.

---

## 5. Time, grid, and warning geometry

### 5.1 Clips

Each episode is 150 frames @ 30 fps (5 s). A training item is anchored at frame \(t\):

| Tensor | Frames used | Meaning |
|---|---|---|
| `frames` | \(t-20, t-15, t-10, t-5, t\) | What the encoder sees “now” |
| `frames_future` | same offsets, anchored at \(t+30\) | EMA teacher input |
| `h_now` | heatmap at \(t\) | current threat |
| `h_mid` | heatmap at \(t+15\) | half-second auxiliary |
| `h_future` | heatmap at \(t+30\) | **the** 1 s target |

Valid \(t\): \(t_{\mathrm{lo}} = -\min(\text{offsets}) = 20\) through \(t_{\mathrm{hi}} = 150-1-30 = 119\). Train samples every 2 frames; val every 3.

### 5.2 5×5 image / heatmap alignment

Row 0 is the **top** of the image (far). Row 4 is the **bottom** (near / feet). Column 0 is image-left.

`warning` for 5×5:

| | col 0 | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|---|
| group | LEFT | LEFT | CENTER | RIGHT | RIGHT |
| row weights (near-weighted) | 0.5, 0.65, 0.8, **1.0**, **1.0** | same | same | same | same |

Direction score = max of (heatmap × row_weights) over that group’s columns. Severity:

- score \(\ge 0.70\) → STOP
- score \(\ge 0.42\) → CAUTION
- else SAFE

`HitLatch`: 3 consecutive CAUTION+ frames to arm, 5 consecutive SAFE to release. Selection metrics use the latch. Raw heatmap metrics (`MetricAccumulator`) do **not**.

### 5.3 Spatial loss map (normalized, mean = 1)

Chebyshev rings around cell `(2,2)`:

```text
[[0.56  0.87  0.87  0.87  0.56]     corners 0.65 → 0.56 after /mean
 [0.87  1.30  1.30  1.30  0.87]     inner ring 1.5 → 1.30
 [0.87  1.30  1.91  1.30  0.87]     center 2.2 → 1.91
 [0.87  1.30  1.30  1.30  0.87]     edges 1.0 → 0.87
 [0.56  0.87  0.87  0.87  0.56]]
```

Walking-corridor mistakes cost more than corner mistakes. Mean is 1 so overall loss scale stays comparable when the flag is toggled.

---

## 6. File map

```text
model_trial_1/
├── configs/
│   ├── default.yaml          # live 5×5 shuffle-0.15 recipe
│   ├── no_shuffle.yaml       # same data/student, shuffle.prob=0
│   ├── grid3.yaml            # frozen 3×3 trial
│   ├── grid3_noshuffle.yaml
│   └── grid3_v2.yaml
├── scripts/
│   ├── 00_prepare_data.py    # catalog + split + cache
│   ├── 01_eval_copy_baseline.py
│   ├── 02_train.py           # the loop
│   ├── 03_eval.py            # copy vs student, val/test, wearable
│   ├── 04_export_onnx.py
│   ├── 05_make_test_split.py # carve test from val; train untouched
│   └── 06_overlay.py
├── rs_jepa/
│   ├── config.py
│   ├── engine.py
│   ├── losses.py
│   ├── metrics.py
│   ├── warning.py
│   ├── data/
│   │   ├── __init__.py       # catalog, split, family, JSON
│   │   ├── video.py          # decode + heatmap cache
│   │   ├── dataset.py        # CollisionDataset + sample_weights
│   │   ├── shuffle.py
│   │   ├── gpu_preprocess.py
│   │   └── augment.py        # CPU ClipAugmentor (tests / unused in train)
│   ├── models/
│   │   ├── tiny_cnn.py
│   │   ├── decoder.py
│   │   └── student.py        # RSJEPA
│   └── viz/overlay.py
└── tests/
    ├── test_shuffle.py
    └── test_wearable.py
```

| File | Role |
|---|---|
| `configs/default.yaml` | Live contract: data mix, 5-frame offsets, copy-residual, losses, gates |
| `rs_jepa/config.py` | Dotted-key YAML loader + `resolve_device` |
| `rs_jepa/models/student.py` | Context encoder, residual predictor, copy-residual heads, EMA |
| `rs_jepa/models/tiny_cnn.py` | Depthwise-separable CNN → \(k\times k\times 128\) |
| `rs_jepa/models/decoder.py` | \(Z\to H\) sigmoid and \(Z\to\Delta\) tanh |
| `rs_jepa/data/shuffle.py` | Permutation, 180° rule, GPU mosaic, unpermute \(Z\) |
| `rs_jepa/data/dataset.py` | Index, `__getitem__`, oversampling weights |
| `rs_jepa/data/video.py` | `preview.mp4` → `frames_160.npy`, JSON → `heatmaps.npy` |
| `rs_jepa/data/__init__.py` | Catalog, stratified split, `source/episode` keys |
| `rs_jepa/data/gpu_preprocess.py` | H2D, `/255`, batched aug |
| `rs_jepa/losses.py` | Focal MSE, spatial map, FP penalty, JEPA, warning alignment |
| `rs_jepa/warning.py` | Direction / severity / `HitLatch` |
| `rs_jepa/metrics.py` | Heatmap PR + wearable nuisance/miss |
| `rs_jepa/engine.py` | Eval loops, checkpoint, seed, wearable comparison |
| `scripts/02_train.py` | Sampler, AMP, full loss, best/last save |
| `tests/test_shuffle.py` | Shuffle invertibility, 3×3/5×5/5-frame shapes |
| `tests/test_wearable.py` | Gates, latch, spatial-weight shape |

---

## 7. `configs/*.yaml`

`Config.get("student.frame_offsets")` walks dotted keys. There is no schema validation beyond what scripts assert (`feature_grid` vs cached heatmap size vs `warning.row_weights` length).

### 7.1 `default.yaml` — live 5×5 recipe

Documented by section. Numbers are the **current** training contract.

**`data`.** Three sources (600 eps). `cache_dir` / `split_file` / `catalog_file` all under `cache_grid5`. `n_frames: 150`, `fps: 30`, `val_fraction: 0.2`, `test_fraction: 0.0` (test is carved later by script 05). `target_grid: 5`, `heatmap_upsample: nearest` (identity on native 5×5).

**`student`.** `img_size: 160`, `cnn_width: 32` (feat channels = 128), `feature_grid: 5`, `z_channels: 24`, `use_loom: true`, `copy_residual: true`, `frame_offsets: [-20, -15, -10, -5, 0]`.

**`horizon`.** `tau_frames: 30` (1.0 s), `mid_tau_frames: 15`.

**`shuffle`.** `prob: 0.15`, `curriculum: false` (no ramp). `rotate_opposite: true` is documentation; rotation is always applied by `rotate_mask_from_perm` whenever a cell actually moves to the opposite half.

**`jepa`.** EMA 0.996. Loss weights: `lambda: 0.12` (JEPA), `now_weight: 0.35`, `mid_weight: 0.25`, `dir_weight: 0.30`, `fp_weight: 0.45`, `fp_safe_threshold: 0.25`, `fp_center_mult: 3.5` on columns `[1,2,3]`, `delta_weight: 0.50`.

**`train`.** `batch_size: 32`, `num_workers: 2`, `lr: 0.001`, `weight_decay: 1e-4`, `epochs: 40`, cosine, AMP. Focal / change / FN, oversampling knobs, spatial-loss knobs, `grad_clip: 1.0`.

**`warning` / `selection`.** See §5.2 and §16. Gates: nuisance ≤ 0.20, miss ≤ 0.25, latch on, score \(= 1 - 0.5\,\mathrm{nuis} - 0.5\,\mathrm{miss}\).

**`paths`.** `checkpoints_grid5`, `export_grid5`, `logs_grid5`, `overlays_grid5`.

### 7.2 `no_shuffle.yaml`

Same student, same data, same losses. `shuffle.prob: 0`. Checkpoints go to `checkpoints_grid5_noshuffle`. Ablation: does the student need the mosaic corruption?

### 7.3 `grid3*.yaml`

Frozen 3×3. Older mixed/side packs, `feature_grid: 3`, often 4-frame offsets `[-20,-10,-5,0]`, shuffle 0.80 on `grid3.yaml`. Do not point `default.yaml` at `cache_grid3`.

---

## 8. `rs_jepa/config.py`

**`Config`.** `load(path)` → `yaml.safe_load`. `get("a.b.c", default)` walks nested dicts. `raw` is the whole tree (stored inside checkpoints). `__getitem__` is top-level only.

**`resolve_device(pref)`.** `"auto"` → `cuda` if `torch.cuda.is_available()` else `cpu`. Any other string is returned as-is (`"cuda:0"`, `"cpu"`).

---

## 9. `rs_jepa/models/tiny_cnn.py`

Maps one RGB frame `[B, 3, 160, 160]` to `[B, 128, k, k]`.

| Module | What |
|---|---|
| `ConvBNAct` | Conv (no bias) + **GroupNorm** + ReLU6. Groups = 8, or halved until it divides `cout`. |
| `DWSeparable` | Depthwise 3×3 then pointwise 1×1, both `ConvBNAct`. |
| `TinyCNN` | Stem stride 2, three stride-2 DW blocks, one stride-1 DW block, then `_pool_to_grid`. |
| `count_params` | `sum(p.numel() for p in module.parameters())` — counts EMA teacher too if you pass the whole `RSJEPA`. |

Spatial sizes at 160 px:

```text
160 → stem/2 → 80 → /2 → 40 → /2 → 20 → /2 → 10 → block4 → 10
_pool_to_grid: bilinear to (2k, 2k), then 2×2 avg-pool → (k, k)
```

For \(k=5\), 10→10 interpolate is a no-op then pool to 5. For \(k=3\), 10→6 then pool to 3. `feat_channels = width * 4 = 128` at `cnn_width=32`.

---

## 10. `rs_jepa/models/decoder.py`

**`HeatmapDecoder`.** `Conv 1×1 (C→32) → ReLU → Conv 3×3 pad1 → ReLU → Conv 1×1 (32→1) → sigmoid → squeeze`. Output `[B, k, k]` in \((0,1)\).

**`HeatmapDelta`.** Same trunk, **tanh** instead of sigmoid. Output \(\Delta \in (-1,1)\). Used only when `copy_residual=True`.

There is no skip from RGB. Both heads see only \(Z\).

---

## 11. `rs_jepa/models/student.py` — the model

This is the file to read if you change architecture.

### 11.1 `pixel_loom(frames, grid) → [B, 4, k, k]`

Luminance \(0.299R+0.587G+0.114B\). Adaptive avg/max pool to \(k\times k\) per frame. Channels:

1. mean luminance **now**
2. mean growth: \(\mathrm{avg}_{now} - \mathrm{avg}_{first}\) (first = \(t-20\))
3. max growth: \(\mathrm{max}_{now} - \mathrm{max}_{first}\)
4. fill: \(\mathrm{max}_{now} - \mathrm{avg}_{now}\)

This is a cheap expansion / occupancy cue that does not go through TinyCNN. It **does** see the (possibly shuffled) mosaic; after unpermute, loom is in real layout only if you unpermute \(Z\) (you do). Loom itself is concatenated **before** the encoder, so it is shuffled with the mosaic. That is intended: loom is a per-cell statistic.

### 11.2 `CollisionEncoder`

`Conv3×3 (in_ch → feat) + GroupNorm + ReLU → Conv3×3 (feat → feat/2) + ReLU → Conv1×1 (feat/2 → Z)`. Then **LayerNorm over channels at each cell** (permute NHWC, LN, permute back). JEPA cosine/MSE on \(Z\) is scale-stable because of this LN.

`in_ch = feat_channels * n_frames + (4 if loom else 0)`. Current: \(128\times 5 + 4 = 644\).

### 11.3 `ResidualPredictor`

\(\hat{Z}_+ = Z_t + P(Z_t)\). \(P\): Conv3×3 → ReLU → **dilated** Conv3×3 (dilation 2, padding 2) → ReLU → Conv1×1. Hidden = \(3Z\). Dilation lets a threat jump a cell in one residual step (jaywalker crossing).

### 11.4 `ContextEncoder`

Shared TinyCNN + CollisionEncoder. This is what the EMA teacher **copies**.

```python
feats = cnn(frames.reshape(B*N, 3, H, W)).reshape(B, N, C, k, k)
now = feats[:, -1]
parts = [now] + [now - feats[:, i] for i in range(N-1)]
x = cat(parts, dim=1)          # [B, C*N, k, k]
if use_loom: x = cat([x, pixel_loom(...)], dim=1)
return encoder(x)
```

`now` is always the **last** offset (must be 0). Diffs are now-minus-each-past, so five frames give four motion maps plus appearance. That is the “linear in time” stack the \(t-15\) frame was added for.

Raises `ValueError` if `N != n_frames`.

### 11.5 `RSJEPA`

Members: `context`, `predictor`, `decoder`, optional `delta_head`, frozen `target = deepcopy(context)`.

**`from_config`.** `n_frames=len(offsets)`. Width, grid, Z, EMA, loom, copy_residual from YAML.

**`_align(z, perm)`.** Identity if `perm is None`, else `unpermute_map`.

**`_future_from_now(z_t, z_plus_hat)`.** Copy-residual vs decode-from-\(Z_+\) (legacy). Mid-horizon is **half delta**, not a third head.

**`forward(frames, frames_future=None, perm=None)`.** Encode now → align → predict \(Z_+\) → heatmaps. If `frames_future` is given, EMA-encode it (no grad, `target.eval()`), align with the **same** perm, store `z_plus`.

**`predict_future_heatmap(frames)`.** Deploy: no perm, returns `h_plus_hat` only.

**`update_ema`.** \(p_{\mathrm{t}} \leftarrow m\, p_{\mathrm{t}} + (1-m)\, p_{\mathrm{s}}\) on parameters; buffers copied. Called after every optimizer step.

---

## 12. `rs_jepa/data/shuffle.py`

Do **not** pre-shuffle videos to disk. A clip needs a new permutation every draw.

**`sample_perm(B, grid, prob, device)`.** Vectorized: `argsort` of uniform keys is a uniform permutation of \(k^2\) cells. With probability \(1-\texttt{prob}\) the perm is identity. Returns `(perm, rotate_mask)`.

**`rotate_mask_from_perm`.** `perm[b, i]` = source cell index that **lands in** destination slot \(i\). Rotate 180° iff the cell **moved** and source/dest are on opposite sides of centre: \((r_s-\bar{c})(r_d-\bar{c}) + (c_s-\bar{c})(c_d-\bar{c}) \le 0\). Adjacent same-side swaps do **not** rotate (tested).

**`extract_cells` / `stitch_cells`.** Centre-crop to a multiple of \(k\), reshape to `[B, N, T, C, ch, cw]`. 160 is divisible by 5 and 3, so crop is the full frame.

**`apply_shuffle`.** Gather cells by `perm`, `rot90(..., 2)` where masked, stitch.

**`apply_shuffle_pair`.** `cat` now and future on time, shuffle once, split. This is C3.

**`unpermute_map` / `permute_map`.** Same gather on `[B, C, k, k]`. Unpermute = gather with `argsort(perm)`.

**`permute_grid`.** Same for a heatmap. Training does **not** permute the target heatmap (C2); these helpers exist for tests and any “shuffle the label too” experiment that must not ship.

---

## 13. `rs_jepa/data/__init__.py` — catalog and split

**`classify_family(scenarios)`.** First keyword hit among vehicle / pedestrian / static / empty, else `other`. Used only to stratify.

**`read_episode_meta`.** `episode.json`. `_has_rare` if `label_histogram` has `NEAR_MISS` or `CRITICAL_THREAT`.

**`build_catalog(sources, seed)`.** For each YAML source: list `episode_*` dirs, optionally `_stratified_sample` to `take`. Key = `f"{name}/{ep.name}"`.

**`make_split`.** Per family, shuffle, hold out `val_fraction+test_fraction`. If `test_fraction==0` (live config), holdout is all val. Script 05 then splits that val.

**`carve_test_from_holdout`.** Stratify by `(family, source, rare)`. Groups of size 1 are assigned with probability `test_fraction`. **Train is never touched** — that is why 05 exists instead of baking test into `make_split` after a run has started.

**`save_json` / `load_json`.** Indent-2.

---

## 14. `rs_jepa/data/video.py`

**`decode_all_frames(path, size)`.** OpenCV, resize square `size`, BGR→RGB, stack uint8 `[N,H,W,3]`.

**`decode_video_full`.** Native resolution + fps (overlay display).

**`load_heatmaps(json)`.** `spatial_annotations.json` → `[N,k,k]` float32, frames sorted by `frame_id`.

**`upsample_heatmap(hm, grid, mode)`.** If \(k=\texttt{grid}\), return as-is. Else OpenCV resize per frame, `INTER_NEAREST` or `INTER_LINEAR`. Live packs are native 5×5.

**`cache_episode(...)`.** Writes `cache_dir / key / frames_{size}.npy` and `heatmaps.npy`. Frames are skipped if the npy already exists (`frames_only_if_missing=True`) so `--heatmaps-only` can rewrite labels without re-decoding video.

---

## 15. `rs_jepa/data/dataset.py`

**`_build_index`.** All `(ep_idx, t)` with stride, \(t \in [t_{\mathrm{lo}}, t_{\mathrm{hi}}]\).

**`CollisionDataset`.** mmap caches per episode. `__getitem__` stacks uint8 NHWC clips for now and future, plus three heatmaps, plus `ep_idx` and `t`. **No `/255`, no aug.** `ClipAugmentor` is accepted but training passes `None`.

**`sample_weights(...)`.** One weight per index entry:

\[
w = (1 + \texttt{hot\_mult}\cdot \max H_{t+\tau} + \texttt{change\_mult}\cdot \max|H_{t+\tau}-H_t|)
\]

then multiply:

| Flag | When | Live value |
|---|---|---|
| `rare_mult` | episode `_has_rare` | 2.0 |
| `quiet_mult` | \(\max H_{t+\tau} < 0.25\) | 4.0 |
| `quiet_ep_mult` | name contains empty_street / safe_walk / car_pass_far / parallel_pedestrian / periph_empty / parked_car | 2.5 |
| `center_cool_mult` | inner 3×3 mean of \(H_{t+\tau} < 0.28\) (`center_cool_mode: inner`) | 5.5 |

`WeightedRandomSampler(..., replacement=True, num_samples=len(weights))` so a 40-epoch run still sees 24 000 draws/epoch but oversamples cool-centre / quiet / changing frames. That is how the vest is pushed off “always beep the corridor.”

---

## 16. `rs_jepa/data/gpu_preprocess.py` and `augment.py`

**`frames_uint8_to_nchw`.** `[B,T,H,W,3] uint8 → [B,T,3,H,W] float / 255`.

**`gpu_clip_augment`.** Per-sample brightness/contrast, Gaussian noise \(\sigma=6/255\), 20% 3×3 avg blur, 50% horizontal flip (frames **and** all heatmaps). Same draw for now and future because they were concatenated.

**`prepare_batch(batch, device, augment=)`.** Non-blocking H2D, concat clips, normalize, maybe aug, split. Train uses `augment=True`. Eval uses `engine.move_batch` → `augment=False`.

**`ClipAugmentor`** in `augment.py` is the CPU twin. Dataset no longer calls it during training. Tests can still use it.

---

## 17. `rs_jepa/losses.py`

### 17.1 `spatial_heatmap_weights(grid, center, ring, edge, corner, normalize=True)`

Chebyshev distance from geometric centre. Centre cell / inner ring / border / corners get the four raw values, then divide by the mean. Returned `[k,k]`. Broadcasts over batch in the MSE.

### 17.2 `weighted_focal_mse(pred, target, ...)`

\[
\mathrm{SE}=(pred-target)^2,\quad
w = 1 + f\cdot target^{\gamma} + c\cdot |change| + f_n\cdot \mathrm{ReLU}(target-pred)\cdot[target>0.3]
\]

then \(w \leftarrow w \odot \texttt{cell\_weights}\) if given. Reduce: mean of \(SE \odot w\).

Live: `focal_weight=1`, `gamma=2`, `change_weight=2.5` (on \(L_H\) only), `fn_weight=0.6`.

### 17.3 `false_positive_penalty`

On cells with \(target < 0.25\), penalize \(pred^2\). Columns `[1,2,3]` (near-centre of 5×5) get ×3.5. This is the “don’t light the empty walking lane” term.

### 17.4 `jepa_distance(z_hat, z)`

MSE + mean \(1 - \cos\) over channel dim. \(Z\) is LayerNormed.

### 17.5 `warning_alignment_loss`

Match LEFT/CENTER/RIGHT scores (`warning_scores` = row-weighted group max, same geometry as `warning.direction_scores`). MSE on the 3-vector, extra peak-FN when GT is CAUTION+, CE on argmax direction among threats, over-prediction penalty on quiet frames. This is why `L_dir` is a separate logged term.

---

## 18. `rs_jepa/warning.py`

**`direction_scores` / `severity_from_score` / `classify`.** Numpy, used by metrics, overlay, eval.

**`HitLatch`.** State machine, per episode:

- Inactive: increment danger streak on CAUTION+; at `confirm_frames` (3) arm with that pending dir/sev.
- Active: update dir/sev while still CAUTION+; at `release_frames` (5) SAFE in a row, disarm.
- `reset()` / `start_episode` rebuilds it so latch does not leak across clips.

CAUTION counts as a **beep** for nuisance. Miss is only **silent on STOP** (CAUTION on a STOP frame is not a miss). That is tested in `tests/test_wearable.py`.

---

## 19. `rs_jepa/metrics.py`

**`MetricAccumulator`.** Per predicted/true heatmap pair:

- MSE over cells
- High-threat PR at 0.5 (`HT-recall` / `HT-F1` in logs)
- Direction accuracy **only on GT CAUTION+**
- Instantaneous STOP PR (no latch)

**`WearableAccumulator`.** Per frame, after optional latch vs GT severity at \(t+\tau\):

- `nuisance = n_beep_on_SAFE / n_SAFE`
- `miss = n_silent_on_STOP / n_STOP`
- `score = 1 - 0.5·nuisance - 0.5·miss`
- `eligible` iff nuisance ≤ 0.20 **and** miss ≤ 0.25

**`wearable_better(cur, best)`** (in `engine.py`): any eligible beats any ineligible; within a class, higher score. That is why an ineligible epoch-14 can remain `student_best.pt` for the rest of a run that never clears the gates.

---

## 20. `rs_jepa/engine.py`

**`predict_episode_heatmaps`.** Full-episode batched deploy inference. Frames before \(t_{\mathrm{lo}}\) stay zeros.

**`evaluate_future_heatmap`.** Loader path: `prepare_batch(augment=False)` → `predict_fn` → `MetricAccumulator` overall + rare.

**`evaluate_wearable`.** **Every** valid frame of **every** listed episode (not the strided loader). Loads `frames_160.npy` + `heatmaps.npy`. `copy_baseline=True` uses \(H(t)\) as if it were the prediction of \(H(t+\tau)\) — the number the student must beat, and a cheat because it sees the true current heatmap.

**`save_checkpoint`.** `torch.save` of `{model, cfg, epoch, val}`.

**`set_seed` / `configure_runtime`.** cudnn benchmark, TF32, OpenCV threads=1.

---

## 21. Scripts

### 21.1 `scripts/00_prepare_data.py`

`build_catalog` → write catalog → `make_split` (unless split exists and no `--reset-split`) → parallel `cache_episode` (`ProcessPoolExecutor`, default 6 workers, spawn). `--heatmaps-only` rewrites `heatmaps.npy` and keeps `split.json`. Prints native grid per source (`identity` vs `will upsample`).

Always `--reset-split` when the source mix changes. Do not reset if you need to compare against a previous run’s exact train list.

### 21.2 `scripts/05_make_test_split.py`

If `split["test"]` already nonempty, print counts and exit. Else copy `split.json` to `split_train_val_only.json`, carve ~half of **current val** into test (seed 1354, fraction 0.5), write `val_legacy`. Train keys unchanged.

Current 600-ep split after 05: **480 train / 59 val / 61 test**.

### 21.3 `scripts/02_train.py` — training loop

`build_loaders`: train stride 2 + `WeightedRandomSampler`; val stride 3, no shuffle sampler.

Per epoch:

1. `shuffle_prob_for_epoch` (constant 0.15 unless curriculum).
2. For each batch: `prepare_batch(augment=True)` → maybe `sample_perm` + `apply_shuffle_pair` → AMP autocast bf16 (or fp16+GradScaler) → `model(frames_s, frames_fs, perm)` → losses → clip 1.0 → step → `update_ema`.
3. Val heatmap metrics on the loader + **full** wearable eval on val episodes.
4. Always write `student_last.pt`. If `wearable_better`, also `student_best.pt` and `student.pt`.

**Full loss (live weights in parentheses):**

\[
\begin{aligned}
L &= L_H &&\text{(spatial focal MSE on }\hat H_{t+\tau}\text{ vs }H_{t+\tau}\text{)} \\
&+ 0.35\, L_{\mathrm{now}} &&\text{(same, }\hat H_t\text{ vs }H_t\text{)} \\
&+ 0.25\, L_{\mathrm{mid}} &&\text{(}\hat H_{t+0.5s}\text{ vs }H_{t+15}\text{)} \\
&+ 0.30\, L_{\mathrm{dir}} &&\text{warning alignment} \\
&+ 0.45\, L_{\mathrm{FP}} &&\text{safe-cell }pred^2\text{, centre }\times 3.5 \\
&+ 0.12\, L_{\mathrm{JEPA}} &&\text{EMA }Z\text{ distance} \\
&+ 0.50\, L_{\Delta} &&\text{MSE}(\Delta,\; \mathrm{clamp}(H_{t+\tau}-H_t, -1,1))
\end{aligned}
\]

Optimizer: AdamW on trainable params (teacher frozen). CosineAnnealingLR to 5% of `lr`. Log every 200 steps and at epoch end.

`--resume` loads `student_last.pt` and continues epoch+1. `--epochs N` overrides YAML.

### 21.4 `scripts/01_eval_copy_baseline.py`

Stride-3 copy \(H(t)\to H(t+\tau)\) on val/test. No GPU. Sanity for “is the label even predictable by copying?”

### 21.5 `scripts/03_eval.py`

`--split val|test|both`, `--ckpt` optional (else `student_best.pt` then `student.pt`). Prints copy heatmap + wearable, student heatmap + wearable, and latched STOP-F1 (`hysteresis_stop_f1`). **Never shuffles.**

### 21.6 `scripts/04_export_onnx.py`

Wraps `predict_future_heatmap`. Dummy `[1, n_clip, 3, 160, 160]`. Opset 17, `dynamo=False`. Optional onnxruntime CPU latency. Reads `paths.ckpt_dir/student.pt` (the best alias).

### 21.7 `scripts/06_overlay.py`

Loads student, predicts a full episode, composites with `rs_jepa.viz.overlay` (same blue/amber/red as `blender_sim/spatial_overlay.py`).

- `--mode aligned`: show pred of \(t+\tau\) on the RGB **at** \(t+\tau\) (fair visual vs GT).
- `--mode wearable`: show the forecast on the RGB **now** (what the vest would overlay).
- `--latch`: HitLatch on PRED banner.
- `--split` / `--episode KEY` / `--video mp4`.

Writes `overlays_grid5/{key}.mp4`.

---

## 22. `rs_jepa/viz/overlay.py`

**`threat_to_rgb`.** Blue → amber at 0.55 → red.

**`haze_rgb`.** Approximate ffmpeg `eq=saturation=0.38:brightness=-0.10:contrast=0.88`.

**`heat_image` / `blend_heat`.** Nearest-neighbour upsample of \(k\times k\), 52% overlay, grid lines, per-cell numeric labels.

**`stack_panel`.** Severity-coloured banner + blended RGB.

**`write_mp4`.** OpenCV `mp4v`, RGB→BGR.

---

## 23. Tests

```bash
cd /storage/BTP/model_trial_1
.venv/bin/python tests/test_shuffle.py
.venv/bin/python tests/test_wearable.py
```

**`test_shuffle.py`.** Opposite-cell 180° rule; heatmap/Z round-trip; identity perm; pair shuffle matches two calls; painted cell survives deshuffle; nearest upsample keeps a 3×3 peak; `RSJEPA` shapes for 3-frame, 4-frame, **5-frame**, 3×3 and 5×5; `prepare_batch` uint8 path; copy-residual clamp to \([0,1]\).

**`test_wearable.py`.** Perfect score; false beeps fail nuisance gate; mute-on-STOP fails miss gate; CAUTION-on-SAFE is nuisance; CAUTION-on-STOP is **not** a miss; eligible beats higher ineligible score; latch needs confirm; centre FP heavier; spatial weights mean≈1 and centre > ring > edge > corner.

These tests construct models with explicit `n_frames=`; they do **not** read YAML. A config with 5 offsets is checked by `from_config` + the 5-frame shape test.

---

## 24. Where is X?

| I want to… | Go here |
|---|---|
| Change history frames | `student.frame_offsets` then **retrain** (encoder `in_ch` changes) |
| Change grid | `data.target_grid` **and** `student.feature_grid` **and** `warning.row_weights` / col groups; recache |
| Turn shuffle off | `configs/no_shuffle.yaml` or `shuffle.prob: 0` |
| Change copy vs decode-from-Z | `student.copy_residual` |
| Spatial loss on/off | `train.spatial_loss` and `spatial_*` |
| Oversample empty corridor | `train.center_cool_*`, `quiet_*` |
| Wearable gates | `selection.max_nuisance` / `max_miss` |
| Beep thresholds | `warning.caution_threshold` / `stop_threshold` |
| EMA / JEPA weight | `jepa.ema_momentum` / `jepa.lambda` |
| Checkpoint rule | `engine.wearable_better` |
| Loom channels | `pixel_loom` + `LOOM_CHANNELS` |
| GPU vs CPU bottleneck | `gpu_preprocess.py` vs `dataset.__getitem__` |
| Overlay colours | `viz/overlay.py` (`threat_to_rgb`) |
| Dataset mix | `data.sources` then `00_prepare_data.py --reset-split` |

---

## 25. Load-bearing pitfalls

1. **Shuffle at deploy is a bug.** `predict_future_heatmap` and ONNX must stay identity layout. Training shuffle is inside `02_train.py`, not inside `RSJEPA.forward` by default (`perm=None`).

2. **Do not permute the heatmap target.** Encoder is scrambled; loss is in real cell coordinates after `unpermute_map`.

3. **BatchNorm will poison shuffle.** Keep GroupNorm.

4. **`copy_residual` detaches \(\hat H(t)\).** Gradients of \(L_H\) go through \(\Delta\) and \(Z\), not through a shortcut that copies GT. If you drop the detach, the model can cheat by putting all of \(H(t+\tau)\) into \(\hat H(t)\).

5. **4-frame weights ≠ 5-frame student.** First conv `in_channels` differs (520 vs 644 with loom). `load_state_dict` will error. Same for 3×3 vs 5×5 (`Z` spatial size).

6. **`--reset-split` after a published run mixes people.** Train keys change. Use 05 to carve test from an existing val instead.

7. **Wearable eval is slow-ish** (full episodes, not the strided loader). That is why each epoch takes longer than the 750 train steps (~1 min/epoch on 4060 including val).

8. **Copy baseline wins safety metrics** if you give it the true \(H(t)\). The student only sees RGB. Compare fairly.

9. **Native 5×5 vs upsample.** `nearest` upsample of old 3×3 labels makes 2×2 blocks, not a true 5×5 threat field. Live s69 packs are native 5×5; do not mix s67 3×3 into `cache_grid5` without knowing that.

10. **Latch state is per episode.** Forgetting `start_episode` would carry a STOP buzz into the next clip’s first frames and inflate nuisance/miss.

---

## 26. Current training result (5×5, this recipe)

Run: `scripts/02_train.py --config configs/default.yaml`, 2026-09-15, RTX 4060, 40 epochs, ~47 min, log `logs_grid5/train.log`.

| | |
|---|---|
| Params | 1,790,778 |
| n_frames | 5 |
| Train / val samples | 24 000 / 2 006 |
| Shuffle | 0.15, no curriculum |
| AMP | bf16 |

**Best wearable checkpoint: epoch 14** (`checkpoints_grid5/student_best.pt`):

| | value |
|---|---|
| score | 0.657 |
| nuisance | 0.432 |
| miss | 0.254 |
| eligible | **False** (miss just over 0.25; nuisance over 0.20) |
| val MSE | 0.0715 |
| HT-recall | 0.304 |
| STOP-F1 | 0.471 |
| dir-acc | 0.691 |

Late epochs (e.g. 40 / `student_last.pt`): STOP-F1 ~0.61, miss ~0.18, **nuisance ~0.58**. The student learned to light threats and then over-beeped the corridor. **No epoch in this run cleared both gates.** Selection therefore kept the ineligible-but-highest-score epoch 14.

That is the live scientific state of the 5-frame / spatial-loss / cool-centre / shuffle-0.15 recipe — not a claim that the vest is done.

---

## 27. Data-flow recap (one page)

```text
blender_sim episode
  preview.mp4  +  spatial_annotations.json  +  episode.json
        │
        ▼  00_prepare_data.py
cache_grid5/{source}/{episode}/
  frames_160.npy     uint8 [150,160,160,3]
  heatmaps.npy       float32 [150,5,5]
        │
        ▼  CollisionDataset  (uint8 slices)
batch: frames, frames_future, h_now, h_mid, h_future
        │
        ▼  prepare_batch  (GPU /255 + aug)
        ▼  apply_shuffle_pair  (p=0.15)
        ▼  RSJEPA.forward  (encode shuffled → unpermute Z → P → H, Δ)
        ▼  EMA teacher on future clip
        ▼  losses  (heatmap + JEPA + warning + FP + Δ)
        ▼  wearable_better → student_best.pt
        │
        ▼  03_eval / 06_overlay / 04_export_onnx
identity layout only: RGB clip → H(t+1s) → LEFT/CENTER/RIGHT × SAFE/CAUTION/STOP
```

---

## 28. Glossary

| Term | Meaning |
|---|---|
| **RS-JEPA** | This student: region-shuffle + JEPA latent prediction + heatmap decode |
| **Cell / token** | One of 25 locations on the 5×5 grid |
| **Shuffle** | Permute those cells in RGB; same perm on all frames of now+future |
| **Unpermute** | Put \(Z\) back in camera layout before \(P\) and \(H\) |
| **Loom** | Four pooled-luminance channels (mean now, growth, max growth, fill) |
| **Copy-residual** | \(\hat H(t+\tau)=\mathrm{clamp}(\hat H(t).\mathrm{detach}+\Delta)\) |
| **EMA teacher** | Exponential moving average of `ContextEncoder`; stop-grad target \(Z\) |
| **Nuisance** | \(P(\text{CAUTION+ beep}\mid\text{GT SAFE})\), latched |
| **Miss** | \(P(\text{SAFE silent}\mid\text{GT STOP})\), latched |
| **Eligible** | nuisance ≤ 0.20 and miss ≤ 0.25 |
| **HT-recall** | Cellwise recall of heatmap ≥ 0.5 |
| **dir-acc** | Predicted direction matches GT on GT CAUTION+ frames |
| **τ** | 30 frames = 1.0 s |
| **Catalog key** | `mixed/episode_0000_...` — unique across packs |
| **Rare** | Episode histogram contains NEAR_MISS or CRITICAL_THREAT |

---

## 29. How to add something (without breaking C1–C10)

- **New input cue (flow, stereo, …):** concatenate onto the CollisionEncoder stack in `ContextEncoder.forward`, bump `in_ch`, retrain. Do not skip unpermute.
- **New loss:** add a scalar in `02_train.py` next to `l_h` / `l_j`; log it on the epoch line. Keep wearable selection as the checkpoint rule unless you are explicitly ablating that.
- **New grid:** recache, fix warning columns, fix `spatial_heatmap_weights` rings, fix overlay. Never interpolate \(Z\).
- **New dataset pack:** add a `data.sources` entry with a **unique** `name`, then `--reset-split` only if you accept a new train set.
- **Deploy change:** `predict_future_heatmap` and `DeployStudent` must stay shuffle-free. If you add preprocessing, it belongs in the ONNX wrapper, not in a training-only `perm` argument.
