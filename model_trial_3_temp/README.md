# model_trial_3 — Collision-JEPA teacher (multi-horizon)

Teacher-only training for a wearable collision-anticipation model. The student /
TinyCNN / distillation path is **out of scope**.

The teacher is pretrained **V-JEPA 2** plus a compact collision adapter. It reads
**16 RGB frames spanning 3 seconds** ending at anchor \(t\) and predicts:

| output | shape | meaning |
| --- | --- | --- |
| `Z_t` | `[B, 32, 5, 5]` | spatial collision latent (future student interface) |
| `Hhat_now` | `[B, 5, 5]` | current 5×5 threat map |
| `Hhat_1s` / `Hhat_1_5s` / `Hhat_2s` | `[B, 5, 5]` | 1.0 / 1.5 / 2.0 s futures |

No future RGB frames enter the encoder. Overlay videos (`spatial_overlay.mp4`)
are never used as input.

## What changed vs Model Trial 2

Trial 2 used an 8-frame history, one 1 s horizon, and immediately averaged
tubelet tokens over time. This trial:

- samples **16 frames**, 200 ms apart, last frame = \(t\) (offsets −90…0)
- predicts **three** futures with copy-residual heads
- keeps a **temporal adapter** on the T′×16×16 token grid
- treats **collision/miss** and **bg_00/bg_01** as first-class pairing axes
- **masks** future losses when the 3 s history is pre-fork and the two outcome
  targets disagree (a deterministic model cannot know which branch occurs)
- **letterboxes** 16:9 video into 256×256 so left/right FOV is not center-cropped
- normalizes with the **V-JEPA processor ImageNet mean/std**, once, in the backbone

## Dataset

Source (override in YAML): `blender_sim/datasets/groups15_n400_s67_n400`

```
sample_XXXX_<scenario>/
  collision|miss_left|miss_right/
    bg_00|bg_01/
      preview.mp4          # RGB input (1920×1080, 30 fps, 450 frames)
      spatial_annotations/spatial_annotations.json   # 5×5 maps
      episode.json
```

Incomplete directories (missing `preview.mp4`, `episode.json`, or heatmaps) are
**skipped** with a logged reason. The planned pack is 400 samples × 2 outcomes ×
2 backgrounds; only finished clips are used.

Heatmap grid: row 0 = far/top, row 4 = near/feet, col 0 = left, col 4 = right.

Valid anchors for a 450-frame clip: **t ∈ [90, 419]** when
`sampling.allow_partial_horizons` is true (default): 1s must exist (`t+30 ≤ 449`)
even if 1.5s/2s do not; missing later targets are masked. Stride 15 last anchor
is **405**. Set the flag false to require all three futures
(**t ∈ [90, 389]**, last stride-15 anchor 375).

## Pairing

**Outcomes.** Collision and miss share RGB until `fork_frame` (default 365 /
12.15 s). If `t < fork_frame` the 16-frame history is identical but a future
heatmap has already split, both clips train on the cell-wise **average** of
collision and miss (`ambiguous_policy: mean_target`). Current-map loss is
kept. Set `collision_target` to copy the collision map instead, or
`mask_ambiguous_pairs` to skip that horizon.

**Backgrounds.** `bg_00`/`bg_01` change clutter and lighting with the same
threat. When heatmaps at the anchor/horizons match within tolerance, a small
per-cell cosine loss on `Z_t` is applied via a stop-grad pair buffer (the two
Giant graphs are not held at once).

IDs: `clip_id`, `scenario_group_id`, `outcome_group_id`, `background_pair_id`.
Splits are by **sample group**, never by clip.

## Architecture

```
preview.mp4  →  16×256 RGB (letterbox)  →  ImageNet normalize
       │
       ▼
V-JEPA 2 encoder  (frozen, or LoRA on last 6 blocks)
tokens [B, 8×16×16, C] = [B, 2048, C]
       │
shared 1×1+3×3 spatial proj per time group → [B, T′, 128, 16, 16]
2× depthwise-temporal residual blocks
mean(T′) ⊕ last(T′) → 5×5 bottleneck LayerNorm → Z_t [B, 32, 5, 5]
       ├─ HeatmapDecoder → Hhat_now
       └─ per horizon ResidualPredictor → Zhat_h
              └─ HeatmapDelta(Zhat_h − Z_t) → Δ_h ∈ [-1,1]
                     Hhat_h = clamp(sg(Hhat_now) + Δ_h, 0, 1)
```

Giant (`facebook/vjepa2-vitg-fpc64-256`, C=1408) is preferred. Large
(`...-vitl-fpc64-256`) is the configured fallback if Giant cannot load. The
backbone stays in **eval** mode. Original weights are never fully unfrozen.

Default loss (coefficients in `configs/default.yaml`):

```
L = L_future + 0.25 L_now + 0.20 L_Δ + 0.10 L_FP + 0.05 L_dir + 0.02 L_bg
L_future = 0.50 L_1s + 0.30 L_1.5s + 0.20 L_2s
```

`L_map` is target-weighted MSE \(w_{ij}=0.05+4H_{ij}^2\) with mild near-row
priority `[0.8, 0.9, 1.0, 1.1, 1.2]`. There is **no** JEPA latent matching loss
in this teacher-only experiment.

Checkpoint selection uses future high-threat recall and STOP F1, gated by
nuisance / miss-rate limits — not MSE alone.

## Commands

All commands are run from `model_trial_3/` (or with that folder on `PYTHONPATH`).

```bash
cd model_trial_3

python tests/run_tests.py

python scripts/00_inspect_dataset.py
python scripts/01_build_manifest.py
python scripts/02_prepare_cache.py

# dummy encoder (no Giant download)
python scripts/03_smoke_test.py --num-clips 2 --dummy
python scripts/07_profile_pipeline.py --dummy

# real V-JEPA 2 (uses paths.hf_cache_dir)
python scripts/03_smoke_test.py --real
python scripts/04_train_teacher.py --phase frozen
python scripts/04_train_teacher.py --phase lora --resume
python scripts/04_train_teacher.py --phase both

python scripts/05_evaluate_teacher.py --checkpoint best
python scripts/06_render_heatmap_overlays.py --checkpoint best
```

`--dry-run` on the train script constructs the model and loaders without
optimizing.

## Outputs

| path | contents |
| --- | --- |
| `cache/manifest.jsonl` | discovered clips |
| `cache/splits.json` | group splits |
| `cache/frames/` | optional 256² uint8 RGB |
| `checkpoints/teacher_phase_a_{last,best}.pt` | frozen phase |
| `checkpoints/teacher_phase_b_{last,best}.pt` | LoRA phase |
| `logs/` | text + JSONL |
| `reports/` | inspect / train / eval JSON+CSV |
| `visualizations/` | RGB + heatmap grids |

Checkpoints store adapter/heads/LoRA, optimizer, scaler, config, preprocessing
signature, and split/manifest fingerprints. Full Giant weights are **not**
duplicated; the HF model id is recorded and reloaded.

## Tests

`python tests/run_tests.py` covers manifest skip/accept, the locked 16-frame
contract, no future leakage, heatmap frame-ID join, adapter/teacher shapes,
losses+masks, pairing IDs, ambiguity masking per horizon, frozen backbone
gradients, LoRA trainable-set, checkpoint round-trip, and letterbox RGB order.

Those tests use `DummyVJEPA2Backbone` and do **not** download Giant.

## Engineering notes

- **Preprocessing.** Default `resize_mode: letterbox`. Official V-JEPA
  shortest-edge + center-crop is `checkpoint_center_crop` (opt-in only).
- **Normalization.** ImageNet stats from `VJEPA2VideoProcessor`, applied in
  `VJEPA2Backbone.forward`. Dataset tensors stay in `[0, 1]`.
- **Tokens.** `get_vision_features` / `skip_predictor=True`. Layout is
  Conv3d `(tubelet=2, patch=16)` → `[B, T′·S·S, C]`. A mismatched count is a
  hard error (no silent slice).
- **16 vs 64 frames.** The checkpoint is `fpc64`; inference length is not
  required to be 64. We feed 16 frames and assert the token geometry.
- **OOM.** Start with `training.micro_batch_size: 1`. Set
  `backbone.model_id` to the Large fallback if Giant will not fit. Do not
  silently claim Giant was used.
- **Cache.** Frame cache is optional and budgeted. Encoder-token cache is off
  by default and must not be reused after LoRA changes the encoder.
