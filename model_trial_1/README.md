# Region-Shuffle Collision JEPA (model_trial_1)

Wearable collision anticipation: 3 RGB frames → 5×5 threat heatmap 1 second ahead
→ `LEFT/CENTER/RIGHT` + `SAFE/CAUTION/STOP`.

This trial tests **one method**: shuffle the 5×5 image regions (same permutation
across time), 180°-rotate a region when it moves to the opposite side of the
frame, predict, **deshuffle**, and match the original future heatmap. If the
network has learned *obstacle motion* (looming / closing) rather than object
identity or the walking-corridor prior, deshuffled predictions stay correct.

## Why this is a JEPA (and how it relates to V-JEPA-2)

V-JEPA-2 (Meta, 2025) does **not** reconstruct pixels. An encoder `E` sees a
corrupted video; a predictor `P` outputs future/masked tokens in representation
space; an EMA target encoder `Ē` provides the stop-grad targets.

RS-JEPA keeps that loop, at wearable scale:

| V-JEPA-2 | This trial |
|---|---|
| ViT-L tubelet tokens | 5×5 collision cells (aligned with labels) |
| Random masking | Region shuffle + 180° on opposite cells |
| EMA encoder `Ē` | EMA copy of the TinyCNN collision encoder |
| Predict in Z, not pixels | Residual predictor `Zhat+ = Zt + P(Zt)` |
| Frozen backbone + probe | Tiny decoder `Z → H` trained jointly |

The frozen 300M `facebook/vjepa2-vitl-fpc64-256` teacher is **not** downloaded
here. Distilling it is a later stage; this experiment isolates the shuffle
method. Deployment still never runs a ViT — only the ~0.6M student.

## Dataset

From `blender_sim/datasets`:

- **all** of `mixed250_s67_n250` (newer)
- **all** of `side40_s11_n50` (newer, peripheral/false-positive hard negatives)
- **500** stratified episodes from `pack_20260911_043022_s67_n1000` (older)

Episode names collide across packs, so cache keys are `source/episode_dir`.
mixed/side heatmaps are 3×3 and are bilinear-upsampled to 5×5; pack is already 5×5.

Episode-level 80/20 split, stratified by family. Never split by frame.

## Train

```bash
cd /storage/BTP/model_trial_1
source .venv/bin/activate
python scripts/00_prepare_data.py
python scripts/01_eval_copy_baseline.py
python scripts/02_train.py
python scripts/03_eval.py
```

Shuffle is training-only. Eval and ONNX export use the real video layout.

## Results (this trial)

Dataset: 800 episodes (250 mixed + 50 side + 500 pack), 641/159 episode split.
Trained 40 epochs on RTX 4060; best checkpoint is epoch 27 by HT-recall + STOP-F1.

| model | MSE | HT-recall | HT-F1 | dir-acc | STOP-F1 |
|---|---|---|---|---|---|
| copy `H(t+1s)=H(t)` val-all | 0.0731 | **0.620** | **0.619** | **0.726** | **0.774** |
| copy val-rare | 0.0869 | 0.627 | 0.627 | 0.719 | 0.780 |
| RS-JEPA val-all (epoch 27) | **0.0674** | 0.549 | 0.546 | 0.663 | 0.625 |
| RS-JEPA val-rare | **0.0724** | 0.548 | 0.572 | 0.647 | 0.667 |
| RS-JEPA hysteresis STOP | — | — | — | — | 0.617 |

Copy still wins the safety metrics (it cheats with the true current heatmap). The shuffle student **beats copy on MSE** and does **not** collapse to a fat center blob (old Collision-JEPA student predicted center ~0.85 vs GT ~0.70). Mean predicted 5×5 follows the GT layout: center ~0.42 vs GT 0.38, sides ~0.20 vs 0.19.

What this says about the method: region shuffle + 180° opposite-cell rotation is a usable training signal for local looming. Under-confidence on cells with GT ≥ 0.5 (pred mean ~0.46) is why recall lags copy. Next knobs, still inside this method: lower `jepa.fp_weight`, or a two-phase schedule (identity-heavy warmup, then shuffle).
