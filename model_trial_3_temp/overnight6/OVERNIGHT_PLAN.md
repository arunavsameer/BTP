# Overnight six-model Pair-CF students

Shared: 22,680 train / 5,120 selection only. Test never loaded as tensors.
RGB `shared_rgb_cache_256_v1`, parent `parent_target_cache_v2`.
Decoder init `77f78c8714667617` independently per experiment.
Student weights random (seeds 1001–1006). Epochs **30**, batch **8**, `lr_student=3e-4`.

## Decoder schedule (all six)

| Epochs | Decoder | LR multiplier | LR at 3e-4 |
|---|---|---:|---:|
| 1–8 | frozen | 0 | 0 |
| 9–10 | unfreeze | 0.001 | 3e-7 |
| 11–12 | | 0.002 | 6e-7 |
| 13–14 | | 0.005 | 1.5e-6 |
| 15–16 | | 0.01 | 3e-6 |
| 17–20 | | 0.02 | 6e-6 |
| 21–25 | | 0.03 | 9e-6 |
| 26–30 | | 0.05 | 1.5e-5 |

`L_decoder_reg = mean((θ-θ0)²)`, `λ=0.01`. Separate decoder AdamW.

## Models

1. STOP-floor + occupied GT + selective KD + latent (identity 1×1, no aux)
2. Ordinal peak margins + latent
3. Hard occupied-cell mining + STOP-floor + empty penalty
4. Parent-correct STOP recovery weighting
5. Dual temporal (last-4 + full-16) gated fusion
6. Task-aware + aux on **detached** features; phase A until epoch 16 **and** selection cosine ≥0.82 and SAFE≥0.40 (failsafe epoch 20); STOP-floor kept; focal not applied to encoder features

## Estimates

- ~190 s/epoch (train+selection) × 30 × 6 ≈ **9.5 h**
- GPU mem ~ similar to prior student_lat batch-8 (≈6–10 GB)
- Disk ~ 30 ckpts × ~8 MB × 6 ≈ **1.5 GB**
