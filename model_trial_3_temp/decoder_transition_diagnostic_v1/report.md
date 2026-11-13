# Decoder transition diagnostic

Selection only. No training. No test. Original Pair-CF decoder hash `77f78c8714667617` (verified).

Artifacts: CSVs and `figures/` in this directory.

## What was compared

For M1 / M7 / M8 at epochs **8, 9, 10, 14, official, 30**, the same selection clips were encoded once. Those student features `F` were decoded with:

1. the decoder **inside the checkpoint** (“trained”);
2. the **original** Pair-CF decoder, evaluation-only, no parameter updates (“original”).

At epoch 8 the two decoders are identical (drift 0), as required.

## Prediction matrix (STOP / SAFE at 0.25 / 0.60)

| model | ep | decoder | SAFE | THR | STOP | prec | pre | post | RIGHT | collapse | p10 | STOP-vs-SAFE ROC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 8 | both | 0.588 | 0.211 | 0.708 | 0.358 | 0.668 | 0.775 | 0.435 | 0.197 | 0.090 | 0.806 |
| M1 | 9 | trained | 0.859 | 0.095 | 0.602 | 0.542 | 0.542 | 0.702 | 0.272 | 0.346 | 0.018 | 0.831 |
| M1 | 10 | trained | 0.868 | 0.117 | **0.611** | 0.537 | 0.545 | 0.720 | 0.413 | 0.338 | 0.012 | 0.836 |
| M7 | 8 | both | 0.000 | 0.768 | 0.427 | 0.335 | 0.339 | 0.572 | **0.098** | **0.000** | 0.486 | 0.643 |
| M7 | 9 | trained ≈ orig | 0.292 | 0.671 | 0.406 | 0.402 | 0.315 | 0.558 | 0.109 | 0.136 | 0.245 | 0.679 |
| M7 | 23 official | trained | 0.863 | 0.115 | 0.342 | 0.578 | 0.241 | 0.509 | 0.043 | 0.513 | 0.017 | 0.741 |
| M7 | 23 | original | 0.813 | 0.186 | 0.314 | 0.639 | 0.215 | 0.477 | 0.043 | 0.460 | 0.061 | 0.733 |
| M8 | 8 | both | 0.000 | 0.592 | 0.626 | 0.257 | 0.551 | 0.751 | 0.467 | **0.000** | 0.367 | 0.682 |
| M8 | 9 | trained ≈ orig | 0.237 | 0.613 | 0.473 | 0.393 | 0.381 | 0.624 | 0.207 | 0.154 | 0.243 | 0.703 |
| M8 | 20 official | trained | 0.850 | 0.174 | 0.326 | 0.533 | 0.233 | 0.480 | 0.120 | 0.516 | 0.028 | 0.767 |
| M8 | 20 | original | 0.824 | 0.217 | 0.300 | 0.535 | 0.215 | 0.439 | 0.120 | 0.487 | 0.057 | 0.761 |

Full rows: `prediction_matrix.csv`. Trained vs original STOP never differs by more than ~0.07, and **at the 8→9 transition they match to three decimals**.

## Features vs parent `F_dec`

| model | ep | cos all | cos GT-STOP | eff-dim PR | feat var | ep8-vs-ep9 self-cos |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 8→9 | 0.797→0.804 | 0.739→0.748 | 2.41→2.27 | 2.34→2.45 | **0.977** |
| M7 | 8→9 | 0.420→0.493 | 0.382→0.442 | 2.68→2.61 | 1.31→1.71 | **0.882** |
| M7 | 23 | 0.657 | 0.563 | **1.20** | 1.75 | — |
| M8 | 8→9 | 0.170→0.178 | 0.156→0.164 | 3.90→4.24 | **9.15→9.37** | **0.936** |
| M8 | 20 | 0.182 | 0.166 | 5.58 | 10.18 | — |

M7 becomes *more* parent-like after unfreeze; M8 never matches parent features.

## Decoder drift vs original Pair-CF

| ckpt | RMS param drift | mean relative L2 |
|---|---:|---:|
| M7/M8 ep8 | 0 | 0 |
| M7 ep9 | 2.4e-5 | ~0 |
| M8 ep9 | 2.2e-5 | ~0 |
| M7 official | 2.6e-3 | small |
| M7/M8 ep30 | ~5e-3 | similar to M1 ep30 (5.2e-3) |

No layer is an outlier at epoch 9 (`decoder_layer_drift.csv`). Functional change at 8→9 cannot be the 2e-5 weight update: swapping the original decoder back in does not restore STOP.

## Implementation (Investigation E)

Measured from `84_train_parent_task_student.py` + logs:

- All decoder parameters share one `requires_grad` flag. Ep 1–8: **g_dec = 0**. Ep 9: g_dec ≈ 0.90 (M7) / 0.81 (M8).
- Separate `opt_d`; `opt_d.step()` only if `decoder_lr_multiplier > 0`. Student AdamW never contains decoder params.
- Decoder LR at ep 9 is `3e-4 × 0.001 = 3e-7`. Student LR is on the cosine schedule (warmup ended at ep 5). **Unfreeze does not bump student LR.**
- `clip_grad_norm_(all parameters, 1.0)` is global after unfreeze (shared clip budget). Not a frozen-update leak.
- `HeatmapDecoder` has **no BatchNorm**. Loop keeps `decoder.eval()`.
- `L_dec_reg = 0.01` is always in the graph; with frozen params it does not move weights.

No evidence that frozen decoder weights were accidentally updated, or that the multiplier was applied to the student optimizer.

## Answers

1. **Do student features change substantially after epoch 8?**  
   **Moderately, not a collapse at the 8→9 step.** Self-cosine of `F` is 0.88 (M7) / 0.94 (M8) / 0.98 (M1). Parent-cosine **rises** for M7 (0.42→0.66 by official) and stays ~0.17 for M8. Later M7 effective dimensionality **does** collapse (2.7→1.2 by ep 23). That is a slow channel-collapse, not an 8→9 jump.

2. **Does the original decoder still produce useful STOP on later features?**  
   **No better than the trained decoder.** At every checkpoint, original ≈ trained. Official M7 STOP 0.342 (trained) vs 0.314 (original). M8 0.326 vs 0.300. Original decoder **does not** retain the early high-STOP operating point.

3. **Does the trained decoder suppress STOP?**  
   **Not specifically.** If it did, original-decoder STOP would stay high on the same `F`. It does not. Late trained decoder is even slightly *higher* STOP than original.

4. **Feature mismatch vs decoder adaptation?**  
   **Student features / calibration, not decoder adaptation.** The 8→9 STOP drop is already present when decoding ep-9 `F` with the untouched Pair-CF decoder. Decoder RMS drift at that step is 2e-5.  
   **Hypothesis (not a separate measurement of loss terms):** parent STOP/SAFE hinges plus FP push peaks across 0.60 while ranking still improves (ROC-AUC 0.64→0.74 on M7). That is amplitude/calibration at a fixed threshold, not a broken decoder.

5. **M7 vs M8?**  
   **Same failure class.** Both: SAFE=0 and high amplitude at ep 8; SAFE rises and STOP@0.60 falls after ep 8; original≈trained; RIGHT is weak (M7 RIGHT already 0.10 at frozen ep 8). Difference: M8 never matches `F_dec` (cos 0.17, variance ~9 vs M1 ~2.5). M7 latent term improves cosine but **does not prevent** the STOP@0.60 drop. High STOP in M7 ep 1–7 was with SAFE=0, so it was not a usable operating point.

6. **Optimizer / unfreeze bug?**  
   **No leak of decoder updates in epochs 1–8.** Unfreeze wiring matches the intended schedule. The coincidence of unfreeze with the SAFE/STOP trade is **confounded** with another student epoch under parent hinges, not explained by decoder weight change.

7. **Spatial vs amplitude?**  
   **Primarily amplitude / calibration.** GT-STOP p10: M7 0.49 (ep8) → 0.25 (ep9) → 0.017 (official); collapse 0 → 0.14 → 0.51. STOP-vs-SAFE ROC **rises** (0.64→0.74), so ranking is not destroyed. Argmax column match **improves** (M7 0.39→0.57). RIGHT STOP is already failed at frozen ep 8 for M7 (0.10); M8 RIGHT drops 0.47→0.21 at 8→9 on **both** decoders.

8. **Smallest next experiment (do not launch here):**  
   **Not “lower decoder LR” and not “permanent freeze as the sole change”** — this swap experiment already applies a frozen original decoder to later features and STOP still falls.  
   **Recommended:** keep the **original Pair-CF decoder frozen for all 30 epochs** and train the student with **parent `L_1s` + blob + FP + dir only** (`L_stop_hinge = L_safe_hinge = 0`). Same architecture, same selection rule. That isolates whether hinge-driven peak pushing is what moves mass across 0.60 while ROC is still improving. Optional readout (no new train): STOP recall vs threshold on existing checkpoints, because ROC says ranking remains.

## Decision

Cause supported by measurement: **thresholded STOP collapse is a student-feature / peak-calibration effect under the parent task objective, visible even with the original decoder.** Decoder unfreeze is temporally aligned with the SAFE recovery but **not** the mechanism (identical original-vs-trained matrices; tiny drift).

Cause not supported: decoder layers “forgetting” STOP; latent distillation as the unique failure (M8 matches M7).

Next experiment: **frozen original decoder + hinge-off parent task** (above). Do not run it in this diagnostic.
