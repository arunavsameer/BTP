# Trial-6 CNN loss change (two-sided pmax hinges)

**Run:** `runs/student_v2_canny_hinge` (replaces `student_v2_canny` after ep18: SAFE≈0.85, STOP≈0.45).

## Why the old run could not lift both metrics

SAFE/STOP are **clip-level**: `pmax = max(Ĥ 5×5)`, then SAFE if `pmax < 0.25`, STOP if `pmax ≥ 0.60`, vs the same cuts on `gt_1s`.

Train mix is **74% SAFE / 16% STOP**. High SAFE + low STOP means the student **under-fires**: real STOP clips never clear 0.60. A SAFE-only hinge (or a dominant 5×5 focal mean over 24 cold cells) just pushes the whole map down, which raises SAFE and **hurts** STOP — that is what happened on Atto/CNN.

Old trial-6 already had imported peak hinges, but focal weight was 1.0, STOP hinge was 0.25, SAFE hinge used margin 0. The heatmap term could still flatten peaks.

## Terms now

Clip-level, **group-mean** (not batch-mean). If a batch has 0 STOP clips, that term is skipped that step. Same for SAFE. Equal weights on the two group means so the SAFE majority cannot drown STOP.

| Term | Weight | What |
|---|---|---|
| `L_stop_hinge` | 1.0 | STOP clips (`gt pmax ≥ 0.60`): `ReLU((0.60 + 0.05) - pmax)` — push the peak **up** through 0.60 |
| `L_safe_hinge` | 1.0 | SAFE clips (`gt pmax < 0.25`): `ReLU(pmax - (0.25 - 0.05))` — push the peak **down** through 0.25 |
| `L_fp` | 0.25 | SAFE clips with any cell ≥ 0.60: extra `ReLU(pmax - 0.60)` (false STOP) |
| Focal 5×5 vs `H_teacher` | **0.25** (was 1.0) | Still matches the teacher map, but down-weighted. Teacher-hot cells (`≥ 0.60`) use their **own mean** so one STOP cell is not drowned by 24 background cells |
| Cosine + 0.5 SmoothL1 on `F_dec` | 0.5 | Student stays aligned to the teacher latent (does not only shrink heatmaps) |
| Whitened L1 on `F_dec` | 0.25 | Same, using trial-5 channel stats |

Not added: Gram, blob, direction, full teacher dump, Z16 BCE.

Logged every interval: `L_stop_hinge`, `L_safe_hinge` (plus `n_stop` / `n_safe` in the batch).

## Why both metrics can rise

The two hinges are **class-conditional**. STOP clips are forced above 0.60; SAFE clips are forced below 0.20. They do not apply a global up/down bias. Feature matching keeps the frozen decoder in a teacher-like regime so the peaks are not invented from noise. Down-weighted, hot-split focal preserves teacher STOP support instead of averaging it away.
