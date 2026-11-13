# StudentLat-PairCF selection audit

Read-only. 5,120 official selection windows. Thresholds 0.25/0.60 unless noted.
Frozen Pair-CF H1s decoder. No test split. Epochs 12, 38, 62.

- Decoder hash: `77f78c8714667617`
- Device: `cuda:6`
- RGB cache: `/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/shared_rgb_cache_256_v1`
- Parent cache: `/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/parent_target_cache_v2`

## Epoch 12

- n=5120  GT-STOP=918  GT-THREAT=608  GT-SAFE=3594
- Student STOP recall **0.602**, SAFE **0.857**, THREAT **0.306**, STOP precision **0.583**
- Parent STOP recall **0.800**, SAFE **0.881**

### 1. GT-STOP confusion (parent vs student at 0.60)

| | student STOP | student non-STOP |
|---|---:|---:|
| parent STOP | 524 | 210 |
| parent non-STOP | 29 | 155 |

GT-STOP n=918. both-nonSTOP detail: `{'parent=THREAT|student=SAFE': 71, 'parent=SAFE|student=SAFE': 51, 'parent=SAFE|student=THREAT': 19, 'parent=THREAT|student=THREAT': 14}`

### 2. Peak-score distributions

| GT class | who | n | mean | p10 | p50 | p90 | max | ≥0.25 | ≥0.60 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GT-SAFE | gt | 3594 | 0.013 | 0.000 | 0.000 | 0.024 | 0.243 | 0.000 | 0.000 |
| GT-SAFE | parent | 3594 | 0.097 | 0.002 | 0.019 | 0.317 | 1.000 | 0.119 | 0.036 |
| GT-SAFE | student | 3594 | 0.122 | 0.002 | 0.027 | 0.452 | 1.000 | 0.143 | 0.078 |
| GT-THREAT | gt | 608 | 0.428 | 0.287 | 0.426 | 0.565 | 0.600 | 1.000 | 0.000 |
| GT-THREAT | parent | 608 | 0.418 | 0.050 | 0.474 | 0.687 | 0.948 | 0.694 | 0.347 |
| GT-THREAT | student | 608 | 0.314 | 0.028 | 0.246 | 0.656 | 0.949 | 0.495 | 0.189 |
| GT-STOP | gt | 918 | 0.857 | 0.637 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 |
| GT-STOP | parent | 918 | 0.737 | 0.286 | 0.780 | 0.999 | 1.000 | 0.905 | 0.800 |
| GT-STOP | student | 918 | 0.587 | 0.065 | 0.680 | 0.992 | 1.000 | 0.728 | 0.602 |

### 3. STOP recall/precision vs threshold (SAFE/caution still 0.25)

| stop_thr | STOP recall | STOP precision | SAFE recall | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.667 | 0.490 | 0.857 | 612 | 636 | 306 |
| 0.45 | 0.649 | 0.505 | 0.857 | 596 | 584 | 322 |
| 0.50 | 0.630 | 0.517 | 0.857 | 578 | 541 | 340 |
| 0.55 | 0.624 | 0.545 | 0.857 | 573 | 478 | 345 |
| 0.60 | 0.602 | 0.583 | 0.857 | 553 | 396 | 365 |
| 0.65 | 0.554 | 0.623 | 0.857 | 509 | 308 | 409 |
| 0.70 | 0.452 | 0.660 | 0.857 | 415 | 214 | 503 |

### 4. Student STOP misses (GT-STOP, student peak < 0.60)

- Count: **365**
- By fork: `{'pre': 280, 'post': 85}`
- By GT side: `{'CENTER': 218, 'LEFT': 86, 'RIGHT': 61}`
- By GT argmax cell: `{'r2_c2': 92, 'r3_c2': 68, 'r4_c2': 54, 'r3_c3': 32, 'r4_c3': 27, 'r0_c0': 24, 'r2_c1': 19, 'r4_c0': 14, 'r3_c1': 9, 'r3_c0': 8, 'r4_c1': 5, 'r2_c0': 4, 'r1_c2': 4, 'r0_c1': 2, 'r2_c3': 2, 'r1_c0': 1}`

Top sample_id miss counts:

- `sample_0073_shape_from_left`: 44
- `sample_0167_periph_child_cut`: 37
- `sample_0133_cube_from_right`: 36
- `sample_0121_child_darting`: 28
- `sample_0172_swerve_vehicle`: 26
- `sample_0097_cyclist_head_on`: 23
- `sample_0072_shape_head_on`: 22
- `sample_0077_car_cross_critical`: 22
- `sample_0095_shape_from_right`: 18
- `sample_0024_jaywalker_turn_toward`: 14
- `sample_0012_car_cut_in`: 12
- `sample_0109_swerve_vehicle`: 12
- `sample_0186_periph_car_turn`: 12
- `sample_0089_cube_head_on`: 10
- `sample_0135_shape_head_on`: 8
- `sample_0149_jaywalker_from_right`: 8
- `sample_0157_shape_from_left`: 8
- `sample_0169_jaywalker_from_left`: 7
- `sample_0019_periph_ped_cut`: 6
- `sample_0070_cube_from_right`: 4

### 5. Amplitude vs spatial

- Primary: **amplitude**
- Kind counts: `{'amplitude_underconfident': 100, 'collapse_low_amplitude': 250, 'spatial_localization': 6, 'spatial_and_weak': 9}`
- Fraction argmax Chebyshev ≤1: 0.8356164383561644
- Fraction same left/center/right side: 0.6575342465753424
- Mean peaks on misses: student 0.19875943954676797, parent 0.5743126438088613, GT 0.8045309824486301
- Amplitude: student blob near GT argmax but peak < 0.60 (or map collapsed <0.25). Spatial: argmax Chebyshev distance >= 2. Mixed otherwise.

## Epoch 38

- n=5120  GT-STOP=918  GT-THREAT=608  GT-SAFE=3594
- Student STOP recall **0.608**, SAFE **0.883**, THREAT **0.227**, STOP precision **0.590**
- Parent STOP recall **0.800**, SAFE **0.881**

### 1. GT-STOP confusion (parent vs student at 0.60)

| | student STOP | student non-STOP |
|---|---:|---:|
| parent STOP | 516 | 218 |
| parent non-STOP | 42 | 142 |

GT-STOP n=918. both-nonSTOP detail: `{'parent=THREAT|student=SAFE': 72, 'parent=SAFE|student=SAFE': 60, 'parent=THREAT|student=THREAT': 6, 'parent=SAFE|student=THREAT': 4}`

### 2. Peak-score distributions

| GT class | who | n | mean | p10 | p50 | p90 | max | ≥0.25 | ≥0.60 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GT-SAFE | gt | 3594 | 0.013 | 0.000 | 0.000 | 0.024 | 0.243 | 0.000 | 0.000 |
| GT-SAFE | parent | 3594 | 0.097 | 0.002 | 0.019 | 0.317 | 1.000 | 0.119 | 0.036 |
| GT-SAFE | student | 3594 | 0.097 | 0.002 | 0.018 | 0.348 | 1.000 | 0.117 | 0.072 |
| GT-THREAT | gt | 608 | 0.428 | 0.287 | 0.426 | 0.565 | 0.600 | 1.000 | 0.000 |
| GT-THREAT | parent | 608 | 0.418 | 0.050 | 0.474 | 0.687 | 0.948 | 0.694 | 0.347 |
| GT-THREAT | student | 608 | 0.279 | 0.016 | 0.172 | 0.660 | 0.984 | 0.438 | 0.211 |
| GT-STOP | gt | 918 | 0.857 | 0.637 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 |
| GT-STOP | parent | 918 | 0.737 | 0.286 | 0.780 | 0.999 | 1.000 | 0.905 | 0.800 |
| GT-STOP | student | 918 | 0.574 | 0.028 | 0.688 | 0.996 | 1.000 | 0.697 | 0.608 |

### 3. STOP recall/precision vs threshold (SAFE/caution still 0.25)

| stop_thr | STOP recall | STOP precision | SAFE recall | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.657 | 0.527 | 0.883 | 603 | 542 | 315 |
| 0.45 | 0.643 | 0.549 | 0.883 | 590 | 485 | 328 |
| 0.50 | 0.632 | 0.554 | 0.883 | 580 | 467 | 338 |
| 0.55 | 0.619 | 0.569 | 0.883 | 568 | 430 | 350 |
| 0.60 | 0.608 | 0.590 | 0.883 | 558 | 387 | 360 |
| 0.65 | 0.556 | 0.635 | 0.883 | 510 | 293 | 408 |
| 0.70 | 0.483 | 0.701 | 0.883 | 443 | 189 | 475 |

### 4. Student STOP misses (GT-STOP, student peak < 0.60)

- Count: **360**
- By fork: `{'pre': 272, 'post': 88}`
- By GT side: `{'CENTER': 223, 'LEFT': 89, 'RIGHT': 48}`
- By GT argmax cell: `{'r2_c2': 95, 'r3_c2': 74, 'r4_c2': 50, 'r2_c1': 24, 'r4_c3': 24, 'r3_c3': 22, 'r0_c0': 19, 'r4_c0': 14, 'r3_c0': 9, 'r3_c1': 9, 'r4_c1': 5, 'r2_c0': 4, 'r1_c2': 4, 'r1_c0': 2, 'r0_c1': 2, 'r2_c3': 2, 'r1_c1': 1}`

Top sample_id miss counts:

- `sample_0073_shape_from_left`: 44
- `sample_0167_periph_child_cut`: 36
- `sample_0121_child_darting`: 34
- `sample_0133_cube_from_right`: 28
- `sample_0097_cyclist_head_on`: 26
- `sample_0172_swerve_vehicle`: 24
- `sample_0077_car_cross_critical`: 22
- `sample_0072_shape_head_on`: 17
- `sample_0109_swerve_vehicle`: 17
- `sample_0089_cube_head_on`: 13
- `sample_0095_shape_from_right`: 13
- `sample_0186_periph_car_turn`: 12
- `sample_0012_car_cut_in`: 11
- `sample_0019_periph_ped_cut`: 10
- `sample_0135_shape_head_on`: 9
- `sample_0107_jaywalker_from_right`: 6
- `sample_0149_jaywalker_from_right`: 6
- `sample_0169_jaywalker_from_left`: 5
- `sample_0190_jaywalker_from_left`: 5
- `sample_0021_jaywalker`: 4

### 5. Amplitude vs spatial

- Primary: **amplitude**
- Kind counts: `{'amplitude_underconfident': 59, 'collapse_low_amplitude': 278, 'spatial_localization': 16, 'spatial_and_weak': 7}`
- Fraction argmax Chebyshev ≤1: 0.7611111111111111
- Fraction same left/center/right side: 0.6305555555555555
- Mean peaks on misses: student 0.15026567247178818, parent 0.5941544426812067, GT 0.8124593098958334
- Amplitude: student blob near GT argmax but peak < 0.60 (or map collapsed <0.25). Spatial: argmax Chebyshev distance >= 2. Mixed otherwise.

## Epoch 62

- n=5120  GT-STOP=918  GT-THREAT=608  GT-SAFE=3594
- Student STOP recall **0.576**, SAFE **0.890**, THREAT **0.176**, STOP precision **0.598**
- Parent STOP recall **0.800**, SAFE **0.881**

### 1. GT-STOP confusion (parent vs student at 0.60)

| | student STOP | student non-STOP |
|---|---:|---:|
| parent STOP | 496 | 238 |
| parent non-STOP | 33 | 151 |

GT-STOP n=918. both-nonSTOP detail: `{'parent=THREAT|student=SAFE': 72, 'parent=SAFE|student=SAFE': 56, 'parent=SAFE|student=THREAT': 14, 'parent=THREAT|student=THREAT': 9}`

### 2. Peak-score distributions

| GT class | who | n | mean | p10 | p50 | p90 | max | ≥0.25 | ≥0.60 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GT-SAFE | gt | 3594 | 0.013 | 0.000 | 0.000 | 0.024 | 0.243 | 0.000 | 0.000 |
| GT-SAFE | parent | 3594 | 0.097 | 0.002 | 0.019 | 0.317 | 1.000 | 0.119 | 0.036 |
| GT-SAFE | student | 3594 | 0.092 | 0.002 | 0.015 | 0.329 | 1.000 | 0.110 | 0.065 |
| GT-THREAT | gt | 608 | 0.428 | 0.287 | 0.426 | 0.565 | 0.600 | 1.000 | 0.000 |
| GT-THREAT | parent | 608 | 0.418 | 0.050 | 0.474 | 0.687 | 0.948 | 0.694 | 0.347 |
| GT-THREAT | student | 608 | 0.269 | 0.015 | 0.133 | 0.676 | 0.996 | 0.380 | 0.204 |
| GT-STOP | gt | 918 | 0.857 | 0.637 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 |
| GT-STOP | parent | 918 | 0.737 | 0.286 | 0.780 | 0.999 | 1.000 | 0.905 | 0.800 |
| GT-STOP | student | 918 | 0.559 | 0.020 | 0.680 | 0.996 | 1.000 | 0.683 | 0.576 |

### 3. STOP recall/precision vs threshold (SAFE/caution still 0.25)

| stop_thr | STOP recall | STOP precision | SAFE recall | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.636 | 0.529 | 0.890 | 584 | 520 | 334 |
| 0.45 | 0.621 | 0.542 | 0.890 | 570 | 481 | 348 |
| 0.50 | 0.601 | 0.554 | 0.890 | 552 | 444 | 366 |
| 0.55 | 0.587 | 0.575 | 0.890 | 539 | 398 | 379 |
| 0.60 | 0.576 | 0.598 | 0.890 | 529 | 356 | 389 |
| 0.65 | 0.534 | 0.648 | 0.890 | 490 | 266 | 428 |
| 0.70 | 0.472 | 0.689 | 0.890 | 433 | 195 | 485 |

### 4. Student STOP misses (GT-STOP, student peak < 0.60)

- Count: **389**
- By fork: `{'pre': 288, 'post': 101}`
- By GT side: `{'CENTER': 246, 'LEFT': 85, 'RIGHT': 58}`
- By GT argmax cell: `{'r2_c2': 101, 'r3_c2': 79, 'r4_c2': 61, 'r3_c3': 30, 'r4_c3': 24, 'r2_c1': 21, 'r0_c0': 17, 'r4_c0': 14, 'r3_c0': 10, 'r3_c1': 10, 'r4_c1': 6, 'r2_c0': 4, 'r1_c2': 4, 'r1_c0': 2, 'r0_c3': 2, 'r2_c3': 2, 'r1_c1': 1, 'r0_c2': 1}`

Top sample_id miss counts:

- `sample_0073_shape_from_left`: 44
- `sample_0167_periph_child_cut`: 38
- `sample_0133_cube_from_right`: 36
- `sample_0121_child_darting`: 34
- `sample_0097_cyclist_head_on`: 26
- `sample_0172_swerve_vehicle`: 26
- `sample_0077_car_cross_critical`: 22
- `sample_0072_shape_head_on`: 17
- `sample_0024_jaywalker_turn_toward`: 14
- `sample_0089_cube_head_on`: 14
- `sample_0109_swerve_vehicle`: 13
- `sample_0135_shape_head_on`: 12
- `sample_0186_periph_car_turn`: 12
- `sample_0012_car_cut_in`: 11
- `sample_0019_periph_ped_cut`: 10
- `sample_0157_shape_from_left`: 10
- `sample_0063_jaywalker`: 9
- `sample_0149_jaywalker_from_right`: 9
- `sample_0095_shape_from_right`: 6
- `sample_0107_jaywalker_from_right`: 6

### 5. Amplitude vs spatial

- Primary: **amplitude**
- Kind counts: `{'amplitude_underconfident': 84, 'collapse_low_amplitude': 291, 'spatial_and_weak': 7, 'spatial_localization': 7}`
- Fraction argmax Chebyshev ≤1: 0.7892030848329049
- Fraction same left/center/right side: 0.6735218508997429
- Mean peaks on misses: student 0.15723793794685895, parent 0.6005386313313383, GT 0.8177568685732648
- Amplitude: student blob near GT argmax but peak < 0.60 (or map collapsed <0.25). Spatial: argmax Chebyshev distance >= 2. Mixed otherwise.

