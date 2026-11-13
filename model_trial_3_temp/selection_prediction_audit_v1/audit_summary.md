# Selection prediction audit v1 (no training, no test)

n=5120 selection windows. Official STOP≥0.60, SAFE<0.25. Checkpoints: M1 ep10, M2 ep19, M3 ep21, M4 ep17, M5 ep17.

## Parity vs published metrics

| Model | Epoch | Audit SAFE | Δ | STOP | Δ | prec | pre | post | match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M1 | 10 | 0.868 | -0.000 | 0.611 | +0.000 | 0.537 | 0.545 | 0.720 | True |
| M2 | 19 | 0.855 | +0.000 | 0.419 | +0.000 | 0.621 | 0.327 | 0.572 | True |
| M3 | 21 | 0.877 | +0.000 | 0.499 | -0.000 | 0.633 | 0.432 | 0.610 | True |
| M4 | 17 | 0.865 | +0.000 | 0.522 | -0.000 | 0.557 | 0.451 | 0.639 | True |
| M5 | 17 | 0.865 | -0.000 | 0.559 | -0.000 | 0.670 | 0.476 | 0.697 | True |

## A. Threshold sweep (SAFE thr fixed 0.25)

**No single-model threshold meets SAFE≥0.82 and STOP≥0.70.** Closest with SAFE≥0.82 (max STOP):

| Model | STOP thr | SAFE | THR | STOP | prec | GT-SAFE→STOP |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.4 | 0.868 | 0.033 | 0.641 | 0.487 | 0.114 |
| M2 | 0.4 | 0.855 | 0.153 | 0.477 | 0.532 | 0.083 |
| M3 | 0.4 | 0.877 | 0.201 | 0.546 | 0.530 | 0.077 |
| M4 | 0.4 | 0.865 | 0.082 | 0.582 | 0.510 | 0.097 |
| M5 | 0.4 | 0.865 | 0.194 | 0.629 | 0.548 | 0.087 |

## B. Ranking (peaks)

| Model | STOP vs SAFE ROC | STOP vs rest ROC | STOP PR-AUC | THREAT vs SAFE ROC |
|---|---:|---:|---:|---:|
| M1 | 0.836 | 0.818 | 0.586 | 0.729 |
| M2 | 0.796 | 0.782 | 0.502 | 0.674 |
| M3 | 0.845 | 0.821 | 0.568 | 0.765 |
| M4 | 0.837 | 0.818 | 0.556 | 0.719 |
| M5 | 0.864 | 0.843 | 0.607 | 0.774 |

Peak stats GT-STOP / GT-SAFE (mean, p10, p50):

- M1 STOP mean=0.584 p10=0.012 p50=0.775; SAFE mean=0.114 p90=0.512
- M2 STOP mean=0.472 p10=0.068 p50=0.361; SAFE mean=0.145 p90=0.350
- M3 STOP mean=0.521 p10=0.027 p50=0.598; SAFE mean=0.107 p90=0.312
- M4 STOP mean=0.525 p10=0.013 p50=0.641; SAFE mean=0.109 p90=0.387
- M5 STOP mean=0.565 p10=0.045 p50=0.672; SAFE mean=0.115 p90=0.346

## C. Pre/post fork

GT-STOP: pre-fork n=572, post-fork n=346.

| Model | Pre rec | Post rec | Pre mean peak | Post mean peak | Pre p10 | Post p10 |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.545 | 0.720 | 0.504 | 0.717 | 0.011 | 0.027 |
| M2 | 0.327 | 0.572 | 0.393 | 0.604 | 0.067 | 0.078 |
| M3 | 0.432 | 0.610 | 0.454 | 0.632 | 0.026 | 0.028 |
| M4 | 0.451 | 0.639 | 0.445 | 0.658 | 0.011 | 0.024 |
| M5 | 0.476 | 0.697 | 0.480 | 0.704 | 0.040 | 0.070 |

Unanimous STOP misses: **275** / 918 (30.0%). Unanimous hits: 310.
Unanimous-miss pre-fork: 202 post-fork: 73.
Top sample_ids among unanimous misses: sample_0073_shape_from_left:40, sample_0133_cube_from_right:36, sample_0097_cyclist_head_on:24, sample_0172_swerve_vehicle:24, sample_0167_periph_child_cut:22, sample_0072_shape_head_on:15, sample_0077_car_cross_critical:15, sample_0095_shape_from_right:14
Direction of GT peak among STOP: {"LEFT": {"n": 218, "M1_rec": 0.591743119266055, "M5_rec": 0.5825688073394495, "unanimous_miss": 69}, "CENTER": {"n": 606, "M1_rec": 0.6501650165016502, "M5_rec": 0.5907590759075908, "unanimous_miss": 154}, "RIGHT": {"n": 94, "M1_rec": 0.40425531914893614, "M5_rec": 0.2978723404255319, "unanimous_miss": 52}}

## D. M1 vs M5 overlap @0.60

- both: n=468 (51.0%) prefork=240 postfork=228
- M1_only: n=93 (10.1%) prefork=72 postfork=21
- M5_only: n=45 (4.9%) prefork=32 postfork=13
- neither: n=312 (34.0%) prefork=228 postfork=84
Exclusive detections (M1-only + M5-only) = 138 / 918 = 15.0% of GT-STOP.
n_models detecting STOP histogram: {0: 275, 1: 81, 2: 81, 3: 81, 4: 90, 5: 310}

## E. Spatial vs amplitude

{
  "M1": {
    "correct_cell_collapsed": 65,
    "near_col": 103,
    "wrong_row_same_col": 142,
    "correct_cell_low_amp": 15,
    "wrong_region": 32
  },
  "M2": {
    "correct_cell_low_amp": 64,
    "wrong_row_same_col": 209,
    "near_col": 129,
    "wrong_region": 45,
    "correct_cell_collapsed": 86
  },
  "M3": {
    "correct_cell_low_amp": 66,
    "wrong_row_same_col": 155,
    "near_col": 147,
    "wrong_region": 30,
    "correct_cell_collapsed": 62
  },
  "M4": {
    "correct_cell_low_amp": 49,
    "wrong_row_same_col": 142,
    "near_col": 114,
    "wrong_region": 42,
    "correct_cell_collapsed": 92
  },
  "M5": {
    "correct_cell_collapsed": 59,
    "wrong_row_same_col": 146,
    "near_col": 116,
    "wrong_region": 23,
    "correct_cell_low_amp": 61
  }
}

## F. Ensembles

**No peak/heatmap ensemble + threshold meets both targets.** Best SAFE≥0.82 ensemble:
{'combo': 'w25', 'stop_thr': 0.4, 'SAFE': 0.8708959376739009, 'THR': 0.18914473684210525, 'STOP': 0.6427015250544662, 'STOP_prec': 0.5728155339805825, 'safe_as_stop': 0.07679465776293823, 'frac_pred_stop': 0.201171875}

## G. Should we run M6?

See final response. Ranking AUCs are moderate-to-strong (STOP vs SAFE ROC typically >0.9 if well separated); fixed 0.60 is in a dense overlap region (STOP p10 often <0.25–0.45). Pre-fork is systematically weaker. M1/M5 exclusive hits exist but joint misses remain large. Spatial misses are mostly low amplitude on the correct/nearby cell.
