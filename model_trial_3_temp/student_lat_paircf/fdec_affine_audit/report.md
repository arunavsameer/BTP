# Epoch-38 F_dec affine calibration audit

Train-only OLS `F'_c = a_c F_c + b_c`. Frozen Pair-CF decoder. Selection eval. No test.

## Feature match (student F vs cached F_dec)

| split | cosine mean | L1 | ||F_s|| mean | ||F_t|| mean | corr mean | a mean |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.8857 | 0.5985 | 340.3 | 398.5 | 0.755 | 1.027 |
| selection | 0.8251 | 0.6980 | 326.1 | 388.6 | 0.586 | — |

Per-channel (train):
- a: mean 1.027 p10 0.959 p90 1.092
- b: mean 0.1739 p10 0.1022 p90 0.2441
- corr: mean 0.755; <0.2: 0/128; >0.8: 30/128

## Selection operating point 0.25/0.60

| | SAFE rec | THREAT rec | STOP rec | STOP prec | pre-STOP | post-STOP |
|---|---:|---:|---:|---:|---:|---:|
| parent | 0.881 | 0.347 | 0.800 | 0.684 | 0.722 | 0.928 |
| epoch38 original | 0.883 | 0.227 | 0.608 | 0.590 | 0.524 | 0.746 |
| epoch38 calibrated | 0.905 | 0.234 | 0.523 | 0.660 | 0.449 | 0.645 |

Peak distributions (heatmap max):

| model | GT-SAFE mean/p50 | GT-THREAT mean/p50 | GT-STOP mean/p50 | GT-STOP frac<0.25 |
|---|---|---|---|---:|
| original | 0.097/0.018 | 0.279/0.172 | 0.574/0.688 | 0.303 |
| calibrated | 0.076/0.012 | 0.226/0.100 | 0.514/0.627 | 0.347 |
| parent | 0.097/0.019 | 0.418/0.474 | 0.737/0.780 | 0.095 |

## Recommendation

Calibration deltas on selection: STOP -0.085, SAFE +0.022, THREAT +0.007, post-STOP -0.101.
