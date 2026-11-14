# ConvNeXt-V2 Atto — stopped early

User stopped the run 2026-11-14 ~13:30 IST (PID 105207 SIGTERM). **Not** a completed 30-epoch job. Do not resume.

| | |
|---|---|
| Recipe | batch 4 / accum 8, lr 3e-4, 4060, `--no-grad-checkpoint` |
| RGB | `/storage/BTP/model_trial_5/shared_rgb_cache_256_v1` (223 samples, extras 207–229 kept) |
| Parent | trial-3 `parent_target_cache_v2` + `pair_cf_treatment_s1337_best.pt` (decoder `77f78c8714667617`) |
| Last completed epoch | **24** / 30 (epoch 25 had started, ~batch 1000) |
| Peak GPU | 2.34 GB |
| Log | `run.log` |

Official selection: SAFE ≥ 0.85, then max STOP. Thresholds SAFE peak `<0.25`, STOP `≥0.60`.

## Official best: epoch 16 (`checkpoint_best.pt`)

Written 10:24 IST; matches `metrics_epoch_16.json` and `checkpoint_epoch_16.pt`.

| Metric | Selection (5120) |
|---|---|
| **SAFE recall** | **0.864** |
| **STOP recall** | **0.569** |
| STOP precision | 0.558 |
| `fdec_cos` | **0.812** |
| `fdec_l1` | 0.747 |
| train `L_feat` | **0.581** |
| train `L_total` | 0.722 |
| train cosine | 0.820 |
| GT heatmap MSE | 0.031 |
| parent heatmap MAE | 0.050 |
| collapsed STOP frac | 0.329 |
| pre-/post-fork STOP | 0.493 / 0.694 |

Ckpt: `/storage/BTP/model_trial_5/student_runs/convnextv2_atto_v1/checkpoint_best.pt`

## Other selection snapshots (eval every 2)

| Epoch | SAFE | STOP | fdec_cos | L_feat |
|---|---:|---:|---:|---:|
| 1 | 0.000 | 0.000 | 0.716 | 0.771 |
| 4 | 0.571 | 0.465 | 0.746 | 0.697 |
| 6 | 0.631 | 0.449 | 0.773 | 0.671 |
| 8 | 0.624 | **0.635** (best STOP; SAFE below 0.85) | 0.784 | 0.647 |
| 10 | 0.831 | 0.532 | 0.798 | 0.625 |
| 12 | 0.845 | 0.534 | 0.801 | 0.608 |
| 14 | 0.849 | 0.521 | 0.809 | 0.593 |
| **16** | **0.864** | **0.569** | **0.812** | **0.581** |
| 18 | 0.858 | 0.533 | 0.814 | 0.572 |
| 20 | 0.871 | 0.521 | 0.816 | 0.563 |
| 22 | 0.874 | 0.554 | 0.818 | 0.556 |
| 24 | **0.897** (best SAFE) | 0.504 | 0.820 | 0.552 |

Later epochs got safer and a slightly better `F_dec` cosine, but STOP recall fell. Collapse on GT-STOP rose (0.33 at ep16 → 0.39 at ep24). SAFE never stayed coupled with a strong STOP floor; user judged this not learning enough.

Last train (ep24): `L_total` 0.670, `L_feat` 0.552, train_cos 0.836.
