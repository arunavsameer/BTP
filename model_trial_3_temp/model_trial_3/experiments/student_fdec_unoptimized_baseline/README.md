# Unoptimized student F_dec pilot baseline (preserved)

This directory archives the **stopped** first pilot attempt before the data-pipeline optimization patch.

## Contents

- `s0/` — partial S0 run (`checkpoint_initial.pt`, `config.json`); **no epoch metrics** (killed during slow epoch 1).
- `shared_backbone_init.pt` — Base concat backbone state (seed 1337) used to initialize both S0 and S1.
- `student_fdec_pilot_master.log` — RGB prepare progress only.

## Known unoptimized pipeline issues

- `letterbox_rgb(256→160)` on **every** `__getitem__` (5× per sample).
- S0 loaded unused `F_dec` [128,16,16] from mmap each sample.
- `num_workers=0`; epoch metrics written only after full dev eval.

Do not delete; optimized runs use the same `shared_backbone_init.pt` for a fair restart.
