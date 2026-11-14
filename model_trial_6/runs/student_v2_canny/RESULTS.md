# CollisionStudentV2 + Canny (`student_v2_canny`) — incomplete

PID 221242 died mid **epoch 20 / 30** (log mtime 16:25 IST; semaphore-leak shutdown, no `.failed`). GPU was then taken by `student_v2_canny_hinge` (PID 258724). **No second CUDA job started.** This file records the first CNN run only.

| | |
|---|---|
| Recipe | batch 4 / accum 8, lr 3e-4, Canny on, 4060 |
| RGB | `/storage/BTP/model_trial_5/shared_rgb_cache_256_v1` |
| Parent | trial-3 `parent_target_cache_v2` |
| Last completed epoch | **19** |
| Peak GPU | 0.58 GB |
| Log | `run.log` |

Official rank: SAFE ≥ 0.85 then max STOP.

## Best under official rule: epoch 18

| Metric | Selection |
|---|---|
| SAFE | **0.851** |
| STOP | **0.454** |
| fdec_cos | 0.799 |
| train L_feat | 0.505 |

Ckpt: `checkpoint_epoch_18.pt` (and `checkpoint_best.pt` if the trainer had selected it).

## Selection snapshots

| Epoch | SAFE | STOP | fdec_cos | L_feat |
|---|---:|---:|---:|---:|
| 1 | 0.000 | 0.000 | 0.672 | 0.920 |
| 4 | 0.565 | 0.326 | 0.749 | 0.681 |
| 6 | **0.857** | 0.263 | 0.774 | 0.616 |
| 8 | 0.799 | 0.357 | 0.784 | 0.581 |
| 10 | 0.808 | 0.321 | 0.786 | 0.558 |
| 12 | 0.722 | **0.463** | 0.789 | 0.540 |
| 14 | **0.891** | 0.338 | 0.795 | 0.525 |
| 16 | 0.852 | 0.386 | 0.796 | 0.514 |
| 18 | 0.851 | 0.454 | 0.799 | 0.505 |

SAFE/STOP still not jointly strong (same pattern as Atto). Live follow-up is `../student_v2_canny_hinge/`.
