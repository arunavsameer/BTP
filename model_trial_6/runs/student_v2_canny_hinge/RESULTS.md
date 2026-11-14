# CNN Canny hinge — RESULTS

**Run:** `model_trial_6/runs/student_v2_canny_hinge`  
**Finished:** 2026-11-14 20:49 IST · `.success` · 30 epochs  
**Cmd:** `train_cnn_edge.py --epochs 30 --batch 4 --accum 8 --lr 3e-4 --device cuda --num-workers 2 --eval-test --run-name student_v2_canny_hinge`  
**PID:** 258724 (sole GPU job after Atto + first CNN were stopped)

## Official selection (`rank_checkpoint`: SAFE ≥ 0.85, then max STOP)

**Best epoch: 20** · ckpt `checkpoint_best.pt`

| Split | SAFE | STOP | fdec_cos | L_feat | collapse |
|---|---|---|---|---|---|
| Selection ep20 | 0.864 | 0.327 | 0.785 | 0.556 | — |
| **Held-out test** (best ckpt) | **0.890** | **0.284** | **0.786** | — | **0.637** |

Test also: stop_precision 0.475 · gt_heatmap_mse 0.034 · parent_heatmap_mae 0.062 · fdec_l1 0.785  
Windows kept **(22680, 5120, 5120)**. decoder_hash `77f78c8714667617`. GPU ~0.58 GB.

## Notes

Hinge (w_stop=1, w_safe=1, margin 0.05) drove early SAFE collapse (ep2 SAFE=1, STOP=0). STOP never matched the **non-hinge** run’s ep18 selection STOP **0.454** (that run was killed mid-ep20). Collapse 0.64 on test is high.

Log: `run.log` · metrics: `final_metrics.json`
