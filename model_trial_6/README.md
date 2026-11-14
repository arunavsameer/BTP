# model_trial_6 — CollisionStudentV2 + Canny

Student maps 16 letterboxed 256 frames with **Canny cyan-on-photo** (order a: after the 256 cache) through a pyramid stem / causal motion / dual readout / MHSA predictor onto parent `F_dec [128,16,16]`, then the **frozen** Pair-CF heatmap decoder.

See `ARCHITECTURE.md` for shapes, Canny, and loss.

Caches (read-only, not copied):

- RGB: `/storage/BTP/model_trial_5/shared_rgb_cache_256_v1`
- Parent: `/storage/BTP/model_trial_3/checkpoints/parent_target_cache_v2`

Do not train on GPU while Atto is running. Launch with `./run_cnn_edge_local.sh` after Atto `RESULTS.md` exists.
