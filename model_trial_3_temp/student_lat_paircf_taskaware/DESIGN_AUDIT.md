# Task-aware student_lat design audit

## Decoder interface (unchanged)
- Pair-CF H1s consumes `F` **[B,128,16,16]** then pools to **[B,5,5]** occupancy.
- No bottleneck, LayerNorm, task_whiten, or Z_t on this path.
- Identity 1×1 is shape-preserving; at init `F_proj == F_ctx`.

## Tensors
- RGB cache uint8 256 letterbox → [0,1] → ImageNet in the model.
- `F_ctx` encoder [B,128,16,16]; `F` = proj(`F_ctx`) matched to cached `F_dec`.
- Heatmap GT is group-mean `gt_1s` [B,5,5], not `H_teacher`.
- Aux logits [B,3] from GAP(`F_ctx`); not decoder outputs.

## Gradient path
- Decoder params frozen (`requires_grad=False`) but **not** under `no_grad`.
- Heatmap / peak / ranking losses backprop through decoder into 1×1 and encoder.
- Latent L1/cos/rel/MGD on `F` / MGD=`proj(predictor(mask·F_ctx))`.
- Focal only through aux head → `F_ctx`.

## Incompatibilities flagged
- None vs Pair-CF decoder contract.

{
  "params": {
    "total": 988622,
    "trainable": 940813,
    "frozen": 47809,
    "encoder": 881290,
    "predictor": 34560,
    "proj_1x1": 16512,
    "aux_head": 8451,
    "decoder": 47809
  },
  "cpu_latency_ms": 60.65721772611141,
  "decoder_hash": "77f78c8714667617",
  "incompatibilities": []
}
