# CollisionStudentV2 float16 inference

Self-contained student package for mobile-ready **fp16 weights**. Source checkpoint (read-only): `/storage/BTP/model_trial_4/ckpts/checkpoint_epoch_22.pt`.

Identified class: **`CollisionStudentV2`** + frozen **`HeatmapDecoder`** (`z=128`, dilated 16×16 → AdaptiveMaxPool 5×5). Keys in the source blob: `backbone`, `decoder` (plus `epoch=22`, `audit`, optimizer — optimizer is **not** copied). No BatchNorm; stem uses GroupNorm / LayerNorm / GELU / Conv3d / Conv2d / Linear (MHSA).

## Layout

- `student_f16/` — V2 backbone, decoder, letterbox, 16-frame offsets, checkpoint loader
- `scripts/export_f16.py` — convert all floating tensors to `torch.float16`
- `scripts/infer_f16.py` — load + dummy forward
- `ckpts/checkpoint_epoch_22_f16.pt` — converted weights

Not included: viz, caches, datasets, logs, reports, `student_runs`, RGB caches, videos.

## Tensor I/O

| Item | Spec |
| --- | --- |
| Input | `[B, 16, 3, 256, 256]` float in **[0, 1]** (`B,T,C,H,W`) |
| Sampling | 16 frames, offsets **[-90, -84, …, 0]** (step 6), clamp to clip bounds |
| Resize | **letterbox** to 256, pad 0 (keep FOV; not center-crop) |
| Norm | **ImageNet** mean/std inside `CollisionStudentV2.normalize` |
| Output `Hhat_1s` | `[B, 5, 5]` sigmoid probabilities (1s heatmap) |
| STOP | peak ≥ **0.60** (THREAT ≥ 0.25; else SAFE) |
| Overlay | **not required** |

Internal permute: `B,T,C,H,W` → `B,C,T,H,W` after normalize.

## Convert / run

```bash
PY=/storage/BTP/impl/venv/bin/python
$PY /storage/BTP/model_trial_4_F16/scripts/export_f16.py
$PY /storage/BTP/model_trial_4_F16/scripts/infer_f16.py
```

On **CUDA**, inference uses `model.half()` and float16 inputs. On **CPU**, many Conv3d kernels lack fp16; the script keeps **fp16 weights on disk** and falls back to fp32 compute (`--compute fp32`).

## Dtype choice

All saved parameters (conv/linear weights, biases, GroupNorm/LayerNorm affine, recency bias, LayerScale gamma) are **fp16**. There are no integer buffers in this student. Non-persistent buffers (`_mean`, `_std`, 2D sincos `pos`) are rebuilt at init and follow `model.half()` / `.float()`.
