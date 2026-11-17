# CollisionStudentV2 int8 weight dump

Independent copy of the V2 student for later Android use. Source (read-only): `/storage/BTP/model_trial_4/ckpts/checkpoint_epoch_22.pt` → **`CollisionStudentV2` + `HeatmapDecoder`**.

## What this is

**Weight-only** symmetric int8 (scale from min/max ≈ max|w|/127, zero_point 0):

- Conv3d / Conv2d / Linear **weights**: per-out-channel scale (`axis=0`)
- Biases and GN/LN affine: per-tensor scale

Python `infer_i8.py` **dequantizes to fp32** (optional fp16) and runs the original modules. This is **not** fused int8 GEMM, QNNPACK quantized conv, QAT, or calibrated PTQ — there was no local calibration RGB cache used.

Torch dynamic quantization (`torch.ao.quantization.quantize_dynamic` on Linear) was skipped as the student is mostly Conv3d/Conv2d; a portable int8+scale dump is easier for Android later.

## Layout

- `student_i8/` — same V2 architecture as F16, plus `quant.py`
- `scripts/export_i8.py` / `scripts/infer_i8.py`
- `ckpts/checkpoint_epoch_22_i8.pt`

Packed tensor record: `{q: int8, scale: fp32, zero_point: int32, axis: 0|None, shape}`.

## Tensor I/O (same as F16)

Input `[B,16,3,256,256]` in [0,1]; offsets **[-90..0 step 6]**; letterbox 256; ImageNet norm inside the model; `Hhat_1s` is `[B,5,5]`; STOP peak ≥ **0.60**. Binary overlay is **not** required.

```bash
PY=/storage/BTP/impl/venv/bin/python
$PY /storage/BTP/model_trial_4_I8/scripts/export_i8.py
$PY /storage/BTP/model_trial_4_I8/scripts/infer_i8.py
```

## Android notes

Consume `backbone_q` / `decoder_q` int8 tensors + scales; dequant or bind to NNAPI/TFLite per-channel affine. Do not expect a ready `.tflite` here.
