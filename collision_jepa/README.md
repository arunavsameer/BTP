# Collision-JEPA (5x5)

A low-compute, low-latency collision-anticipation model for a wearable that helps a
blind user avoid obstacles. It predicts a **5x5 threat heatmap 1 second into the
future** from three cheap RGB frames, and turns that into a
`LEFT / CENTER / RIGHT` + `SAFE / CAUTION / STOP` warning.

The system is trained with a teacher/student (JEPA) scheme:

- **Teacher (offline, frozen):** a frozen V-JEPA-2 video encoder + a small learned
  spatial pool + bottleneck produces a compact collision state `Z in R^{5x5x16}`.
  A tiny decoder maps `Z -> H` (the 5x5 heatmap). We only ever run the teacher
  offline to cache the future collision state `Z+`.
- **Student (wearable):** a ~1M-param depthwise-separable CNN sees 3 downsampled
  frames, forms feature differences, builds its own `Z_t`, and predicts the future
  `Zhat+ = Z_t + P(...)` with a residual predictor. The frozen teacher decoder maps
  `Zhat+ -> Hhat+`.

Loss: `L = L_H + lambda * L_JEPA`, heatmap first.

## Why "collision-worthy only"
We never run an object detector. Focus on real hazards (cars, people, cyclists,
potholes) and not clutter (leaves, texture) comes from: (1) the dataset's 5x5 labels
already exclude clutter, (2) focal high-threat loss weighting, (3) V-JEPA-2 semantic
motion features that the student distills, and (4) keeping safe/empty episodes as
hard negatives to suppress false alarms.

## Layout
```
collision_jepa/            # python package
  config.py                # YAML config loader
  data/                    # unzip, split, video cache, dataset, augment
  models/                  # tiny_cnn, decoder, teacher, student, baseline
  losses.py metrics.py warning.py engine.py
configs/default.yaml       # all knobs
scripts/                   # 00..07 pipeline entrypoints
tests/                     # shape + hysteresis tests
```

## Pipeline
Run from the repo root (`e:\btp\collision_jepa`). A dedicated venv is used because
the system `python` (msys2) has no pip. On Python 3.13 the CUDA wheels live under
the `cu124` index (there are no `cu121` cp313 wheels):

```powershell
# 0. Create venv + install deps (CUDA build of torch for the RTX 2050)
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 1. Prepare data: unzip, split, cache frames + heatmaps
python scripts/00_prepare_data.py

# 2. The mandatory copy baseline: H(t+1s) = H(t)
python scripts/01_eval_copy_baseline.py

# 3. Stage 0: tiny CNN -> future 5x5 directly (no V-JEPA). Must beat copy.
python scripts/02_train_stage0.py

# 4. Stage A: frozen V-JEPA-2 teacher reconstructs H_t
python scripts/03_train_teacher.py

# 5. Stage B: cache Z_t and Z+ from the frozen teacher
python scripts/04_cache_teacher_z.py

# 6. Stage C: train the tiny student (L = L_H + 0.1 L_JEPA)
python scripts/05_train_student.py

# 7. Evaluate everything + export INT8
python scripts/06_eval.py
python scripts/07_export_onnx.py
```

Steps 1-3 run immediately on a 4 GB GPU with no large download. Steps 4-6 download
the V-JEPA-2 weights once.

## Results so far (smoke run on this machine)
Validated end-to-end on an RTX 2050 (4 GB), Python 3.13, torch 2.6 + cu124:

| model | HT-recall | HT-F1 | dir-acc | STOP-F1 |
|-------|-----------|-------|---------|---------|
| copy `H(t+1s)=H(t)` | **0.652** | 0.672 | 0.734 | **0.837** |
| stage0 (RGB only, 30 ep) | 0.522 | 0.550 | 0.603 | 0.661 |

Two important takeaways:
- **The copy baseline is very strong** because it is *privileged*: it uses the true
  current heatmap `H(t)`, and 1-second-ahead threat maps are highly autocorrelated.
  RGB-only models start at a disadvantage on the "nothing changed" majority of frames
  and must earn their keep on the frames where the scene *changes* (a car pulls out, a
  pedestrian steps in). This is exactly what the residual predictor + V-JEPA distillation
  (stages 3-5) target. Evaluate improvement specifically on changing/rare frames.
- **Latency is a non-issue for the student.** Exported fp32 ONNX runs in ~2.7 ms/frame
  single-threaded on CPU — already under the 5 ms goal — so `student_fp32.onnx` is the
  recommended artifact. Dynamic INT8 is *slower* here (depthwise-conv int8 kernels in
  onnxruntime), so it is opt-in via `python scripts/07_export_onnx.py --int8`.

Stages 3-5 (V-JEPA-2 teacher, Z caching, student distillation) are fully implemented
and shape-tested but require a one-time multi-GB download of the V-JEPA-2 weights, so
they are not part of the quick smoke run.

## Data facts
- 101 episodes, 150 frames each, 30 fps, 1920x1080 `preview.mp4`.
- Per-frame label is a `5x5` matrix in `[0,1]` (`spatial_annotations.json`, `k=5`).
- Rows are far -> near, columns are left -> right; the center corridor has a strong
  prior, so the **copy baseline** is the number to beat, especially on
  `NEAR_MISS` / `CRITICAL_THREAT` episodes.
