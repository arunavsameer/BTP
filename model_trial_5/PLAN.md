# Execution plan — Atto 30ep then trial-6 CNN (laptop only)

Depends on `PREFLIGHT.md`. Touch **only** `model_trial_5` and `model_trial_6`. Read-only elsewhere. Venv: `/storage/BTP/impl/venv` (do not modify). **This runner is the only process that starts GPU training.** No sbatch, no DGX, no extra scp unless the trial-5 RGB dest is broken (then rsync to that dest only).

Do **not** skip RGB clips 207–229. Do **not** start Atto until RGB copy is complete. Do **not** start CNN GPU training until Atto’s 30-epoch run has finished (success, or results recorded after a real completion). Trial-6 **code** may be written on CPU while waiting for copy or while Atto runs.

---

## Phase A — wait / verify RGB copy

**Dest (only):** `/storage/BTP/model_trial_5/shared_rgb_cache_256_v1`  
**Copy:** PID 43154 `scp -r asrivastava@10.206.5.228:/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/shared_rgb_cache_256_v1 /storage/BTP/model_trial_5/`

Done when **all** of:

1. PID 43154 gone, **exit 0** (if the parent shell is still open, check that shell’s status; do not assume `COMPLETE` on disk means local-done — that file was copied from the source).
2. `du -sh` ≈ **48G** (metadata 47.91 GiB).
3. **223** `sample_*` dirs (metadata `unique_sample_ids`).
4. **823** clips with `frames.npy` matching parent `index.jsonl` `clip_id`s.
5. Extra samples `sample_0207`…`sample_0229` present (one clip each is expected).
6. Previously missing dirs present: `sample_0054_car_cut_in`, `sample_0079_child_darting`, `sample_0092_cube_on_path`, `sample_0114_shape_head_on`, `sample_0135_shape_head_on`, `sample_0149_jaywalker_from_right`.

**Fallbacks**

- scp still running: poll (minutes, not busy-spin). Do not launch Atto.
- scp died incomplete: **report**, then `rsync -a --partial` from the same remote path **into this dest only**. Do not recopy onto trial 3/4/temp or a second local folder.
- Local cache corrupt after a “successful” scp: rsync repair into the same dest; do not drop samples.

While waiting: may write trial-6 **CPU code** (Phase D) but **no CUDA**.

---

## Phase B — ConvNeXt-V2 Atto 30 epochs on the 4060

### B.0 Patches in trial 5 (required; blocking if skipped)

1. **`_local_rgb_ok`** in `scripts/90_train_convnextv2_student.py`: treat shared cache as `root / clip_id / frames.npy` (and keep the old `frames_uint8.npy` slug path as a fallback). **Do not skip** windows that have RGB.
2. **`SpatialHeatmaps` / `WinDS`:** if `spatial_annotations.json` is missing (extra samples 207–229), use **zero `h_now`** and keep the window. GT remains `gt_1s.npy`. Do not drop 920 train windows.
3. **`run_convnextv2_atto.sh`:** point `STUDENT_RGB_CACHE` at the trial-5 shared cache; `STUDENT_PYTHON` at the venv; batch **8** / accum **4** / workers **2** / `--device cuda`. Keep parent cache/ckpt as **read-only trial 3** paths (or the existing trial-5 ckpt copy). Do **not** set RGB to `model_trial_3/cache/frames`.
4. Rotate the prior **smoke** under `student_runs/convnextv2_atto_v1/` (`run.log`, `checkpoint_last.pt`, smoke `final_metrics.json`). **Do not `--resume`** that run (it trained on the 800-clip cache and skipped extras).

### B.1 Launch (only after Phase A + B.0)

CWD: `/storage/BTP/model_trial_5`

```bash
export STUDENT_PYTHON=/storage/BTP/impl/venv/bin/python
export STUDENT_RGB_CACHE=/storage/BTP/model_trial_5/shared_rgb_cache_256_v1
export STUDENT_PARENT_CACHE=/storage/BTP/model_trial_3/checkpoints/parent_target_cache_v2
export STUDENT_PARENT_CKPT=/storage/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337_best.pt
export STUDENT_RUN_DIR=/storage/BTP/model_trial_5/student_runs/convnextv2_atto_v1
export STUDENT_FDEC_STATS=/storage/BTP/model_trial_5/student_runs/fdec_channel_stats.npz

mkdir -p "$STUDENT_RUN_DIR"
cd /storage/BTP/model_trial_5
"$STUDENT_PYTHON" -u scripts/90_train_convnextv2_student.py \
  --run-name convnextv2_atto_v1 \
  --variant atto \
  --epochs 30 \
  --batch 8 \
  --accum 4 \
  --lr 3e-4 \
  --eval-every 2 \
  --no-grad-checkpoint \
  --num-workers 2 \
  --eval-test \
  --device cuda \
  >> "$STUDENT_RUN_DIR/run.log" 2>&1
```

Keep the job in the **background**. Poll the log. Do not stop because an epoch is slow. Do not start a second GPU job.

Startup must log **no** `skip N train windows missing local RGB` (or skip=0). Windows must stay **(22680, 5120, 5120)** after the RGB filter. `rgb_cache=` must be the trial-5 shared dest. `decoder_hash=77f78c8714667617`.

### B.2 Success criteria

Official selection (`student_convnextv2/metrics.py` `rank_checkpoint`): **SAFE ≥ 0.85**, then max **STOP**, then STOP precision, then `fdec_cosine`. Thresholds: SAFE if peak `< 0.25`, STOP if peak `≥ 0.60`.

Record from **best ckpt** (script writes `checkpoint_best.pt` / `selected_epoch`) and held-out test (`--eval-test`): SAFE, STOP, `fdec_cos`, `L_feat`, heatmap MSE/MAE vs GT and parent, `collapsed_stop_frac`, best epoch, ckpt path.

### B.3 Fallbacks

- **OOM:** lower `--batch` in trial-5 script/shell only (try 4, accum 8). Do not blindly restart forever; if OOM persists after batch 2, stop and report.
- CUDA init fail: do not silently train on CPU for 30 epochs.
- Path/FileNotFound on extras: fix trial-5 data/filter (B.0); **never** filter extras out.
- Hash mismatch on decoder: stop; do not train against the wrong teacher.

---

## Phase C — note Atto results (trial 5 only)

Write `/storage/BTP/model_trial_5/student_runs/convnextv2_atto_v1/RESULTS.md` with SAFE, STOP, `fdec_cos`, `L_feat`, heatmap metrics, collapse, best epoch, ckpt path, log path. CNN GPU training does not start until this file exists after a **real** 30-epoch (or best-ckpt) completion.

---

## Phase D — implement trial 6 CNN + Canny (CPU OK anytime after docs)

Create `/storage/BTP/model_trial_6` from scratch (slim). **Do not implement until Phase A inventory is written** (this file). May code while Atto trains. **No GPU** here.

### Layout (no viz dump dirs)

- `cnn_edge/{model,data,losses,metrics,paths}.py`
- `scripts/train_cnn_edge.py`
- `run_cnn_edge_local.sh` (venv python, `--device cuda`, laptop only)
- small README

Copy **small code only** (Canny helper can be a local 80-line module matching `EdgeHighlightConfig`; do not vendor trial 3). Read teacher cache from trial 3 / RGB from trial 5 by **config path**, no GB copies. Do not write into trial 3/4. Do not modify trial 5 except Atto patches above.

### Architecture (ChatGPT share unreadable → this is the plan)

Compact **2.5D CNN**, not ConvNeXt-V2:

- Input `highlight_edges` on cached 256 RGB (Canny 40/120, cyan on photo, 3 channels).
- Stem `Conv3d` 3→32, downsample spatially 256→16 (total stride 16) and time 16→~4, a few (2+1)D residual blocks, temporal softmax pool, `1×1` → `F [B,128,16,16]`.
- Frozen Pair-CF `HeatmapDecoder` (same hash). Modest size for 4060.

### Loss (inspired by 1s teacher / M1, **not** a dump of Pair-CF)

Let `F_s` be student `F`, `F_t` cached `F_dec` (channel-whitened with existing `fdec_channel_stats.npz`). `Ĥ` / logits from the frozen decoder. `Y = gt_1s`. Mask ambiguous `amb_1s` like the parent.

Proposed terms:

| Term | Weight | Why |
|---|---|---|
| Whitened spatial L1 `L_feat` | 1.0 | Main latent match (M1) |
| Per-location cosine `L_cos` | 0.3 | Direction of F_dec, scale-invariant |
| BCE-on-logits vs `Y` | 0.5 | Occupancy, teacher-style heatmap head |
| STOP peak hinge (≥0.60) | 0.5 | Don’t collapse STOP |
| SAFE peak hinge (<0.25) | 0.25 | Don’t false-alarm SAFE |

**Omit** for the small CNN: close-head, bg-reg, Pair-CF PCA / token pair-diff, direction loss, blob-BCE, gram/MGD, 1.5s/2s residuals. Optional later: light parent-heatmap KD when parent class agrees with GT.

### ChatGPT fallback

Share is login-walled. Do not block Phase D on it. If it becomes readable later, take **ideas only**; do not paste an oversized net onto an 8GB card.

---

## Phase E — train CNN on 4060 **after** Atto

Same GPU ownership rule. Start only when Phase C is done.

Example:

```bash
cd /storage/BTP/model_trial_6
/storage/BTP/impl/venv/bin/python -u scripts/train_cnn_edge.py \
  --epochs 30 --batch 8 --accum 4 --lr 3e-4 --device cuda --num-workers 2 \
  --rgb-cache /storage/BTP/model_trial_5/shared_rgb_cache_256_v1 \
  --parent-cache /storage/BTP/model_trial_3/checkpoints/parent_target_cache_v2 \
  --parent-ckpt /storage/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337_best.pt
```

Log: `model_trial_6/runs/cnn_edge_v1/run.log`. Results: `.../RESULTS.md` (same metrics as Atto). Debug OOM/path errors in trial 6 only. OOM: drop batch, keep extras.

---

## Sequence lock

```
A copy complete  →  B.0 patches  →  B.1 Atto GPU  →  C RESULTS.md
                                              ↘ D trial-6 CPU code (may overlap A/B)
C done + D code  →  E CNN GPU
```

If A or B.0 still failing: **stop and report, do not train.**
