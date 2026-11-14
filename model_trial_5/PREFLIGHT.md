# Atto + trial-6 CNN preflight (laptop RTX 4060)

Facts only. Checked 2026-11-14 ~04:05 IST. **No GPU training launched this preflight.**

Statuses: **PASS** / **FAIL** / **WAIT** / **MISSING**.

---

## Summary table

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | RGB cache dest complete | **WAIT** | Dir exists, `COMPLETE`+`metadata.json` copied from source, scp PID **43154 still running**. 47G / ~48G. 217/223 sample dirs; **798/823** index clips have `frames.npy`. |
| 2 | Disk space `/storage` | **PASS** | 576G fs, **140G avail** (76% used). Enough for remaining ~1–2G RGB + Atto/CNN ckpts. |
| 3 | Teacher / parent target cache | **PASS** (read-only trial 3) | Full v2 cache at trial 3. Trial 5 has **no** local `parent_target_cache_v2`. |
| 4 | Pair-CF decoder ckpt | **PASS** | `pair_cf_treatment_s1337_best.pt` exists in trial 3 and trial 5. Prior smoke: decoder hash `77f78c8714667617`. |
| 5 | Trial 5 Atto code / RGB wiring | **FAIL** (code) | Scripts/package present. `_local_rgb_ok` does not understand shared 256 cache. `run_convnextv2_atto.sh` points at old trial-3 `cache/frames` and batch 32. Extra-sample `h_now` spatial JSON missing. |
| 6 | Venv + GPU idle | **PASS** | `/storage/BTP/impl/venv`: torch 2.13.0+cu130, **cuda True**, **NVIDIA GeForce RTX 4060 Laptop GPU**. nvidia-smi **36 MiB / 8188 MiB, 0%**. No Python GPU job. |
| 7 | Trial 3 docs | **PASS** | Three markdown docs readable. Canny contract summarized below. |
| 8 | ChatGPT architecture share | **FAIL** (login-walled) | `https://chatgpt.com/s/t_6af78f1c59c08191a40935b3249dbf7f` requires login. Fallback: M1 + Canny docs + compact CNN plan below. |
| 9 | `model_trial_6` | **MISSING** | Folder does not exist. Not created this preflight. |
| 10 | Risks | — | Listed in §10. None of them is a reason to drop samples 207–229. |

**Blocking for a naive Atto launch:** RGB copy still in progress (**WAIT**), plus code **FAIL** items in §5. Do not train until copy is complete **and** the §5 patches in `PLAN.md` are applied. Do not use DGX.

---

## 1. RGB cache — `/storage/BTP/model_trial_5/shared_rgb_cache_256_v1`

| Fact | Value |
|---|---|
| Exists | yes |
| `du -sh` | **47G** (source estimate **47.91 GiB** / 51447791616 bytes) |
| `COMPLETE` | present, contents `ok`, mtime 03:41 IST — **copied from source, not local completion** |
| `metadata.json` | present. `unique_clips=823`, `unique_sample_ids=223`, `unique_frames=261677`, windows 22680/5120/5120, size 256, offsets 16 |
| `estimate.json` | present, matches ~47.91 GiB |
| `loader.py` | present (`clip_id/frames.npy` + `frame_indices.npy`, not `frames_uint8.npy`) |
| Sample dirs | **217** of **223** unique `sample_id`s |
| Index clips with `frames.npy` | **798 / 823** (25 clips still missing) |
| `sample_0208` | **present** (`sample_0208_same_dir_closing`). Extra 207–229: **23/23 clips already have `frames.npy`** |
| scp PID 43154 | **still running** ~1h13m: `scp -r asrivastava@10.206.5.228:/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/shared_rgb_cache_256_v1 /storage/BTP/model_trial_5/` |

Sample dirs **not yet on disk** (6):

- `sample_0054_car_cut_in`
- `sample_0079_child_darting`
- `sample_0092_cube_on_path`
- `sample_0114_shape_head_on`
- `sample_0135_shape_head_on`
- `sample_0149_jaywalker_from_right`

Typical complete sample has 4 clips (`collision`/`miss_*` × `bg_00`/`bg_01`). Extra samples 207–229 have **one clip each** in the parent index (920 train windows total) — one `frames.npy` is the expected layout, not a skip.

`/storage/BTP/model_trial_3_temp/shared_rgb_cache_256_v1` is a **dangling symlink** to `/raid/...` (not mounted on this laptop). `paths.rgb_cache()` therefore falls through to this trial-5 dest. Do **not** recopy onto any other path.

---

## 2. Disk

```
/dev/nvme0n1p5  576G  437G  140G  76%  /storage   (fuseblk)
```

Headroom: remaining RGB (~1–2G) + Atto 30 epoch ckpts (Atto last ckpt already ~50 MB in a prior smoke) + trial-6 CNN. **PASS**.

---

## 3. Teacher / parent target cache

**Usable read-only path (do not copy GB):**

`/storage/BTP/model_trial_3/checkpoints/parent_target_cache_v2`  (**2.1G**, `COMPLETE` = `ok`)

| File | Present | Size |
|---|---|---|
| `COMPLETE` | yes | 3 B |
| `mapping.json` | yes | 1.9M |
| `F_dec.npy` | yes | 2.1G  shape contract `[32920,128,16,16]` float16 |
| `H_teacher.npy` | yes | 1.6M  `[32920,5,5]` |
| `gt_1s.npy` | yes | 1.6M |
| `index.jsonl` | yes | 19M  **32920** rows, **223** sample_ids, **823** clip_ids |
| `metadata.json` | yes | decoder ckpt epoch 8, `missing: 0` |
| `Z_t.npy` | yes | 51M (parent-only; student does not need it) |
| `fdec_channel_stats.npz` | **yes, already in trial 5** | `/storage/BTP/model_trial_5/student_runs/fdec_channel_stats.npz` |

`paths.parent_cache()` resolves here. Trial 5 `checkpoints/` has decoder/student ckpts only — **no** local parent_target_cache_v2.

Extra sample_ids `sample_0207`…`sample_0229` (23 ids, **920 train windows**, 0 selection/test) are **in the index**. `video_path` points at `/home/asrivastava/VJEPA2_BTP/BTP/blender_sim/datasets/groups15_n400_s67_n400/...` which **does not exist** on this laptop. `groups15-n400` has no `sample_02xx`. GT 1s maps still live in `gt_1s.npy`. Spatial JSON for `h_now` / appear weights does **not**.

---

## 4. Decoder / Pair-CF teacher ckpt

| Path | Status |
|---|---|
| `/storage/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt` | **MISSING** (nested dir) |
| `/storage/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337_best.pt` | **EXISTS** 5.6M — this is what `paths.parent_ckpt()` returns |
| `/storage/BTP/model_trial_5/checkpoints/pair_cf_treatment_s1337_best.pt` | **EXISTS** 5.6M (local copy) |
| Expected decoder hash | `77f78c8714667617` (`student_convnextv2/paths.py` + trial-3 `parent_pair_cf_model.md`) |
| Prior smoke | hash matched |

`load_parent_decoder` requires key `decoder` and the hash. Prior smoke loaded it on CUDA successfully.

---

## 5. Code (trial 5)

Present:

- `scripts/90_train_convnextv2_student.py`
- `student_convnextv2/{__init__,model,data,losses,metrics,parent,paths,blocks,sparse}.py`
- `run_convnextv2_atto.sh`
- `tests/test_convnextv2_student.py`
- `student_model/edge_highlight.py` (Canny; trial 5 already has a copy)

Import smoke from `/storage/BTP/impl/venv/bin/python` with `cwd=model_trial_5`: **all OK** (`student_convnextv2.*`, `collision_jepa3.models.heatmap_decoder`, heatmap losses, `student_model.loss`, `student_model.edge_highlight`). `Config.load()` OK (yaml `paths.cache_root` still points at `/raid/...`; train script does not read RGB from that yaml).

**FAIL — RGB path filter vs shared cache**

`scripts/90_train_convnextv2_student.py` `_local_rgb_ok` looks for:

- `rgb_root/slug/frames_uint8.npy` if root name is `frames`, else
- `rgb_root/frames/slug/frames_uint8.npy`

Shared cache layout is `rgb_root/<clip_id>/frames.npy` (clip_id has slashes, e.g. `sample_0000_jaywalker/collision/bg_00`). Pointing the train script at `shared_rgb_cache_256_v1` **without a patch would drop every window**.

`student_convnextv2/data.py` `RGBCache256` **does** load `loader.py` when present (`get_frames(..., normalize=False)`). `loader.py` has **no** `has_clip`.

**FAIL — `run_convnextv2_atto.sh` defaults (do not run as-is)**

```
STUDENT_PYTHON defaults to python3 (not the venv)
STUDENT_RGB_CACHE=$TRIAL3/cache/frames     # old 800-clip cache; extra 207–229 ABSENT
STUDENT_PARENT_CACHE=$TRIAL3/checkpoints/parent_target_cache_v2
batch 32 / accum 1 / num-workers 4 / --device auto
```

Trial-3 `cache/frames`: 66G, **800** `frames_uint8.npy`, **0** clips for sample_0207–0229. A previous smoke in `student_runs/convnextv2_atto_v1/` used that cache, logged **`skip 920 train windows missing local RGB frames`** (21760/22680), batch 8 / accum 4, max-batches smoke only (`selected_epoch: null`). That is the skip the user forbade. **Do not `--resume` that smoke.**

**FAIL — extra-sample `h_now`**

`WinDS` always calls `SpatialHeatmaps.get_now` when heatmaps is set (train script always sets it). Extra 207–229 `video_path` + `LOCAL_DATASET_ROOTS` have **no** `spatial_annotations.json` → `FileNotFoundError` if those 920 windows are kept. GT for the 1s map is still in `gt_1s.npy`; only appear-cell weights need `h_now`. Must fall back to zeros, **not** drop the clips.

Laptop recipe that already fitted 8GB (prior smoke config): `--batch 8 --accum 4 --num-workers 2 --device cuda --no-grad-checkpoint`. Shell script’s batch 32 is not that recipe.

Atto param audit (prior smoke): total **4 422 222**, trainable 4 374 413, frozen decoder 47 809.

---

## 6. Venv / GPU

| Item | Value |
|---|---|
| Python | `/storage/BTP/impl/venv/bin/python` → `/usr/bin/python` |
| torch | 2.13.0+cu130 |
| `torch.cuda.is_available()` | **True** |
| GPU name | **NVIDIA GeForce RTX 4060 Laptop GPU** |
| CUDA runtime (torch) | 13.0 |
| nvidia-smi | 620.70.06, **36 MiB / 8188 MiB**, util **0%**, 51C P8 4W |
| GPU processes | Xorg 4 MiB, kwin_wayland 7 MiB. **No training.** |
| cv2 | 5.0.0 (Canny available) |
| numpy | 2.5.2 |

Venv treated as **read-only**. scp not killed.

---

## 7. Trial 3 docs (read-only)

Used:

- `/storage/BTP/model_trial_3/docs/student_edge_street_vlog.md` — Canny input contract
- `/storage/BTP/model_trial_3/docs/parent_pair_cf_model.md` — 1s teacher, Pair-CF, F_dec, decoder, STOP/SAFE
- `/storage/BTP/model_trial_3/docs/student_m1_model.md` — student contract (F_dec match, heatmap, selection rule)

Also present, not required: `architecture_explainer.html`.

Trial 5 mirrors: `model_trial_5/docs/parent_pair_cf_model.md`, `student_m1_model.md`.

### Canny input contract (from street vlog + `edge_highlight.py`)

- Preprocess only: OpenCV Canny, **not** in the trained weights; does not resize.
- Blur: `cv2.GaussianBlur` kernel **(5,5)**, `sigma=0` (OpenCV derives σ≈1.1).
- Canny: low **40**, high **120**, Sobel aperture 3, `L2gradient=False`.
- Paint: RGB **(0, 255, 255)** cyan where the mask is nonzero; photograph otherwise (`highlight_edges`). `edges_only` is cyan-on-black (order c, not the training default).
- Resolution: caller’s frame, unchanged. Street order **(a)** = Canny **after** 256 letterbox on the content box (pad not edged). Cache frames are **already** letterboxed 256×256, so training Canny = order (a) on the cached uint8 RGB.
- Network tensor after paint: **`[B, 16, 3, 256, 256]`** float `[0,1]` then ImageNet mean/std inside the model. Still **3 channels**, not a 4th edge channel, unless we choose to add one later.
- Offsets: `[-90,-84,…,-6,0]` (16 frames). SAFE peak `<0.25`, STOP `≥0.60`.

---

## 8. ChatGPT share + CNN architecture intent

URL `https://chatgpt.com/s/t_6af78f1c59c08191a40935b3249dbf7f` fetched 2026-11-14: **login wall** (“Log in to get answers based on saved chats”). No architecture text recovered.

**Intended CNN (from docs, not a copy of ConvNeXt-V2 or the full teacher):**

M1 student maps `clip [B,16,3,256,256]` → `F [B,128,16,16]` → frozen Pair-CF `HeatmapDecoder` → `Hhat_1s [B,5,5]`. Parent F_dec is the distillation target. M1 loss is whitened L1 + cosine + a STOP floor, not the full Pair-CF composite.

**Improved compact plan (trial 6, implement later):**

- Keep a **small 2.5D/3D CNN** (stem + a few inverted-residual / (2+1)D blocks + temporal softmax pool + 1×1 to 128). **Not** ConvNeXt-V2. Target ≲1M params so batch 8 fits 8GB.
- Input: cached RGB 256 + **Canny `highlight_edges`** per frame (order a). Stay 3-channel cyan-on-photo to match the street-vlog student contract.
- Frozen copied Pair-CF decoder (hash `77f78c8714667617`); match `F_dec [128,16,16]`.
- Loss (simpler than teacher, logical for a tiny CNN): whitened L1 + cosine on F_dec; BCE-on-logits vs `gt_1s`; STOP hinge (0.60) + SAFE hinge (0.25). Skip close-head, bg-reg, Pair-CF PCA, direction, gram/MGD unless a later ablation needs them.
- Data: **read** trial-5 RGB cache + trial-3 `parent_target_cache_v2`. Do not duplicate GB. No viz dump dirs.

---

## 9. Trial 6

`/storage/BTP/model_trial_6` **does not exist**. Will be created in Phase D of `PLAN.md`. No CNN code this preflight. No GPU.

---

## 10. Risks

| Risk | Notes |
|---|---|
| RGB 207–229 missing | Extra **clips are already in the trial-5 cache**. Prior FileNotFound was trial-3 `cache/frames` `frames_uint8.npy`. Remaining missing dirs are **other** samples (0054, 0079, 0092, 0114, 0135, 0149) plus 25 index clips still in flight. **Do not drop extras. Wait for scp.** |
| scp dies incomplete | Resume with **rsync to the same dest only**. Do not recopy to other paths. |
| 8GB VRAM | Atto smoke used batch 8 / accum 4. Shell default batch 32 is OOM-prone. If OOM, lower batch in trial 5 only. |
| Other Cursor chat restarts Atto | GPU is idle now. This runner owns launches. Monitor must not start a second GPU job. |
| DGX / sbatch | Not to be used. Laptop only. TEMP3 `/raid` symlinks are dead here. |
| `COMPLETE` file present while copy runs | Source COMPLETE copied early. **Do not treat COMPLETE as local-done** until scp exits 0 and clip count is 823. |
| Extra-sample spatial JSON | Appear weights will crash unless `h_now` falls back to zeros. |
| Naive RGB filter | Would skip **all** shared-cache windows, or skip 920 extras if pointed at trial-3 `cache/frames`. |
| Sibling overwrite of `convnextv2_atto_v1` | Prior smoke ckpt/log live there. Rotate before the real 30-epoch run; do not resume smoke. |

---

## Verdict

**Do not start Atto or CNN training now.** RGB copy is **WAIT**. Code wiring for the shared 256 cache and extra-sample `h_now` is **FAIL** until patched in trial 5 (see `PLAN.md`). Teacher cache, decoder ckpt, venv CUDA, and GPU idle are **PASS**.
