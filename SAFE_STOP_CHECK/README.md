# SAFE / STOP check

Folder: `/storage/BTP/SAFE_STOP_CHECK`

Scores a student 5×5 `Hhat_1s` against GT occupancy **1 s later**. A blob that
spreads one cell (Chebyshev ≤ 1, including diagonals) still passes. Heat on
BLUE cells (outside that neighborhood) is a false hot. Rising vs falling
intent over the last ~3 s is a secondary check.

THREAT (`0.25 ≤ max < 0.60`) is labeled on the HUD and **not** averaged into
SAFE or STOP.

## How SAFE and STOP are computed

1. Window class from **GT** peak of `h_1s`: SAFE if max &lt; 0.25, THREAT if
   0.25 ≤ max &lt; 0.60, STOP if max ≥ 0.60.
2. **Allowed region** = GT cells ≥ `hot_threshold` (0.50) dilated by a 3×3
   square. Student heat there is not a spatial error.
3. **STOP spatial** (GT-STOP frames only): `coverage = clip(max(pred[allowed]) / 0.60, 0, 1)`;
   `leak = fraction of BLUE cells with pred ≥ 0.50`;
   `spatial = clip(coverage * (1 - leak), 0, 1)`. Empty allowed → 0.
4. **SAFE spatial** (GT-SAFE frames only):
   `peak_term = clip(1 - relu(max(pred) - 0.25) / 0.35, 0, 1)`;
   `spatial = peak_term * (1 - fraction of cells ≥ 0.25)`.
5. **Intent**: 6 samples at offsets `(-90,-72,-54,-36,-18,0)` (3 s history,
   on the 16-frame grid). Slope = last − first. Deadzone 0.05.
   Same sign or both flat → 1; opposite → 0; one flat / one moving → 0.5.
6. Per-frame quality `= 0.70 * spatial + 0.30 * intent`.
7. Clip STOP = mean quality on GT-STOP frames (N/A if none). Clip SAFE =
   mean quality on GT-SAFE frames (N/A if none).

## Hyperparameters (`config.py`)

| knob | value | why |
|---|---:|---|
| `HOT_THRESHOLD` | 0.50 | Pair-CF occupied/blob cut; visibly hot on 5×5 without treating every 0.26 THREAT cell as a core red blob |
| `CHEBYSHEV_RADIUS` | 1 | requested 3×3 neighborhood including diagonals |
| `SAFE_FALSE_HOT_THRESHOLD` | 0.25 | a BLUE cell at ≥ 0.25 is already out of SAFE |
| `STOP_FALSE_HOT_THRESHOLD` | 0.50 | student-red on BLUE uses the same occupied cut |
| `N_INTENT_SAMPLES` / offsets | 6 / −90…0 step 18 | 3 s history; 0.5 s (15 frames) is not on the step-6 input grid |
| `INTENT_DEADZONE` | 0.05 | 0.02 jitter is flat |
| `SPATIAL_WEIGHT` / `INTENT_WEIGHT` | 0.70 / 0.30 | spatial was the primary request |
| `SAFE_PEAK` / `STOP_PEAK` | 0.25 / 0.60 | `configs/default.yaml` warning cuts |

## How to run

```bash
cd /storage/BTP/SAFE_STOP_CHECK
/storage/BTP/impl/venv/bin/python test_metrics.py
/storage/BTP/impl/venv/bin/python run.py --model m1_stop_floor \
  --checkpoint /storage/BTP/model_trial_3/checkpoints/m1_stop_floor_epoch_10.pt
```

Default writes `output/m1_stop_floor/` (mp4 + `scores.csv` + `scores.json` + `index.md`).

Smoke one clip first: `run.py --smoke`.

## Swap the student (one command)

Copy `models/m1_stop_floor.py` to `models/your_name.py`, implement
`load(checkpoint, device)` and `predict(clip_nchw01) → Hhat_1s`, register
the name in `models/registry.py`, then:

```bash
/storage/BTP/impl/venv/bin/python run.py --model your_name --checkpoint /path/to/new.pt
```

That writes `output/your_name/` and leaves the M1 videos in place.
`metrics.py`, `render.py`, and clip picking never import a checkpoint path
or `CollisionStudent`.

M1 is overnight `StudentOvernight` (encoder + 1×1 proj + Pair-CF decoder).
It is **not** `CollisionStudent` (different heatmap path). Giant is not loaded.
Dummy backbones are not used.

## Layout

```
config.py           score knobs
metrics.py          spatial + intent + clip scores
data.py             manifest / cache / GT (read-only)
render.py           side-by-side mp4 + HUD
run.py              CLI
infer_student.py    load_model + predict re-export
models/             adapters only
  base.py           StudentModel
  m1_stop_floor.py  current checkpoint
  registry.py       name → adapter
tests/test_metrics.py
output/<model>/     videos + scores for that model
```
