"""SAFE / STOP evaluation knobs. Change numbers here; do not scatter them in scorers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
_BTP = ROOT.parent
if not (_BTP / "model_trial_3").is_dir() and Path("/storage/BTP/model_trial_3").is_dir():
    _BTP = Path("/storage/BTP")
TRIAL3 = _BTP / "model_trial_3"
TRIAL4 = _BTP / "model_trial_4"
_PY_CANDIDATES = (
    Path("/raid/money/miniconda3/bin/python"),
    Path("/raid/money/miniconda3/envs/btp-student/bin/python"),
    Path("/storage/BTP/impl/venv/bin/python"),
)
PYTHON = next((p for p in _PY_CANDIDATES if p.is_file()), Path("python3"))

# --- class cuts: same as model_trial_3/configs/default.yaml warning.* ---
SAFE_PEAK = 0.25  # caution_threshold: max(h_1s) below this is SAFE
STOP_PEAK = 0.60  # stop_threshold: max(h_1s) at/above this is STOP
# THREAT is the leftover band: SAFE_PEAK <= max < STOP_PEAK. Not averaged into SAFE/STOP.

# --- spatial leeway ---
HOT_THRESHOLD = 0.50  # core RED blob cut (Pair-CF occupied); 0.26 THREAT cells are not a core blob
CHEBYSHEV_RADIUS = 1  # allowed = RED plus 3x3 square (diagonals included)
SAFE_FALSE_HOT_THRESHOLD = 0.25  # BLUE cell at this peak is already out of SAFE; penalize it
STOP_FALSE_HOT_THRESHOLD = 0.50  # student-RED on BLUE: same cut as HOT_THRESHOLD
SAFE_PEAK_PENALTY_SPAN = 0.35  # SAFE spatial goes 1→0 as max(pred) goes 0.25→0.60

# --- temporal intent ---
N_INTENT_SAMPLES = 6  # six samples over the 3 s history the student sees, including t
# 3.0 s / 5 gaps = 0.6 s. A literal 0.5 s step (15 frames) is not on the 16-frame
# input grid (-90..0 step 6). Every third input frame lands on both.
INTENT_FRAME_OFFSETS = (-90, -72, -54, -36, -18, 0)
INTENT_DEADZONE = 0.05  # |last-first| below this is flat; 0.02 jitter is not a trend

# --- mix ---
SPATIAL_WEIGHT = 0.70  # spatial leeway was the primary request
INTENT_WEIGHT = 0.30  # rising/falling intent is a secondary check

# --- sampling contract (must match the student; not a score invention) ---
FPS = 30
HISTORY_SECONDS = 3.0
INPUT_FRAMES = 16
INPUT_SPACING_FRAMES = 6
FRAME_OFFSETS = (
    -90, -84, -78, -72, -66, -60, -54, -48,
    -42, -36, -30, -24, -18, -12, -6, 0,
)
HORIZON_1S_FRAMES = 30  # student at t predicts occupancy at t+1 s
N_FRAMES = 450
T_MIN = 90  # first 90 frames have no 3 s history, so no prediction
INPUT_SIZE = 256
BATCH_SIZE = 4  # RTX 4060 8 GB

# --- display (wide composite, not a 256 banner) ---
DISPLAY_WIDTH = 854
DISPLAY_HEIGHT = 480
HUD_HEIGHT = 64
OVERLAY_ALPHA = 0.45
VIDEO_FPS = 30.0

# --- default IO; adapters may override checkpoint ---
DEFAULT_MODEL = "m1_stop_floor"
DEFAULT_CHECKPOINT = TRIAL3 / "checkpoints" / "m1_stop_floor_epoch_10.pt"
MANIFEST_PATH = TRIAL3 / "cache" / "manifest.jsonl"
SPLITS_PATH = TRIAL3 / "cache" / "splits.json"
FRAME_CACHE_ROOT = TRIAL3 / "cache"
_GROUPS15 = _BTP / "blender_sim" / "datasets" / "groups15_n400_s67_n400"
_GROUPS15_ALT = _BTP / "blender_sim" / "datasets" / "groups15-n400"
_GROUPS15_STORAGE = Path("/storage/BTP/groups15-n400")
DATASET_ROOT = (
    _GROUPS15 if _GROUPS15.is_dir()
    else _GROUPS15_ALT if _GROUPS15_ALT.is_dir()
    else _GROUPS15_STORAGE
)
N_CLIPS = 12
PREFERRED_SPLIT = "test"
