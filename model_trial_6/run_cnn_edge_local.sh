#!/bin/bash
# Local 4060 CollisionStudentV2 + Canny. Do NOT start while Atto still owns the GPU.
set -euo pipefail
ROOT=/storage/BTP/model_trial_6
T5=/storage/BTP/model_trial_5
T3=/storage/BTP/model_trial_3
RUN="$ROOT/runs/student_v2_canny"
PY=/storage/BTP/impl/venv/bin/python
cd "$ROOT"
mkdir -p "$RUN"
if pgrep -f 'scripts/90_train_convnextv2_student.py' >/dev/null; then
  echo "REFUSE: Atto still running; CNN waits"
  exit 1
fi
export STUDENT_RGB_CACHE="$T5/shared_rgb_cache_256_v1"
export STUDENT_PARENT_CACHE="$T3/checkpoints/parent_target_cache_v2"
export STUDENT_PARENT_CKPT="$T3/checkpoints/pair_cf_treatment_s1337_best.pt"
export STUDENT_FDEC_STATS="$T5/student_runs/fdec_channel_stats.npz"
export STUDENT_RUN_DIR="$RUN"
export PYTHONUNBUFFERED=1
# Small batch: stem keeps T=8 at 128×128 before the next downsample.
setsid nohup "$PY" -u scripts/train_cnn_edge.py \
  --epochs 30 --batch 4 --accum 8 --lr 3e-4 --device cuda --num-workers 2 --eval-test \
  >> "$RUN/run.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
disown
echo "CNN LAUNCHED $(cat "$RUN/train.pid")"
