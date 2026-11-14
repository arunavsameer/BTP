#!/bin/bash
# Detached Atto trainer that survives the launching shell (setsid + nohup).
# Do not start this if a 90_train_convnextv2 process is already running.
set -euo pipefail
ROOT=/storage/BTP/model_trial_5
RUN="$ROOT/student_runs/convnextv2_atto_v1"
PY=/storage/BTP/impl/venv/bin/python
cd "$ROOT"
mkdir -p "$RUN"
if pgrep -f 'scripts/90_train_convnextv2_student.py' >/dev/null; then
  echo "REFUSE: trainer already running"
  pgrep -af '90_train_convnextv2_student.py'
  exit 1
fi
BATCH="${1:-8}"
ACCUM="${2:-4}"
printf '\n===== detach_atto.sh batch%s accum%s %s =====\n' "$BATCH" "$ACCUM" "$(date -Iseconds)" >> "$RUN/run.log"
setsid nohup env \
  STUDENT_RGB_CACHE="$ROOT/shared_rgb_cache_256_v1" \
  STUDENT_PARENT_CACHE=/storage/BTP/model_trial_3/checkpoints/parent_target_cache_v2 \
  STUDENT_PARENT_CKPT=/storage/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337_best.pt \
  STUDENT_RUN_DIR="$RUN" \
  STUDENT_FDEC_STATS="$ROOT/student_runs/fdec_channel_stats.npz" \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  OMP_NUM_THREADS=4 \
  TOKENIZERS_PARALLELISM=false \
  "$PY" -u scripts/90_train_convnextv2_student.py \
    --run-name convnextv2_atto_v1 \
    --variant atto \
    --epochs 30 \
    --batch "$BATCH" \
    --accum "$ACCUM" \
    --lr 3e-4 \
    --eval-every 2 \
    --no-grad-checkpoint \
    --num-workers 2 \
    --eval-test \
    --device cuda \
  >> "$RUN/run.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
disown
echo "LAUNCHED $(cat "$RUN/train.pid") batch=$BATCH accum=$ACCUM"
ps -p "$(cat "$RUN/train.pid")" -o pid,ppid,sid,stat,cmd
