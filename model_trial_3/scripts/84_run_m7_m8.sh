#!/usr/bin/env bash
# Independent M7 (GPU 6) and M8 (GPU 7) parent-task students.
set -euo pipefail
ROOT="/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3"
TEMP="/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp"
PY="${PYTHON:-/raid/money/miniconda3/envs/jepa-align/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT"
cd "$ROOT"

mkdir -p "$TEMP/m7_parent_task_latent_v1" "$TEMP/m8_parent_task_only_v1"

nohup "$PY" scripts/84_train_parent_task_student.py --model 7 --epochs 30 --batch 8 --lr 3e-4 --device 6 --seed 1001 \
  > "$TEMP/m7_parent_task_latent_v1/nohup.out" 2>&1 &
echo $! > "$TEMP/m7_parent_task_latent_v1/pid"
echo "M7 pid=$(cat "$TEMP/m7_parent_task_latent_v1/pid") log=$TEMP/m7_parent_task_latent_v1/run.log"

nohup "$PY" scripts/84_train_parent_task_student.py --model 8 --epochs 30 --batch 8 --lr 3e-4 --device 7 --seed 1001 \
  > "$TEMP/m8_parent_task_only_v1/nohup.out" 2>&1 &
echo $! > "$TEMP/m8_parent_task_only_v1/pid"
echo "M8 pid=$(cat "$TEMP/m8_parent_task_only_v1/pid") log=$TEMP/m8_parent_task_only_v1/run.log"
