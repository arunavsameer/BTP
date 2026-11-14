#!/bin/bash
# 3D ConvNeXt-V2 Large student vs cached Pair-CF F_dec. Writes to /raid.
#
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_5
#   sbatch submit_convnextv2_student.sh
#
#SBATCH --job-name=cjepa5_cnx_l
#SBATCH --output=logs/slurm_convnextv2_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_5/: sbatch submit_convnextv2_student.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_5
PY=/raid/money/miniconda3/bin/python
TAG=convnextv2_large_v1
RUN=/raid/asrivastava/VJEPA2_BTP/model_trial_5/student_runs/${TAG}

cd "$ROOT"
mkdir -p "logs/${TAG}" logs "$RUN"

export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export STUDENT_RGB_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/shared_rgb_cache_256_v1
export STUDENT_PARENT_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2
export STUDENT_PARENT_CKPT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt
export STUDENT_RUN_DIR="$RUN"
export STUDENT_FDEC_STATS=/raid/asrivastava/VJEPA2_BTP/model_trial_5/fdec_channel_stats.npz

echo "=== SLURM job $SLURM_JOB_ID ${TAG} on $(hostname) ==="
date
nvidia-smi -L || true
nvidia-smi || true

"$PY" tests/test_convnextv2_student.py -v
"$PY" scripts/90_train_convnextv2_student.py \
    --run-name "$TAG" \
    --variant large \
    --epochs 30 \
    --batch 2 \
    --accum 4 \
    --lr 1e-4 \
    --device auto \
    --num-workers 4 \
    --eval-test

echo "=== ${TAG} complete ==="
date
