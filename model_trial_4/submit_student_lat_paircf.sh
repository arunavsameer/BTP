#!/bin/bash
# Trial-4 student: L_feat + Trial-3 G heatmap, constant weights.
# Reads Trial-3 caches/checkpoint. Writes only model_trial_4/student_runs/.
#
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
#   sbatch submit_student_lat_paircf.sh
#
#SBATCH --job-name=cjepa4_stu_ghm
#SBATCH --output=logs/slurm_student_feat_gheatmap_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_student_lat_paircf.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
PY=/raid/money/miniconda3/bin/python
TAG=feat_gheatmap_v2

cd "$ROOT"
mkdir -p "logs/${TAG}" "student_runs/${TAG}" logs

export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export STUDENT_RGB_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/shared_rgb_cache_256_v1
export STUDENT_PARENT_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2
export STUDENT_PARENT_CKPT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt

echo "=== SLURM job $SLURM_JOB_ID ${TAG} on $(hostname) ==="
date
nvidia-smi -L || true

"$PY" scripts/10_smoke_student.py
"$PY" scripts/10_train_student.py \
    --run-name "$TAG" \
    --epochs 100 \
    --batch 8 \
    --device auto \
    --num-workers 4

echo "=== ${TAG} complete ==="
date
