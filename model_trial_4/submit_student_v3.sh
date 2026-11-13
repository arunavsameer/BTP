#!/bin/bash
# CollisionStudentV3: CoordConv + Δ1/Δ3 + time embed. Isolated from V2 jobs.
#
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
#   sbatch submit_student_v3.sh
#
#SBATCH --job-name=cjepa4_stu_v3
#SBATCH --output=logs/slurm_student_v3_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_student_v3.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
PY=/raid/money/miniconda3/bin/python
TAG=student_v3_coord_v1
RUN=/raid/asrivastava/VJEPA2_BTP/model_trial_4/student_runs/${TAG}

cd "$ROOT"
mkdir -p "logs/${TAG}" logs "$RUN"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export STUDENT_RGB_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/shared_rgb_cache_256_v1
export STUDENT_PARENT_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2
export STUDENT_PARENT_CKPT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt
export STUDENT_Z16_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_4/z16_sidecar_v1
export STUDENT_RUN_DIR="$RUN"

echo "=== SLURM job $SLURM_JOB_ID ${TAG} on $(hostname) ==="
date
nvidia-smi -L || true

"$PY" tests/test_student_v3.py -v
if [[ ! -f "$STUDENT_Z16_CACHE/COMPLETE" ]]; then
  "$PY" scripts/11_build_z16_cache.py --device cuda --batch 64
fi
"$PY" scripts/12_train_student_v3.py \
    --run-name "$TAG" \
    --epochs 100 \
    --batch 16 \
    --accum 2 \
    --device auto \
    --num-workers 4

echo "=== ${TAG} complete ==="
date
