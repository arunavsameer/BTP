#!/bin/bash
# CollisionStudentV2: resume epoch 20, cosine-decay L_feat weight 0.50 → 0.05 over 25 epochs.
# Isolated from job 10252 (student_v2_v1). Writes under /raid/.../model_trial_4_decay/.
#
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
#   sbatch submit_student_v2_featdecay.sh
#
#SBATCH --job-name=cjepa4_v2_fd
#SBATCH --output=logs/slurm_student_v2_featdecay_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_student_v2_featdecay.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
PY=/raid/money/miniconda3/bin/python
TAG=student_v2_featdecay
RUN=/raid/asrivastava/VJEPA2_BTP/model_trial_4_decay/${TAG}
RESUME=/raid/asrivastava/VJEPA2_BTP/model_trial_4_decay/resume_ep20.pt

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
ls -lh "$RESUME"

"$PY" tests/test_student_v2.py -v
if [[ ! -f "$STUDENT_Z16_CACHE/COMPLETE" ]]; then
  "$PY" scripts/11_build_z16_cache.py --device cuda --batch 64
fi
"$PY" scripts/11_train_student_v2.py \
    --run-name "$TAG" \
    --resume "$RESUME" \
    --feat-w 0.5 \
    --feat-decay-start 20 \
    --feat-decay-duration 25 \
    --feat-w-end 0.05 \
    --epochs 100 \
    --batch 16 \
    --accum 2 \
    --device auto \
    --num-workers 4

echo "=== ${TAG} complete ==="
date
