#!/bin/bash
# Gate 2: 32-clip overfit for CollisionStudentV2.
#SBATCH --job-name=cjepa4_v2_of
#SBATCH --output=logs/slurm_student_v2_overfit_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Use sbatch submit_student_v2_overfit.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
PY=/raid/money/miniconda3/bin/python
TAG=student_v2_overfit
RUN=/raid/asrivastava/VJEPA2_BTP/model_trial_4/student_runs/${TAG}
cd "$ROOT"
mkdir -p logs "$RUN"
export STUDENT_RGB_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/shared_rgb_cache_256_v1
export STUDENT_PARENT_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2
export STUDENT_PARENT_CKPT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt
export STUDENT_Z16_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_4/z16_sidecar_v1
export STUDENT_RUN_DIR="$RUN"

"$PY" tests/test_student_v2.py -v
"$PY" scripts/11_train_student_v2.py --overfit --overfit-steps 200 --batch 16 --accum 2 --device auto --num-workers 2 --run-name "$TAG"
echo "overfit done"
date
