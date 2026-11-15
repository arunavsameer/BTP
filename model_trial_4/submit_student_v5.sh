#!/bin/bash
# CollisionStudentV5: V4 gated backbone + decoder unfreeze / QFL / λ_feat floor 0.10.
# Isolated from V3/V4 run dirs.
#
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
#   sbatch submit_student_v5.sh
#
#SBATCH --job-name=cjepa4_stu_v5
#SBATCH --output=logs/slurm_student_v5_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
# GPU 0 on DGX2 is wedged (100% util, CUDA busy). Request 2 so the cgroup
# can include GPU 1 (empty); the job then binds to the first working UUID.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_student_v5.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
PY=/raid/money/miniconda3/bin/python
TAG=student_v5_unfreeze_v1
RUN=/raid/asrivastava/VJEPA2_BTP/model_trial_4/student_runs/${TAG}
GATE_BIAS="${STUDENT_V5_GATE_BIAS:--2.0}"

cd "$ROOT"
mkdir -p "logs/${TAG}" logs "$RUN"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export STUDENT_RGB_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp/shared_rgb_cache_256_v1
export STUDENT_PARENT_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2
export STUDENT_PARENT_CKPT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt
export STUDENT_Z16_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_4/z16_sidecar_v1
export STUDENT_RUN_DIR="$RUN"

echo "=== SLURM job $SLURM_JOB_ID ${TAG} on $(hostname) gate_bias=${GATE_BIAS} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
date
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv || true

# ConstrainDevices=yes: only allocated /dev/nvidia* exist. GPU 0 is wedged;
# pick the first visible GPU whose UUID is not the dead card and that cudaMallocs.
export WEDGED_UUID="GPU-84084747-8ddf-30a2-3fd6-b9c93922d7ac"
PICK="$("$PY" - <<'PY'
import os, subprocess, sys
wedged = os.environ.get("WEDGED_UUID", "GPU-84084747-8ddf-30a2-3fd6-b9c93922d7ac")
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
    text=True,
)
rows = []
for line in out.strip().splitlines():
    idx, uuid = [x.strip() for x in line.split(",", 1)]
    rows.append((idx, uuid))
    if uuid == wedged:
        continue
    print(idx, flush=True)
    sys.exit(0)
# fall back: try last visible (not first/wedged)
if len(rows) > 1:
    print(rows[-1][0], flush=True)
    sys.exit(0)
sys.exit("no non-wedged GPU in this job cgroup")
PY
)"
export CUDA_VISIBLE_DEVICES="${PICK}"
echo "bound CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L || true

"$PY" tests/test_student_v5.py -v

if [[ ! -f "$STUDENT_Z16_CACHE/COMPLETE" ]]; then
  "$PY" scripts/11_build_z16_cache.py --device cuda --batch 64
fi
"$PY" scripts/16_train_student_v5.py \
    --run-name "$TAG" \
    --init-gate-bias "$GATE_BIAS" \
    --feat-w 0.5 \
    --feat-decay-start 20 \
    --feat-decay-duration 25 \
    --feat-w-end 0.10 \
    --epochs 100 \
    --batch 16 \
    --accum 2 \
    --device auto \
    --num-workers 4

echo "=== ${TAG} complete ==="
date
