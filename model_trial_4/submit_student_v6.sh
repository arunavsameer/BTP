#!/bin/bash
# CollisionStudentV6: V2 + ResidualGatedFusionV6. Isolated from V5 job 10383.
#
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
#   sbatch submit_student_v6.sh
#
#SBATCH --job-name=cjepa4_stu_v6
#SBATCH --output=logs/slurm_student_v6_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
# Train on 1 GPU. If SLURM hands us the wedged GPU 0 UUID, the bind
# step below fails fast instead of CUDA-busy looping.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_student_v6.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
PY=/raid/money/miniconda3/bin/python
TAG=student_v6_clean_gated
RUN=/raid/asrivastava/VJEPA2_BTP/model_trial_4/student_runs/${TAG}
GATE_BIAS="${STUDENT_V6_GATE_BIAS:--0.5}"

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

# SLURM GRES can point at a card that still has another user's process.
# Hold the allocation and wait until this GPU actually has room.
MIN_FREE_MB="${MIN_FREE_MB:-16000}"
WAIT_SECS="${WAIT_SECS:-60}"
WAIT_TRIES="${WAIT_TRIES:-1440}"
for ((i=1; i<=WAIT_TRIES; i++)); do
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "wait ${i}/${WAIT_TRIES} free=${FREE_MB} MiB need=${MIN_FREE_MB} $(date)"
  if [[ "${FREE_MB}" -ge "${MIN_FREE_MB}" ]]; then
    break
  fi
  if [[ "${i}" -eq "${WAIT_TRIES}" ]]; then
    echo "timed out waiting for ${MIN_FREE_MB} MiB free on allocated GPU" >&2
    exit 1
  fi
  sleep "${WAIT_SECS}"
done

"$PY" tests/test_student_v6.py -v

if [[ ! -f "$STUDENT_Z16_CACHE/COMPLETE" ]]; then
  "$PY" scripts/11_build_z16_cache.py --device cuda --batch 64
fi
"$PY" scripts/17_train_student_v6.py \
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
