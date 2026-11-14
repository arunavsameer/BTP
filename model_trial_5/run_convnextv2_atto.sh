#!/bin/bash
# Local Atto 3D ConvNeXt-V2 student vs Trial-3 cached Pair-CF F_dec.
# Teacher cache and RGB frames stay in model_trial_3 (read-only).
#
#   cd /storage/BTP/model_trial_5
#   ./run_convnextv2_atto.sh
#   ./run_convnextv2_atto.sh --resume
#
set -euo pipefail

ROOT=/storage/BTP/model_trial_5
TRIAL3=/storage/BTP/model_trial_3
TAG=convnextv2_atto_v1
RUN="$ROOT/student_runs/${TAG}"
PY="${STUDENT_PYTHON:-python3}"

cd "$ROOT"
mkdir -p "$RUN" logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export STUDENT_PARENT_CACHE="$TRIAL3/checkpoints/parent_target_cache_v2"
export STUDENT_PARENT_CKPT="$TRIAL3/checkpoints/pair_cf_treatment_s1337_best.pt"
export STUDENT_RGB_CACHE="$TRIAL3/cache/frames"
export STUDENT_RUN_DIR="$RUN"
export STUDENT_FDEC_STATS="$ROOT/student_runs/fdec_channel_stats.npz"

echo "=== Atto ConvNeXt-V2 ${TAG} on $(hostname) ==="
date
nvidia-smi -L 2>/dev/null || echo "no GPU / nvidia driver"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

"$PY" tests/test_convnextv2_student.py -v
"$PY" scripts/90_train_convnextv2_student.py \
    --run-name "$TAG" \
    --variant atto \
    --epochs 30 \
    --batch 32 \
    --accum 1 \
    --lr 3e-4 \
    --eval-every 2 \
    --no-grad-checkpoint \
    --num-workers 4 \
    --eval-test \
    --device auto \
    "$@"

echo "=== ${TAG} complete ==="
date
