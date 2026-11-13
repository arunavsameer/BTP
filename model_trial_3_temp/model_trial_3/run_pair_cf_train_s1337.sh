#!/usr/bin/env bash
# Scratch control (if needed) then pair-CF treatment on assigned GPU.
set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
# shellcheck source=pair_cf_gpu_lib.sh
source "$(dirname "$0")/pair_cf_gpu_lib.sh"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  pair_cf_assign_free_gpu
fi
GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}

export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache
export PROJECT_SEED=1337

COMMON=(
  --phase frozen --frozen-epochs 100 --heads-mode direct_h1s
  --extra-same-dir --extra-same-dir-min 207 --extra-same-dir-max 229
  --close-weight 1.0 --early-stop-metric balanced_warning
  --post-fork-stop-floor 0.88 --stop-recall-floor 0.55 --safe-recall-floor 0.82
  --max-safe-stop-rate 0.04 --patience 6 --patience-start-epoch 8
  --stop-hinge-weight 1.0 --safe-hinge-weight 0.5 --small-blob-weight 2.0
  --adapter-lr 0.0003 --warmup-epochs 5 --val-subset-fraction 1.0
  --num-workers 4 --max-steps 28350
)

run_train() {
  local name=$1 ckpt=$2 logroot=$3 report=$4
  shift 4
  local extra=("$@")
  mkdir -p "$ckpt" "$logroot" "$report"
  local done_flag="$ckpt/.train_success"
  if [ -f "$done_flag" ]; then
    echo "=== $name already marked success ($done_flag); skip ==="
    return 0
  fi
  local resume=()
  if [ -f "$ckpt/teacher_phase_a_last.pt" ]; then
    echo "=== $name resume from last checkpoint ==="
    resume=(--resume)
  else
    echo "=== $name from scratch ==="
  fi
  echo "=== $name on GPU $GPU_ID $(date) ==="
  if python scripts/04_train_teacher.py "${COMMON[@]}" "${extra[@]}" "${resume[@]}" \
    --checkpoint-root "$ckpt" \
    --logs-root "$logroot" \
    --reports-root "$report"; then
    touch "$done_flag"
    echo "=== $name finished OK ==="
    return 0
  fi
  echo "=== $name FAILED ==="
  return 1
}

CTRL_CKPT=checkpoints/pair_cf_scratch_control_s1337
TRT_CKPT=checkpoints/pair_cf_treatment_s1337

run_train control "$CTRL_CKPT" logs/pair_cf_scratch_control_s1337 reports/pair_cf_scratch_control_s1337 || exit 1
run_train treatment "$TRT_CKPT" logs/pair_cf_treatment_s1337 reports/pair_cf_treatment_s1337 \
  --bg-reg-pair-cf \
  --bg-reg-pair-cf-subspace reports/pair_background_counterfactual/subspace/learned_k16.npz \
  --bg-reg-pair-cf-frac 0.5 \
  --bg-reg-pair-cf-grad-scale 0.10 || exit 1

echo "=== pair_cf s1337 control+treatment complete $(date) ==="
