#!/usr/bin/env bash
# R0–R3 background-pair regularization on first usable GPU (memory.used < 1000 MiB, not ghost-busy).
set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/checkpoints reports

# Ghost-busy: high util + very low reported mem (DGX-style); same rule as submit_direct_h1s_pair_signals.sh
is_ghost() {
  local mem=$1 util=$2
  [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]
}

pick_gpu() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # First id only (waiter passes a single physical index)
    echo "${CUDA_VISIBLE_DEVICES%%,*}"
    return
  fi
  echo "1 2 3 4 5 6 7"
}

FREE=""
for i in $(pick_gpu); do
  mem_util=$(nvidia-smi -i "$i" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
  mem=${mem_util%%,*}; util=${mem_util##*,}
  mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
  if [ -z "${mem:-}" ]; then
    continue
  fi
  if [ "${mem:-999999}" -lt 1000 ] && ! is_ghost "$mem" "$util"; then
    FREE=$i
    break
  fi
done
if [ -z "$FREE" ]; then
  echo "No usable GPU: need memory.used < 1000 MiB and not ghost (util>=50 & mem<400). Set CUDA_VISIBLE_DEVICES or free a GPU."
  exit 1
fi
export CUDA_VISIBLE_DEVICES=$FREE
export SKIP_GPU_GUARD=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache
echo "Using physical GPU $FREE (CUDA_VISIBLE_DEVICES=$FREE)"
nvidia-smi -i "$FREE" --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv

SEEDS=(1337 1338)
for seed in "${SEEDS[@]}"; do
  for run in R0 R1 R2 R3; do
    case "$run" in
      R0)
        BG_ARGS=()
        ;;
      R1)
        BG_ARGS=(--bg-reg --bg-reg-no-early-stop --bg-reg-pair-consistency)
        ;;
      R2)
        BG_ARGS=(--bg-reg --bg-reg-no-early-stop --bg-reg-targeted-dropout)
        ;;
      R3)
        BG_ARGS=(--bg-reg --bg-reg-no-early-stop --bg-reg-random-dropout)
        ;;
    esac
    OUT="checkpoints/bg_reg_${run}_s${seed}"
    LOG="logs/train_bg_reg_${run}_s${seed}.log"
    mkdir -p "$OUT" "$OUT/logs" "$OUT/reports"
    echo "=== $run seed=$seed → $OUT"
    python scripts/04_train_teacher.py \
      --config configs/default.yaml \
      --phase frozen \
      --frozen-epochs 14 \
      --patience 99 \
      --patience-start-epoch 999 \
      --early-stop-metric val_mse \
      --device cuda:0 \
      --seed "$seed" \
      --checkpoint-root "$OUT" \
      --logs-root "$OUT/logs" \
      --reports-root "$OUT/reports" \
      "${BG_ARGS[@]}" \
      2>&1 | tee "$LOG"
  done
done
