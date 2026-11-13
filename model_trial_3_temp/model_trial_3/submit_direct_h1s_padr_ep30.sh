#!/bin/bash
# PADR only, from scratch. Early-stop patience counts only after epoch 30 (PADR warmup 18 + ramp 12).
# Does not overwrite direct_h1s_padr or direct_h1s_g_revised.
set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

G=checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt
GREV=checkpoints/direct_h1s_g_revised/teacher_phase_a_best.pt
RUN=padr_ep30
CKPT=checkpoints/direct_h1s_${RUN}
LOGS=logs/direct_h1s_${RUN}
REPORTS=reports/direct_h1s_${RUN}

COMMON=(
    --phase frozen
    --frozen-epochs 100
    --heads-mode direct_h1s
    --extra-same-dir
    --extra-same-dir-min 207
    --extra-same-dir-max 229
    --close-weight 1.0
    --early-stop-metric balanced_warning
    --post-fork-stop-floor 0.88
    --stop-recall-floor 0.55
    --safe-recall-floor 0.82
    --max-safe-stop-rate 0.04
    --patience 6
    --patience-start-epoch 30
    --stop-hinge-weight 1.0
    --safe-hinge-weight 0.5
    --small-blob-weight 2.0
    --adapter-lr 0.0003
    --warmup-epochs 5
    --val-subset-fraction 1.0
    --num-workers 4
    --bg-weight 0.0
    --logit-hinges
    --logit-hinge-margin 0.1
    --heatmap-balanced
)

gpu_guard() {
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
  PHYS=${SLURM_JOB_GPUS%%,*}
  GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}
  GPU_ID=${GPU_ID:-0}
  if [ "${PHYS:-}" = "0" ]; then
    echo "refusing physical GPU 0"
    exit 3
  fi
}

echo "=== PADR ep30-patience on $(hostname) ==="
date
if [ "${SKIP_GPU_GUARD:-0}" = "1" ]; then
  echo "SKIP_GPU_GUARD=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
else
  gpu_guard
fi

mkdir -p "$LOGS" "$REPORTS" "$CKPT"
echo "=== PADR (patience after epoch 30) ==="
python scripts/04_train_teacher.py \
    "${COMMON[@]}" \
    --padr \
    --padr-warmup-epochs 18 \
    --padr-ramp-epochs 12 \
    --checkpoint-root "$CKPT" \
    --logs-root "$LOGS" \
    --reports-root "$REPORTS"

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out "$REPORTS/full_val_operating.json" \
    --checkpoints best="$CKPT/teacher_phase_a_best.pt"

WC_ARGS=(--device auto --out "$REPORTS/warning_confusion.json" --checkpoints "padr=$CKPT/teacher_phase_a_best.pt")
if [ -f "$GREV" ]; then
  WC_ARGS+=( "grevised=$GREV" )
fi
if [ -f "$G" ]; then
  WC_ARGS+=( "gr40g=$G" )
fi
python scripts/17_warning_confusion.py "${WC_ARGS[@]}"

echo "=== PADR ep30-patience complete ==="
date
