#!/bin/bash
#SBATCH --job-name=cjepa3_task_whiten
#SBATCH --output=logs/slurm_direct_h1s_task_whiten_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_task_whiten reports/direct_h1s_task_whiten checkpoints/direct_h1s_task_whiten
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job ${SLURM_JOB_ID:-local} task-whitened G fine-tune on $(hostname) ==="
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}
GPU_ID=${GPU_ID:-0}
mem_util=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
mem=${mem_util%%,*}; util=${mem_util##*,}
mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
echo "checking GPU $GPU_ID mem=${mem}MiB util=${util}"
if [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]; then
  echo "refusing ghost-busy GPU mem=${mem}MiB util=${util}"
  exit 3
fi
if [ "${mem:-0}" -ge 1000 ]; then
  echo "assigned GPU occupied mem=${mem}MiB util=${util}; waiting up to 3 h"
  freed=0
  for i in $(seq 1 180); do
    sleep 60
    mem_util=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    mem=${mem_util%%,*}; util=${mem_util##*,}
    mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
    echo "wait $i min mem=${mem}MiB util=${util}"
    if [ "${mem:-0}" -lt 1000 ]; then
      freed=1
      break
    fi
  done
  if [ "$freed" -ne 1 ]; then
    echo "refusing occupied GPU mem=${mem}MiB"
    exit 2
  fi
fi
echo "GPU looks idle; starting task-whitened G"

# Init from G. New run dir. Does not overwrite gr40g / Phase A / caches.
# Frozen V-JEPA. Train adapter+decoder through frozen task-whitened drop-16.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 20 \
    --heads-mode direct_h1s \
    --init-from checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
    --task-whiten \
    --task-whiten-drop-k 16 \
    --extra-same-dir \
    --extra-same-dir-min 207 \
    --extra-same-dir-max 229 \
    --close-weight 1.0 \
    --early-stop-metric balanced_warning \
    --post-fork-stop-floor 0.88 \
    --stop-recall-floor 0.55 \
    --safe-recall-floor 0.82 \
    --max-safe-stop-rate 0.04 \
    --patience 6 \
    --patience-start-epoch 3 \
    --stop-hinge-weight 1.0 \
    --safe-hinge-weight 0.5 \
    --small-blob-weight 2.0 \
    --adapter-lr 0.0001 \
    --warmup-epochs 2 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_task_whiten \
    --logs-root logs/direct_h1s_task_whiten \
    --reports-root reports/direct_h1s_task_whiten

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_task_whiten/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_task_whiten/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_task_whiten/warning_confusion.json \
    --checkpoints \
        tw=checkpoints/direct_h1s_task_whiten/teacher_phase_a_best.pt \
        gr40g=checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt

echo "=== task-whitened G complete ==="
date
