#!/bin/bash
# Sequential from-scratch control (revised G loss) then PADR. Does not overwrite gr40g.
#SBATCH --job-name=cjepa3_padr
#SBATCH --output=logs/slurm_direct_h1s_padr_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

G=checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt
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
    --patience-start-epoch 8
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
  mem_util=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  mem=${mem_util%%,*}; util=${mem_util##*,}
  mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
  echo "checking visible=$GPU_ID physical=${PHYS:-?} mem=${mem}MiB util=${util}"
  # Ghost-busy is physical GPU 0 only. A remapped CUDA index of 0 is not GPU 0.
  if [ "${PHYS:-}" = "0" ]; then
    echo "refusing physical GPU 0"
    exit 3
  fi
  if [ "${mem:-0}" -ge 1000 ]; then
    echo "assigned GPU occupied mem=${mem}MiB; waiting up to 3 h"
    freed=0
    for i in $(seq 1 180); do
      sleep 60
      mem_util=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
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
}

echo "=== PADR from-scratch runs on $(hostname) ==="
date
if [ "${SKIP_GPU_GUARD:-0}" = "1" ]; then
  echo "SKIP_GPU_GUARD=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
else
  gpu_guard
fi

# ----- Control: revised G task loss, no PADR, no L_bg -----
mkdir -p logs/direct_h1s_g_revised reports/direct_h1s_g_revised checkpoints/direct_h1s_g_revised
echo "=== CONTROL revised G loss, from scratch ==="
python scripts/04_train_teacher.py \
    "${COMMON[@]}" \
    --checkpoint-root checkpoints/direct_h1s_g_revised \
    --logs-root logs/direct_h1s_g_revised \
    --reports-root reports/direct_h1s_g_revised

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_g_revised/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_g_revised/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_g_revised/warning_confusion.json \
    --checkpoints \
        grevised=checkpoints/direct_h1s_g_revised/teacher_phase_a_best.pt \
        gr40g="$G"

# ----- PADR: same revised loss + pair consistency + local tangent -----
mkdir -p logs/direct_h1s_padr reports/direct_h1s_padr checkpoints/direct_h1s_padr
echo "=== PADR from scratch ==="
python scripts/04_train_teacher.py \
    "${COMMON[@]}" \
    --padr \
    --padr-warmup-epochs 18 \
    --padr-ramp-epochs 12 \
    --checkpoint-root checkpoints/direct_h1s_padr \
    --logs-root logs/direct_h1s_padr \
    --reports-root reports/direct_h1s_padr

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_padr/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_padr/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_padr/warning_confusion.json \
    --checkpoints \
        padr=checkpoints/direct_h1s_padr/teacher_phase_a_best.pt \
        grevised=checkpoints/direct_h1s_g_revised/teacher_phase_a_best.pt \
        gr40g="$G"

echo "=== PADR from-scratch runs complete ==="
date
