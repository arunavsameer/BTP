#!/bin/bash
#SBATCH --job-name=cjepa3_alpha
#SBATCH --output=logs/slurm_direct_h1s_alpha_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_alpha reports/direct_h1s_alpha checkpoints/direct_h1s_alpha
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID alpha BCE+fine stride 15 on $(hostname) ==="
date
nvidia-smi -L

# Direct H_1s, current default: soft BCE + 16×16 decode then max-pool.
# Train/val stride 15 (original). Reuses the existing token cache.
# Phase A best.pt is not touched.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --early-stop-metric recall_at_5fa \
    --patience 3 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_alpha \
    --logs-root logs/direct_h1s_alpha \
    --reports-root reports/direct_h1s_alpha

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_alpha/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_alpha/teacher_phase_a_best.pt

echo "=== alpha BCE+fine stride 15 complete ==="
date
