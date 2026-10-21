#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob reports/direct_h1s_appear_blob checkpoints/direct_h1s_appear_blob
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID appear-weight + blob BCE stride 15 on $(hostname) ==="
date
nvidia-smi -L

# Direct H_1s, current default: soft BCE + appear extra + 3x3 blob BCE.
# Train/val stride 15. Reuses the existing token cache.
# Phase A best.pt and direct_h1s_alpha are not touched.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --early-stop-metric recall_at_5fa \
    --patience 3 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob \
    --logs-root logs/direct_h1s_appear_blob \
    --reports-root reports/direct_h1s_appear_blob

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob/teacher_phase_a_best.pt

echo "=== appear-weight + blob BCE stride 15 complete ==="
date
