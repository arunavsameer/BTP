#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob_gr40
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_gr40_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob_gr40 reports/direct_h1s_appear_blob_gr40 checkpoints/direct_h1s_appear_blob_gr40
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID appear+blob group-random 40 on $(hostname) ==="
date
nvidia-smi -L

# Same loss/heads as appear_blob. New group-random token cache.
# Writes only under *_gr40. Does not touch Phase A or previous run dirs.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --early-stop-metric recall_at_5fa \
    --patience 3 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob_gr40 \
    --logs-root logs/direct_h1s_appear_blob_gr40 \
    --reports-root reports/direct_h1s_appear_blob_gr40

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob_gr40/teacher_phase_a_best.pt

echo "=== appear+blob group-random 40 complete ==="
date
