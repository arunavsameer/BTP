#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob_gr40c
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_gr40c_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob_gr40c reports/direct_h1s_appear_blob_gr40c checkpoints/direct_h1s_appear_blob_gr40c
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID appear+blob gr40c pre-fork STOP hinge on $(hostname) ==="
date
nvidia-smi -L

# Same group-random 40 token cache. New dir only.
# Does not touch Phase A, appear_blob, gr40, or gr40b.
# Init from gr40b best (weights only). STOP hinge + small-blob upweight.
# Checkpoint on pre-fork STOP recall with post-fork STOP floor 0.90.
# Patience starts after warmup (epoch 5).
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --init-from checkpoints/direct_h1s_appear_blob_gr40b/teacher_phase_a_best.pt \
    --early-stop-metric prefork_stop \
    --post-fork-stop-floor 0.90 \
    --patience 4 \
    --patience-start-epoch 5 \
    --stop-hinge-weight 1.0 \
    --small-blob-weight 4.0 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob_gr40c \
    --logs-root logs/direct_h1s_appear_blob_gr40c \
    --reports-root reports/direct_h1s_appear_blob_gr40c

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40c/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob_gr40c/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40c/warning_confusion.json \
    --checkpoints \
        gr40c=checkpoints/direct_h1s_appear_blob_gr40c/teacher_phase_a_best.pt \
        gr40b=checkpoints/direct_h1s_appear_blob_gr40b/teacher_phase_a_best.pt

echo "=== appear+blob gr40c complete ==="
date
