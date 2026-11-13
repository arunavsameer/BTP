#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob_gr40h
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_gr40h_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob_gr40h reports/direct_h1s_appear_blob_gr40h checkpoints/direct_h1s_appear_blob_gr40h
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID gr40h from-scratch Beta evidential heatmap on $(hostname) ==="
date
nvidia-smi -L

# From-scratch adapter (no --init-from). Does not touch Phase A / appear_blob / gr40b-g.
# Same G recipe (close aux, extras 207-229, dual hinges, FA cap 0.04) plus:
#   Beta NLL on H_1s (Ferrari mean-precision; Gneiting proper scoring)
#   evidential |μ-y|φ (Amini NeurIPS 2020 analogue)
#   noisy-OR P(any cell ≥ 0.60) STOP/SAFE hinges (Kendall & Gal tails)
# Inference still uses max(μ) like G. Precision is train-only.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --extra-same-dir \
    --extra-same-dir-min 207 \
    --extra-same-dir-max 229 \
    --close-weight 1.0 \
    --beta-nll-weight 0.4 \
    --evidential-weight 0.05 \
    --prob-stop-weight 0.5 \
    --prob-safe-weight 0.5 \
    --early-stop-metric balanced_warning \
    --post-fork-stop-floor 0.88 \
    --stop-recall-floor 0.55 \
    --safe-recall-floor 0.82 \
    --max-safe-stop-rate 0.04 \
    --patience 6 \
    --patience-start-epoch 8 \
    --stop-hinge-weight 1.0 \
    --safe-hinge-weight 0.5 \
    --small-blob-weight 2.0 \
    --adapter-lr 0.0003 \
    --warmup-epochs 5 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob_gr40h \
    --logs-root logs/direct_h1s_appear_blob_gr40h \
    --reports-root reports/direct_h1s_appear_blob_gr40h

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40h/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob_gr40h/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40h/warning_confusion.json \
    --checkpoints \
        gr40h=checkpoints/direct_h1s_appear_blob_gr40h/teacher_phase_a_best.pt \
        gr40g=checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
        gr40e=checkpoints/direct_h1s_appear_blob_gr40e/teacher_phase_a_best.pt \
        gr40d=checkpoints/direct_h1s_appear_blob_gr40d/teacher_phase_a_best.pt

echo "=== appear+blob gr40h complete ==="
date
