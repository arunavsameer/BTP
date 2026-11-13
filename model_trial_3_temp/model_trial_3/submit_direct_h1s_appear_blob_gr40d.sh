#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob_gr40d
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_gr40d_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob_gr40d reports/direct_h1s_appear_blob_gr40d checkpoints/direct_h1s_appear_blob_gr40d
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID appear+blob gr40d dual SAFE/STOP hinge on $(hostname) ==="
date
nvidia-smi -L

# Same token cache. New dir only. Does not touch Phase A / appear_blob / gr40 / gr40b / gr40c.
# Start from gr40c epoch 0 (already high STOP, lower FA than epoch 1).
# Dual peak hinge: keep STOP >= 0.65 on GT-STOP, push GT-SAFE peaks below 0.25.
# small_blob 2 (not 4) to limit smear. Checkpoint on min(SAFE, STOP) with FA cap.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --init-from checkpoints/direct_h1s_appear_blob_gr40c/teacher_phase_a_epoch_00.pt \
    --early-stop-metric balanced_warning \
    --post-fork-stop-floor 0.90 \
    --patience 4 \
    --patience-start-epoch 5 \
    --stop-hinge-weight 1.0 \
    --safe-hinge-weight 1.5 \
    --small-blob-weight 2.0 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob_gr40d \
    --logs-root logs/direct_h1s_appear_blob_gr40d \
    --reports-root reports/direct_h1s_appear_blob_gr40d

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40d/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob_gr40d/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40d/warning_confusion.json \
    --checkpoints \
        gr40d=checkpoints/direct_h1s_appear_blob_gr40d/teacher_phase_a_best.pt \
        gr40c=checkpoints/direct_h1s_appear_blob_gr40c/teacher_phase_a_best.pt \
        gr40b=checkpoints/direct_h1s_appear_blob_gr40b/teacher_phase_a_best.pt

echo "=== appear+blob gr40d complete ==="
date
