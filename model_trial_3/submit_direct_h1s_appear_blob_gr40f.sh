#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob_gr40f
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_gr40f_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob_gr40f reports/direct_h1s_appear_blob_gr40f checkpoints/direct_h1s_appear_blob_gr40f
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID gr40f hold-only extras + disappear from gr40e on $(hostname) ==="
date
nvidia-smi -L

# New dir only. Does not touch Phase A / appear_blob / gr40 / gr40b / gr40c / gr40d / gr40e.
# Init gr40e best (receding already learned). Hold-only 210-219 as a light reminder.
# No closing 207-209. No pulling 220-229. Disappear 8x on occupied-now empty-1s.
# SAFE hinge 0.5 (was 1.5). STOP hinge stays 0.65. LR 1e-4, warmup 2.
# post-fork floor 0.88 so an epoch-5-style STOP gain can win. STOP floor 0.67.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --init-from checkpoints/direct_h1s_appear_blob_gr40e/teacher_phase_a_best.pt \
    --extra-same-dir \
    --extra-same-dir-min 210 \
    --extra-same-dir-max 219 \
    --disappear-weight 8.0 \
    --early-stop-metric balanced_warning \
    --post-fork-stop-floor 0.88 \
    --stop-recall-floor 0.67 \
    --patience 4 \
    --patience-start-epoch 3 \
    --stop-hinge-weight 1.0 \
    --safe-hinge-weight 0.5 \
    --small-blob-weight 2.0 \
    --adapter-lr 0.0001 \
    --warmup-epochs 2 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob_gr40f \
    --logs-root logs/direct_h1s_appear_blob_gr40f \
    --reports-root reports/direct_h1s_appear_blob_gr40f

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40f/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob_gr40f/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40f/warning_confusion.json \
    --checkpoints \
        gr40f=checkpoints/direct_h1s_appear_blob_gr40f/teacher_phase_a_best.pt \
        gr40e=checkpoints/direct_h1s_appear_blob_gr40e/teacher_phase_a_best.pt \
        gr40d=checkpoints/direct_h1s_appear_blob_gr40d/teacher_phase_a_best.pt

echo "=== appear+blob gr40f complete ==="
date
