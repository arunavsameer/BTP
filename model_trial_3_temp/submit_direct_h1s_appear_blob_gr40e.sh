#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob_gr40e
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_gr40e_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob_gr40e reports/direct_h1s_appear_blob_gr40e checkpoints/direct_h1s_appear_blob_gr40e
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID gr40e same-dir extras 207-229 from gr40d on $(hostname) ==="
date
nvidia-smi -L

# New dir only. Does not touch Phase A / appear_blob / gr40 / gr40b / gr40c / gr40d.
# Init gr40d best. Same dual-hinge recipe. extra_same_dir default on (train only).
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --init-from checkpoints/direct_h1s_appear_blob_gr40d/teacher_phase_a_best.pt \
    --extra-same-dir \
    --early-stop-metric balanced_warning \
    --post-fork-stop-floor 0.90 \
    --patience 4 \
    --patience-start-epoch 5 \
    --stop-hinge-weight 1.0 \
    --safe-hinge-weight 1.5 \
    --small-blob-weight 2.0 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob_gr40e \
    --logs-root logs/direct_h1s_appear_blob_gr40e \
    --reports-root reports/direct_h1s_appear_blob_gr40e

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40e/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob_gr40e/teacher_phase_a_best.pt

python scripts/17_warning_confusion.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40e/warning_confusion.json \
    --checkpoints \
        gr40e=checkpoints/direct_h1s_appear_blob_gr40e/teacher_phase_a_best.pt \
        gr40d=checkpoints/direct_h1s_appear_blob_gr40d/teacher_phase_a_best.pt

echo "=== appear+blob gr40e complete ==="
date
