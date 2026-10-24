#!/bin/bash
#SBATCH --job-name=cjepa3_appear_blob_gr40b
#SBATCH --output=logs/slurm_direct_h1s_appear_blob_gr40b_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_appear_blob_gr40b reports/direct_h1s_appear_blob_gr40b checkpoints/direct_h1s_appear_blob_gr40b
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID appear+blob gr40b logits-BCE lr3e-4 on $(hostname) ==="
date
nvidia-smi -L

# Group-random 40 cache. Logits BCE, global rec@5FA with FA/mean gates, lr 3e-4, warmup 5.
# New dir only. Does not touch Phase A, appear_blob, or gr40.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --early-stop-metric recall_at_5fa \
    --patience 3 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_appear_blob_gr40b \
    --logs-root logs/direct_h1s_appear_blob_gr40b \
    --reports-root reports/direct_h1s_appear_blob_gr40b

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out reports/direct_h1s_appear_blob_gr40b/full_val_operating.json \
    --checkpoints best=checkpoints/direct_h1s_appear_blob_gr40b/teacher_phase_a_best.pt

echo "=== appear+blob gr40b complete ==="
date
