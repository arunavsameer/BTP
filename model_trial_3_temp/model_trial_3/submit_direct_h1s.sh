#!/bin/bash
#SBATCH --job-name=cjepa3_h1s
#SBATCH --output=logs/slurm_direct_h1s_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s reports/direct_h1s checkpoints/direct_h1s
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID on $(hostname) ==="
echo "Date: $(date)"
nvidia-smi -L

# Direct Z_t -> H_1s. Side folder so Phase A epoch-5 best.pt is not overwritten.
# Reuses the existing token cache (fingerprint does not include heads.mode).
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 8 \
    --heads-mode direct_h1s \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s \
    --logs-root logs/direct_h1s \
    --reports-root reports/direct_h1s

echo "=== direct H_1s frozen run complete ==="
echo "Date: $(date)"
