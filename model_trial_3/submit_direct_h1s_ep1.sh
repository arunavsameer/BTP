#!/bin/bash
#SBATCH --job-name=cjepa3_ep1
#SBATCH --output=logs/slurm_direct_h1s_ep1_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_ep1 reports/direct_h1s_ep1 checkpoints/direct_h1s_ep1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 2 \
    --heads-mode direct_h1s \
    --early-stop-metric val_mse \
    --patience 99 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_ep1 \
    --logs-root logs/direct_h1s_ep1 \
    --reports-root reports/direct_h1s_ep1
