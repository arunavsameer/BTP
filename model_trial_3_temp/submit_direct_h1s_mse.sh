#!/bin/bash
#SBATCH --job-name=cjepa3_h1sm
#SBATCH --output=logs/slurm_direct_h1s_mse_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_mse reports/direct_h1s_mse checkpoints/direct_h1s_mse
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID on $(hostname) ==="
echo "Date: $(date)"
nvidia-smi -L

# Same direct Z_t -> H_1s run, but stop when val heatmap MSE does not fall
# for 3 consecutive epochs. best.pt is the last improving epoch (x-3).
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 8 \
    --heads-mode direct_h1s \
    --early-stop-metric val_mse \
    --patience 3 \
    --num-workers 4 \
    --checkpoint-root checkpoints/direct_h1s_mse \
    --logs-root logs/direct_h1s_mse \
    --reports-root reports/direct_h1s_mse

echo "=== direct H_1s val-MSE early-stop run complete ==="
echo "Date: $(date)"
