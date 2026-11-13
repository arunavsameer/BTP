#!/bin/bash
#SBATCH --job-name=cjepa3_dump
#SBATCH --output=logs/slurm_dense_dump_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs reports/dense_dump
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

python scripts/13_dense_dump.py \
    --stride 6 \
    --device auto \
    --encode-missing \
    --batch 8 \
    --out reports/dense_dump \
    --models \
        "phase_a=checkpoints/teacher_phase_a_best.pt:residual" \
        "mse17=checkpoints/direct_h1s_mse_until/teacher_phase_a_best.pt:direct_h1s" \
        "beta2=checkpoints/direct_h1s_beta2/teacher_phase_a_best.pt:direct_h1s"
