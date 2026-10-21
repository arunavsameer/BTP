#!/bin/bash
#SBATCH --job-name=cjepa3_tokencache
#SBATCH --output=logs/slurm_prepare_token_cache_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs reports
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID group-random token cache on $(hostname) ==="
date
nvidia-smi -L

python scripts/02_prepare_token_cache.py --device auto

echo "=== token cache complete ==="
date
