#!/bin/bash
#SBATCH --job-name=cjepa3_zbg
#SBATCH --output=logs/slurm_zbg_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs reports
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

python scripts/09_probe_z_bg.py --checkpoint best --split val --n-clips 64 --batch-size 4
