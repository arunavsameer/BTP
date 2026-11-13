#!/bin/bash
#SBATCH --job-name=cjepa3_viz
#SBATCH --output=logs/slurm_viz_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
PROJECT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
cd "$PROJECT"
mkdir -p logs visualizations/test_videos
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

python scripts/08_render_test_comparison_videos.py \
    --checkpoint best \
    --split test \
    --n-clips 5 \
    --stride 1 \
    --batch-size 4
