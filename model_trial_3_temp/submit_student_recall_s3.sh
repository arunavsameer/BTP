#!/bin/bash
#SBATCH --job-name=s3_recall
#SBATCH --output=logs/slurm_student_recall_s3_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp
mkdir -p logs experiments/student_recall_s3 reports/student_recall_experiments
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache
export PROJECT_SEED=1337

echo "=== Student Recall S3 SLURM GPU $(hostname) ==="
date
nvidia-smi -L
nvidia-smi

/raid/money/miniconda3/bin/python -u scripts/69_student_recall_s3.py --device cuda --epochs 20 --batch 8 --lr 3e-4 --seed 1337 --force

