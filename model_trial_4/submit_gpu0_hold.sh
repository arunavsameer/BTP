#!/bin/bash
# Occupies the lowest Slurm GPU index (usually the dead GPU 0) so a
# real train job can land on a working card. Does not run CUDA.
#
#SBATCH --job-name=gpu0_hold
#SBATCH --output=logs/slurm_gpu0_hold_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
mkdir -p logs
echo "HOLD job=${SLURM_JOB_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} HOST=$(hostname)"
date
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
sleep 7200
