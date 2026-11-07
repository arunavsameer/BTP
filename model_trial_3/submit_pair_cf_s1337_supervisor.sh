#!/bin/bash
# CPU-only SLURM wrapper: keeps polling for a free non-ghost GPU and runs/relaunches pair-CF training.
#SBATCH --job-name=pair_cf_sup_s1337
#SBATCH --output=logs/slurm_pair_cf_supervisor_s1337_%j.out
# Prefer DGX2 head/login-style nodes where all GPUs are visible without holding a GPU allocation.
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --nodelist=DGX2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
# Note: no --gres=gpu — polls nvidia-smi and runs training with CUDA_VISIBLE_DEVICES set.

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs

if [ -f logs/wait_and_run_pair_cf_s1337.pid ]; then
  old=$(cat logs/wait_and_run_pair_cf_s1337.pid || true)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "Supervisor already running pid=$old"
    exit 0
  fi
fi

exec bash wait_and_run_pair_cf_s1337.sh
