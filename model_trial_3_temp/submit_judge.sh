#!/bin/bash
#SBATCH --job-name=cjepa3_judge
#SBATCH --output=logs/slurm_judge_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/model_judge reports/model_judge/preds
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job ${SLURM_JOB_ID:-local} on $(hostname) ==="
date
nvidia-smi -L || true

python scripts/12_judge_models.py \
    --device auto \
    --n-boot 200 \
    --n-folds 5 \
    --dump-npz-dir reports/model_judge/preds \
    --out reports/model_judge/verdict.json \
    --models \
        "phase_a=checkpoints/teacher_phase_a_best.pt:residual" \
        "direct_score=checkpoints/direct_h1s/teacher_phase_a_best.pt:direct_h1s" \
        "mse17=checkpoints/direct_h1s_mse_until/teacher_phase_a_best.pt:direct_h1s" \
        "ep1=checkpoints/direct_h1s_ep1/teacher_phase_a_epoch_01.pt:direct_h1s" \
        "beta2=checkpoints/direct_h1s_beta2/teacher_phase_a_best.pt:direct_h1s" \
        "beta4=checkpoints/direct_h1s_beta4/teacher_phase_a_best.pt:direct_h1s" \
        "beta8=checkpoints/direct_h1s_beta8/teacher_phase_a_best.pt:direct_h1s"

echo "=== judge complete ==="
date
