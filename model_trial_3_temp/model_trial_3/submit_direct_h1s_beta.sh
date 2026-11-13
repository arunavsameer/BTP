#!/bin/bash
#SBATCH --job-name=cjepa3_bX
#SBATCH --output=logs/slurm_direct_h1s_beta_%x_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
: "${BETA:?set BETA}"
TAG="beta${BETA%.*}"
mkdir -p "logs/direct_h1s_${TAG}" "reports/direct_h1s_${TAG}" "checkpoints/direct_h1s_${TAG}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID beta=$BETA on $(hostname) ==="
date
nvidia-smi -L

python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 100 \
    --heads-mode direct_h1s \
    --recall-beta "$BETA" \
    --early-stop-metric recall_at_5fa \
    --patience 3 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --checkpoint-root "checkpoints/direct_h1s_${TAG}" \
    --logs-root "logs/direct_h1s_${TAG}" \
    --reports-root "reports/direct_h1s_${TAG}"

python scripts/11_score_h1s_operating.py \
    --device auto \
    --out "reports/direct_h1s_${TAG}/full_val_operating.json" \
    --checkpoints "best=checkpoints/direct_h1s_${TAG}/teacher_phase_a_best.pt"

echo "=== beta $BETA complete ==="
date
