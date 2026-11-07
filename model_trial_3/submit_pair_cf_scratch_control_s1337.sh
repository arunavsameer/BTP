#!/bin/bash
#SBATCH --job-name=pair_cf_ctrl_s1337
#SBATCH --output=logs/slurm_pair_cf_scratch_control_s1337_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache
export PROJECT_SEED=1337

CKPT=checkpoints/pair_cf_scratch_control_s1337
mkdir -p "$CKPT" logs/pair_cf_scratch_control_s1337 reports/pair_cf_scratch_control_s1337

echo "=== pair_cf scratch control s1337 on $(hostname) ==="
date
nvidia-smi -L

COMMON=(
  --phase frozen --frozen-epochs 100 --heads-mode direct_h1s
  --extra-same-dir --extra-same-dir-min 207 --extra-same-dir-max 229
  --close-weight 1.0 --early-stop-metric balanced_warning
  --post-fork-stop-floor 0.88 --stop-recall-floor 0.55 --safe-recall-floor 0.82
  --max-safe-stop-rate 0.04 --patience 6 --patience-start-epoch 8
  --stop-hinge-weight 1.0 --safe-hinge-weight 0.5 --small-blob-weight 2.0
  --adapter-lr 0.0003 --warmup-epochs 5 --val-subset-fraction 1.0
  --num-workers 4 --max-steps 28350
)

python scripts/04_train_teacher.py "${COMMON[@]}" \
  --checkpoint-root "$CKPT" \
  --logs-root logs/pair_cf_scratch_control_s1337 \
  --reports-root reports/pair_cf_scratch_control_s1337
