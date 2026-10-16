#!/bin/bash
#SBATCH --job-name=cjepa3_nolbg
#SBATCH --output=logs/slurm_nolbg_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/ablation_nolbg reports/ablation_nolbg checkpoints/ablation_nolbg
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

# 1 epoch, L_bg off. Writes to a side folder so epoch-5 best.pt is not overwritten.
python scripts/04_train_teacher.py \
    --phase frozen \
    --frozen-epochs 1 \
    --bg-weight 0 \
    --num-workers 4 \
    --checkpoint-root checkpoints/ablation_nolbg \
    --logs-root logs/ablation_nolbg \
    --reports-root reports/ablation_nolbg

python scripts/09_probe_z_bg.py \
    --checkpoint checkpoints/ablation_nolbg/teacher_phase_a_last.pt \
    --split val \
    --n-clips 64 \
    --batch-size 4 \
    --out reports/ablation_nolbg/z_bg_probe_val.json
