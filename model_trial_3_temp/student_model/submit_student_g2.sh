#!/bin/bash
#SBATCH --job-name=cjepa3_student_g2
#SBATCH --output=logs/slurm_student_g2_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/student_g2 reports/student_g2 checkpoints/student_g2
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID student_g2 (G-init decoder trainable, cosine distill) on $(hostname) ==="
date
nvidia-smi -L

# Does not touch student_g / Phase A / gr40b-h / token cache / frame cache (reused).
# Decoder+bottleneck+close start from G then train. Feature loss is cosine, not raw MSE.
python student_model/train.py \
    --teacher-ckpt checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
    --frozen-epochs 100 \
    --extra-same-dir \
    --extra-same-dir-min 207 \
    --extra-same-dir-max 229 \
    --close-weight 1.0 \
    --feat-weight 1.0 \
    --z-weight 0.3 \
    --kd-weight 1.0 \
    --gt-weight 1.0 \
    --at-weight 0.2 \
    --distill-warmup-epochs 0 \
    --decoder-lr 0.0001 \
    --early-stop-metric balanced_warning \
    --post-fork-stop-floor 0.88 \
    --stop-recall-floor 0.55 \
    --safe-recall-floor 0.82 \
    --max-safe-stop-rate 0.04 \
    --patience 6 \
    --patience-start-epoch 8 \
    --stop-hinge-weight 1.0 \
    --safe-hinge-weight 0.5 \
    --small-blob-weight 2.0 \
    --adapter-lr 0.0003 \
    --warmup-epochs 5 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --cache-frames \
    --prepare-frame-cache \
    --frame-cache-workers 8 \
    --frame-cache-max-gb 80 \
    --checkpoint-root checkpoints/student_g2 \
    --logs-root logs/student_g2 \
    --reports-root reports/student_g2

echo "=== student_g2 complete ==="
date
