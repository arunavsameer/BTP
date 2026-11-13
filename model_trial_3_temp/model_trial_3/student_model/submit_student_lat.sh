#!/bin/bash
#SBATCH --job-name=cjepa3_student_lat
#SBATCH --output=logs/slurm_student_lat_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/student_lat reports/student_lat checkpoints/student_lat
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID student_lat (JEPA L1 on F, heatmap eval-only) on $(hostname) ==="
date
nvidia-smi -L

# Does not touch student_g / g2 / g3 / Phase A / gr40b-h / caches.
# Train encoder+predictor only. Frozen G decoder used at val to score heatmaps, not in the loss.
# Losses: V-JEPA L1(F,F_t) + L1(Z,Z_t) + cosine + MGD + spatial relations.
python student_model/train.py \
    --teacher-ckpt checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
    --frozen-epochs 100 \
    --extra-same-dir \
    --extra-same-dir-min 207 \
    --extra-same-dir-max 229 \
    --close-weight 1.0 \
    --feat-weight 1.0 \
    --z-weight 0.3 \
    --cos-weight 0.2 \
    --mgd-weight 0.5 \
    --rel-weight 0.2 \
    --kd-weight 0.0 \
    --gt-weight 0.0 \
    --at-weight 0.0 \
    --distill-warmup-epochs 0 \
    --predictor-lr 0.0003 \
    --early-stop-metric latent_l1 \
    --patience 8 \
    --patience-start-epoch 5 \
    --adapter-lr 0.0003 \
    --warmup-epochs 5 \
    --val-subset-fraction 1.0 \
    --num-workers 4 \
    --cache-frames \
    --prepare-frame-cache \
    --frame-cache-workers 8 \
    --frame-cache-max-gb 80 \
    --checkpoint-root checkpoints/student_lat \
    --logs-root logs/student_lat \
    --reports-root reports/student_lat

echo "=== student_lat complete ==="
date
