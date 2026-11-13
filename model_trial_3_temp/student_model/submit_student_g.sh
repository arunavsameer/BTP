#!/bin/bash
#SBATCH --job-name=cjepa3_student_g
#SBATCH --output=logs/slurm_student_g_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/student_g reports/student_g checkpoints/student_g
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID student distilled from gr40g on $(hostname) ==="
date
nvidia-smi -L

# RGB student <1M params. Frozen G decoder/bottleneck/close head.
# Teacher Giant is NOT loaded: token cache + dummy 1408-D shell for the adapter.
# Builds 256^2 uint8 frame cache on /raid (~68 GB, preview.mp4 only), then trains.
# Does not touch Phase A / appear_blob / gr40b-h or the token cache.
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
    --gt-weight 0.5 \
    --at-weight 0.1 \
    --distill-warmup-epochs 3 \
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
    --checkpoint-root checkpoints/student_g \
    --logs-root logs/student_g \
    --reports-root reports/student_g

echo "=== student_g complete ==="
date
