#!/bin/bash
#SBATCH --job-name=cjepa3_student_g3
#SBATCH --output=logs/slurm_student_g3_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/student_g3 reports/student_g3 checkpoints/student_g3
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID student_g3 (SimKD hint + frozen G decoder) on $(hostname) ==="
date
nvidia-smi -L

# Does not touch student_g / student_g2 / Phase A / gr40b-h / token or frame cache.
# Frozen G decoder (g2 unfreeze drifted). Identity-init 1x1 hint + extra IR.
# Feature loss is teacher-standardized smooth-L1 (g: raw MSE drowned GT; g2: cosine dropped scale).
python student_model/train.py \
    --teacher-ckpt checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
    --frozen-epochs 100 \
    --extra-same-dir \
    --extra-same-dir-min 207 \
    --extra-same-dir-max 229 \
    --close-weight 1.0 \
    --feat-weight 1.0 \
    --z-weight 0.4 \
    --kd-weight 2.0 \
    --gt-weight 1.0 \
    --at-weight 0.4 \
    --distill-warmup-epochs 0 \
    --hint-lr 0.001 \
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
    --checkpoint-root checkpoints/student_g3 \
    --logs-root logs/student_g3 \
    --reports-root reports/student_g3

echo "=== student_g3 complete ==="
date
