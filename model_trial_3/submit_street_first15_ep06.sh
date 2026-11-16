#!/bin/bash
#SBATCH --job-name=street15_ep06
#SBATCH --output=/raid/asrivastava/VJEPA2_BTP/street_first15min_teacher_ep06/slurm_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

OUT_DIR=/raid/asrivastava/VJEPA2_BTP/street_first15min_teacher_ep06
mkdir -p "$OUT_DIR" logs

echo "=== SLURM job ${SLURM_JOB_ID:-local} on $(hostname) ==="
date
nvidia-smi -L
nvidia-smi

python scripts/20_render_street_first15.py \
  --checkpoint checkpoints/pair_cf_treatment_s1337/teacher_phase_a_epoch_06.pt \
  --source /raid/asrivastava/VJEPA2_BTP/street_walking_480p.mp4 \
  --hf-cache /raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache \
  --device cuda \
  --batch-size 8 \
  --duration 900 \
  --tag teacher_phase_a_epoch_06 \
  --out "$OUT_DIR/street_walking_first15min_ep06.mp4"

echo "=== done ==="
ls -lh "$OUT_DIR"
date
