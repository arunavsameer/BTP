#!/bin/bash
#SBATCH --job-name=cjepa3_phA
#SBATCH --output=logs/slurm_phaseA_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

echo "=== SLURM job $SLURM_JOB_ID on $(hostname) ==="
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# ── paths ──
PROJECT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
cd "$PROJECT"
mkdir -p logs checkpoints reports /raid/asrivastava/VJEPA2_BTP/model_trial_3/cache

# ── environment ──
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

# ── step 1: manifest + splits (first 200 samples, ~800 clips) ──
echo ""
echo "=== Step 1: build manifest (200 samples) ==="
python scripts/01_build_manifest.py --reset-split --max-samples 200

# ── step 2: prepare frame cache (letterbox 256×256, ~67 GB) ──
echo ""
echo "=== Step 2: prepare frame cache ==="
python scripts/02_prepare_cache.py

# ── step 3: train Phase A (frozen V-JEPA, 8 epochs) ──
echo ""
echo "=== Step 3: train Phase A (frozen, 8 epochs) ==="
python scripts/04_train_teacher.py \
    --phase frozen \
    --num-workers 4

echo ""
echo "=== Phase A training complete ==="
echo "Date: $(date)"
