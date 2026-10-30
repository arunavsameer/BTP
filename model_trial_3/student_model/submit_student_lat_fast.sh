#!/bin/bash
#SBATCH --job-name=cjepa3_student_lat_fast
#SBATCH --output=logs/slurm_student_lat_fast_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/student_lat_fast reports/student_lat_fast checkpoints/student_lat_fast
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID student_lat_fast (thin 64-ch trunk, original lat losses) on $(hostname) ==="
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
mem_util=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
mem=${mem_util%%,*}; util=${mem_util##*,}
mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
if [ "${mem:-0}" -ge 1000 ]; then
  echo "refusing occupied GPU mem=${mem}MiB util=${util}"
  exit 2
fi
if [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]; then
  echo "refusing ghost-busy GPU mem=${mem}MiB util=${util}"
  exit 3
fi
echo "GPU looks idle; starting train"

# Does not touch student_lat / lat2 / lat3 / lat4 / g / g2 / g3 / Phase A / caches.
# Same frozen G decoder. Original latent-only JEPA losses. Thin 64-ch 3D encoder.
python student_model/train.py \
    --teacher-ckpt checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
    --encoder-variant fast \
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
    --h-weight 0.0 \
    --peak-stop-weight 0.0 \
    --peak-safe-weight 0.0 \
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
    --checkpoint-root checkpoints/student_lat_fast \
    --logs-root logs/student_lat_fast \
    --reports-root reports/student_lat_fast

echo "=== student_lat_fast complete ==="
date
