#!/bin/bash
#SBATCH --job-name=cjepa3_u1
#SBATCH --output=logs/slurm_u1_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs/direct_h1s_u1 reports/direct_h1s_u1 checkpoints/direct_h1s_u1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID U1 counterfactual variance on $(hostname) ==="
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
mem_util=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
mem=${mem_util%%,*}; util=${mem_util##*,}
mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
DEVICE=auto
if [ "${mem:-0}" -ge 1000 ]; then
  echo "assigned GPU occupied mem=${mem}MiB util=${util}; U1 is tiny, falling back to CPU"
  DEVICE=cpu
  export CUDA_VISIBLE_DEVICES=""
elif [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]; then
  echo "assigned GPU ghost-busy mem=${mem}MiB util=${util}; falling back to CPU"
  DEVICE=cpu
  export CUDA_VISIBLE_DEVICES=""
else
  echo "GPU looks idle; starting U1 on GPU"
fi

# Does not touch G / Phase A / student_* / token cache / frame cache.
ORACLE_FLAGS=()
if [ -f reports/direct_h1s_u1/oracles.json ] && [ -f reports/direct_h1s_u1/dump_val.npz ]; then
  ORACLE_FLAGS+=(--reuse-dump)
fi
python scripts/20_u1_oracles.py \
    --device "$DEVICE" \
    --teacher-ckpt checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
    --out-root reports/direct_h1s_u1 \
    "${ORACLE_FLAGS[@]}"

python scripts/21_train_u1.py \
    --device "$DEVICE" \
    --teacher-ckpt checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt \
    --out-root reports/direct_h1s_u1 \
    --checkpoint-root checkpoints/direct_h1s_u1 \
    --logs-root logs/direct_h1s_u1 \
    --epochs 30 \
    --patience 8 \
    --lr 0.001 \
    --pos-weight 9.0 \
    --batch-size 16 \
    --num-workers 4

echo "=== U1 complete ==="
date
