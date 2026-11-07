#!/bin/bash
# 15-epoch 144-ch 1s teacher. Requests 2 Slurm GPUs so we can skip a dead/busy
# card and train on the other allocated one. Does not remap onto a third job.
#
#SBATCH --job-name=cjepa4_h1s_v3
#SBATCH --output=logs/slurm_teacher_h1s_v3_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_teacher_h1s_v3_gpu2.sh" >&2
  exit 1
fi

cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
echo "=== SLURM job $SLURM_JOB_ID teacher_h1s_v3 2-GPU picker on $(hostname) ==="
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv

pick=""
while IFS=, read -r idx uuid mem util; do
  idx=${idx// /}; uuid=${uuid// /}; mem=${mem// /}; util=${util// /}
  mem=${mem%.*}; util=${util%.*}
  echo "probe idx=$idx mem=${mem}MiB util=$util uuid=$uuid"
  if [ "${mem:-0}" -ge 1000 ]; then
    echo "skip occupied $idx"
    continue
  fi
  if [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]; then
    echo "skip ghost-dead $idx"
    continue
  fi
  pick=$idx
  break
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)

if [ -z "$pick" ]; then
  echo "no idle GPU in this 2-GPU allocation" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=$pick
echo "training on allocated GPU index $pick (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
exec bash submit_teacher_h1s_v3.sh
