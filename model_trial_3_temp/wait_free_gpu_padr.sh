#!/bin/bash
# CPU waiter: do not hold a SLURM GPU. Launch PADR on the first free GPU in 1..7.
set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs
echo "=== PADR free-GPU waiter start $(date) pid=$$ ==="

while true; do
  for i in 1 2 3 4 5 6 7; do
    mem_util=$(nvidia-smi -i "$i" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    mem=${mem_util%%,*}; util=${mem_util##*,}
    mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
    echo "$(date '+%F %T') gpu $i mem=${mem}MiB util=${util}"
    if [ "${mem:-0}" -lt 1000 ]; then
      echo "GPU $i is free; launching PADR"
      export CUDA_VISIBLE_DEVICES=$i
      export SKIP_GPU_GUARD=1
      export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
      export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache
      nvidia-smi -i "$i" --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
      exec bash submit_direct_h1s_padr.sh
    fi
  done
  sleep 30
done
