#!/bin/bash
# Wait for GPU 1-7, then run submit_direct_h1s_padr_ep30.sh
set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs
LOG=logs/padr_ep30_wait_free_gpu.log
echo "=== PADR ep30 waiter start $(date) pid=$$ ===" | tee -a "$LOG"

while true; do
  for i in 1 2 3 4 5 6 7; do
    mem_util=$(nvidia-smi -i "$i" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    mem=${mem_util%%,*}; util=${mem_util##*,}
    mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
    echo "$(date '+%F %T') gpu $i mem=${mem}MiB util=${util}" | tee -a "$LOG"
    if [ "${mem:-0}" -lt 1000 ]; then
      echo "GPU $i free; launching PADR ep30" | tee -a "$LOG"
      export CUDA_VISIBLE_DEVICES=$i
      export SKIP_GPU_GUARD=1
      exec bash submit_direct_h1s_padr_ep30.sh 2>&1 | tee -a "$LOG"
    fi
  done
  sleep 30
done
