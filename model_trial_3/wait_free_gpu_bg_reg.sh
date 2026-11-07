#!/usr/bin/env bash
# Poll GPUs 1–7; launch bg_reg R0–R3 when one is free (low mem, not ghost-busy). Skips GPU 0.
set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs
LOG=logs/wait_free_gpu_bg_reg.log
exec > >(tee -a "$LOG") 2>&1

is_ghost() {
  local mem=$1 util=$2
  [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]
}

echo "=== bg_reg GPU waiter start $(date) pid=$$ log=$LOG ==="

while true; do
  for i in 1 2 3 4 5 6 7; do
    if ! nvidia-smi -i "$i" &>/dev/null; then
      continue
    fi
    mem_util=$(nvidia-smi -i "$i" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    mem=${mem_util%%,*}; util=${mem_util##*,}
    mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
    echo "$(date '+%F %T') gpu $i mem=${mem}MiB util=${util}%"
    if [ "${mem:-999999}" -lt 1000 ] && ! is_ghost "$mem" "$util"; then
      echo "GPU $i is free; launching submit_bg_reg_runs.sh"
      export CUDA_VISIBLE_DEVICES=$i
      export SKIP_GPU_GUARD=1
      export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
      export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache
      nvidia-smi -i "$i" --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
      exec bash submit_bg_reg_runs.sh
    fi
  done
  sleep 30
done
