# shellcheck shell=bash
# Poll for a free GPU (1–7) and set CUDA_VISIBLE_DEVICES.

pair_cf_is_ghost() {
  local mem=$1 util=$2
  [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]
}

pair_cf_gpu_mem_util() {
  local i=$1
  local mem_util
  mem_util=$(nvidia-smi -i "$i" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
  if [ -z "$mem_util" ] || [[ "$mem_util" == *"No devices"* ]] || [[ "$mem_util" == *"not found"* ]]; then
    return 1
  fi
  local mem=${mem_util%%,*}; local util=${mem_util##*,}
  mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
  echo "$mem $util"
}

# Free = mem < threshold (default 1000 MiB), not ghost, skip physical GPU 0 by default.
pair_cf_gpu_usable() {
  local i=$1
  local thresh=${PAIR_CF_FREE_MEM_MIB:-1000}
  if [ "${PAIR_CF_SKIP_GPU0:-1}" = "1" ] && [ "$i" = "0" ]; then
    return 1
  fi
  local mu
  mu=$(pair_cf_gpu_mem_util "$i") || return 1
  local mem=${mu%% *}; local util=${mu##* }
  [[ "$mem" =~ ^[0-9]+$ ]] || return 1
  [[ "$util" =~ ^[0-9]+$ ]] || return 1
  [ "$mem" -lt "$thresh" ] || return 1
  pair_cf_is_ghost "$mem" "$util" && return 1
  return 0
}

# First free GPU in 1..7 (then 0 if PAIR_CF_ALLOW_GPU0=1 and free).
pair_cf_pick_free_gpu() {
  local i
  for i in 1 2 3 4 5 6 7; do
    if pair_cf_gpu_usable "$i"; then
      echo "$i"
      return 0
    fi
  done
  if [ "${PAIR_CF_ALLOW_GPU0:-0}" = "1" ] && pair_cf_gpu_usable 0; then
    echo "0"
    return 0
  fi
  return 1
}

pair_cf_assign_free_gpu() {
  local gpu
  gpu=$(pair_cf_pick_free_gpu) || return 1
  export CUDA_VISIBLE_DEVICES=$gpu
  nvidia-smi -i "$gpu" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
  echo "assigned CUDA_VISIBLE_DEVICES=$gpu"
  return 0
}
