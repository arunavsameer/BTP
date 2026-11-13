#!/usr/bin/env bash
# Poll until a free non-ghost GPU appears; run treatment (control skipped if done); retry on failure.
set -uo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs
# shellcheck source=pair_cf_gpu_lib.sh
source "$(dirname "$0")/pair_cf_gpu_lib.sh"

LOG=logs/wait_and_run_pair_cf_s1337.log
PIDFILE=logs/wait_and_run_pair_cf_s1337.pid
DONE=logs/pair_cf_s1337_supervisor.done

exec >>"$LOG" 2>&1
echo "=== pair_cf supervisor start $(date) pid=$$ ==="
echo $$ >"$PIDFILE"

POLL_SEC=${PAIR_CF_POLL_SEC:-30}

training_complete() {
  [ -f checkpoints/pair_cf_treatment_s1337/.train_success ]
}

if training_complete; then
  echo "Treatment already complete; exit"
  touch "$DONE"
  exit 0
fi

fail_streak=0
while true; do
  if training_complete; then
    echo "=== pair_cf supervisor SUCCESS $(date) ==="
    touch "$DONE"
    rm -f "$PIDFILE"
    exit 0
  fi

  picked=""
  for i in 1 2 3 4 5 6 7; do
    if pair_cf_gpu_usable "$i"; then
      picked=$i
      break
    fi
    mu=$(pair_cf_gpu_mem_util "$i" || echo "? ?")
    echo "$(date '+%F %T') poll gpu $i mem/util=$mu"
  done
  if [ -z "$picked" ] && [ "${PAIR_CF_ALLOW_GPU0:-0}" = "1" ] && pair_cf_gpu_usable 0; then
    picked=0
  fi

  if [ -z "$picked" ]; then
    sleep "$POLL_SEC"
    continue
  fi

  export CUDA_VISIBLE_DEVICES=$picked
  echo "$(date '+%F %T') free GPU $picked — launching"
  nvidia-smi -i "$picked" --query-gpu=index,memory.used,utilization.gpu --format=csv

  if bash "$(dirname "$0")/run_pair_cf_train_s1337.sh"; then
    fail_streak=0
    if training_complete; then
      echo "=== pair_cf supervisor SUCCESS $(date) ==="
      touch "$DONE"
      rm -f "$PIDFILE"
      exit 0
    fi
  else
    fail_streak=$((fail_streak + 1))
    echo "$(date '+%F %T') run failed (streak=$fail_streak); resume polling"
    sleep "$POLL_SEC"
  fi
done
