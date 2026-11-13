#!/usr/bin/env bash
# Single Python process: preflight (if needed) + 30-epoch train — one RGB prepare.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
EXP="$ROOT/experiments/student_final_from_scratch"
REP="$ROOT/reports/student_final_from_scratch"
mkdir -p "$EXP" "$REP"
LOG="$EXP/run.log"
PIDFILE="$EXP/run.pid"
DEVICE="${STUDENT_FINAL_DEVICE:-auto}"
EXTRA_ARGS="${STUDENT_FINAL_EXTRA_ARGS:-}"

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "=== smoke $(date -Is) ===" | tee -a "$LOG"
  python3 scripts/70_student_final_smoke.py --config configs/default.yaml --device "$DEVICE" | tee -a "$LOG"
fi

{
  echo "=== train $(date -Is) device=$DEVICE ==="
  python3 scripts/69_student_final_from_scratch.py \
    --config configs/default.yaml \
    --epochs 30 \
    --device "$DEVICE" \
    $EXTRA_ARGS
} >>"$LOG" 2>&1 &

echo $! >"$PIDFILE"
echo "Started PID=$(cat "$PIDFILE")"
echo "Status: $ROOT/check_student_final_status.sh"
echo "Results: $REP/RESULTS_SUMMARY.txt"
