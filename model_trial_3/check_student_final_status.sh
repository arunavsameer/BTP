#!/usr/bin/env bash
# Quick status for student_final_from_scratch job.
ROOT="$(cd "$(dirname "$0")" && pwd)"
EXP="$ROOT/experiments/student_final_from_scratch"
REP="$ROOT/reports/student_final_from_scratch"
PIDFILE="$EXP/run.pid"
if [[ -f "$PIDFILE" ]]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "RUNNING pid=$PID"
  else
    echo "NOT RUNNING (stale pid=$PID)"
  fi
else
  pgrep -af "69_student_final_from_scratch" || echo "NOT RUNNING (no pid file)"
fi
[[ -f "$EXP/.success" ]] && echo "DONE: .success present"
[[ -f "$EXP/.failed" ]] && echo "FAILED: see $EXP/.failed"
[[ -f "$REP/RESULTS_SUMMARY.txt" ]] && echo "--- RESULTS ---" && cat "$REP/RESULTS_SUMMARY.txt"
[[ -f "$REP/preflight_report.md" ]] && echo "--- Preflight (head) ---" && head -15 "$REP/preflight_report.md"
echo "Log tail: $EXP/run.log"
tail -5 "$EXP/run.log" 2>/dev/null || true
