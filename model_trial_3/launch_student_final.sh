#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DEVICE="${STUDENT_FINAL_DEVICE:-auto}"
export SKIP_SMOKE=1

echo "Running test suite..."
python3 scripts/71_student_final_test_suite.py --device "$DEVICE"

rm -f experiments/student_final_from_scratch/.failed
export STUDENT_FINAL_EXTRA_ARGS="${STUDENT_FINAL_EXTRA_ARGS:-}"

if [[ -f reports/student_final_from_scratch/preflight_report.json ]]; then
  PROCEED="$(python3 -c "import json;print(json.load(open('reports/student_final_from_scratch/preflight_report.json'))['proceed_to_training'])")"
  if [[ "$PROCEED" == "True" ]]; then
    export STUDENT_FINAL_EXTRA_ARGS="--skip-preflight $STUDENT_FINAL_EXTRA_ARGS"
    echo "Will skip preflight (already passed)."
  fi
fi

exec "$ROOT/run_student_final_from_scratch.sh"
