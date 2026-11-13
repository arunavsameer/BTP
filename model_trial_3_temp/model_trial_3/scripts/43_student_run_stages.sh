#!/usr/bin/env bash
# Student architecture research — staged execution (no final model selection).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=reports/student_architecture_tests
mkdir -p logs "$OUT"

echo "Stage A: Parts 1–2"
python3 scripts/30_student_part1_parent.py --out "$OUT" --device "${CUDA_DEVICE:-cuda}"
python3 scripts/31_student_part2_audit.py --out "$OUT"
python3 -m unittest tests.test_student_grid_align -q

echo "Stage B–D: pilots (requires teacher_cache/)"
python3 scripts/42_student_research_suite.py --out "$OUT" --stage all --epochs "${PILOT_EPOCHS:-4}" --max-rows "${PILOT_MAX_ROWS:-2000}" --device "${CUDA_DEVICE:-cuda}"

echo "Artifacts → $OUT"
