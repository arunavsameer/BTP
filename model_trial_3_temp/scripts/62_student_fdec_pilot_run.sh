#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
FORCE=()
if [[ "${1:-}" == "--force" ]]; then
  FORCE=(--force)
fi
python3 -u scripts/61_student_fdec_pilot.py --device cuda --variant S0 --epochs 15 --batch 8 --lr 3e-4 --seed 1337 "${FORCE[@]}"
python3 -u scripts/61_student_fdec_pilot.py --device cuda --variant S1 --epochs 15 --batch 8 --lr 3e-4 --seed 1337 "${FORCE[@]}"
python3 -u scripts/61_student_fdec_pilot.py --variant compare
