#!/usr/bin/env bash
# Part 3: one resumable experiment per frame set (Micro, train-pilot cache only).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=reports/student_architecture_tests
CACHE="${PILOT_CACHE_DIR:-$OUT/teacher_cache_train_pilot}"
export PYTHONUNBUFFERED=1
DEVICE="${CUDA_DEVICE:-cuda}"

if [[ ! -f "$OUT/pilot_split_train_groups.json" ]] || ! grep -q '"cache_built": true' "$OUT/pilot_split_train_groups.json" 2>/dev/null; then
  echo "Train-pilot teacher cache missing; run scripts/49 and 50 first." >&2
  exit 1
fi

python3 -u scripts/45_student_smoke_test.py --out "$OUT" --device "$DEVICE"

for FS in F1 F2 F3 F4 F5; do
  python3 -u scripts/46_student_run_experiment.py \
    --exp-id "part3_${FS}" \
    --kind part3_frame \
    --frame-set "$FS" \
    --arch micro_concat \
    --epochs "${PILOT_EPOCHS:-4}" \
    --max-rows "${PILOT_MAX_ROWS:-800}" \
    --cache-dir "$CACHE" \
    --device "$DEVICE" \
    --alpha-kd 0.5 \
    --eta-latent 0 \
    --out "$OUT"
  python3 -u scripts/51_student_part3_summary.py --out "$OUT"
done
