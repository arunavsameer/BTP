#!/usr/bin/env bash
# Parts 0–11: temporal supernet (single shared student).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=reports/student_architecture_tests
export PYTHONUNBUFFERED=1
DEVICE="${CUDA_DEVICE:-cuda}"
EPOCHS="${SUPERNET_EPOCHS:-25}"

python3 -u scripts/52_temporal_supernet_part0_shutdown.py --out "$OUT"
python3 -u scripts/53_temporal_supernet_manifests.py --out "$OUT"
python3 -u scripts/54_temporal_supernet_rgb_cache.py --out "$OUT"
python3 -u scripts/55_temporal_supernet_smoke.py --out "$OUT" --device "$DEVICE"

python3 -u scripts/56_temporal_supernet_train.py \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --batch 8 \
  --device "$DEVICE"
