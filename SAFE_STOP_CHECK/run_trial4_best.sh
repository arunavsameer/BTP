#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-/raid/money/miniconda3/bin/python}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
RAID="/raid/asrivastava/VJEPA2_BTP"
cd "$ROOT"
# CPU so we do not touch occupied / wedged GPUs.
export CUDA_VISIBLE_DEVICES=""

run_one() {
  local name="$1" ckpt="$2"
  echo "======== ${name} ${ckpt} ========"
  "$PY" run.py --model "$name" --checkpoint "$ckpt" --device cpu --batch-size 8 --no-render \
    --out-dir "$ROOT/output/${name}"
}

run_one trial4_v2 "$RAID/model_trial_4_decay/student_v2_featdecay/checkpoint_epoch_22.pt"
run_one trial4_v3 "$RAID/model_trial_4/student_runs/student_v3_coord_v1/checkpoint_epoch_28.pt"
run_one trial4_v4 "$RAID/model_trial_4/student_runs/student_v4_gated_v1/checkpoint_epoch_20.pt"
run_one trial4_v5 "$RAID/model_trial_4/student_runs/student_v5_unfreeze_v1/checkpoint_best.pt"
echo "ALL DONE"
