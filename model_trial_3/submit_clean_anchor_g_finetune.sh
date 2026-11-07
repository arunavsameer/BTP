#!/usr/bin/env bash
# Fine-tune from G checkpoint: clean path + auxiliary ICC channel dropout (no pair consistency).
set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3

OUT="${1:-checkpoints/clean_anchor_g_finetune_s1337}"
LOG="${2:-logs/train_clean_anchor_g_finetune_s1337.log}"
G_CKPT="${3:-checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt}"
SEED="${4:-1337}"

mkdir -p "$OUT" "$OUT/logs" "$OUT/reports"

python scripts/04_train_teacher.py \
  --config configs/default.yaml \
  --phase frozen \
  --frozen-epochs 6 \
  --heads-mode direct_h1s \
  --init-from "$G_CKPT" \
  --adapter-lr 0.00003 \
  --warmup-epochs 1 \
  --patience 99 \
  --patience-start-epoch 999 \
  --early-stop-metric val_mse \
  --device cuda:0 \
  --seed "$SEED" \
  --checkpoint-root "$OUT" \
  --logs-root "$OUT/logs" \
  --reports-root "$OUT/reports" \
  --bg-reg \
  --bg-reg-clean-anchor \
  --bg-reg-dropout-prob 0.25 \
  --bg-reg-no-early-stop \
  --bg-reg-p4 reports/p4_channel_icc/p4_report.json \
  2>&1 | tee "$LOG"
