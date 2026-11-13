#!/usr/bin/env bash
# Sequential overnight chain: smoke all six, then train 1→6. Model failure continues; infra failure stops.
set -u
ROOT="/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3"
TEMP="/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp/overnight6"
PY="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-8}"
LR="${LR:-3e-4}"
mkdir -p "$TEMP"
CHAIN="$TEMP/chain.log"
STATUS="$TEMP/chain_status.json"
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "$(date -Is) chain start device=$DEVICE epochs=$EPOCHS batch=$BATCH" | tee -a "$CHAIN"
echo '{"stage":"smoke"}' > "$STATUS"

if ! "$PY" scripts/81_smoke_overnight6.py --device "$DEVICE"; then
  echo "$(date -Is) SMOKE_FAILED stop chain" | tee -a "$CHAIN"
  echo '{"stage":"smoke_failed","ok":false}' > "$STATUS"
  exit 2
fi
echo "$(date -Is) ALL_SMOKE_OK" | tee -a "$CHAIN"

infra_fail=0
for m in 1 2 3 4 5 6; do
  echo "$(date -Is) START model $m" | tee -a "$CHAIN"
  echo "{\"stage\":\"train\",\"model\":$m}" > "$STATUS"
  if "$PY" scripts/81_train_overnight6.py --model "$m" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --device "$DEVICE"; then
    echo "$(date -Is) OK model $m exit=0" | tee -a "$CHAIN"
    echo 0 > "$TEMP/m${m}.exit"
  else
    ec=$?
    echo "$(date -Is) FAIL model $m exit=$ec" | tee -a "$CHAIN"
    echo "$ec" > "$TEMP/m${m}.exit"
    if [[ $ec -eq 137 || $ec -eq 9 || $ec -eq 127 ]]; then
      echo "$(date -Is) INFRA-like exit $ec stop chain" | tee -a "$CHAIN"
      infra_fail=1
      break
    fi
  fi
done

"$PY" scripts/81_report_overnight6.py | tee -a "$CHAIN" || true
if [[ $infra_fail -eq 1 ]]; then
  echo '{"stage":"stopped_infra","ok":false}' > "$STATUS"
  exit 3
fi
echo "$(date -Is) CHAIN_DONE" | tee -a "$CHAIN"
echo '{"stage":"done","ok":true}' > "$STATUS"
exit 0
