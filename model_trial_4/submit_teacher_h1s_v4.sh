#!/bin/bash
# Frozen 1s-ahead teacher (trial 4), last-tubelet dilated trunk, 15 epochs, FULL val.
# Reuses the trial 3 token cache. New run dir only.
# Does not touch teacher_h1s_v1/v2/v3, teacher_h1s_128, teacher_h1s_128_full, or trial 3.
#
# From the login node:
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
#   sbatch submit_teacher_h1s_v4.sh
#
#SBATCH --job-name=cjepa4_h1s_v4
#SBATCH --output=logs/slurm_teacher_h1s_v4_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_teacher_h1s_v4.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
MT3=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
RAID_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3/cache
TOKEN_ID=f0878f26dd0029c5_gr16333c81
TOKEN_DIR=${RAID_CACHE}/tokens/${TOKEN_ID}
TAG=teacher_h1s_v4
PY=/raid/money/miniconda3/bin/python

cd "$ROOT"
mkdir -p "logs/${TAG}" "reports/${TAG}" "checkpoints/${TAG}" logs

export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID ${TAG} last-tubelet dilated 128-ch FULL val on $(hostname) ==="
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
mem_util=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
mem=${mem_util%%,*}; util=${mem_util##*,}
mem=${mem// /}; util=${util// /}; mem=${mem%.*}; util=${util%.*}
if [ "${util:-0}" -ge 50 ] && [ "${mem:-0}" -lt 400 ]; then
  echo "refusing ghost-busy GPU mem=${mem}MiB util=${util}" >&2
  exit 3
fi
if [ "${mem:-0}" -ge 36000 ]; then
  echo "refusing nearly-full GPU mem=${mem}MiB util=${util}" >&2
  exit 2
fi
echo "GPU has ${mem}MiB used util=${util}; probing CUDA"

if [[ -e "checkpoints/${TAG}/teacher_phase_a_best.pt" || -e "checkpoints/${TAG}/teacher_phase_a_last.pt" ]]; then
  echo "refusing to overwrite checkpoints/${TAG}" >&2
  exit 1
fi

if [[ ! -d "$TOKEN_DIR" ]]; then
  echo "token cache missing: $TOKEN_DIR" >&2
  echo "refusing to encode a new cache" >&2
  exit 1
fi

mkdir -p cache
link_existing() {
  local src=$1 dst=$2
  if [[ ! -f "$src" ]]; then
    echo "missing $src" >&2
    exit 1
  fi
  if [[ -L "$dst" ]]; then
    ln -sfn "$src" "$dst"
  elif [[ -e "$dst" ]]; then
    echo "refusing to replace existing $dst" >&2
    exit 1
  else
    ln -s "$src" "$dst"
  fi
}
if [[ -f ${MT3}/cache/manifest.jsonl ]]; then
  link_existing "${MT3}/cache/manifest.jsonl" cache/manifest.jsonl
  link_existing "${MT3}/cache/splits.json" cache/splits.json
elif [[ -f ${RAID_CACHE}/manifest.jsonl ]]; then
  link_existing "${RAID_CACHE}/manifest.jsonl" cache/manifest.jsonl
  link_existing "${RAID_CACHE}/splits.json" cache/splits.json
else
  echo "manifest/splits not found under model_trial_3/cache or ${RAID_CACHE}" >&2
  exit 1
fi

"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from collision_jepa4.config import Config
from collision_jepa4.data.cache import TokenCache, token_cache_id
from collision_jepa4.data.manifest import load_manifest
from collision_jepa4.data.sampling import anchors_from_cfg
from collision_jepa4.models.trunk import CollisionTrunk, TRUNK_PARAMS_1408

EXPECTED = "f0878f26dd0029c5_gr16333c81"
N_TOKENS = 8 * 16 * 16
HIDDEN = 1408
cfg = Config.load("configs/default.yaml")
cid = token_cache_id(cfg, dummy=False)
print("token_cache_id", cid)
if cid != EXPECTED:
    raise SystemExit(
        f"fingerprint {cid} != existing cache {EXPECTED}; refusing to build a new token cache"
    )
n = sum(p.numel() for p in CollisionTrunk(hidden_size=1408, channels=128).parameters())
print("trunk_128_params", n, "spec", TRUNK_PARAMS_1408)
if n < 350000:
    raise SystemExit(f"128-ch trunk params {n} look wrong")
cache_root = cfg.path("paths.cache_root")
if cache_root != Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3/cache"):
    raise SystemExit(f"cache_root {cache_root} is not the trial 3 raid cache")
if bool(cfg.get("cache.cache_frames", False)):
    raise SystemExit("cache.cache_frames is true; refusing to touch the frame cache")
records = load_manifest(cfg.path("paths.manifest"))
cache = TokenCache(cache_root, cid)
n_frames = int(cfg.get("data.expected_frames", 450))
valid = [r for r in records if r.valid]
missing = []
group_anchors = {}
for rec in valid:
    gid = rec.scenario_group_id
    if gid not in group_anchors:
        group_anchors[gid] = anchors_from_cfg(cfg, gid, n_frames=n_frames)
    if not cache.has_clip(rec.clip_id, group_anchors[gid], N_TOKENS, HIDDEN):
        missing.append(rec.clip_id)
print(f"token cache reuse check: valid={len(valid)} missing={len(missing)} dir={cache.root}")
if len(valid) < 830 or missing:
    preview = ", ".join(missing[:8])
    raise SystemExit(
        f"refusing to encode: valid={len(valid)} missing={len(missing)} ({preview})"
    )
print("token cache complete; trainer will not write tokens")
PY

"$PY" - <<'PY'
import sys
import torch

try:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("CUDA unavailable; not training")
    x = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    print("cuda probe ok", torch.cuda.get_device_name(0), "allocated", torch.cuda.memory_allocated())
    del x
except SystemExit:
    raise
except Exception as exc:
    print(f"cuda probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(4) from exc
PY

"$PY" scripts/04_train_teacher.py \
    --trunk-channels 128 \
    --frozen-epochs 15 \
    --patience 6 \
    --patience-start-epoch 8 \
    --adapter-lr 0.0003 \
    --num-workers 4 \
    --val-subset-fraction 1.0 \
    --checkpoint-root "checkpoints/${TAG}" \
    --logs-root "logs/${TAG}" \
    --reports-root "reports/${TAG}"

echo "=== ${TAG} complete ==="
date
