#!/bin/bash
# Frozen now-only teacher (trial 4). Reuses the trial 3 token cache.
# Does not encode tokens, does not build a frame cache, does not touch trial 3 checkpoints.
#
# From the login node, after model_trial_4 is on the box:
#   cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
#   squeue -u asrivastava          # must be < 2 running jobs
#   mkdir -p logs
#   sbatch submit_teacher_now_v1.sh
#
#SBATCH --job-name=cjepa4_now_v1
#SBATCH --output=logs/slurm_teacher_now_v1_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside SLURM. From model_trial_4/: sbatch submit_teacher_now_v1.sh" >&2
  exit 1
fi

ROOT=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_4
MT3=/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
RAID_CACHE=/raid/asrivastava/VJEPA2_BTP/model_trial_3/cache
TOKEN_ID=f0878f26dd0029c5_gr16333c81
TOKEN_DIR=${RAID_CACHE}/tokens/${TOKEN_ID}
TAG=teacher_now_v1
PY=/raid/money/miniconda3/bin/python

cd "$ROOT"
mkdir -p "logs/${TAG}" "reports/${TAG}" "checkpoints/${TAG}" logs

export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

echo "=== SLURM job $SLURM_JOB_ID ${TAG} on $(hostname) ==="
date
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

if [[ -e "checkpoints/${TAG}/teacher_phase_a_best.pt" || -e "checkpoints/${TAG}/teacher_phase_a_last.pt" ]]; then
  echo "refusing to overwrite checkpoints/${TAG}" >&2
  exit 1
fi

if [[ ! -d "$TOKEN_DIR" ]]; then
  echo "token cache missing: $TOKEN_DIR" >&2
  echo "refusing to encode a new cache" >&2
  exit 1
fi

# Manifest and splits stay the trial 3 files. Symlink only; never copy or rebuild.
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

# Fail before the trainer loads V-JEPA if this checkout would not hit the existing tokens.
"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from collision_jepa4.config import Config
from collision_jepa4.data.cache import TokenCache, token_cache_id
from collision_jepa4.data.manifest import load_manifest
from collision_jepa4.data.sampling import anchors_from_cfg

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

# Dead DGX2 GPU: Slurm can hand out a card that shows ~0 MiB and then CUDA is busy.
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA unavailable; not training")
x = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print("cuda probe ok", torch.cuda.get_device_name(0), "allocated", torch.cuda.memory_allocated())
del x
PY

"$PY" scripts/04_train_teacher.py \
    --frozen-epochs 8 \
    --patience 4 \
    --adapter-lr 0.0003 \
    --num-workers 4 \
    --val-subset-fraction 0.25 \
    --checkpoint-root "checkpoints/${TAG}" \
    --logs-root "logs/${TAG}" \
    --reports-root "reports/${TAG}"

echo "=== ${TAG} complete ==="
date
