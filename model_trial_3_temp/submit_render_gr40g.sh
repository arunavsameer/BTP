#!/bin/bash
#SBATCH --job-name=cjepa3_render_gr40g
#SBATCH --output=logs/slurm_render_gr40g_%j.out
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/asrivastava/VJEPA2_BTP/BTP/model_trial_3
mkdir -p logs visualizations/gr40g_more
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/raid/asrivastava/VJEPA2_BTP/model_trial_2/hf_cache

CKPT="gr40g=checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt"
OUT=visualizations/gr40g_more

python scripts/19_render_checkpoint_videos.py \
    --clip-id sample_0210_same_dir_hold/miss_left/bg_00 \
    --checkpoints "$CKPT" --out-dir "$OUT"

python scripts/19_render_checkpoint_videos.py \
    --clip-id sample_0003_jaywalker_turn_toward/collision/bg_00 \
    --checkpoints "$CKPT" --out-dir "$OUT"

python scripts/19_render_checkpoint_videos.py \
    --clip-id sample_0098_car_cross_critical/collision/bg_00 \
    --checkpoints "$CKPT" --out-dir "$OUT"

python scripts/19_render_checkpoint_videos.py \
    --clip-id sample_0220_same_dir_pulling/miss_left/bg_00 \
    --checkpoints "$CKPT" --out-dir "$OUT"

echo "=== gr40g render complete ==="
ls -lh "$OUT"/*.mp4
date
