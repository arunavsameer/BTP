#!/bin/sh
# Resume groups15_n400 against the existing pack (skips clips that already
# have preview.mp4 + spatial_overlay.mp4).
#SBATCH --job-name=groups15_resume
#SBATCH --output=groups15_resume_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo $CUDA_VISIBLE_DEVICES
NV_GPU=$CUDA_VISIBLE_DEVICES

cd /home/asrivastava/VJEPA2_BTP/BTP/blender_sim
export BTP_GPU=egl
export BLENDER=/home/asrivastava/opt/blender/blender
export OMP_NUM_THREADS=16
export PATH="/home/asrivastava/opt/blender:/raid/money/miniconda3/bin:/usr/bin:$PATH"

PACK=/home/asrivastava/VJEPA2_BTP/BTP/blender_sim/datasets/groups15_n400_s67_n400

./run.sh \
  --plan "$PACK/plan.json" \
  --output "$PACK" \
  --seed 67 \
  --start-episode 0 \
  --episodes 400 \
  --media video \
  --threat-grid 5 \
  --spatial-overlay \
  --no-rgb \
  --ego-mode walk
