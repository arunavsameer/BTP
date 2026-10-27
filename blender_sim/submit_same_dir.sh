#!/bin/sh
# Append same-direction follow clips into the existing groups pack.
# IDs 0200-0209 closing, 0210-0219 hold, 0220-0229 pulling.
#SBATCH --job-name=same_dir_200
#SBATCH --output=same_dir_200_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L | head -3

cd /home/asrivastava/VJEPA2_BTP/BTP/blender_sim
export BTP_GPU=egl
export BLENDER=/home/asrivastava/opt/blender/blender
export OMP_NUM_THREADS=16
export PATH="/home/asrivastava/opt/blender:/raid/money/miniconda3/bin:/usr/bin:$PATH"

OUT=/home/asrivastava/VJEPA2_BTP/BTP/blender_sim/datasets/groups15_n400_s67_n400
start=200
for s in same_dir_closing same_dir_hold same_dir_pulling; do
  echo "[same_dir] scenario=$s start-episode=$start"
  ./run.sh --episodes 10 --start-episode "$start" --scenario "$s" \
    --ego-mode walk --biome street \
    --media video --no-rgb --spatial-overlay --threat-grid 5 \
    --output "$OUT"
  start=$((start + 10))
done
echo "[same_dir] done"
