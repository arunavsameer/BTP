#!/bin/sh
#SBATCH --job-name=groups15_n400
#SBATCH --output=groups15_n400_%j.out
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

# No xvfb: EEVEE on this DGX needs NVIDIA EGL, not a dummy X server.
./gen_dataset.py --groups --n 400 --bg-variants 2 --seed 67 --name groups15_n400 \
  --threat-grid 5 --spatial-overlay --media video --no-rgb --biome auto
