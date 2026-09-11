#!/usr/bin/env bash
# Launch the pipeline with whatever `blender` is on PATH (Blender 4.x / 5.x).
#
# EEVEE is a GPU rasterizer. On hybrid laptops the default GL context is the
# iGPU (here: AMD 780M). Set BTP_GPU=nvidia (default: auto) to PRIME-offload
# onto a discrete NVIDIA card *before* Blender creates that context.
#
#   BTP_GPU=auto|nvidia|amd   (default auto)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BLENDER="${BLENDER:-blender}"
if ! command -v "$BLENDER" >/dev/null 2>&1; then
  echo "blender not found on PATH. Install Blender 4.x or set BLENDER=/path/to/blender" >&2
  exit 127
fi

gpu_mode="${BTP_GPU:-auto}"
case "$gpu_mode" in
  auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      gpu_mode=nvidia
    else
      gpu_mode=amd
    fi
    ;;
  nvidia|amd) ;;
  *)
    echo "BTP_GPU must be auto|nvidia|amd (got '$gpu_mode')" >&2
    exit 2
    ;;
esac

if [[ "$gpu_mode" == "nvidia" ]]; then
  # PRIME render offload: GL/Vulkan on the dGPU, display stays on the iGPU.
  export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
  export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
  echo "[gpu] PRIME offload → NVIDIA  (__NV_PRIME_RENDER_OFFLOAD=1)"
else
  echo "[gpu] using default GL device (usually the iGPU)"
fi

exec "$BLENDER" --background --python "$ROOT/main.py" -- "$@"
