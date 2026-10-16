#!/usr/bin/env bash
# Launch the pipeline with whatever `blender` is on PATH (Blender 4.x / 5.x).
#
# EEVEE is a GPU rasterizer. On hybrid laptops the default GL context is the
# iGPU (here: AMD 780M). Set BTP_GPU=nvidia (default: auto) to PRIME-offload
# onto a discrete NVIDIA card *before* Blender creates that context.
#
#   BTP_GPU=auto|nvidia|amd|egl   (default auto)
#   egl = headless NVIDIA (DGX / no DISPLAY). nvidia = laptop PRIME offload.
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
      if [[ -z "${DISPLAY:-}" ]]; then
        gpu_mode=egl
      else
        gpu_mode=nvidia
      fi
    else
      gpu_mode=amd
    fi
    ;;
  nvidia|amd|egl) ;;
  *)
    echo "BTP_GPU must be auto|nvidia|amd|egl (got '$gpu_mode')" >&2
    exit 2
    ;;
esac

if [[ "$gpu_mode" == "egl" ]]; then
  # Headless NVIDIA: EGL device, no Xvfb, no laptop PRIME vars.
  unset DISPLAY || true
  unset __NV_PRIME_RENDER_OFFLOAD || true
  unset __GLX_VENDOR_LIBRARY_NAME || true
  unset __VK_LAYER_NV_optimus || true
  export EGL_PLATFORM="${EGL_PLATFORM:-device}"
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    for cand in \
      /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
      /usr/share/glvnd/egl_vendor.d/10_nvidia_wayland.json
    do
      if [[ -f "$cand" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$cand"
        break
      fi
    done
  fi
  echo "[gpu] NVIDIA EGL device (headless)  EGL_PLATFORM=${EGL_PLATFORM}"
elif [[ "$gpu_mode" == "nvidia" ]]; then
  # PRIME render offload: GL/Vulkan on the dGPU, display stays on the iGPU.
  export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
  export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
  echo "[gpu] PRIME offload → NVIDIA  (__NV_PRIME_RENDER_OFFLOAD=1)"
else
  echo "[gpu] using default GL device (usually the iGPU)"
fi

THREADS="${BLENDER_THREADS:-${OMP_NUM_THREADS:-}}"
if [[ -n "$THREADS" ]]; then
  exec "$BLENDER" --threads "$THREADS" --background --python "$ROOT/main.py" -- "$@"
fi
exec "$BLENDER" --background --python "$ROOT/main.py" -- "$@"
