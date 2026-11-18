#!/usr/bin/env bash
# Launch the pipeline with whatever `blender` is on PATH (Blender 4.x / 5.x).
#
# EEVEE is a GPU rasterizer. On hybrid laptops the default GL context is the
# iGPU (here: AMD 780M). Set BTP_GPU=nvidia (default: auto) to PRIME-offload
# onto a discrete NVIDIA card *before* Blender creates that context.
#
#   BTP_GPU=auto|nvidia|amd|egl|xvfb   (default auto)
#   BTP_THREADS=N   Blender -t N (0 = AUTO). Not OMP_NUM_THREADS.
#   BTP_GPU_BACKEND=opengl|vulkan     forwarded as --gpu-backend
#   egl  = NVIDIA EGL device (headless DGX). This is the path that rendered
#          groups15 and the 1-frame EEVEE smoke on DGX2 (job 10546).
#   xvfb = dummy X + NVIDIA GLX. On this DGX epoxy asserts (no GLX/EGL ctx).
#   nvidia = laptop PRIME offload (needs a real DISPLAY).
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
  nvidia|amd|egl|xvfb) ;;
  *)
    echo "BTP_GPU must be auto|nvidia|amd|egl|xvfb (got '$gpu_mode')" >&2
    exit 2
    ;;
esac

if [[ "$gpu_mode" == "egl" ]]; then
  # Headless NVIDIA: EGL device, no Xvfb. Job 10519 SIGSEGV'd here only on a
  # GPU already running vLLM; a 1-frame EEVEE smoke on another A100 succeeded.
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
elif [[ "$gpu_mode" == "xvfb" ]]; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "BTP_GPU=xvfb requires xvfb-run" >&2
    exit 127
  fi
  # Dummy X + NVIDIA GLX. Do not set EGL_PLATFORM=device.
  unset EGL_PLATFORM || true
  unset __EGL_VENDOR_LIBRARY_FILENAMES || true
  export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
  export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
  if [[ -z "${BTP_GPU_BACKEND:-}" ]]; then
    export BTP_GPU_BACKEND=opengl
  fi
  echo "[gpu] Xvfb + NVIDIA GLX  backend=${BTP_GPU_BACKEND}"
elif [[ "$gpu_mode" == "nvidia" ]]; then
  # PRIME render offload: GL/Vulkan on the dGPU, display stays on the iGPU.
  export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
  export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
  echo "[gpu] PRIME offload → NVIDIA  (__NV_PRIME_RENDER_OFFLOAD=1)"
else
  echo "[gpu] using default GL device (usually the iGPU)"
fi

# BTP_THREADS from gen_dataset (or env). Do not use OMP_NUM_THREADS=1 — that caps Blender.
THREADS="${BTP_THREADS:-${BLENDER_THREADS:-}}"
blender_args=()
if [[ -n "${BTP_GPU_BACKEND:-}" ]]; then
  blender_args+=(--gpu-backend "$BTP_GPU_BACKEND")
fi
if [[ -n "$THREADS" && "$THREADS" != "0" ]]; then
  blender_args+=(--threads "$THREADS")
fi
blender_args+=(--background)
if [[ -n "${BTP_BLEND_FILE:-}" ]]; then
  blender_args+=("$BTP_BLEND_FILE")
fi
blender_args+=(--python "$ROOT/main.py" -- "$@")

if [[ "$gpu_mode" == "xvfb" ]]; then
  exec xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" "$BLENDER" "${blender_args[@]}"
fi
exec "$BLENDER" "${blender_args[@]}"
