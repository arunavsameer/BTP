#!/usr/bin/env bash
# Arch Linux bootstrap for the headless ModernGL synthetic generator.
# Offscreen rendering uses EGL (mesa) via moderngl/glcontext.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> Installing system GL stack (mesa + libglvnd)"
if command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed --noconfirm mesa libglvnd python python-pip python-virtualenv \
        python-numpy python-scipy python-pillow
else
    echo "pacman not found; skipping system packages. Ensure libEGL is available." >&2
fi

echo "==> Creating project virtualenv (system site-packages for numpy/scipy)"
python3 -m venv --system-site-packages "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'moderngl>=5.10' 'glcontext>=2.5' 'pyglm>=2.7'

echo
echo "Setup complete. Smoke test (EGL offscreen MRT):"
echo "  source $ROOT/.venv/bin/activate"
echo "  python $ROOT/main.py --episodes 2 --frames 30 --output $ROOT/dataset_output"
echo
echo "Full 5-second episodes (150 frames @ 30 FPS, class-balanced):"
echo "  python $ROOT/main.py --episodes 32 --output $ROOT/dataset_output --seed 20260910"
