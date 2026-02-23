#!/usr/bin/env bash
set -euo pipefail

# This script assumes pyenv is installed and available in PATH.
# It sets up a fresh arm64 Python 3.11.10 env, installs GPU-enabled packages,
# and runs quick checks for TensorFlow (Metal) and PyTorch (MPS).

PYVER="3.11.10"
ENV_ROOT="$(pwd)"

# Ensure pyenv is initialized (common in interactive shells). Adjust if needed.
if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found in PATH. Please load pyenv and rerun." >&2
  exit 1
fi

echo "Using pyenv version: $PYVER"
if ! pyenv versions --bare | grep -q "^${PYVER}$"; then
  echo "Installing Python ${PYVER} via pyenv..."
  pyenv install "${PYVER}"
fi

cd "${ENV_ROOT}"
pyenv local "${PYVER}"

# Upgrade pip/setuptools/wheel first.
pyenv exec python -m pip install --upgrade pip setuptools wheel

# Install Metal/MPS-capable packages.
pyenv exec python -m pip install --upgrade \
  torch torchvision torchaudio \
  tensorflow-macos==2.16.1 tensorflow-metal==1.1.0

# Quick checks
cat > /tmp/gpu_quick_check.py <<'PY'
import platform
print("python:", platform.python_version(), platform.platform(), platform.machine())

import tensorflow as tf
print("tensorflow:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))

import torch
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps") * 2
    print("mps tensor:", x, "value:", x.item())
PY

pyenv exec python /tmp/gpu_quick_check.py
