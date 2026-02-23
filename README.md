# Tensor Analytics

Design Patterns for Tensor Analytics

## Apple Silicon GPU setup (Metal/MPS)

This repo includes a simple setup script to get TensorFlow (Metal) and PyTorch (MPS) working on Apple Silicon.

### Prerequisites
- Apple Silicon Mac (M-series) with a logged-in GUI session.
- `pyenv` installed and on your `PATH`.
- Command Line Tools (`xcode-select -p` should resolve).

### One-command setup and check
From the repo root (`tensor_analytics`):
```
./gpu_setup_check.sh
```
What it does:
- Ensures `pyenv local 3.11.10` (writes `.python-version`).
- Upgrades `pip/setuptools/wheel`.
- Installs GPU-capable packages: `torch`, `torchvision`, `torchaudio`, `tensorflow-macos==2.16.1`, `tensorflow-metal==1.1.0`.
- Runs quick GPU checks for TensorFlow (Metal) and PyTorch (MPS).

### Manual steps (if you prefer)
1) Install Python 3.11.10 via pyenv if missing:
   ```
   pyenv install 3.11.10
   ```
2) Set the local Python version:
   ```
   pyenv local 3.11.10
   ```
3) Install the GPU packages:
   ```
   pyenv exec pip install --upgrade pip setuptools wheel
   pyenv exec pip install --upgrade torch torchvision torchaudio
   pyenv exec pip install --upgrade tensorflow-macos==2.16.1 tensorflow-metal==1.1.0
   ```
4) Verify GPU availability:
   ```
   pyenv exec python gpu_check.py
   ```
   Expected on Apple Silicon with Metal exposed: TensorFlow lists a GPU, PyTorch shows `mps_is_available: True`.

### Helper scripts
- `gpu_setup_check.sh`: Automates env setup and runs the GPU checks.
- `gpu_check.py`: Prints TensorFlow GPU devices and PyTorch MPS status, runs a small TensorFlow matmul on `/GPU:0` when present, a small PyTorch matmul on `mps` when available, and compares matmul timings CPU vs GPU for both frameworks (calling out which side is faster).
