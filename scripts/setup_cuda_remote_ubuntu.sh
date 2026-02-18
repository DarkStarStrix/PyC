#!/usr/bin/env bash
set -euo pipefail

# Deterministic CUDA bench setup for rented Ubuntu GPU hosts.
# Usage:
#   bash scripts/setup_cuda_remote_ubuntu.sh

echo "[1/6] System info"
uname -a
lsb_release -a || true

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; GPU driver/CUDA runtime missing." >&2
  exit 1
fi

nvidia-smi

echo "[2/6] Installing base tooling"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv build-essential cmake git

echo "[3/6] Creating virtualenv"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "[4/6] Installing benchmark deps"
# Pick a CUDA wheel index as needed for your host/toolkit.
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install numpy

echo "[5/6] Validating torch CUDA"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY

echo "[6/6] Ready"
echo "Run: source .venv/bin/activate && python3 benchmark/benchmarks/gpu/run_gpu_suite.py --device cuda --tag gpu_baseline"
