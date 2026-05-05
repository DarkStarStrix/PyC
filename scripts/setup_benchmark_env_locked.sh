#!/usr/bin/env bash
set -euo pipefail

# Deterministic benchmark environment bootstrap for Ubuntu GPU hosts.
# Usage:
#   bash scripts/setup_benchmark_env_locked.sh
# Optional:
#   ENABLE_TVM_CUDA_BUILD=1 bash scripts/setup_benchmark_env_locked.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y python3 python3-venv python3-pip build-essential cmake git tmux jq

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Pin Torch/CUDA wheel set to avoid resolver drift between runs.
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
pip install -r benchmark/benchmarks/gpu/requirements-locked.txt

# Optional source build for TVM CUDA backend (required for native TVM CUDA mode).
if [ "${ENABLE_TVM_CUDA_BUILD:-0}" = "1" ]; then
  bash scripts/setup_tvm_cuda_remote_ubuntu.sh
else
  pip install apache-tvm==0.14.dev273
fi

python - <<'PY'
import importlib.util
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
for mod in ["tvm", "torch_xla", "tensorrt", "torch_tensorrt"]:
    print(f"module_{mod}", bool(importlib.util.find_spec(mod)))
PY
