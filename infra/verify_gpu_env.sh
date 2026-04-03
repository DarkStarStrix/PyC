#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

STRICT="${VERIFY_STRICT:-0}"

fail() {
  echo "[verify][ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[verify][WARN] $*"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

echo "[verify] host=$(hostname)"
uname -a || true

echo "[verify] checking core toolchain"
for cmd in python3 cmake git; do
  if ! have "$cmd"; then
    fail "missing required command: $cmd"
  fi
  echo "[verify] found $cmd: $(command -v "$cmd")"
done

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[verify] checking MPI"
if have mpirun; then
  mpirun --version | head -n 1 || true
else
  if [[ "$STRICT" == "1" ]]; then
    fail "mpirun not found"
  fi
  warn "mpirun not found"
fi

echo "[verify] checking GPU stack"
GPU_OK=0
if have nvidia-smi; then
  echo "[verify] NVIDIA detected"
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || nvidia-smi || true
  GPU_OK=1
  if have nvcc; then
    nvcc --version | tail -n 1 || true
  else
    warn "nvcc not found (runtime-only CUDA may still work)"
  fi
fi

if have rocm-smi || have rocminfo; then
  echo "[verify] ROCm detected"
  if have rocm-smi; then
    rocm-smi || true
  fi
  if have rocminfo; then
    rocminfo | head -n 40 || true
  fi
  GPU_OK=1
  if have hipcc; then
    hipcc --version | head -n 2 || true
  else
    warn "hipcc not found"
  fi
fi

if [[ "$GPU_OK" -ne 1 ]]; then
  if [[ "$STRICT" == "1" ]]; then
    fail "no GPU runtime detected (nvidia-smi/rocm-smi/rocminfo missing)"
  fi
  warn "no GPU runtime detected"
fi

echo "[verify] checking communication runtime libs"
python3 - <<'PY'
import ctypes.util
libs = ["nccl", "rccl", "mpi"]
for lib in libs:
    print(f"[verify] lib{lib}: {ctypes.util.find_library(lib) or 'not-found'}")
PY

echo "[verify] checking python packages"
python3 - <<'PY'
import importlib.util
mods = ["torch", "numpy"]
for mod in mods:
    print(f"[verify] module_{mod}={bool(importlib.util.find_spec(mod))}")
PY

if python3 -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') else 1)"; then
  python3 - <<'PY'
import torch
print("[verify] torch", torch.__version__)
print("[verify] torch_cuda_available", torch.cuda.is_available())
print("[verify] torch_cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("[verify] torch_cuda_device_0", torch.cuda.get_device_name(0))
PY
else
  warn "torch not installed"
fi

echo "[verify] complete"
