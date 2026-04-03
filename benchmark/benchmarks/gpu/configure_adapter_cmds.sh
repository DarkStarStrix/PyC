#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY_BIN="${PY_BIN:-python3}"
XLA_PY_BIN="${XLA_PY_BIN:-${PY_BIN}}"
XLA_LD_LIBRARY_PATH=""

# Optional split environment for JAX/XLA to avoid CUDA runtime clashes with Torch.
if [ -x "${ROOT}/.venv_jax/bin/python" ]; then
  XLA_PY_BIN="${ROOT}/.venv_jax/bin/python"
  for lib_dir in \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cublas/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cuda_cupti/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cuda_runtime/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cudnn/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cufft/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cusolver/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/cusparse/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/nccl/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/nvjitlink/lib" \
    "${ROOT}/.venv_jax/lib/python3.11/site-packages/nvidia/nvshmem/lib"
  do
    if [ -d "${lib_dir}" ]; then
      if [ -z "${XLA_LD_LIBRARY_PATH}" ]; then
        XLA_LD_LIBRARY_PATH="${lib_dir}"
      else
        XLA_LD_LIBRARY_PATH="${XLA_LD_LIBRARY_PATH}:${lib_dir}"
      fi
    fi
  done
fi

# Auto-wire CUDA toolchain on common Linux layouts.
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/bin" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# If TVM was built from source on remote host, auto-wire its runtime path.
if [ -z "${TVM_LIBRARY_PATH:-}" ] && [ -f "/tmp/tvm/build/libtvm.so" ]; then
  export TVM_LIBRARY_PATH="/tmp/tvm/build"
fi

export PYC_GPU_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_pyc_cmd.py"
export TVM_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_tvm_cmd.py"
if [ -n "${XLA_LD_LIBRARY_PATH}" ]; then
  export XLA_BENCH_CMD="env LD_LIBRARY_PATH=${XLA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-} ${XLA_PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_xla_cmd.py"
else
  export XLA_BENCH_CMD="${XLA_PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_xla_cmd.py"
fi
export TENSORRT_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_tensorrt_cmd.py"
export GLOW_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_glow_cmd.py"

echo "Configured adapter commands:"
echo "  PYC_GPU_BENCH_CMD=${PYC_GPU_BENCH_CMD}"
echo "  TVM_BENCH_CMD=${TVM_BENCH_CMD}"
echo "  CUDA_HOME=${CUDA_HOME:-unset}"
echo "  NVCC=$(command -v nvcc || echo unset)"
echo "  TVM_LIBRARY_PATH=${TVM_LIBRARY_PATH:-unset}"
echo "  XLA_BENCH_CMD=${XLA_BENCH_CMD}"
echo "  XLA_PY_BIN=${XLA_PY_BIN}"
echo "  XLA_LD_LIBRARY_PATH=${XLA_LD_LIBRARY_PATH:-unset}"
echo "  TENSORRT_BENCH_CMD=${TENSORRT_BENCH_CMD}"
echo "  GLOW_BENCH_CMD=${GLOW_BENCH_CMD}"
