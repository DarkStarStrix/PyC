#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ENABLE_BASE_ENV="${ENABLE_BASE_ENV:-1}"
ENABLE_CUTLASS_HARNESS="${ENABLE_CUTLASS_HARNESS:-1}"
ENABLE_CUTLASS_PROFILER="${ENABLE_CUTLASS_PROFILER:-1}"
ENABLE_TVM_CUDA_BUILD="${ENABLE_TVM_CUDA_BUILD:-0}"
ENABLE_TENSORRT_PYTHON="${ENABLE_TENSORRT_PYTHON:-0}"
ENABLE_JAX_CUDA_LOCAL="${ENABLE_JAX_CUDA_LOCAL:-0}"
CUTLASS_DIR="${CUTLASS_DIR:-/tmp/cutlass}"
CUTLASS_REF="${CUTLASS_REF:-v4.2.1}"
CUTLASS_ARCHS="${CUTLASS_ARCHS:-90a}"
CUTLASS_BUILD_JOBS="${CUTLASS_BUILD_JOBS:-2}"
CUTLASS_LIBRARY_OPERATIONS="${CUTLASS_LIBRARY_OPERATIONS:-gemm}"
CUTLASS_LIBRARY_KERNELS="${CUTLASS_LIBRARY_KERNELS:-cutlass3x_sm90_tensorop_gemm_*}"
CUTLASS_UNITY_BUILD_ENABLED="${CUTLASS_UNITY_BUILD_ENABLED:-OFF}"

if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

if [ "${ENABLE_BASE_ENV}" = "1" ]; then
  ENABLE_TVM_CUDA_BUILD="${ENABLE_TVM_CUDA_BUILD}" bash scripts/setup_benchmark_env_locked.sh
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

if [ "${ENABLE_TENSORRT_PYTHON}" = "1" ]; then
  .venv/bin/python -m pip install --upgrade pip
  TRT_TORCH_VERSION="$(".venv/bin/python" - <<'PY'
try:
    import torch
    print(getattr(torch, "__version__", "").strip())
except Exception:
    print("")
PY
)"
  if [ -n "${TRT_TORCH_VERSION}" ]; then
    .venv/bin/python -m pip install --upgrade \
      tensorrt \
      "torch-tensorrt==${TRT_TORCH_VERSION}" \
      --extra-index-url https://download.pytorch.org/whl/cu124
  else
    .venv/bin/python -m pip install --upgrade \
      tensorrt \
      torch-tensorrt \
      --extra-index-url https://download.pytorch.org/whl/cu124
  fi
fi

if [ "${ENABLE_JAX_CUDA_LOCAL}" = "1" ]; then
  if [ ! -x ".venv_jax/bin/python" ]; then
    python3 -m venv .venv_jax
  fi
  .venv_jax/bin/python -m pip install --upgrade pip
  env \
    PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" \
    .venv_jax/bin/python -m pip install --upgrade "jax[cuda12-local]"
fi

if [ "${ENABLE_CUTLASS_HARNESS}" = "1" ] && ! command -v cutlass_gemm_bench >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y git build-essential
  bash scripts/build_cutlass_harness_remote.sh
fi

if [ "${ENABLE_CUTLASS_PROFILER}" = "1" ] && ! command -v cutlass_profiler >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y git cmake ninja-build build-essential

  if [ ! -d "${CUTLASS_DIR}" ]; then
    git clone --recursive https://github.com/NVIDIA/cutlass "${CUTLASS_DIR}"
  fi

  cd "${CUTLASS_DIR}"
  git fetch --tags --all
  git checkout "${CUTLASS_REF}"
  git submodule update --init --recursive
  rm -rf build

  cmake_args=(
    -S .
    -B build
    "-DCMAKE_CUDA_COMPILER=${CUDACXX:-nvcc}"
    "-DCUDAToolkit_ROOT=${CUDA_HOME}"
    "-DCUTLASS_NVCC_ARCHS=${CUTLASS_ARCHS}"
    "-DCUTLASS_UNITY_BUILD_ENABLED=${CUTLASS_UNITY_BUILD_ENABLED}"
  )
  if [ -n "${CUTLASS_LIBRARY_OPERATIONS}" ]; then
    cmake_args+=("-DCUTLASS_LIBRARY_OPERATIONS=${CUTLASS_LIBRARY_OPERATIONS}")
  fi
  if [ -n "${CUTLASS_LIBRARY_KERNELS}" ]; then
    cmake_args+=("-DCUTLASS_LIBRARY_KERNELS=${CUTLASS_LIBRARY_KERNELS}")
  fi

  cmake "${cmake_args[@]}"
  cmake --build build --parallel "${CUTLASS_BUILD_JOBS}" --target cutlass_profiler

  install -m 0755 "${CUTLASS_DIR}/build/tools/profiler/cutlass_profiler" /usr/local/bin/cutlass_profiler
fi

python3 - <<'PY'
import importlib.util
import json
import shutil
import subprocess


def has_module(name: str) -> bool:
    return bool(importlib.util.find_spec(name))


payload = {
    "cutlass_gemm_bench": shutil.which("cutlass_gemm_bench"),
    "cutlass_profiler": shutil.which("cutlass_profiler"),
    "trtexec": shutil.which("trtexec"),
    "torch_xla": has_module("torch_xla"),
    "tensorrt": has_module("tensorrt"),
    "torch_tensorrt": has_module("torch_tensorrt"),
    "tvm": has_module("tvm"),
}

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    payload["gpu"] = out.stdout.strip()
except Exception:
    payload["gpu"] = ""

print(json.dumps(payload, indent=2))
PY
