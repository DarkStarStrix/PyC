#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CUTLASS_DIR="${CUTLASS_DIR:-/tmp/cutlass}"
CUTLASS_REF="${CUTLASS_REF:-v4.2.1}"
OUT_BIN="${OUT_BIN:-/usr/local/bin/cutlass_gemm_bench}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TMP_BIN="/tmp/cutlass_gemm_bench"

if [ ! -x "${CUDA_HOME}/bin/nvcc" ] && [ -x "/usr/local/cuda/bin/nvcc" ]; then
  CUDA_HOME="/usr/local/cuda"
fi

export PATH="${CUDA_HOME}/bin:${PATH}"

if [ ! -d "${CUTLASS_DIR}/.git" ]; then
  git clone --recursive https://github.com/NVIDIA/cutlass "${CUTLASS_DIR}"
fi

cd "${CUTLASS_DIR}"
git fetch --tags --all
git checkout "${CUTLASS_REF}"
git submodule update --init --recursive

cd "${ROOT}"
"${CUDA_HOME}/bin/nvcc" \
  -O3 \
  -std=c++17 \
  --expt-relaxed-constexpr \
  -DNDEBUG \
  -arch=sm_90a \
  -I"${CUTLASS_DIR}/include" \
  -I"${CUTLASS_DIR}/tools/util/include" \
  benchmark/benchmarks/gpu/external/cutlass_gemm_bench.cu \
  -o "${TMP_BIN}"

install -m 0755 "${TMP_BIN}" "${OUT_BIN}"
echo "${OUT_BIN}"
