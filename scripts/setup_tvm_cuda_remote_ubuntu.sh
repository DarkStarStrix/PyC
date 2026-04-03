#!/usr/bin/env bash
set -euo pipefail

# Build TVM with CUDA enabled on Ubuntu GPU hosts.
# Usage:
#   source .venv/bin/activate
#   bash scripts/setup_tvm_cuda_remote_ubuntu.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TVM_DIR="${TVM_DIR:-/tmp/tvm}"
TVM_REF="${TVM_REF:-v0.14.0}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git cmake ninja-build build-essential llvm-dev libtinfo-dev zlib1g-dev

if [ ! -d "${TVM_DIR}" ]; then
  git clone --recursive https://github.com/apache/tvm "${TVM_DIR}"
fi

cd "${TVM_DIR}"
git fetch --tags --all
git checkout "${TVM_REF}"
git submodule update --init --recursive

mkdir -p build
cp cmake/config.cmake build/config.cmake
{
  echo "set(USE_CUDA ON)"
  echo "set(USE_CUBLAS ON)"
  echo "set(USE_CUDNN OFF)"
  echo "set(USE_LLVM llvm-config)"
  echo "set(CMAKE_BUILD_TYPE Release)"
} >> build/config.cmake

cmake -S . -B build -G Ninja
cmake --build build --parallel

cd python
# Remove wheel build if present so editable CUDA build is authoritative.
pip uninstall -y apache-tvm tvm || true
pip install -e .

export TVM_LIBRARY_PATH="${TVM_DIR}/build"

python - <<'PY'
import tvm
print("tvm_version", tvm.__version__)
print("tvm_cuda_runtime_enabled", bool(tvm.runtime.enabled("cuda")))
dev = tvm.cuda(0)
print("tvm_cuda_device_exist", bool(dev.exist))
PY
