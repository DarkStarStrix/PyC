#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${PY_BIN:-python3}"

export PYC_GPU_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_pyc_cmd.py"
export TVM_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_tvm_cmd.py"
export XLA_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_xla_cmd.py"
export TENSORRT_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_tensorrt_cmd.py"
export GLOW_BENCH_CMD="${PY_BIN} ${ROOT}/benchmark/benchmarks/gpu/external/bench_glow_cmd.py"

echo "Configured adapter commands:"
echo "  PYC_GPU_BENCH_CMD=${PYC_GPU_BENCH_CMD}"
echo "  TVM_BENCH_CMD=${TVM_BENCH_CMD}"
echo "  XLA_BENCH_CMD=${XLA_BENCH_CMD}"
echo "  TENSORRT_BENCH_CMD=${TENSORRT_BENCH_CMD}"
echo "  GLOW_BENCH_CMD=${GLOW_BENCH_CMD}"
