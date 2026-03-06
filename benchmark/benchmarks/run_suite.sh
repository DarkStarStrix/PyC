#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${1:-${ROOT}/benchmark/benchmarks/results}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
ITERS="${ITERS:-40}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-3}"
BATCH="${BATCH:-64}"
HIDDEN="${HIDDEN:-2048}"
CPU_ADAPTERS="${CPU_ADAPTERS:-torch_eager,torch_compile,pyc,tvm,xla,tensorrt,glow}"
GPU_ADAPTERS="${GPU_ADAPTERS:-torch_eager,torch_compile,pyc,tvm,xla,tensorrt,glow}"
BENCH_STRICT_NATIVE="${BENCH_STRICT_NATIVE:-0}"
STRICT_NATIVE_REQUIRED_CPU="${STRICT_NATIVE_REQUIRED_CPU:-torch_eager,torch_compile,pyc,tvm}"
STRICT_NATIVE_REQUIRED_GPU="${STRICT_NATIVE_REQUIRED_GPU:-torch_eager,torch_compile,pyc,tvm}"
BENCH_PROGRESS="${BENCH_PROGRESS:-1}"
BENCH_PRUNE_FLAT_HISTORY="${BENCH_PRUNE_FLAT_HISTORY:-1}"

CPU_NATIVE_ARGS=()
GPU_NATIVE_ARGS=()
PROGRESS_ARGS=()
if [ "${BENCH_STRICT_NATIVE}" = "1" ]; then
  CPU_NATIVE_ARGS+=(--require-native-adapter "${STRICT_NATIVE_REQUIRED_CPU}")
  GPU_NATIVE_ARGS+=(--require-native-adapter "${STRICT_NATIVE_REQUIRED_GPU}")
fi
if [ "${BENCH_PROGRESS}" = "1" ]; then
  PROGRESS_ARGS+=(--progress)
fi

mkdir -p "${OUT_ROOT}/json" "${OUT_ROOT}/reports" "${OUT_ROOT}/images"

source "${ROOT}/benchmark/benchmarks/gpu/configure_adapter_cmds.sh"

python3 "${ROOT}/benchmark/benchmarks/gpu/run_gpu_suite.py" \
  --device cpu \
  --adapters "${CPU_ADAPTERS}" \
  --batch "${BATCH}" \
  --hidden "${HIDDEN}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --run-id "${RUN_ID}" \
  --tag cpu \
  "${PROGRESS_ARGS[@]}" \
  "${CPU_NATIVE_ARGS[@]}" \
  --output-root "${OUT_ROOT}"

python3 "${ROOT}/benchmark/benchmarks/gpu/run_gpu_suite.py" \
  --device cuda \
  --adapters "${GPU_ADAPTERS}" \
  --batch "${BATCH}" \
  --hidden "${HIDDEN}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --run-id "${RUN_ID}" \
  --tag gpu \
  "${PROGRESS_ARGS[@]}" \
  "${GPU_NATIVE_ARGS[@]}" \
  --output-root "${OUT_ROOT}"

# Normalize and re-render canonical latest artifacts after each suite run.
STANDARDIZE_ARGS=(--results-root "${OUT_ROOT}")
if [ "${BENCH_PRUNE_FLAT_HISTORY}" = "1" ]; then
  STANDARDIZE_ARGS+=(--prune-flat-history)
fi
python3 "${ROOT}/scripts/standardize_benchmark_results.py" "${STANDARDIZE_ARGS[@]}"

echo "Benchmark suite complete. Run ID: ${RUN_ID}"
