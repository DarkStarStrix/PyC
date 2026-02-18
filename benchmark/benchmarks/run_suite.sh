#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${1:-${ROOT}/benchmark/benchmarks/results}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
ITERS="${ITERS:-40}"
WARMUP="${WARMUP:-10}"
BATCH="${BATCH:-64}"
HIDDEN="${HIDDEN:-2048}"
PYC_REQUIRE_NATIVE_CUDA="${PYC_REQUIRE_NATIVE_CUDA:-0}"
CUDA_NATIVE_ARGS=()
if [ "${PYC_REQUIRE_NATIVE_CUDA}" = "1" ]; then
  CUDA_NATIVE_ARGS+=(--require-native-adapter pyc)
fi

mkdir -p "${OUT_ROOT}/json" "${OUT_ROOT}/reports" "${OUT_ROOT}/images"

source "${ROOT}/benchmark/benchmarks/gpu/configure_adapter_cmds.sh"

python3 "${ROOT}/benchmark/benchmarks/gpu/run_gpu_suite.py" \
  --device cpu \
  --batch "${BATCH}" \
  --hidden "${HIDDEN}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --run-id "${RUN_ID}" \
  --tag cpu \
  --output-root "${OUT_ROOT}"

python3 "${ROOT}/benchmark/benchmarks/gpu/run_gpu_suite.py" \
  --device cuda \
  --batch "${BATCH}" \
  --hidden "${HIDDEN}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --run-id "${RUN_ID}" \
  --tag gpu \
  "${CUDA_NATIVE_ARGS[@]}" \
  --output-root "${OUT_ROOT}"

echo "Benchmark suite complete. Run ID: ${RUN_ID}"
