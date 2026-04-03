#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BUILD_DIR="${PYC_BUILD_DIR:-build-distributed}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_BENCH="${RUN_BENCH:-1}"
DIST_ITERS="${DIST_ITERS:-4000}"
DIST_COUNT="${DIST_COUNT:-1024}"
DIST_REPEATS="${DIST_REPEATS:-3}"
DIST_TAG="${DIST_TAG:-distributed_comm_smoke}"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[build] configure ${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" \
  -D PYC_BUILD_EXPERIMENTAL=OFF \
  -D PYC_BUILD_COMPILER_NEXT=ON \
  -D PYC_BUILD_COMPILER_NEXT_TESTS=ON \
  -D PYC_BUILD_DISTRIBUTED_SCAFFOLD=ON \
  -D PYC_BUILD_BENCHMARKS=ON

echo "[build] compile"
cmake --build "${BUILD_DIR}" --parallel

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "[build] ctest"
  ctest --test-dir "${BUILD_DIR}" -C Release --output-on-failure
fi

if [[ "${RUN_BENCH}" == "1" ]]; then
  echo "[build] distributed benchmark suite"
  python3 benchmark/benchmarks/distributed/run_distributed_comm_suite.py \
    --build-dir "${BUILD_DIR}" \
    --iters "${DIST_ITERS}" \
    --count "${DIST_COUNT}" \
    --repeats "${DIST_REPEATS}" \
    --tag "${DIST_TAG}"
fi

echo "[build] latest artifacts"
echo "  - benchmark/benchmarks/results/json/latest_distributed_comm.json"
echo "  - benchmark/benchmarks/results/reports/latest_distributed_comm.md"
echo "  - benchmark/benchmarks/results/images/latest_distributed_comm.svg"
