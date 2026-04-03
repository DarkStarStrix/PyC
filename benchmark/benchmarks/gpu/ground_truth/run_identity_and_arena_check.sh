#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CAPTURE="${ROOT}/benchmark/benchmarks/gpu/ground_truth/run_ground_truth_capture.sh"
VERIFY="${ROOT}/benchmark/benchmarks/gpu/ground_truth/verify_kernel_identity.py"
OUT_ROOT="${1:-${ROOT}/benchmark/benchmarks/results/profiles/ground_truth}"
RUN_ID_BASE="${2:-$(date -u +%Y%m%dT%H%M%SZ)_identity}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 required" >&2
  exit 1
fi

if [ ! -x "${CAPTURE}" ]; then
  echo "Missing capture runner: ${CAPTURE}" >&2
  exit 1
fi

if [ ! -f "${VERIFY}" ]; then
  echo "Missing verifier: ${VERIFY}" >&2
  exit 1
fi

COMMON_ENV=(
  DEVICE="${DEVICE:-cuda}"
  DTYPE="${DTYPE:-float16}"
  HIDDEN="${HIDDEN:-2048}"
  PAD_HIDDEN_TO="${PAD_HIDDEN_TO:-16}"
  BATCH="${BATCH:-64}"
  BUCKET_LENGTHS="${BUCKET_LENGTHS:-256}"
  BATCH_SIZES="${BATCH_SIZES:-64}"
  PROFILE_SEQ_LEN="${PROFILE_SEQ_LEN:-256}"
  WARMUP="${WARMUP:-30}"
  ITERS="${ITERS:-120}"
  ALLOW_TF32="${ALLOW_TF32:-1}"
  USE_INPLACE_RESIDUAL="${USE_INPLACE_RESIDUAL:-1}"
)

EAGER_RUN="${RUN_ID_BASE}_eager"
COMPILED_RUN="${RUN_ID_BASE}_compiled"

echo "[identity-check] eager capture: ${EAGER_RUN}"
env "${COMMON_ENV[@]}" \
  USE_TORCH_COMPILE=0 \
  USE_CUDA_GRAPH_ARENA="${USE_CUDA_GRAPH_ARENA_EAGER:-0}" \
  "${CAPTURE}" "${OUT_ROOT}" "${EAGER_RUN}"

echo "[identity-check] compiled capture: ${COMPILED_RUN}"
env "${COMMON_ENV[@]}" \
  USE_TORCH_COMPILE=1 \
  COMPILE_MODE="${COMPILE_MODE:-default}" \
  COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}" \
  INDUCTOR_GEMM_BACKENDS="${INDUCTOR_GEMM_BACKENDS:-ATEN}" \
  USE_CUDA_GRAPH_ARENA="${USE_CUDA_GRAPH_ARENA_COMPILED:-0}" \
  "${CAPTURE}" "${OUT_ROOT}" "${COMPILED_RUN}"

python3 "${VERIFY}" \
  --eager-stats "${OUT_ROOT}/${EAGER_RUN}/nsys_stats.csv" \
  --compiled-stats "${OUT_ROOT}/${COMPILED_RUN}/nsys_stats.csv" \
  --top-k "${TOP_K:-2}" \
  --output "${OUT_ROOT}/${RUN_ID_BASE}_identity_report.json"

echo "[identity-check] wrote ${OUT_ROOT}/${RUN_ID_BASE}_identity_report.json"
