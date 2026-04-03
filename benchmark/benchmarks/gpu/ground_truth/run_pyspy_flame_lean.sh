#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
SCRIPT="${ROOT}/benchmark/benchmarks/gpu/ground_truth/encoder_ground_truth.py"
OUT_DIR="${1:-${ROOT}/benchmark/benchmarks/results/profiles/ground_truth}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)_pyspy_lean}"

RATE="${RATE:-49}"
DURATION="${DURATION:-60}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
HIDDEN="${HIDDEN:-2048}"
PAD_HIDDEN_TO="${PAD_HIDDEN_TO:-16}"
WARMUP="${WARMUP:-80}"
ITERS="${ITERS:-8000}"
SEQ_LEN="${SEQ_LEN:-256}"
BATCH="${BATCH:-64}"
ALLOW_TF32="${ALLOW_TF32:-1}"
USE_INPLACE_RESIDUAL="${USE_INPLACE_RESIDUAL:-1}"

mkdir -p "${OUT_DIR}/${RUN_ID}"

if ! command -v py-spy >/dev/null 2>&1; then
  echo "py-spy is required for this script" >&2
  exit 1
fi

ARGS=(
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --hidden "${HIDDEN}"
  --pad-hidden-to "${PAD_HIDDEN_TO}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --bucket-lengths "${SEQ_LEN}"
  --batch-sizes "${BATCH}"
  --single-shape-seq-len "${SEQ_LEN}"
  --skip-batch-scaling
  --output "${OUT_DIR}/${RUN_ID}/pyspy_profile.json"
)
if [ "${ALLOW_TF32}" = "1" ]; then
  ARGS+=(--allow-tf32)
fi
if [ "${USE_INPLACE_RESIDUAL}" = "1" ]; then
  ARGS+=(--use-inplace-residual)
fi

py-spy record \
  --rate "${RATE}" \
  --duration "${DURATION}" \
  --format flamegraph \
  -o "${OUT_DIR}/${RUN_ID}/pyspy_flame.svg" \
  -- python3 "${SCRIPT}" "${ARGS[@]}"

echo "[pyspy-lean] wrote ${OUT_DIR}/${RUN_ID}/pyspy_flame.svg"
