#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-benchmark/remote_results/runpod_h100_8x/campaign_v4}"
MODEL="${MODEL:-distilbert-base-uncased}"
DATASET="${DATASET:-ag_news}"
NPROC="${NPROC:-8}"
MODE="${MODE:-nexa_vortex}"
DIST_MODE="${DIST_MODE:-ddp}"
BACKEND="${BACKEND:-nccl}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
if [[ -n "${HUGGINGFACE_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_TOKEN}"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

CMD=(
  torchrun --standalone --nproc_per_node "${NPROC}" scripts/train.py
  --run-id "${RUN_ID}"
  --out-root "${OUT_ROOT}"
  --mode "${MODE}"
  --dist "${DIST_MODE}"
  --backend "${BACKEND}"
  --model-name "${MODEL}"
  --dataset-name "${DATASET}"
  --epochs "${EPOCHS:-1.0}"
  --max-train-samples "${MAX_TRAIN_SAMPLES:-120000}"
  --max-eval-samples "${MAX_EVAL_SAMPLES:-7600}"
  --max-length "${MAX_LENGTH:-256}"
  --per-device-batch "${PER_DEVICE_BATCH:-8}"
  --grad-accum "${GRAD_ACCUM:-4}"
  --dataloader-workers "${DATALOADER_WORKERS:-1}"
  --prefetch-factor "${PREFETCH_FACTOR:-2}"
  --dataloader-timeout-sec "${DATALOADER_TIMEOUT_SEC:-120}"
  --pin-memory "${PIN_MEMORY:-true}"
  --persistent-workers "${PERSISTENT_WORKERS:-false}"
  --non-blocking-h2d "${NON_BLOCKING_H2D:-true}"
  --mixed-precision "${MIXED_PRECISION:-bf16}"
  --torch-compile "${TORCH_COMPILE:-none}"
  --progress "${PROGRESS:-on}"
)

echo "[campaign] run_id=${RUN_ID}"
echo "[campaign] out_root=${OUT_ROOT}"
printf '[campaign] cmd=%q ' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[campaign] done"
