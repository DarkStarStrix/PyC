#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

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
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --model_name "${MODEL_NAME:-Qwen/Qwen2.5-14B}"
  --dataset_name "${DATASET_NAME:-nvidia/OpenCodeInstruct}"
  --precision "${PRECISION:-bf16}"
  --dist "${DIST_MODE:-fsdp}"
  --backend "${BACKEND:-nccl}"
  --fsdp full_shard auto_wrap
  --fsdp_transformer_layer_cls_to_wrap "${FSDP_LAYER_CLS:-Qwen2DecoderLayer}"
  --gradient_checkpointing true
  --num_workers "${NUM_WORKERS:-32}"
  --pin_memory true
  --prefetch_factor "${PREFETCH_FACTOR:-8}"
  --preprocessing_workers "${PREPROCESSING_WORKERS:-48}"
  --preprocessing_batch_size "${PREPROCESSING_BATCH_SIZE:-128}"
  --compile "${COMPILE_MODE:-reduce-overhead}"
)

if [[ "${MODE}" == "preflight" ]]; then
  exec torchrun --standalone --nproc_per_node="${NPROC:-8}" scripts/preflight_qwen_sft.py \
    --model_name "${MODEL_NAME:-Qwen/Qwen2.5-14B}" \
    --dataset_name "${DATASET_NAME:-nvidia/OpenCodeInstruct}" \
    --seq_length "${SEQ_LENGTH:-1024}" \
    --max_samples "${MAX_SAMPLES:-8}"
fi

if [[ "${MODE}" == "smoke" ]]; then
  exec torchrun --standalone --nproc_per_node="${NPROC:-8}" scripts/train_sft.py \
    "${COMMON_ARGS[@]}" \
    --dist "${DIST_MODE:-fsdp}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES:-4096}" \
    --max_eval_samples "${MAX_EVAL_SAMPLES:-256}" \
    --seq_length "${SEQ_LENGTH:-1024}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-1}" \
    --max_steps "${MAX_STEPS:-20}" \
    --learning_rate "${LEARNING_RATE:-2e-5}" \
    --warmup_steps "${WARMUP_STEPS:-5}" \
    --logging_steps 1 \
    --eval_steps 10 \
    --save_steps 20 \
    --output_dir "${OUTPUT_DIR:-runs/smoke_qwen14b}" \
    --run_name "${RUN_NAME:-smoke_qwen14b}"
fi

if [[ "${MODE}" == "distributed-smoke" ]]; then
  exec torchrun --standalone --nproc_per_node="${NPROC:-8}" scripts/train_sft.py \
    "${COMMON_ARGS[@]}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES:-2048}" \
    --max_eval_samples "${MAX_EVAL_SAMPLES:-0}" \
    --seq_length "${SEQ_LENGTH:-1024}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-1}" \
    --max_steps "${MAX_STEPS:-5}" \
    --learning_rate "${LEARNING_RATE:-2e-5}" \
    --warmup_steps "${WARMUP_STEPS:-2}" \
    --logging_steps 1 \
    --eval_steps 0 \
    --save_steps 0 \
    --output_dir "${OUTPUT_DIR:-runs/distributed_smoke_qwen14b}" \
    --run_name "${RUN_NAME:-distributed_smoke_qwen14b}"
fi

if [[ "${MODE}" == "production" ]]; then
  exec torchrun --standalone --nproc_per_node="${NPROC:-8}" scripts/train_sft.py \
    "${COMMON_ARGS[@]}" \
    --dataset_streaming true \
    --seq_length "${SEQ_LENGTH:-4096}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-8}" \
    --max_steps "${MAX_STEPS:-2000}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --warmup_steps "${WARMUP_STEPS:-200}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --eval_steps "${EVAL_STEPS:-200}" \
    --save_steps "${SAVE_STEPS:-500}" \
    --output_dir "${OUTPUT_DIR:-runs/qwen14b_codeinstruct}" \
    --run_name "${RUN_NAME:-qwen14b_codeinstruct}"
fi

echo "unsupported mode: ${MODE}" >&2
exit 1
