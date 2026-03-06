#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
SCRIPT="${ROOT}/benchmark/benchmarks/gpu/ground_truth/encoder_ground_truth.py"
OUT_ROOT="${1:-${ROOT}/benchmark/benchmarks/results/profiles/ground_truth}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
HIDDEN="${HIDDEN:-2048}"
BATCH="${BATCH:-64}"
WARMUP="${WARMUP:-40}"
ITERS="${ITERS:-300}"
BUCKET_LENGTHS="${BUCKET_LENGTHS:-128,256,512}"
BATCH_SIZES="${BATCH_SIZES:-16,32,64,96,128}"
NSYS_WARMUP="${NSYS_WARMUP:-20}"
NCU_WARMUP="${NCU_WARMUP:-20}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_BACKEND="${COMPILE_BACKEND:-inductor}"
INDUCTOR_GEMM_BACKENDS="${INDUCTOR_GEMM_BACKENDS:-}"
ALLOW_TF32="${ALLOW_TF32:-0}"
DISABLE_INFERENCE_MODE="${DISABLE_INFERENCE_MODE:-0}"
USE_INPLACE_RESIDUAL="${USE_INPLACE_RESIDUAL:-0}"
PAD_HIDDEN_TO="${PAD_HIDDEN_TO:-0}"
PROFILE_SEQ_LEN="${PROFILE_SEQ_LEN:-}"
USE_CUDA_GRAPH_ARENA="${USE_CUDA_GRAPH_ARENA:-0}"

if [ ! -f "${SCRIPT}" ]; then
  echo "Missing ground-truth runner: ${SCRIPT}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

RUN_DIR="${OUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

COMMON_ARGS=(
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --hidden "${HIDDEN}"
  --batch "${BATCH}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --bucket-lengths "${BUCKET_LENGTHS}"
  --batch-sizes "${BATCH_SIZES}"
  --output "${RUN_DIR}/ground_truth.json"
)

EXTRA_ARGS=()
if [ "${USE_TORCH_COMPILE}" = "1" ]; then
  EXTRA_ARGS+=(--use-torch-compile)
  EXTRA_ARGS+=(--compile-mode "${COMPILE_MODE}")
  EXTRA_ARGS+=(--compile-backend "${COMPILE_BACKEND}")
  if [ -n "${INDUCTOR_GEMM_BACKENDS}" ]; then
    EXTRA_ARGS+=(--inductor-gemm-backends "${INDUCTOR_GEMM_BACKENDS}")
  fi
fi
if [ "${ALLOW_TF32}" = "1" ]; then
  EXTRA_ARGS+=(--allow-tf32)
fi
if [ "${DISABLE_INFERENCE_MODE}" = "1" ]; then
  EXTRA_ARGS+=(--disable-inference-mode)
fi
if [ "${USE_INPLACE_RESIDUAL}" = "1" ]; then
  EXTRA_ARGS+=(--use-inplace-residual)
fi
if [ "${PAD_HIDDEN_TO}" != "0" ]; then
  EXTRA_ARGS+=(--pad-hidden-to "${PAD_HIDDEN_TO}")
fi
if [ "${USE_CUDA_GRAPH_ARENA}" = "1" ]; then
  EXTRA_ARGS+=(--use-cuda-graph-arena)
fi
COMMON_ARGS+=("${EXTRA_ARGS[@]}")

if [ -z "${PROFILE_SEQ_LEN}" ]; then
  PROFILE_SEQ_LEN="$(python3 - <<'PY' "${BUCKET_LENGTHS}"
import sys
vals=[int(x.strip()) for x in sys.argv[1].split(",") if x.strip()]
if not vals:
    vals=[256]
print(vals[len(vals)//2])
PY
)"
fi

echo "[ground-truth] running baseline checks -> ${RUN_DIR}/ground_truth.json"
python3 "${SCRIPT}" "${COMMON_ARGS[@]}"

if [ "${DEVICE}" = "cuda" ] && command -v nsys >/dev/null 2>&1; then
  echo "[ground-truth] capturing Nsight Systems timeline"
  if nsys profile \
      --force-overwrite=true \
      --sample=none \
      --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --output "${RUN_DIR}/nsys_timeline" \
      python3 "${SCRIPT}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}" \
        --hidden "${HIDDEN}" \
        --batch "${BATCH}" \
        --warmup "${NSYS_WARMUP}" \
        --iters 1 \
        --bucket-lengths "${PROFILE_SEQ_LEN}" \
        --batch-sizes "${BATCH}" \
        --single-shape-seq-len "${PROFILE_SEQ_LEN}" \
        --skip-batch-scaling \
        --single-pass-after-warmup \
        --cuda-profiler-range \
        "${EXTRA_ARGS[@]}" \
      --output "${RUN_DIR}/nsys_single_pass.json"; then
    nsys stats \
      --report cuda_gpu_kern_sum,cuda_api_sum \
      --format csv \
      "${RUN_DIR}/nsys_timeline.nsys-rep" \
      > "${RUN_DIR}/nsys_stats.csv"
  else
    echo "[ground-truth] nsys capture failed" | tee "${RUN_DIR}/nsys_error.txt"
  fi
else
  echo "[ground-truth] skipping nsys timeline capture (requires cuda + nsys)"
fi

if [ "${DEVICE}" = "cuda" ] && command -v ncu >/dev/null 2>&1; then
  echo "[ground-truth] capturing Nsight Compute kernel metrics"
  if ! ncu \
      --force-overwrite \
      --target-processes all \
      --set full \
      --export "${RUN_DIR}/ncu_profile" \
      python3 "${SCRIPT}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}" \
        --hidden "${HIDDEN}" \
        --batch "${BATCH}" \
        --warmup "${NCU_WARMUP}" \
        --iters 1 \
        --bucket-lengths "${PROFILE_SEQ_LEN}" \
        --batch-sizes "${BATCH}" \
        --single-shape-seq-len "${PROFILE_SEQ_LEN}" \
        --skip-batch-scaling \
        --single-pass-after-warmup \
        "${EXTRA_ARGS[@]}" \
        --output "${RUN_DIR}/ncu_single_pass.json"; then
    echo "[ground-truth] ncu capture failed (likely GPU perf counter permissions)." \
      | tee "${RUN_DIR}/ncu_error.txt"
  fi
else
  echo "[ground-truth] skipping ncu kernel metrics capture (requires cuda + ncu)"
fi

python3 - <<'PY' "${RUN_DIR}" "${RUN_ID}" "${DEVICE}" "${DTYPE}" "${HIDDEN}" "${BATCH}" "${WARMUP}" "${ITERS}" "${BUCKET_LENGTHS}" "${BATCH_SIZES}" "${USE_TORCH_COMPILE}" "${COMPILE_MODE}" "${ALLOW_TF32}" "${DISABLE_INFERENCE_MODE}" "${USE_INPLACE_RESIDUAL}" "${PAD_HIDDEN_TO}" "${COMPILE_BACKEND}" "${INDUCTOR_GEMM_BACKENDS}" "${USE_CUDA_GRAPH_ARENA}"
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

run_dir = sys.argv[1]
run_id = sys.argv[2]
device = sys.argv[3]
dtype = sys.argv[4]
hidden = int(sys.argv[5])
batch = int(sys.argv[6])
warmup = int(sys.argv[7])
iters = int(sys.argv[8])
buckets = sys.argv[9]
batches = sys.argv[10]
torch_compile = bool(int(sys.argv[11]))
compile_mode = sys.argv[12]
allow_tf32 = bool(int(sys.argv[13]))
disable_inference_mode = bool(int(sys.argv[14]))
use_inplace_residual = bool(int(sys.argv[15]))
pad_hidden_to = int(sys.argv[16])
compile_backend = sys.argv[17]
inductor_gemm_backends = sys.argv[18]
use_cuda_graph_arena = bool(int(sys.argv[19]))

def cmd_out(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

meta = {
    "run_id": run_id,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "host": platform.node(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "device": device,
    "dtype": dtype,
    "hidden": hidden,
    "batch": batch,
    "warmup": warmup,
    "iters": iters,
    "bucket_lengths": buckets,
    "batch_sizes": batches,
    "use_torch_compile": torch_compile,
    "compile_mode": compile_mode,
    "compile_backend": compile_backend,
    "inductor_gemm_backends": inductor_gemm_backends,
    "allow_tf32": allow_tf32,
    "inference_mode": not disable_inference_mode,
    "use_inplace_residual": use_inplace_residual,
    "pad_hidden_to": pad_hidden_to,
    "use_cuda_graph_arena": use_cuda_graph_arena,
    "tool_versions": {
        "nsys": cmd_out(["nsys", "--version"]) if shutil.which("nsys") else "",
        "ncu": cmd_out(["ncu", "--version"]) if shutil.which("ncu") else "",
        "nvidia_smi": cmd_out(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"]) if shutil.which("nvidia-smi") else "",
    },
}

with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PY

echo "[ground-truth] complete: ${RUN_DIR}"
