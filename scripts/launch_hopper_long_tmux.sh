#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-pyc-hopper}"
WINDOW_NAME="${WINDOW_NAME:-hopper-long}"
REPO_ROOT="${REPO_ROOT:-/root/work/PyC/repo}"
MATRIX_FILE="${MATRIX_FILE:-benchmark/benchmarks/gpu/configs/hopper_bf16_gemm_shapes_long.json}"
TAG="${TAG:-hopper_bf16_long_native}"
PYC_GPU_BENCH_TIMEOUT_SEC="${PYC_GPU_BENCH_TIMEOUT_SEC:-180}"

tmux has-session -t "${SESSION}" 2>/dev/null || tmux new-session -d -s "${SESSION}" -n main bash
if tmux list-windows -t "${SESSION}" -F "#{window_name}" | grep -qx "${WINDOW_NAME}"; then
  tmux kill-window -t "${SESSION}:${WINDOW_NAME}"
fi

read -r -d '' WINDOW_CMD <<'EOF' || true
while ! command -v cutlass_gemm_bench >/dev/null 2>&1 && ! command -v cutlass_profiler >/dev/null 2>&1; do
  echo "[hopper-long] waiting for CUTLASS benchmark binary"
  sleep 20
done
cd "__REPO_ROOT__"
source .venv/bin/activate
source benchmark/benchmarks/gpu/configure_adapter_cmds.sh
export BENCH_STRICT_NATIVE=1
export PYC_GPU_BENCH_TIMEOUT_SEC="__TIMEOUT__"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
tag="__TAG__"
python benchmark/benchmarks/gpu/run_gemm_suite.py \
  --device cuda \
  --matrix-file "__MATRIX_FILE__" \
  --arena-mode \
  --parity-strict \
  --progress \
  --run-id "${run_id}" \
  --tag "${tag}"
EOF

WINDOW_CMD="${WINDOW_CMD//__REPO_ROOT__/${REPO_ROOT}}"
WINDOW_CMD="${WINDOW_CMD//__MATRIX_FILE__/${MATRIX_FILE}}"
WINDOW_CMD="${WINDOW_CMD//__TAG__/${TAG}}"
WINDOW_CMD="${WINDOW_CMD//__TIMEOUT__/${PYC_GPU_BENCH_TIMEOUT_SEC}}"

tmux new-window -t "${SESSION}" -n "${WINDOW_NAME}" "bash -lc '$WINDOW_CMD'"
tmux select-window -t "${SESSION}:${WINDOW_NAME}"
tmux list-windows -t "${SESSION}" -F "#{window_index}:#{window_name}:#{window_active}"
