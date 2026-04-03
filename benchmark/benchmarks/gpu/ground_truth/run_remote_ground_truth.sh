#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
LOCAL_GT_DIR="${ROOT}/benchmark/benchmarks/gpu/ground_truth"
LOCAL_RESULTS_ROOT="${ROOT}/benchmark/benchmarks/results/profiles/ground_truth"

HOST=""
PORT=""
KEY=""
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_ground_truth}"
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/pyc_ground_truth}"

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "${HOST}" ] || [ -z "${PORT}" ] || [ -z "${KEY}" ]; then
  echo "Usage: $0 --host <ip> --port <port> --key <private_key_path> [--run-id <id>] [--remote-root <dir>]" >&2
  exit 1
fi

SSH_OPTS=(-i "${KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "${PORT}")
SCP_OPTS=(-i "${KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${PORT}")

REMOTE_GT_DIR="${REMOTE_ROOT}/benchmark/benchmarks/gpu/ground_truth"
REMOTE_RESULTS_ROOT="${REMOTE_ROOT}/benchmark/benchmarks/results/profiles/ground_truth"

echo "[remote-ground-truth] staging scripts on ${HOST}:${REMOTE_ROOT}"
ssh "${SSH_OPTS[@]}" "root@${HOST}" "mkdir -p '${REMOTE_GT_DIR}' '${REMOTE_RESULTS_ROOT}'"
scp "${SCP_OPTS[@]}" "${LOCAL_GT_DIR}/encoder_ground_truth.py" "root@${HOST}:${REMOTE_GT_DIR}/"
scp "${SCP_OPTS[@]}" "${LOCAL_GT_DIR}/run_ground_truth_capture.sh" "root@${HOST}:${REMOTE_GT_DIR}/"

echo "[remote-ground-truth] executing run id=${RUN_ID}"
ssh "${SSH_OPTS[@]}" "root@${HOST}" \
  "cd '${REMOTE_ROOT}' && chmod +x '${REMOTE_GT_DIR}/run_ground_truth_capture.sh' && PATH=/usr/local/cuda/bin:\$PATH '${REMOTE_GT_DIR}/run_ground_truth_capture.sh' '${REMOTE_RESULTS_ROOT}' '${RUN_ID}'"

mkdir -p "${LOCAL_RESULTS_ROOT}"
echo "[remote-ground-truth] pulling artifacts"
scp "${SCP_OPTS[@]}" -r "root@${HOST}:${REMOTE_RESULTS_ROOT}/${RUN_ID}" "${LOCAL_RESULTS_ROOT}/"

if [ -f "${LOCAL_RESULTS_ROOT}/${RUN_ID}/nsys_stats.csv" ]; then
  python3 "${LOCAL_GT_DIR}/render_nsys_flamegraph.py" \
    --input "${LOCAL_RESULTS_ROOT}/${RUN_ID}/nsys_stats.csv" \
    --output "${LOCAL_RESULTS_ROOT}/${RUN_ID}/nsys_kernel_flame.svg" \
    --title "PyC Ground Truth (${HOST})"
fi

echo "[remote-ground-truth] complete: ${LOCAL_RESULTS_ROOT}/${RUN_ID}"
