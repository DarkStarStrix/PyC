#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PYC_GPU_BENCH_BUILD_DIR:-$ROOT/build}"
EXE_NAME="${PYC_GPU_BENCH_EXE:-pyc_compiler_next_bench}"
EXE_PATH="$BUILD_DIR/$EXE_NAME"

if [[ ! -x "$EXE_PATH" ]]; then
  echo "[pyc-bench] missing executable: $EXE_PATH" >&2
  exit 1
fi

if [[ -n "${PYC_BENCH_JSON_OUT:-}" ]]; then
  TMP_JSON="$PYC_BENCH_JSON_OUT"
  KEEP_JSON=1
  mkdir -p "$(dirname "$TMP_JSON")"
else
  TMP_JSON="${TMPDIR:-/tmp}/pyc-bench-pretty-$$.json"
  KEEP_JSON=0
fi
TMP_STDERR="${TMPDIR:-/tmp}/pyc-bench-pretty-$$.stderr"
if [[ "$KEEP_JSON" -eq 0 ]]; then
  trap 'rm -f "$TMP_JSON" "$TMP_STDERR"' EXIT
else
  trap 'rm -f "$TMP_STDERR"' EXIT
fi

if ! "$EXE_PATH" "$@" >"$TMP_JSON" 2>"$TMP_STDERR"; then
  if [[ -s "$TMP_JSON" ]]; then
    python3 - "$TMP_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
status = payload.get("status", "unknown")
print(f"[pyc-bench] status={status} error={payload.get('error', 'unknown')}")
PY
  elif [[ -s "$TMP_STDERR" ]]; then
    cat "$TMP_STDERR" >&2
  fi
  exit 1
fi

python3 - "$TMP_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
status = payload.get("status", "unknown")
if status != "ok":
    print(f"[pyc-bench] status={status} error={payload.get('error', 'unknown')}")
    raise SystemExit(1)

task = payload.get("task", "-")
device = payload.get("device", "-")
m = payload.get("m", 0)
k = payload.get("k", 0)
n = payload.get("n", 0)
lat = payload.get("latency_ms", {}) or {}
profile = payload.get("profile", {}) or {}
kernel = (payload.get("kernel_selection", {}) or {}).get("symbol", "-")
path_name = payload.get("execution_path", "-")
fallback = (payload.get("reliability", {}) or {}).get("fallback_count", 0)
tflops = float(payload.get("throughput_tflops_per_sec", 0.0) or 0.0)
mean_ms = float(lat.get("mean", 0.0) or 0.0)
dispatch_ms = float(profile.get("dispatch_ms_mean", 0.0) or 0.0)
kernel_select_ms = float(profile.get("kernel_select_ms_mean", 0.0) or 0.0)

print(f"[pyc-bench] task={task} device={device} shape={m}x{k}x{n}")
print(
    f"[pyc-bench] mean_ms={mean_ms:.4f} tflops={tflops:.4f} "
    f"dispatch_ms={dispatch_ms:.4f} kernel_select_ms={kernel_select_ms:.4f}"
)
print(f"[pyc-bench] path={path_name} kernel={kernel} fallback={fallback}")
print(f"[pyc-bench] json={path}")
PY
