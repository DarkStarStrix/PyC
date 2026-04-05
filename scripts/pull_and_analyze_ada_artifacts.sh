#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-64.247.206.171}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/ubuntu/work/PyC}"
SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/prime_next}"
LOCAL_HOST_SLUG="${LOCAL_HOST_SLUG:-host0356_kci2_ty6k_prxmx100056}"
LOCAL_REMOTE_ROOT="${LOCAL_REMOTE_ROOT:-$ROOT/benchmark/benchmarks/results/remote_results/hosts/$LOCAL_HOST_SLUG/runs}"
LOCAL_ANALYSIS_ROOT="${LOCAL_ANALYSIS_ROOT:-$ROOT/benchmark/benchmarks/results/analysis/ada}"

RUN_ID="${RUN_ID:-}"
TAG="${TAG:-}"
KERNEL_PATTERN="${KERNEL_PATTERN:-*ada_gemm*.json}"
SKIP_PULL=0
SKIP_ANALYSIS=0

usage() {
  cat <<EOF
Usage: bash scripts/pull_and_analyze_ada_artifacts.sh [options]

Pull one Ada sweep from the remote GPU box, stage the latest kernel-lab result,
and generate the local analysis bundle with graphs, sheets, and rankings.

This is the canonical operator flow for both fixed-shape sweeps and mixed-shape
`gemm_sequence` runs.

Options:
  --host HOST                 Remote host (default: ${REMOTE_HOST})
  --user USER                 Remote user (default: ${REMOTE_USER})
  --identity PATH             SSH identity (default: ${SSH_IDENTITY})
  --remote-repo-root PATH     Remote repo root (default: ${REMOTE_REPO_ROOT})
  --local-host-slug SLUG      Local host folder name (default: ${LOCAL_HOST_SLUG})
  --run-id RUN_ID             Explicit run id to pull
  --tag TAG                   Explicit tag to pull
  --kernel-pattern GLOB       Remote kernel-lab glob (default: ${KERNEL_PATTERN})
  --skip-pull                 Skip SSH/SCP and analyze local files only
  --skip-analysis             Pull only, do not run analysis
  -h, --help                  Show this help

Examples:
  bash scripts/pull_and_analyze_ada_artifacts.sh
  bash scripts/pull_and_analyze_ada_artifacts.sh --run-id 20260404T213005Z --tag ada-sm89-fp32-comparable-pyc-sweep-cublaslt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --user)
      REMOTE_USER="$2"
      shift 2
      ;;
    --identity)
      SSH_IDENTITY="$2"
      shift 2
      ;;
    --remote-repo-root)
      REMOTE_REPO_ROOT="$2"
      shift 2
      ;;
    --local-host-slug)
      LOCAL_HOST_SLUG="$2"
      LOCAL_REMOTE_ROOT="$ROOT/benchmark/benchmarks/results/remote_results/hosts/$LOCAL_HOST_SLUG/runs"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --kernel-pattern)
      KERNEL_PATTERN="$2"
      shift 2
      ;;
    --skip-pull)
      SKIP_PULL=1
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[pull-ada] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SSH_OPTS=(-i "$SSH_IDENTITY" -o StrictHostKeyChecking=no)
REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

discover_latest_remote_run() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" \
    "python3 - <<'PY'
from pathlib import Path
root = Path('$REMOTE_REPO_ROOT') / 'benchmark' / 'benchmarks' / 'results' / 'json'
candidates = []
for path in root.glob('*.json'):
    name = path.name
    if name.startswith('latest_') or name.endswith('.metadata.json') or name.endswith('.progress.json'):
        continue
    stem = path.stem
    if '__' not in stem:
        continue
    run_id, tag = stem.split('__', 1)
    if '__' in tag:
        continue
    candidates.append((path.stat().st_mtime, run_id, tag, str(path)))
if not candidates:
    raise SystemExit(1)
candidates.sort()
_, run_id, tag, path = candidates[-1]
print(run_id)
print(tag)
print(path)
PY"
}

if [[ "$SKIP_PULL" -eq 0 ]]; then
  if [[ -z "$RUN_ID" || -z "$TAG" ]]; then
    mapfile -t DISCOVERED < <(discover_latest_remote_run)
    if [[ "${#DISCOVERED[@]}" -lt 3 ]]; then
      echo "[pull-ada] failed to discover latest remote run" >&2
      exit 1
    fi
    RUN_ID="${RUN_ID:-${DISCOVERED[0]}}"
    TAG="${TAG:-${DISCOVERED[1]}}"
    echo "[pull-ada] discovered latest run: run_id=${RUN_ID} tag=${TAG}"
    echo "[pull-ada] source sweep: ${DISCOVERED[2]}"
  fi
else
  if [[ -z "$RUN_ID" || -z "$TAG" ]]; then
    echo "[pull-ada] --skip-pull requires --run-id and --tag" >&2
    exit 2
  fi
fi

LOCAL_RUN_DIR="$LOCAL_REMOTE_ROOT/$RUN_ID/$TAG"
LOCAL_SWEEP_JSON="$LOCAL_RUN_DIR/${RUN_ID}__${TAG}.json"
LOCAL_KERNEL_JSON="$LOCAL_RUN_DIR/kernel_lab_ada_gemm_result.json"
LOCAL_OUTPUT_DIR="$LOCAL_ANALYSIS_ROOT/$RUN_ID/$TAG"

if [[ "$SKIP_PULL" -eq 0 ]]; then
  REMOTE_RESULTS_DIR="$REMOTE_REPO_ROOT/benchmark/benchmarks/results"
  REMOTE_RUN_DIR="$REMOTE_RESULTS_DIR/remote_results/hosts/$LOCAL_HOST_SLUG/runs/$RUN_ID/$TAG"
  REMOTE_KERNEL_DIR="$REMOTE_REPO_ROOT/kernels/lab/results"

  mkdir -p "$LOCAL_RUN_DIR"

  echo "[pull-ada] pulling run dir -> $LOCAL_RUN_DIR"
  scp "${SSH_OPTS[@]}" -r "$REMOTE_TARGET:$REMOTE_RUN_DIR/." "$LOCAL_RUN_DIR/"

  echo "[pull-ada] pulling latest kernel result matching $KERNEL_PATTERN"
  REMOTE_KERNEL_PATH="$(ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" \
    "python3 - <<'PY'
from pathlib import Path
root = Path('$REMOTE_KERNEL_DIR')
matches = sorted(root.glob('$KERNEL_PATTERN'), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit(1)
print(matches[-1])
PY")"
  scp "${SSH_OPTS[@]}" "$REMOTE_TARGET:$REMOTE_KERNEL_PATH" "$LOCAL_KERNEL_JSON"
fi

if [[ ! -f "$LOCAL_SWEEP_JSON" ]]; then
  echo "[pull-ada] missing sweep json: $LOCAL_SWEEP_JSON" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_KERNEL_JSON" ]]; then
  echo "[pull-ada] missing kernel json: $LOCAL_KERNEL_JSON" >&2
  exit 1
fi

python3 - "$LOCAL_SWEEP_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
task = payload.get("task", "unknown")
shapes = payload.get("shapes") or []
print(f"[pull-ada] sweep kind: {task}")
print(f"[pull-ada] sweep shapes: {len(shapes)}")
PY

if [[ "$SKIP_ANALYSIS" -eq 0 ]]; then
  echo "[pull-ada] running analysis bundle"
  python3 "$ROOT/benchmark/tools/analyze_ada_gemm_results.py" \
    --run-dir "$LOCAL_RUN_DIR" \
    --sweep-json "$LOCAL_SWEEP_JSON" \
    --kernel-json "$LOCAL_KERNEL_JSON" \
    --output-dir "$LOCAL_OUTPUT_DIR"
fi

echo "[pull-ada] run dir: $LOCAL_RUN_DIR"
echo "[pull-ada] sweep json: $LOCAL_SWEEP_JSON"
echo "[pull-ada] kernel json: $LOCAL_KERNEL_JSON"
echo "[pull-ada] analysis dir: $LOCAL_OUTPUT_DIR"
