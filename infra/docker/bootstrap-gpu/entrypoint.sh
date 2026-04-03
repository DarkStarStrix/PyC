#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [[ $# -eq 0 ]]; then
  set -- bash infra/bootstrap_gpu_box.sh
fi

exec "$@"
