#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT}/infra/nexa_insight"

cd "${APP_DIR}"

if ! command -v go >/dev/null 2>&1; then
  echo "[nexa-insight][ERROR] go is required" >&2
  exit 1
fi

echo "[nexa-insight] building"
go build -o nexa-insight ./cmd/nexa-insight

echo "[nexa-insight] starting"
exec ./nexa-insight "$@"
