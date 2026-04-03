#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="${ROOT}/infra/nexa_insight"

cd "${APP_DIR}"

if ! command -v go >/dev/null 2>&1; then
  echo "[nexa-insight-tui][ERROR] go is required (or copy prebuilt binary to infra/nexa_insight_tui/nexa-insight-tui)." >&2
  exit 1
fi

echo "[nexa-insight-tui] building"
go build -o nexa-insight-tui ./cmd/nexa-insight-tui

echo "[nexa-insight-tui] starting"
exec ./nexa-insight-tui "$@"
