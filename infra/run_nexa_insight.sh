#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT}/infra/nexa_insight/local_tui.py"

PY_BIN="${ROOT}/.venv-observer/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="${ROOT}/.venv/bin/python"
fi
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PY_BIN}" ]]; then
  echo "[nexa-insight][ERROR] python3 is required" >&2
  exit 1
fi

echo "[nexa-insight] starting"
exec "${PY_BIN}" "${APP}" "$@"
