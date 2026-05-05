#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-pyc-hopper}"
WINDOW_NAME="${WINDOW_NAME:-main}"
TMUX_DEFAULT_CMD="${TMUX_DEFAULT_CMD:-bash}"
TMUX_KEEPALIVE_INTERVAL_SEC="${TMUX_KEEPALIVE_INTERVAL_SEC:-15}"

ensure_tmux_session() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[tmux-keepalive][ERROR] tmux is not installed" >&2
    return 1
  fi

  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}" "${TMUX_DEFAULT_CMD}"
    return 0
  fi

  if ! tmux list-windows -t "${SESSION_NAME}" -F "#{window_name}" | grep -qx "${WINDOW_NAME}"; then
    tmux new-window -d -t "${SESSION_NAME}" -n "${WINDOW_NAME}" "${TMUX_DEFAULT_CMD}"
  fi
}

trap 'exit 0' INT TERM

ensure_tmux_session

while true; do
  ensure_tmux_session
  sleep "${TMUX_KEEPALIVE_INTERVAL_SEC}"
done
