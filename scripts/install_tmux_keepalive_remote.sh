#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVICE_NAME="${SERVICE_NAME:-pyc-tmux-keepalive.service}"
SERVICE_TEMPLATE="${ROOT}/infra/systemd/pyc-tmux-keepalive.service.template"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
REPO_ROOT="${REPO_ROOT:-${ROOT}}"

if [[ ! -f "${SERVICE_TEMPLATE}" ]]; then
  echo "[tmux-keepalive][ERROR] missing service template: ${SERVICE_TEMPLATE}" >&2
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

chmod 0755 "${ROOT}/infra/tmux_keepalive.sh"

tmp_unit="$(mktemp)"
trap 'rm -f "${tmp_unit}"' EXIT

sed "s|__REPO_ROOT__|${REPO_ROOT}|g" "${SERVICE_TEMPLATE}" > "${tmp_unit}"

${SUDO} install -D -m 0644 "${tmp_unit}" "${SERVICE_PATH}"
${SUDO} systemctl daemon-reload
${SUDO} systemctl enable --now "${SERVICE_NAME}"
${SUDO} systemctl restart "${SERVICE_NAME}"

echo "[tmux-keepalive] service=${SERVICE_NAME}"
${SUDO} systemctl --no-pager --full status "${SERVICE_NAME}" | sed -n '1,20p'
echo "[tmux-keepalive] tmux sessions"
for _ in $(seq 1 10); do
  if tmux ls >/dev/null 2>&1; then
    tmux ls
    exit 0
  fi
  sleep 1
done

echo "[tmux-keepalive][ERROR] tmux server did not appear after service start" >&2
exit 1
