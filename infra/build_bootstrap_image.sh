#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-pyc/bootstrap-gpu:latest}"
DOCKERFILE="${ROOT}/infra/docker/bootstrap-gpu/Dockerfile"

echo "[docker-build] root=${ROOT}"
echo "[docker-build] tag=${IMAGE_TAG}"

docker build \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_TAG}" \
  "${ROOT}"

echo "[docker-build] done"
