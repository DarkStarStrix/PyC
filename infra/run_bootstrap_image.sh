#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-pyc/bootstrap-gpu:latest}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
GPU_FLAG="${GPU_FLAG:---gpus all}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
RUN_BUILD="${RUN_BUILD:-1}"
VERIFY_STRICT="${VERIFY_STRICT:-0}"
PYC_BUILD_DIR="${PYC_BUILD_DIR:-build-distributed}"

echo "[docker-run] image=${IMAGE_TAG}"
echo "[docker-run] workdir=${CONTAINER_WORKDIR}"

docker run --rm -it \
  ${GPU_FLAG} \
  -v "${ROOT}:${CONTAINER_WORKDIR}" \
  -w "${CONTAINER_WORKDIR}" \
  -e INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS}" \
  -e INSTALL_TORCH="${INSTALL_TORCH}" \
  -e RUN_BUILD="${RUN_BUILD}" \
  -e VERIFY_STRICT="${VERIFY_STRICT}" \
  -e PYC_BUILD_DIR="${PYC_BUILD_DIR}" \
  "${IMAGE_TAG}" \
  bash infra/bootstrap_gpu_box.sh
