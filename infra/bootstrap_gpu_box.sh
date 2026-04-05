#!/usr/bin/env bash
set -euo pipefail

# One-shot bootstrap for rented Ubuntu GPU boxes.
# Installs build + benchmark deps, sets up virtualenv, verifies GPU env,
# and optionally builds/tests/benchmarks distributed comm module.
#
# Usage:
#   bash infra/bootstrap_gpu_box.sh
# Optional env:
#   INSTALL_SYSTEM_DEPS=1     # default: 1
#   INSTALL_TORCH=1           # default: 1
#   RUN_BUILD=1               # default: 1
#   VERIFY_STRICT=0           # default: 0
#   PYC_BUILD_DIR=build-distributed

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

export DEBIAN_FRONTEND=noninteractive

if [[ "${INSTALL_SYSTEM_DEPS:-1}" == "1" ]]; then
  echo "[bootstrap] installing system dependencies..."
  ${SUDO} apt-get update -y
  ${SUDO} apt-get install -y \
    python3 python3-venv python3-pip \
    build-essential cmake ninja-build git jq \
    curl ca-certificates pkg-config \
    pciutils lsb-release \
    openmpi-bin libopenmpi-dev
else
  echo "[bootstrap] skipping system dependency install (INSTALL_SYSTEM_DEPS=0)"
fi

if [[ "${INSTALL_TORCH:-1}" == "1" ]]; then
  echo "[bootstrap] setting up Python venv + torch..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
  pip install tqdm
else
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
fi

echo "[bootstrap] verifying environment..."
bash infra/verify_gpu_env.sh

if [[ "${RUN_BUILD:-1}" == "1" ]]; then
  echo "[bootstrap] building + testing + benchmarking distributed module..."
  bash infra/build_distributed_module.sh
fi

echo "[bootstrap] done."
