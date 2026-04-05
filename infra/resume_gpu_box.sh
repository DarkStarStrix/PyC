#!/usr/bin/env bash
set -euo pipefail

# Guarded, resumable setup for Ubuntu GPU boxes.
# Keeps step markers and log files so tmux sessions can be resumed safely.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STATE_DIR="${PYC_STATE_DIR:-${HOME}/.pyc_gpu_state}"
LOG_DIR="${PYC_LOG_DIR:-${HOME}/.pyc_gpu_logs}"
STATUS_FILE="${STATE_DIR}/status.txt"
LAST_EXIT_FILE="${STATE_DIR}/last_exit.txt"
RUN_LOG="${LOG_DIR}/resume_$(date +%Y%m%d_%H%M%S).log"

INSTALL_BASE_DEPS="${INSTALL_BASE_DEPS:-1}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
RUN_DOCTOR="${RUN_DOCTOR:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

mkdir -p "${STATE_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

write_status() {
  printf "%s\n" "$1" > "${STATUS_FILE}"
}

step_done() {
  [[ -f "${STATE_DIR}/$1.done" ]]
}

mark_done() {
  touch "${STATE_DIR}/$1.done"
}

run_step() {
  local name="$1"
  shift
  if step_done "${name}"; then
    echo "[resume] skip ${name}"
    return 0
  fi
  echo "[resume] run ${name}"
  "$@"
  mark_done "${name}"
}

finish() {
  local rc="$1"
  printf "%s\n" "${rc}" > "${LAST_EXIT_FILE}"
  if [[ "${rc}" -eq 0 ]]; then
    write_status "done"
    echo "[resume] complete"
  else
    write_status "failed"
    echo "[resume][ERROR] exit=${rc}"
  fi
}

trap 'rc=$?; finish "${rc}"' EXIT

write_status "running"
echo "[resume] repo=${ROOT}"
echo "[resume] state=${STATE_DIR}"
echo "[resume] log=${RUN_LOG}"

run_step system_info bash -lc 'uname -a; lsb_release -a || true; nvidia-smi || true'

if [[ "${INSTALL_BASE_DEPS}" == "1" ]]; then
  run_step base_deps bash -lc "${SUDO} apt-get update -y && ${SUDO} apt-get install -y python3 python3-venv python3-pip build-essential cmake ninja-build git jq curl ca-certificates pkg-config pciutils lsb-release openmpi-bin libopenmpi-dev wget"
fi

if [[ "${INSTALL_CUDA_TOOLKIT}" == "1" ]]; then
  if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then
    echo "[resume] skip cuda_toolkit (nvcc present)"
    mark_done cuda_toolkit
  else
    run_step cuda_toolkit bash -lc "cd /tmp && rm -f cuda-keyring_1.1-1_all.deb && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && ${SUDO} dpkg -i cuda-keyring_1.1-1_all.deb && ${SUDO} apt-get update -y && ${SUDO} apt-get install -y cuda-toolkit"
  fi

  run_step cuda_env bash -lc 'if ! grep -q "/usr/local/cuda/bin" "${HOME}/.bashrc"; then printf "\nexport PATH=/usr/local/cuda/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}\n" >> "${HOME}/.bashrc"; fi'
  export PATH="/usr/local/cuda/bin:${PATH}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

run_step venv_create bash -lc 'if [[ ! -d .venv ]]; then python3 -m venv .venv; fi'
run_step pip_upgrade bash -lc 'source .venv/bin/activate && python -m pip install --upgrade pip'

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  if .venv/bin/python - <<'PY'
import importlib
import sys
for name in ("torch", "torchvision"):
    try:
        importlib.import_module(name)
    except Exception:
        sys.exit(1)
sys.exit(0)
PY
  then
    echo "[resume] skip torch_install (torch + torchvision import)"
    mark_done torch_install
  else
    run_step torch_install bash -lc "source .venv/bin/activate && pip install --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}"
fi

run_step tqdm_install bash -lc 'source .venv/bin/activate && python -c "import tqdm" >/dev/null 2>&1 || (source .venv/bin/activate && pip install tqdm)'
fi

if [[ "${RUN_DOCTOR}" == "1" ]]; then
  run_step kernel_doctor bash -lc 'source .venv/bin/activate && export PATH=/usr/local/cuda/bin:$PATH && python3 kernels/lab/kernel_lab.py doctor'
fi
