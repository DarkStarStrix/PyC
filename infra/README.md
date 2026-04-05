# Infra Bootstrap (GPU Boxes)

One-shot setup for rented GPU hosts (Ubuntu-focused) to install dependencies, verify GPU/runtime, then build/test/benchmark distributed comm.

## Quick start

```bash
bash infra/bootstrap_gpu_box.sh
```

Prebuild the common toolchain image for faster VM bring-up:

```bash
bash infra/build_bootstrap_image.sh
```

Run the bootstrap from that image on a GPU VM:

```bash
INSTALL_SYSTEM_DEPS=0 bash infra/run_bootstrap_image.sh
```

If the host session exits mid-setup, resume from the current machine state instead of restarting:

```bash
bash infra/resume_gpu_box.sh
```

## Scripts

- `infra/bootstrap_gpu_box.sh`: system install + venv + verify + build pipeline.
- `infra/resume_gpu_box.sh`: guarded, resumable host bootstrap with step markers and persistent logs.
- `infra/build_bootstrap_image.sh`: builds the reusable Ubuntu bootstrap image.
- `infra/run_bootstrap_image.sh`: runs the bootstrap image against the checked-out repo on a VM.
- `infra/verify_gpu_env.sh`: checks GPU runtime/tools/libs/Python environment.
- `infra/build_distributed_module.sh`: configures, builds, runs compiler-next tests, runs distributed benchmark suite.
- `infra/run_nexa_insight.sh`: launch the local SSH-backed Nexa Insight TUI.
- `infra/run_nexa_insight_tui.sh`: launch the same local SSH-backed TUI explicitly.
- `infra/run_nexa_insight_local_tui.sh`: direct entrypoint for the same observer.

## Useful env vars

- `VERIFY_STRICT=1`: fail if GPU runtime or `mpirun` is missing.
- `RUN_BUILD=0`: skip build/test/bench stage during bootstrap.
- `RUN_TESTS=0`: skip `ctest` in build script.
- `RUN_BENCH=0`: skip distributed benchmark run.
- `PYC_BUILD_DIR=build-distributed`: choose build directory.
- `DIST_ITERS`, `DIST_COUNT`, `DIST_REPEATS`, `DIST_TAG`: benchmark settings.
- `INSTALL_TORCH=0`: skip torch install in bootstrap.
- `INSTALL_SYSTEM_DEPS=0`: skip apt install when running inside the prebuilt bootstrap image.
- `IMAGE_TAG=pyc/bootstrap-gpu:latest`: override the bootstrap image tag.
- `GPU_FLAG='--gpus all'`: override the Docker GPU runtime flags.

## Nexa Insight

Run from repo root:

```bash
bash infra/run_nexa_insight.sh --refresh 1
```

Dependencies:

```bash
source .venv/bin/activate
pip install textual nvidia-ml-py
```

Run from your local machine to observe the remote box outside tmux:

```bash
bash infra/run_nexa_insight_local_tui.sh
```

This starts a local Textual TUI that polls the remote host over SSH and renders the latest task or benchmark progress, GPU telemetry, active processes, windows, and pane tail:

- live `nvidia-smi` GPU telemetry
- active compute processes
- current `pyc-ada` tmux windows
- a live tail of the active tmux pane
- structured benchmark progress from `latest_ada_fp32_gemm.progress.json`
- recent benchmark completions for quick judgment

For direct, human-readable PyC bench runs in the tmux pane, prefer `scripts/run_pyc_bench_pretty.sh`. It keeps the full JSON artifact but prints compact summary lines instead of flooding the terminal.
