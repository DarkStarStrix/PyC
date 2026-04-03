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

## Scripts

- `infra/bootstrap_gpu_box.sh`: system install + venv + verify + build pipeline.
- `infra/build_bootstrap_image.sh`: builds the reusable Ubuntu bootstrap image.
- `infra/run_bootstrap_image.sh`: runs the bootstrap image against the checked-out repo on a VM.
- `infra/verify_gpu_env.sh`: checks GPU runtime/tools/libs/Python environment.
- `infra/build_distributed_module.sh`: configures, builds, runs compiler-next tests, runs distributed benchmark suite.
- `infra/run_nexa_insight.sh`: build and run Nexa Insight telemetry TUI.
- `infra/run_nexa_insight_tui.sh`: run the Bubble Tea cyberpunk TUI with ML metrics.

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

## Nexa Insight (single-node telemetry TUI)

Run from repo root:

```bash
bash infra/run_nexa_insight.sh --refresh 1s --top 20
```

Optional snapshot export:

```bash
bash infra/run_nexa_insight.sh \
  --json-out benchmark/remote_results/runpod_h100_8x/insight/live.ndjson
```

## Nexa Insight TUI (Bubble Tea)

Run from repo root:

```bash
bash infra/run_nexa_insight_tui.sh --refresh 1s \
  --runs-root benchmark/remote_results/runpod_h100_8x/campaign_v4
```
