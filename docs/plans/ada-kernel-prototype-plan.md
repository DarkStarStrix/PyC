# Ada Kernel Prototype Plan

## Current repo map

- `kernels/`: low-friction CUDA prototype area driven by `kernels/lab/kernel_lab.py`
- `src/compiler/runtime/kernel_registry.c`: runtime-side kernel scoring and selection
- `src/compiler/cutlass_kernels/`: promotion target for production CUDA kernels
- `benchmark/benchmarks/gpu/`: standardized GPU measurement harness
- `infra/README.md`: reusable GPU bootstrap image path for faster VM bring-up

## Ada constraints driving this prototype

- Target native `sm_89` code generation and also emit `compute_89` PTX for portability.
- Start with a shared-memory GEMM because it is the cleanest path to shape-aware matmul promotion later.
- Favor high-SM occupancy without immediately depending on CUTLASS or Tensor Core-specific fragments.
- Keep correctness self-contained so the first GPU pass can fail loudly on bad numerics.

## Prototype staged in `kernels/`

- `kernels/prototypes/ada/gemm/kernel.cu`
  - FP32 shared-memory tiled GEMM
  - `64 x 64 x 16` CTA tile
  - `16 x 16` thread block
  - standalone harness for correctness, timing, and GFLOPS reporting
- `kernels/lab/manifests/kernels.json`
  - `ada_gemm` entry with `sm_89` compile flags
  - default run shape: `1024 x 1024 x 1024`
  - Ada Tensor Core FP16/BF16 lane also staged for later VM validation

## Promotion plan after VM access

1. Compile and run `ada_gemm` on real Ada hardware.
2. Sweep shapes and tile sizes to find the first stable winner.
3. Compare against the existing PyC CUDA path and any CUTLASS baselines.
4. If the prototype is competitive, split the winning configuration into:
   - a production kernel registration entry in `src/compiler/cutlass_kernels/` or a new runtime CUDA kernel path
   - a benchmark-backed registry metadata update
5. Wire the chosen symbol into `kernel_registry` with measured occupancy/shared-memory values.

## Current winning prototype

- Winner: `ada_gemm_k64_warp32_async`
- Source: `kernels/prototypes/ada/gemm_k64_warp32_async/kernel.cu`
- It preserves the `64 x 64 x 64` / `32 x 8` / `8 x 2` family and replaces the old single-stage shared-memory loop with a double-buffered async/shared-memory path.
- Baseline result on RTX 6000 Ada at `1024 x 1024 x 1024`:
  - `33.825 TFLOPS`
  - `best_ms=0.063`
  - `max_abs_diff=0.000000`

## Current measured comparison

- Prior baseline: `ada_gemm_k64_warp32_store2`
- Prior `nsys` kernel time: `77.72 us`
- Current `nsys` kernel time: `68.41 us`
- Decision: the async path is now the prototype to beat before any further promotion or CUTLASS wiring.

## First VM commands

```bash
bash infra/build_bootstrap_image.sh
INSTALL_SYSTEM_DEPS=0 bash infra/run_bootstrap_image.sh
python3 kernels/lab/kernel_lab.py doctor
python3 kernels/lab/kernel_lab.py compile ada_gemm
python3 kernels/lab/kernel_lab.py run ada_gemm
python3 kernels/lab/kernel_lab.py bench-suite --tag ada --dry-run
python3 benchmark/benchmarks/gpu/run_gpu_suite.py --device cuda --tag ada_proto_baseline
```

## Immediate follow-up experiments

1. Increase `BLOCK_K` to 32 and re-measure.
2. Add vectorized global loads for aligned shapes.
3. Prototype a Tensor Core path for FP16/BF16 once the scalar FP32 baseline is validated.
4. Decide whether productionization belongs in `kernels/` first or directly in `src/compiler/cutlass_kernels/`.
