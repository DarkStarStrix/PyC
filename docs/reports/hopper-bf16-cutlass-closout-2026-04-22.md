# Hopper BF16 CUTLASS Closeout

Date: `2026-04-22`

## Scope

This closeout covers the Hopper BF16 bring-up loop that ended with:

- BF16 execution unblocked on the Hopper box,
- CUTLASS GEMM integrated as the working tensor-core lane for BF16,
- better runtime failure observability in the benchmark path,
- a clean separation between the owned WMMA guardrail and the stronger native reference lanes,
- remote artifact sync and box shutdown.

Primary result:

- PyC BF16 now completes through the CUTLASS tensor-core lane on Hopper for the control shape instead of failing through the old native path.

## Environment

- Host: `31.22.104.38`
- Remote repo: `/root/work/PyC/repo`
- SSH key used: `~/.ssh/prime_next`
- Primary tmux session: `pyc-hopper:main`
- Target device class: Hopper / `sm_90`

Relevant local surfaces:

- `src/compiler/runtime/cuda_backend.c`
- `src/compiler/cutlass_kernels/CMakeLists.txt`
- `src/compiler/cutlass_kernels/gemm/kernel.cu`
- `src/compiler/cutlass_kernels/registry/init.cu`
- `benchmark/benchmarks/gpu/workloads/pyc_compiler_next_bench.c`

## Starting Failure State

The first Hopper BF16 main-loop analysis showed that PyC was failing all BF16 shapes even while native comparison lanes were healthy.

Analysis artifact:

- `benchmark/benchmarks/results/analysis/hopper/20260422T014002Z/hopper_bf16_main_loop/sheets/analysis.md`

Summary from that run:

- all `pyc` BF16 adapter attempts failed,
- native references such as CUTLASS profiler, TensorRT, Torch, and TVM produced valid numbers,
- the runtime path was the blocker, not the box or the benchmark harness.

The direct failure chain was captured in the diagnostic trace:

- `benchmark/benchmarks/results/optimizer_diag/hopper_bf16_cutlass_trace_20260422T021606Z.log`

Observed reason:

- `cutlass_gemm_failed:cutlass_gemm_tensorcore_bf16;cuda_gemm_failed:not_supported;cpu_fallback_failed`

Interpretation:

- the legacy native CUDA GEMM path did not support this BF16 case,
- the fallback path then failed because the CPU executor only supports `F32`,
- CUTLASS existed conceptually in the repo but the remote build was still using the minimal registry, so the BF16 CUTLASS symbol was not actually available in practice.

## What Changed

### 1. Runtime failure reporting was made explicit

`benchmark/benchmarks/gpu/workloads/pyc_compiler_next_bench.c` now emits structured failure information instead of collapsing failures into a vague bench-level error.

Added fields:

- `pyc_status`
- `pyc_status_code`
- `execution_path`
- `decision_log`
- shape metadata on failures

This made it possible to distinguish:

- native CUDA unsupported failures,
- CUTLASS dispatch failures,
- CPU fallback failures,
- compile/setup failures.

The bench path also now zero-initializes the runtime result state before execution.

### 2. BF16 runtime routing was repaired

`src/compiler/runtime/cuda_backend.c` was updated so BF16 routing preserves detailed failure reasons and selects the BF16 CUTLASS matmul symbol when available.

Important effects:

- native failure reasons are preserved instead of being overwritten,
- combined failure chains remain visible,
- BF16 can route into the CUTLASS tensor-core lane instead of dying immediately on the old native CUDA lane.

### 3. CUTLASS BF16 GEMM kernel dispatch was corrected

`src/compiler/cutlass_kernels/gemm/kernel.cu` was updated to use a row-major BF16 GEMM path that matches the runtime’s actual operand layout expectations.

This removed a bad disconnect between:

- what the runtime requested,
- what the BF16 CUTLASS wrapper instantiated,
- and what the registry could validly dispatch.

### 4. Full CUTLASS build was made usable in practice

The Hopper box originally built against the minimal registry because `PYC_CUTLASS_PATH` was unset and the build never exposed the full BF16 GEMM lane.

On the remote box:

- CUTLASS was cloned to `/root/work/cutlass`,
- the repo was reconfigured with `PYC_CUTLASS_PATH=/root/work/cutlass/include`.

Then the local repo was patched so full CUTLASS defaults to GEMM-only instead of also forcing experimental conv and attention kernels that were not clean against the remote CUTLASS checkout.

Files changed:

- `src/compiler/cutlass_kernels/CMakeLists.txt`
- `src/compiler/cutlass_kernels/registry/init.cu`

New behavior:

- full CUTLASS mode builds GEMM by default,
- conv and attention are optional behind `PYC_CUTLASS_ENABLE_EXTRA_KERNELS`,
- the registry init log now makes it clear whether the build is minimal, GEMM-only full CUTLASS, or full CUTLASS with extra kernels.

This was the build-system fix that turned the BF16 runtime route from theoretical to real.

## Validation

### Local validation

The repo changes were validated locally with:

```bash
cmake --build build --parallel --target pyc_compiler_next_bench
ctest --test-dir build --output-on-failure -R pyc_compiler_next_test_cuda_backend
```

These passed after the runtime and bench changes landed.

### Remote validation

The decisive validation happened on the Hopper box after syncing the runtime and CUTLASS changes and rebuilding with the real CUTLASS include path.

Successful BF16 diagnostic artifact:

- `benchmark/benchmarks/results/optimizer_diag/hopper_bf16_full_cutlass_20260422T022134Z.log`

Key result from that run:

- `status: ok`
- `shape: 512x512x512`
- `throughput_tflops_per_sec: 16.1812`
- `execution_path: cuda_cutlass_bf16_graph:cutlass_gemm_tensorcore_bf16`

Meaning:

- BF16 execution is no longer falling out of the runtime,
- the Hopper runtime can now execute the control BF16 path through the intended CUTLASS tensor-core lane,
- the next bottleneck is competitive performance and shape coverage, not basic correctness or dispatch failure.

## Performance Readout

There are now three clearly separate Hopper lanes:

1. Native best-case references
- CUTLASS profiler / vendor-grade native paths remain the throughput ceiling.

2. Owned WMMA guardrail
- documented in `docs/plans/hopper-gap-close-path.md`
- useful as an owned baseline and for controlled iteration
- still materially behind top Hopper-native tensor-core lanes

3. Runtime-integrated PyC BF16 CUTLASS path
- now valid and working
- currently proven on the control BF16 case
- ready for broader shape validation and routing policy work

Important boundary:

- this loop solved the BF16 execution gap,
- it did not solve the full Hopper competitiveness gap.

That is the correct outcome for this pass. The system moved from failing BF16 dispatch to successfully executing BF16 on the intended tensor-core route.

## What Was Learned

- The BF16 failure was mostly a systems integration problem, not a pure kernel problem.
- The runtime needed better failure visibility before the actual blocking layer could be identified cleanly.
- Full CUTLASS cannot be treated as “enabled” just because source exists in the tree; the registry and build path must expose the exact symbol the runtime needs.
- Experimental extra kernels in the CUTLASS build were contaminating the bring-up path. Making the full build GEMM-first was the right control measure.
- The owned WMMA lane and the runtime CUTLASS lane solve different problems and should not be conflated.

## Remaining Risks

- BF16 coverage is still narrow relative to the full Hopper benchmark matrix.
- There is still a warning in `src/compiler/runtime/cuda_backend.c` around possible `snprintf` truncation when appending `cpu_fallback_failed` to a long native reason string.
- Full CUTLASS on remote machines still depends on a valid external checkout and `PYC_CUTLASS_PATH` being configured correctly.
- Competitive Hopper performance still requires broader shape routing, better ops-layer reuse, and eventually a stronger Hopper-native mainloop where that effort is justified.

## Recommended Next Loop

1. Expand BF16 runtime validation beyond the control shape.
2. Keep the owned WMMA guardrail as the regression baseline instead of reopening old variants.
3. Use the runtime-integrated CUTLASS BF16 lane for honest system-level comparisons.
4. Push broader Hopper work into two separate tracks:
   - ops-layer overhead removal and replay stability,
   - shape-aware high-throughput kernel routing.

Do not reopen the old failure chain unless BF16 regressions reappear. The repo now has a working BF16 tensor-core execution lane on Hopper, and future work should build from that stable floor.

## Artifacts

Primary analysis and logs referenced in this closeout:

- `benchmark/benchmarks/results/analysis/hopper/20260422T014002Z/hopper_bf16_main_loop/sheets/analysis.md`
- `benchmark/benchmarks/results/optimizer_diag/hopper_bf16_cutlass_trace_20260422T021606Z.log`
- `benchmark/benchmarks/results/optimizer_diag/hopper_bf16_full_cutlass_20260422T022134Z.log`
- `docs/plans/hopper-gap-close-path.md`

## Shutdown State

After sync and validation, the Hopper box was shut down cleanly:

```bash
ssh -i ~/.ssh/prime_next -o StrictHostKeyChecking=no root@31.22.104.38 'shutdown -h now'
```

Final operational state:

- artifacts synced locally,
- code changes preserved in the repo,
- remote box powered down.
