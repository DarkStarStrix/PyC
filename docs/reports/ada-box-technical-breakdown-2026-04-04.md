# Ada Box Technical Breakdown

Date: `2026-04-04`

## Scope

This report covers the Ada GPU work completed on the RTX 6000 Ada box:

- compiler-next runtime feature buildout to an experimentally real state,
- FP32 PyC CUDA benchmarking against PyTorch on a standardized sweep,
- mixed-shape phantom/speculative/runtime-pressure validation,
- custom Ada FP32 kernel prototyping and promotion through the runtime kernel-registry path,
- final artifact sync and shutdown.

The synced remote artifact bundle is:

- `benchmark/benchmarks/results/remote_results/hosts/host0356_kci2_ty6k_prxmx100056/final_sync/20260405T034611Z`

## Environment

- Host: `0356-kci2-ty6k-prxmx100056`
- GPU: `NVIDIA RTX 6000 Ada Generation`
- Target ISA: `sm_89`
- Primary runtime surfaces:
  - `src/compiler/compiler_api.c`
  - `src/compiler/runtime/cuda_backend.c`
  - `src/compiler/runtime/kernel_registry.c`
- Benchmark surfaces:
  - `benchmark/benchmarks/gpu/workloads/pyc_compiler_next_bench.c`
  - `benchmark/benchmarks/gpu/run_gemm_suite.py`
  - `benchmark/tools/analyze_ada_gemm_results.py`
- Kernel prototype surface:
  - `kernels/prototypes/ada/gemm_k64_warp32_async/kernel.cu`

## What Was Built

The repo ended this pass with the following compiler-next/runtime features in a working experimental state:

- deterministic run-boundary guards,
- in-memory compile cache,
- speculative plan variants,
- shadow-first phantom graph tracking,
- budget-sensitive rematerialization telemetry,
- runtime controller rails,
- kernel + allocator co-selection,
- CUDA fast-path execution with a cuBLASLt-first FP32 GEMM path,
- promoted custom GEMM execution through the kernel registry path.

The buildout state is reflected in:

- `docs/plans/compiler-next-feature-buildout.md`
- `docs/compiler-next/overview.md`
- `docs/compiler-next/runtime-integration-spec.md`
- `docs/compiler-next/phantom-graph-rationale.md`

## Measurement Surfaces

There were two important measurement surfaces:

1. Standardized benchmark JSON
- produced by `pyc_compiler_next_bench` and the GPU suite
- comparable across `torch_eager`, `torch_compile`, and `pyc`
- used for runtime-path decisions

2. Standalone kernel harness timing
- produced by the prototype `.cu` harness
- now reports both CUDA-event timing and host-wall timing
- used for kernel-family iteration

Important rule:

- do not compare standalone CUDA-event timing directly against full runtime wall time.
- compare standalone wall time against promoted runtime wall time.

That measurement correction turned out to be critical near the end.

## Runtime Path Progression

### 1. FP32 comparable sweep before the fast path landed

From the FP32 comparable sweep v3:

- source: `benchmark/benchmarks/results/remote_results/hosts/host0356_kci2_ty6k_prxmx100056/runs/20260404T201555Z/ada-sm89-fp32-comparable-pyc-sweep-v3/20260404T201555Z__ada-sm89-fp32-comparable-pyc-sweep-v3.json`
- analysis: `benchmark/benchmarks/results/analysis/ada/20260404T201555Z/ada-sm89-fp32-comparable-pyc-sweep-v3/sheets/analysis.md`

Key `square-1024` result:

| adapter | TFLOPS |
|---|---:|
| `pyc` | `2.4965` |
| `torch_eager` | `24.5344` |
| `torch_compile` | `24.5956` |

At this point the runtime path was correct but far too slow.

Relevant graph bundle:

- `benchmark/benchmarks/results/analysis/ada/20260404T201555Z/ada-sm89-fp32-comparable-pyc-sweep-v3/graphs/adapter_summary.svg`
- `benchmark/benchmarks/results/analysis/ada/20260404T201555Z/ada-sm89-fp32-comparable-pyc-sweep-v3/graphs/square_1024_focus.svg`

![Pre-fastpath adapter summary](../../benchmark/benchmarks/results/analysis/ada/20260404T201555Z/ada-sm89-fp32-comparable-pyc-sweep-v3/graphs/adapter_summary.svg)

![Pre-fastpath square-1024 focus](../../benchmark/benchmarks/results/analysis/ada/20260404T201555Z/ada-sm89-fp32-comparable-pyc-sweep-v3/graphs/square_1024_focus.svg)

### 2. cuBLASLt-first runtime path

The real throughput breakthrough came from changing the runtime GEMM path, not the compiler control logic.

From the cuBLASLt sweep:

- source: `benchmark/benchmarks/results/remote_results/hosts/host0356_kci2_ty6k_prxmx100056/runs/20260404T213005Z/20260404T213005Z__ada-sm89-fp32-comparable-pyc-sweep-cublaslt.json`
- analysis: `benchmark/benchmarks/results/analysis/ada/20260404T213005Z/ada-sm89-fp32-comparable-pyc-sweep-cublaslt/sheets/analysis.md`

Key results:

| shape | PyC TFLOPS | torch_eager TFLOPS | torch_compile TFLOPS |
|---|---:|---:|---:|
| `square-1024` | `28.3142` | `24.5726` | `24.4594` |
| `square-2048` | `47.3470` | `44.1649` | `44.2265` |

PyC won every shape in that FP32 sweep.

Relevant graph bundle:

- `benchmark/benchmarks/results/analysis/ada/20260404T213005Z/ada-sm89-fp32-comparable-pyc-sweep-cublaslt/graphs/throughput_winners.svg`
- `benchmark/benchmarks/results/analysis/ada/20260404T213005Z/ada-sm89-fp32-comparable-pyc-sweep-cublaslt/graphs/square_1024_focus.svg`
- `benchmark/benchmarks/results/analysis/ada/20260404T213005Z/ada-sm89-fp32-comparable-pyc-sweep-cublaslt/graphs/adapter_summary.svg`

![Post-cuBLASLt throughput winners](../../benchmark/benchmarks/results/analysis/ada/20260404T213005Z/ada-sm89-fp32-comparable-pyc-sweep-cublaslt/graphs/throughput_winners.svg)

![Post-cuBLASLt adapter summary](../../benchmark/benchmarks/results/analysis/ada/20260404T213005Z/ada-sm89-fp32-comparable-pyc-sweep-cublaslt/graphs/adapter_summary.svg)

![Post-cuBLASLt square-1024 focus](../../benchmark/benchmarks/results/analysis/ada/20260404T213005Z/ada-sm89-fp32-comparable-pyc-sweep-cublaslt/graphs/square_1024_focus.svg)

### 3. Prod-like FP32 sweeps

The final prod-like fixed-shape sweeps were:

- baseline: `20260404T234632Z__ada-sm89-prod-fp32-baseline.json`
- utilization shadow: `20260404T234759Z__ada-sm89-prod-fp32-util-shadow.json`

Baseline PyC vs PyTorch:

| shape | PyC | torch_eager | torch_compile |
|---|---:|---:|---:|
| `square-256` | `1.5100` | `0.8955` | `0.8955` |
| `square-512` | `9.1135` | `7.1252` | `7.1623` |
| `square-1024` | `26.8108` | `24.5663` | `24.6909` |
| `square-2048` | `46.6286` | `44.4861` | `44.2794` |
| `tall-skinny-4096x1024x4096` | `32.4172` | `30.1473` | `31.2735` |
| `wide-skinny-1024x4096x1024` | `35.2024` | `33.4815` | `33.5194` |

Utilization-shadow PyC vs PyTorch:

| shape | PyC | torch_eager | torch_compile |
|---|---:|---:|---:|
| `square-256` | `1.4729` | `0.9249` | `0.9755` |
| `square-512` | `9.3526` | `7.3058` | `7.3640` |
| `square-1024` | `26.5158` | `24.6517` | `24.4583` |
| `square-2048` | `46.6837` | `44.1834` | `43.7230` |
| `tall-skinny-4096x1024x4096` | `40.4725` | `36.9825` | `30.5954` |
| `wide-skinny-1024x4096x1024` | `35.0207` | `33.4180` | `33.3280` |

Read:

- the utilization-biased shadow policy did not dominate universally,
- but it materially helped the tall-skinny case,
- the fixed-shape runtime path is now strong enough that the remaining work is shape-aware policy and integration detail, not emergency throughput repair.

## Mixed-Shape Runtime Behavior

The most realistic control-path run was:

- `benchmark/benchmarks/results/remote_results/hosts/host0356_kci2_ty6k_prxmx100056/final_sync/20260405T034611Z/benchmark_results/json/20260404T234926Z__ada-sm89-prod-mixed-pressure.json`

Sequence:

- `512 -> 1024 -> 2048 -> 1024 -> 512`

Summary:

- `phantom_match_count = 116`
- `phantom_mismatch_count = 4`
- `phantom_reshape_count = 4`
- `final_confidence = 0.9928`
- `guard_miss_count = 0`
- `fallback_count = 0`
- rematerialization stayed at `0` on all five steps in this run

Per-step readout:

| step | shape | TFLOPS | mismatch_delta | reshape_delta |
|---|---|---:|---:|---:|
| `0` | `512x512x512` | `9.0607` | `0` | `0` |
| `1` | `1024x1024x1024` | `26.3988` | `1` | `1` |
| `2` | `2048x2048x2048` | `46.5525` | `1` | `1` |
| `3` | `1024x1024x1024` | `21.4565` | `1` | `1` |
| `4` | `512x512x512` | `9.0982` | `1` | `1` |

Read:

- phantom graph adaptation behaved correctly under drift,
- speculative-plan hits stayed live,
- there were no runtime failures,
- the return trip to `1024` was measurably worse than the first `1024` step, which is a good future target for controller and replay stability work.

## Custom Kernel Progression

The important kernel-family progression was:

- `ada_gemm`: validated FP32 shared-memory baseline
- `ada_gemm_k64_warp32_store2`: improved baseline family
- `ada_gemm_k64_warp32_async`: double-buffered async/shared-memory winner

Current winning prototype:

- source: `kernels/prototypes/ada/gemm_k64_warp32_async/kernel.cu`
- shape: `1024 x 1024 x 1024`
- geometry: `64 x 64 x 64` tile, `32 x 8` threads, `8 x 2` thread tile

Measured results:

- standalone event-timed: `30.394 TFLOPS`, `best_ms=0.071`
- standalone host-wall: `0.076 ms`
- `nsys` kernel time improved from `77.72 us` on the old winner to `68.41 us` on the async winner

Important shape behavior:

- strong on larger steady-state GEMMs,
- notably weak on `512^3`,
- therefore this is a strong large-GEMM kernel family, not a universal “best for every shape” kernel.

## Runtime Promotion Through The Registry Path

The promoted symbol path now works end-to-end:

- runtime source: `src/compiler/runtime/cuda_backend.c`
- hot-path model logic: `src/compiler/compiler_api.c`
- promoted kernel registration path: `src/compiler/cutlass_kernels/gemm/dispatch_registry.cu`

Final promoted runtime status:

- `execution_path = cuda_promoted_gemm_graph:ada_gemm_k64_warp32_async_f32`
- `mean wall ms = 0.0754`
- `28.4903 TFLOPS`

Critical lesson:

- the earlier “standalone 33.8 vs promoted 23.6” story was mostly a measurement mismatch,
- because the standalone harness was using CUDA-event timing while the runtime path was measured with full wall-clock run time,
- once the standalone harness also reported wall-clock and the runtime fastpath was tightened, the promoted path matched and slightly edged the standalone wall-clock surface.

The last direct comparison artifacts synced from the box are:

- `benchmark/benchmarks/results/remote_results/hosts/host0356_kci2_ty6k_prxmx100056/final_sync/20260405T034611Z/tmp_logs/pyc_async_compare.log`
- `benchmark/benchmarks/results/remote_results/hosts/host0356_kci2_ty6k_prxmx100056/final_sync/20260405T034611Z/tmp_logs/pyc_promoted_compare.json`

## Code-Level End State

The highest-signal code surfaces at the end of the session were:

- `src/compiler/runtime/cuda_backend.c`
  - cuBLASLt-first FP32 path
  - promoted-kernel graph replay path
  - CUDA dispatch trace with copy/kernel/sync timing
- `src/compiler/compiler_api.c`
  - stable-repeat fastpath
  - runtime controller observation
  - phantom/speculative bookkeeping
  - exact tensor-descriptor cache for repeated identical buffers
- `benchmark/benchmarks/gpu/workloads/pyc_compiler_next_bench.c`
  - direct controller/execution-path telemetry
  - standardized JSON emission for runtime and reliability fields
- `kernels/prototypes/ada/gemm_k64_warp32_async/kernel.cu`
  - final async prototype kernel
  - event timing plus host-wall timing in one harness

## Artifact Inventory

Final synced artifact counts:

- benchmark `json`: `291`
- kernel-lab `results`: `30`
- kernel-lab `runs`: `5`
- comparison logs: `2`

Primary sync root:

- `benchmark/benchmarks/results/remote_results/hosts/host0356_kci2_ty6k_prxmx100056/final_sync/20260405T034611Z`

## Final Technical Readout

1. The runtime feature stack is no longer a paper design. `A2`, `A3`, and `B2` are experimentally real, tested, and exercised on the GPU hot path.
2. The biggest runtime throughput gain came from changing the CUDA GEMM execution path, not from controller logic.
3. The custom async Ada FP32 kernel is a real win and a legitimate promotion target.
4. The promoted runtime path now preserves that kernel’s wall-clock performance well enough that the “runtime integration gap” is basically closed for `1024^3`.
5. Small-shape behavior remains weaker than large-shape behavior, so future kernel work should be shape-aware rather than trying to force one kernel to dominate every surface.
6. Phantom/speculative/controller work is justified primarily by control stability and runtime robustness, not by raw FLOPS. The mixed-shape results support that framing.
7. The repo is in a good experimental state, but future work should bias toward convergence and cleanup rather than adding more feature surface area blindly.

## Recommended Next Move

If work resumes from this point, the best next order is:

1. add a small-shape GEMM lane or fallback policy for the `512` class,
2. promote the async kernel path more broadly where the shape family is large enough to benefit,
3. keep controller/phantom work focused on stability under shape drift,
4. only then go deeper into full-kernel expansion beyond the current async family.
