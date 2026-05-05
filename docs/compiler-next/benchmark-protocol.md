# Compiler-Next Benchmark Protocol

## Objective

Produce reproducible, decision-grade performance signals for compiler-next components.

## Measurement Rules

1. Use fixed repeats and warm-up policy.
2. Record machine metadata with every run.
3. Compare only like-for-like hardware classes.
4. Report median and spread, not single-run bests.
5. Separate microbenchmark lanes from end-to-end evaluation lanes.
6. Reject proxy adapters in decision-grade arena runs.

## Trust Model

- Hopper GEMM arena runs should use the Hopper arena profile, not the legacy arena.
- Legacy or archived lanes such as Glow must not appear in Hopper arena mode.
- Native-only gates are the default for serious comparisons:
  - framework baselines must execute natively
  - PyC must execute natively
  - optional challenger lanes may be unavailable, but they must not silently downgrade to proxies

## Recommended Harnesses

- Official kernel-space harness:
  - CUTLASS Profiler for dense GEMM sweeps and kernel-space reference comparisons
- Framework/compiler baselines:
  - `torch_eager`
  - `torch_compile`
  - TensorRT, TVM, and XLA only when they report `mode=native`
- System and correctness validation:
  - Nsight Compute / Nsight Systems for profiling
  - MLPerf-style end-to-end evaluation discipline for model-serving claims

## Hopper GEMM Arena

- Use `benchmark/benchmarks/gpu/configs/hopper_bf16_gemm_shapes.json`.
- Shapes are grouped by:
  - launch-sensitive control
  - sustained square throughput
  - transformer-style projection and MLP regimes
- BF16 Hopper shapes are intentionally tensor-core-friendly and avoid odd, naive dimensions that mainly benchmark quantization artifacts instead of execution quality.

## Required Outputs

- JSON artifact for automation
- Markdown summary for human review
- Visualization artifact for quick inspection

## Acceptance Gates

1. Correctness must pass before perf comparison.
2. End-to-end model latency is the primary KPI.
3. Memory reduction and kernel wins are tracked as secondary KPIs.

## CI Integration Path

- Early phases: non-blocking perf jobs.
- Later phases: threshold-based regression gates.

## GPU Execution Path

For real CUDA benchmarking on rented Linux hosts, use:

- `docs/compiler-next/gpu-testing-playbook.md`
- `scripts/setup_cuda_remote_ubuntu.sh`
- `benchmark/benchmarks/gpu/run_gpu_suite.py`
