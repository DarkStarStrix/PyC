# Kernel Optimization Playbook

## Priority Order

1. GEMM + fused epilogues
2. Reductions + LayerNorm
3. Conv2D family

## Registry Interface

Defined in `include/pyc/kernel_registry.h`:

- register kernels by op key + backend
- select highest-priority compatible kernel
- store benchmark signals per op/backend

## Experiment Loop

1. Generate candidate configuration.
2. Validate numerical correctness.
3. Benchmark with fixed protocol.
4. Persist best candidate.
5. Route runtime selection by op key + backend.

## Benchmark Signals

- latency (p50/p95)
- throughput
- memory footprint
- stability (variance)

## Guardrail

No candidate is promoted unless it is both correct and faster than baseline on target workload.
