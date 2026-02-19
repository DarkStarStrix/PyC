# R&D Landscape: Classical vs AI Compilers (and Where PyC Can Win)

Last updated: 2026-02-17

## 1) How Classical Compilers Are Usually Built

Classical compiler architecture is typically organized as:

1. Frontend
- lexical analysis
- parsing
- semantic analysis
2. Intermediate representation (IR)
3. Optimization passes
4. Backend code generation

LLVM exemplifies this model with a typed SSA IR and pass pipelines as a central optimization mechanism.

Practical implication for PyC:
- Keep strict phase boundaries and deterministic pass ordering.
- Invest in IR-level invariants/verifiers early.

## 2) How AI Compilers Differ

AI compilers still follow frontend -> IR -> passes -> lowering, but add specialized layers:

1. Graph/tensor semantics and shape propagation.
2. Hardware-specific kernel generation/tuning.
3. Runtime memory planning and buffer management.
4. Runtime interfaces and portability layers (e.g., StableHLO/PJRT ecosystems).

Observed patterns:
- TVM emphasizes schedule search and cost-model-driven program generation.
- XLA/OpenXLA emphasizes graph-level optimization and backend portability (HLO/StableHLO + PJRT).
- MLIR emphasizes multi-level IR composition and reusable rewrite infrastructure.

Practical implication for PyC:
- Your current direction (custom IR + pass manager + kernel registry + allocator) is structurally aligned with successful systems.

## 3) Dynamic Memory Optimization: Why Your Focus Is High-Leverage

Memory planning/rematerialization is a major leverage point in AI workloads, especially under constrained memory and dynamic shapes.

Evidence direction:
- Dynamic Tensor Rematerialization (DTR) shows online rematerialization can closely match strong static methods while supporting dynamic models.
- Recent research and systems work continue to optimize memory-footprint-sensitive compilation and scheduling.

Inference for PyC (from literature + system patterns):
- A runtime-aware planner that supports policy modes and telemetry feedback can be a strong differentiator versus fixed/static planning pipelines.

## 4) Ecosystem Sentiment Snapshot (What People Are Saying)

Sentiment is mixed-positive:

Positive themes:
1. Strong performance gains from compiler optimization/fusion/autotuning are real and repeatedly demonstrated.
2. Interop/portability layers (e.g., StableHLO + PJRT) are seen as increasingly important.

Pain themes:
1. Build complexity and environment brittleness remain common complaints in large compiler stacks.
2. Tuning workflows can be operationally heavy (time, debugging complexity, infra dependencies).
3. Users need stronger debugging/inspection surfaces (HLO/IR dumps, error taxonomies, explainability).

Inference for PyC:
- Your deterministic CI/test rules + explicit runtime diagnostics are not overhead; they are product differentiation.

## 5) Priority Experiments for PyC (Backlog-Driven)

The following are the highest ROI items from your innovation backlog:

1. Adaptive planner modes (`latency_first`, `memory_first`, `balanced`)
- Expected outcome: stronger flexibility with explicit tradeoff control.
- KPI: >=20% memory reduction in `memory_first` and <=5% latency hit in `latency_first`.

2. Reuse + rematerialization policy
- Expected outcome: meaningful memory-footprint reduction under pressure.
- KPI: >=25% peak-memory reduction with <10% latency penalty.

3. Shape-clustered multi-plan execution
- Expected outcome: better p95 on dynamic-shape workloads.
- KPI: >=1.2x p95 latency improvement.

4. Kernel + allocator co-selection
- Expected outcome: better end-to-end behavior than isolated kernel timing decisions.
- KPI: >=10% E2E latency gain on memory-pressure workloads.

## 6) Uniqueness Strategy for PyC

To be both experimental and practically useful, PyC should position itself as:

1. Deterministic-by-default experimental compiler.
2. Memory-policy-driven runtime (not just kernel-speed-driven).
3. Explainable optimization system (policy, decision logs, reproducible outputs).

This is a clear contrast to many larger ecosystems where flexibility and operational clarity are harder for small teams.

## 7) Integration into Current Roadmap

Suggested mapping:

- Phase 3
  - adaptive planner modes
  - rematerialization policy
- Phase 4
  - shape-clustered multi-plan
  - kernel+allocator co-selection
- Phase 5
  - telemetry feedback planner
  - deterministic what-if simulator
  - policy plugins + optimization contracts

This mapping is now reflected in `docs/compiler-next/innovation-backlog.md` and `docs/compiler-next/roadmap-phases.md`.

## 8) Phase 5 Translation (What Was Built and Why)

Phase 5 implementation was prioritized around the highest-confidence pain points from the R&D review:

1. Dispatch overhead and repeated-launch cost: added CUDA graph capture/replay for stable execution signatures to reduce launch-path tax.
2. Opaque fallback behavior: expanded graph-break and fallback diagnostics (taxonomy + first-break metadata) so unsupported patterns are explicit and actionable.
3. Real-world operator chains: extended native CUDA path coverage to matmul-centric chains with `add` and `relu` epilogues, which are common in AI inference paths.
4. Promotion ambiguity: bound promotion to deterministic tests, benchmark metadata stamps, and compatibility checks rather than anecdotal speedup claims.

Expected practical outcome:

- Better repeat-run stability, clearer failure surfaces, and a tighter loop for performance iteration without losing determinism.

## 9) High-Resolution Target for Next Iteration

Given measured behavior, the next high-resolution target remains:

1. Keep deterministic contracts strict (no silent wrong outputs, explicit reason-coded fallback).
2. Reduce GPU dispatch overhead further via deeper fusion and persistent-kernel strategies.
3. Improve arithmetic intensity on hot paths (matmul-activation families first).
4. Maintain benchmark integrity with fixed protocol, variance gates, and stamped artifacts.

## 10) Innovation Tracker Snapshot

| ID | Item | Phase | KPI |
|---|---|---|---|
| A1 | Adaptive planner modes | 3-4 | `memory_first` >=20% peak-memory reduction; `latency_first` <=5% latency regression |
| A2 | Reuse + rematerialization | 3-4 | >=25% peak-memory reduction with <10% latency penalty |
| A3 | Shape-clustered multi-plan | 4 | >=1.2x p95 latency improvement |
| B1 | Kernel + allocator co-selection | 4-5 | >=10% end-to-end latency gain |
| B2 | Online feedback planner | 5 | >=15% tail-latency drift reduction |
| B3 | Deterministic what-if simulator | 5 | >=80% predicted-vs-measured winner agreement |
| C1 | Optimization contracts | 5 | 100% contract compliance, >=90% explainable logs |
| C2 | Policy plugin interfaces | 5 | <200 LOC integration per policy, no CI determinism regressions |

## Sources

Primary/official sources used:

- LLVM LangRef: https://llvm.org/docs/LangRef.html
- LLVM docs hub: https://www.llvm.org/docs/
- MLIR CGO 2021 paper: https://research.google/pubs/mlir-scaling-compiler-infrastructure-for-domain-specific-computation/
- MLIR docs: https://mlir.llvm.org/docs/
- TVM OSDI 2018 paper page: https://www.usenix.org/conference/osdi18/presentation/chen
- Ansor OSDI 2020 paper page: https://www.usenix.org/conference/osdi20/presentation/zheng
- TVM auto-scheduler blog (template complexity context): https://tvm.apache.org/2021/03/03/intro-auto-scheduler.html
- OpenXLA overview: https://openxla.org/
- XLA tf2xla optimization example: https://openxla.org/xla/tf2xla
- OpenXLA StableHLO: https://openxla.org/stablehlo
- OpenXLA PJRT: https://openxla.org/xla/pjrt
- OpenXLA HLO dumps (debug workflows): https://openxla.org/xla/hlo_dumps
- OpenXLA error codes: https://openxla.org/xla/error_codes
- XLA repo: https://github.com/openxla/xla
- IREE VM design doc: https://iree.dev/developers/design-docs/vm/
- TensorRT timing cache/tactics docs: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/TimingCache.html
- TensorRT builder config/tactic sources: https://developer.nvidia.com/docs/drive/drive-os/6.0.8/public/drive-os-tensorrt/api-reference/docs/python/infer/Core/BuilderConfig.html
- Dynamic Tensor Rematerialization project page: https://sampl.cs.washington.edu/projects/dtr.html
- Dynamic Tensor Rematerialization paper summary/index: https://www.emergentmind.com/papers/2006.09616
- DynaTune (ICLR 2021): https://www.microsoft.com/en-us/research/publication/dynatune-dynamic-tensor-program-optimization-in-deep-neural-network-compilation/
- AdaTune (NeurIPS 2020): https://www.microsoft.com/en-us/research/publication/adatune-adaptive-tensor-program-compilation-made-efficient/
- Hummingbird OSDI 2020: https://www.microsoft.com/en-us/research/publication/a-tensor-compiler-for-unified-machine-learning-prediction-serving/

Sentiment evidence inputs (issues and operational pain points):

- XLA build issue example: https://github.com/openxla/xla/issues/10592
- XLA build issue example (macOS): https://github.com/openxla/xla/issues/17820
- XLA build issue example (CUDA tool build): https://github.com/openxla/xla/issues/13889
- TVM tuning hang issue example: https://github.com/apache/tvm/issues/12330
