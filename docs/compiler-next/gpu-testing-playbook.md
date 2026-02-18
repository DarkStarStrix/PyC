# GPU Testing Playbook (Rented Linux + CUDA)

This is the operational path to test PyC ideas on real GPUs with repeatable metrics.

## 1) Provision a Linux GPU machine

Recommended:

- Ubuntu 22.04+
- NVIDIA GPU (A10/L4/A100/H100)
- SSH access

## 2) Setup deterministic CUDA test environment

On remote host in repo root:

```bash
bash scripts/setup_cuda_remote_ubuntu.sh
source .venv/bin/activate
```

The script validates:

- driver/runtime (`nvidia-smi`)
- Python env
- CUDA-enabled PyTorch install

## 3) Run standardized workload suite

```bash
python3 benchmark/benchmarks/gpu/run_gpu_suite.py --device cuda --batch 64 --hidden 2048 --iters 80 --warmup 20 --tag gpu_baseline
```

Outputs:

- `benchmark/benchmarks/gpu/results/gpu_baseline.json`
- `benchmark/benchmarks/gpu/results/gpu_baseline.md`

Tracked KPIs:

- p50/p95 latency
- throughput (tokens/s)
- peak memory bytes
- GPU metadata and environment

## 4) Benchmark against other compiler stacks

The suite now runs adapter scripts:

- `torch_eager`
- `torch_compile`
- `pyc`
- `tvm`
- `xla`
- `tensorrt`
- `glow`

Run all adapters:

```bash
python3 benchmark/benchmarks/gpu/run_gpu_suite.py --device cuda --tag gpu_compare_all
```

### External adapter commands (TVM/XLA/TensorRT/PyC)

For non-PyTorch stacks, set the adapter command env vars so each tool can run your standardized command and emit JSON.
The repo now includes a helper:

```bash
source benchmark/benchmarks/gpu/configure_adapter_cmds.sh
```

Manual form (if needed):

```bash
export TVM_BENCH_CMD="python3 path/to/tvm_bench.py --json"
export XLA_BENCH_CMD="python3 path/to/xla_bench.py --json"
export TENSORRT_BENCH_CMD="python3 path/to/trt_bench.py --json"
export PYC_GPU_BENCH_CMD="python3 path/to/pyc_cuda_bench.py --json"
export GLOW_BENCH_CMD="python3 path/to/glow_bench.py --json"
```

Each command must print JSON with at least:

- `status`
- `latency_ms` (`mean`, `p50`, `p95`, `min`, `max`)
- `throughput_tokens_per_sec`
- `peak_memory_bytes`

For fair comparisons, keep this protocol:

1. Same host and GPU.
2. Same workload shapes (`batch`, `hidden`, iterations).
3. Same warmup/run counts.
4. Same dtype and precision policy.
5. Save each run under unique `--tag`.

Example tags:

- `gpu_torch_compile`
- `gpu_tvm`
- `gpu_xla`
- `gpu_tensorrt`

## 5) Apply PyC ideas and compare

For each idea (allocator policy, kernel scoring, rematerialization):

1. Run baseline tag.
2. Enable one idea only.
3. Re-run suite with new tag.
4. Compare p95 latency, throughput, peak memory.

Promotion guideline:

- Keep only ideas with stable gain across at least 3 runs.
- Revert ideas with regressions or high variance.

## 6) Suggested iteration loop

1. Implement one change.
2. Run local deterministic tests (`ctest`).
3. Run remote GPU suite (`run_gpu_suite.py`).
4. Compare with prior tags.
5. Update roadmap/backlog status with measured evidence.

## 7) Important constraints

- Apple Silicon (local) cannot run CUDA; use remote Linux GPU for CUDA phases.
- Always pin tool versions for fair comparisons.
- Do not compare numbers across different GPU classes without normalization.
