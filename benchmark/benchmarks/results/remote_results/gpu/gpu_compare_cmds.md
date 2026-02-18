# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-17T23:47:21.587686+00:00
- Host: b5060ab798c6
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- Not available: nvidia-smi query failed

## Adapter Results

- `PyTorch Eager`: p50 95.2503 ms, p95 100.6381 ms, mean 81.3846 ms, throughput 1610525.74 tokens/s
- `PyTorch Compile`: p50 91.6883 ms, p95 97.0664 ms, mean 66.1312 ms, throughput 1981998.22 tokens/s
- `PyC CUDA`: p50 141.2969 ms, p95 147.8048 ms, mean 141.9971 ms, throughput 923060.83 tokens/s
- `TVM`: unavailable (TVM not installed; install TVM or change TVM_BENCH_CMD)
- `XLA`: unavailable (torch_xla not installed; install torch_xla or change XLA_BENCH_CMD)
- `TensorRT`: unavailable (torch_tensorrt not installed; install it or change TENSORRT_BENCH_CMD)
- `Glow`: unavailable (Glow runtime benchmark integration is not wired in this environment yet)

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
