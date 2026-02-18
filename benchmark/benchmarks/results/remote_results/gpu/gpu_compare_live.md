# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-17T23:40:05.421133+00:00
- Host: b5060ab798c6
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- Not available: nvidia-smi query failed

## Adapter Results

- `PyTorch Eager`: p50 92.2615 ms, p95 99.3985 ms, mean 72.277 ms, throughput 1813468.09 tokens/s
- `PyTorch Compile`: p50 88.6281 ms, p95 94.1412 ms, mean 52.3751 ms, throughput 2502562.32 tokens/s
- `PyC CUDA`: unavailable (Set PYC_GPU_BENCH_CMD to a command that outputs benchmark JSON for PyC CUDA path)
- `TVM`: unavailable (Install TVM or set TVM_BENCH_CMD to external benchmark command)
- `XLA`: unavailable (Install torch_xla or set XLA_BENCH_CMD to external benchmark command)
- `TensorRT`: unavailable (Install TensorRT python bindings or set TENSORRT_BENCH_CMD)
- `Glow`: unavailable (Set GLOW_BENCH_CMD to external benchmark command for Glow)

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
