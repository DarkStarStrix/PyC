# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:19:19.479882+00:00
- Host: Allans-Air.attlocal.net
- OS: macOS-26.3-arm64-arm-64bit-Mach-O
- Python: 3.14.0

## GPU

- Not available: nvidia-smi not found

## Adapter Results

- `PyTorch Eager`: error (Torch not compiled with CUDA enabled)
- `PyTorch Compile`: error (Torch not compiled with CUDA enabled)
- `PyC CUDA`: p50 79.2881 ms, p95 83.3326 ms, mean 81.3104 ms, throughput 6296.86 tokens/s
- `TVM`: unavailable (TVM not installed; install TVM or change TVM_BENCH_CMD)
- `XLA`: p50 0.0238 ms, p95 0.0298 ms, mean 0.0268 ms, throughput 19110537.35 tokens/s
- `TensorRT`: unavailable (CUDA not available)
- `Glow`: unavailable (CUDA not available for Glow proxy benchmark)

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
