# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:28:04.683525+00:00
- Host: Allans-Air.attlocal.net
- OS: macOS-26.3-arm64-arm-64bit-Mach-O
- Python: 3.14.0

## GPU

- Not available: nvidia-smi not found

## Adapter Results

- `PyTorch Eager`: error (Torch not compiled with CUDA enabled)
- `PyTorch Compile`: error (Torch not compiled with CUDA enabled)
- `PyC CUDA`: p50 77.0491 ms, p95 79.152 ms, mean 78.1006 ms, throughput 6555.65 tokens/s
- `TVM`: unavailable (TVM not installed; install TVM or change TVM_BENCH_CMD)
- `XLA`: p50 0.0236 ms, p95 0.0302 ms, mean 0.0269 ms, throughput 19050807.55 tokens/s
- `TensorRT`: unavailable (CUDA not available)
- `Glow`: unavailable (CUDA not available for Glow proxy benchmark)

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
