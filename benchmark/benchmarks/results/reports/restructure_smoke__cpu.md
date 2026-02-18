# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:27:57.555296+00:00
- Host: Allans-Air.attlocal.net
- OS: macOS-26.3-arm64-arm-64bit-Mach-O
- Python: 3.14.0

## GPU

- Not available: nvidia-smi not found

## Adapter Results

- `PyTorch Eager`: p50 0.032 ms, p95 0.0705 ms, mean 0.0512 ms, throughput 9994339.0 tokens/s
- `PyTorch Compile`: p50 0.0755 ms, p95 0.1387 ms, mean 0.1071 ms, throughput 4780378.09 tokens/s
- `PyC CUDA`: p50 75.1349 ms, p95 79.017 ms, mean 77.076 ms, throughput 6642.8 tokens/s
- `TVM`: unavailable (TVM not installed; install TVM or change TVM_BENCH_CMD)
- `XLA`: p50 0.0259 ms, p95 0.0311 ms, mean 0.0285 ms, throughput 17952000.23 tokens/s
- `TensorRT`: p50 0.0971 ms, p95 0.1208 ms, mean 0.1089 ms, throughput 4699964.21 tokens/s
- `Glow`: p50 0.0253 ms, p95 0.0292 ms, mean 0.0273 ms, throughput 18788990.74 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
