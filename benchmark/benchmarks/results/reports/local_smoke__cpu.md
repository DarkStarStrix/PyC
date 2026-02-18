# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:19:09.066245+00:00
- Host: Allans-Air.attlocal.net
- OS: macOS-26.3-arm64-arm-64bit-Mach-O
- Python: 3.14.0

## GPU

- Not available: nvidia-smi not found

## Adapter Results

- `PyTorch Eager`: p50 0.0302 ms, p95 0.0655 ms, mean 0.0478 ms, throughput 10708497.08 tokens/s
- `PyTorch Compile`: p50 0.0781 ms, p95 0.1299 ms, mean 0.104 ms, throughput 4923100.49 tokens/s
- `PyC CUDA`: p50 75.8652 ms, p95 78.0785 ms, mean 76.9718 ms, throughput 6651.79 tokens/s
- `TVM`: unavailable (TVM not installed; install TVM or change TVM_BENCH_CMD)
- `XLA`: p50 0.0265 ms, p95 0.0297 ms, mean 0.0281 ms, throughput 18231346.28 tokens/s
- `TensorRT`: p50 0.0861 ms, p95 0.1197 ms, mean 0.1029 ms, throughput 4975922.12 tokens/s
- `Glow`: p50 0.0249 ms, p95 0.0289 ms, mean 0.0269 ms, throughput 19021435.29 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
