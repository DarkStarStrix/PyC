# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:47:28.628866+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 16.3671 ms, p95 75.1722 ms, mean 36.0145 ms, throughput 3639425.85 tokens/s
- `PyTorch Compile`: p50 13.2568 ms, p95 76.628 ms, mean 32.7525 ms, throughput 4001894.65 tokens/s
- `PyC CUDA`: p50 1070.868 ms, p95 1083.914 ms, mean 1072.2825 ms, throughput 122236.44 tokens/s
- `TVM`: p50 10.5717 ms, p95 68.4926 ms, mean 26.5001 ms, throughput 4946099.73 tokens/s
- `XLA`: p50 19.6563 ms, p95 75.7144 ms, mean 39.9168 ms, throughput 3283631.07 tokens/s
- `TensorRT`: p50 12.105 ms, p95 71.0655 ms, mean 29.3551 ms, throughput 4465052.76 tokens/s
- `Glow`: p50 14.3688 ms, p95 75.3568 ms, mean 33.087 ms, throughput 3961435.53 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
