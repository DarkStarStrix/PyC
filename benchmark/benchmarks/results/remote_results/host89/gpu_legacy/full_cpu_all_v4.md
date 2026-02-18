# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:15:33.786564+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 15.9215 ms, p95 73.9385 ms, mean 36.2089 ms, throughput 3619888.08 tokens/s
- `PyTorch Compile`: p50 10.3385 ms, p95 71.5978 ms, mean 26.4611 ms, throughput 4953391.43 tokens/s
- `PyC CUDA`: p50 139.8214 ms, p95 152.4895 ms, mean 141.0321 ms, throughput 929377.29 tokens/s
- `TVM`: p50 9.98 ms, p95 69.3122 ms, mean 23.6425 ms, throughput 5543914.26 tokens/s
- `XLA`: p50 16.9793 ms, p95 76.5611 ms, mean 38.9585 ms, throughput 3364402.13 tokens/s
- `TensorRT`: p50 13.5502 ms, p95 75.9196 ms, mean 31.2176 ms, throughput 4198661.45 tokens/s
- `Glow`: p50 14.4597 ms, p95 73.7702 ms, mean 33.41 ms, throughput 3923135.71 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
