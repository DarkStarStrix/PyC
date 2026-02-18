# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:48:58.301294+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.1098 ms, p95 0.1237 ms, mean 0.1113 ms, throughput 1177409204.73 tokens/s
- `PyTorch Compile`: p50 0.1543 ms, p95 0.1734 ms, mean 0.1619 ms, throughput 809775992.05 tokens/s
- `PyC CUDA`: p50 1070.001 ms, p95 1199.584 ms, mean 1101.8633 ms, throughput 118954.86 tokens/s
- `TVM`: p50 10.2136 ms, p95 69.3832 ms, mean 24.8347 ms, throughput 5277766.16 tokens/s
- `XLA`: p50 0.1105 ms, p95 0.1146 ms, mean 0.1113 ms, throughput 1177730901.64 tokens/s
- `TensorRT`: p50 0.1683 ms, p95 0.1818 ms, mean 0.1697 ms, throughput 772206057.46 tokens/s
- `Glow`: p50 0.1091 ms, p95 0.1254 ms, mean 0.118 ms, throughput 1110418890.83 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
