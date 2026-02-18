# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:23:23.264950+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.1086 ms, p95 0.1253 ms, mean 0.1111 ms, throughput 1179952591.08 tokens/s
- `PyTorch Compile`: p50 0.1582 ms, p95 0.1764 ms, mean 0.1606 ms, throughput 816146514.4 tokens/s
- `PyC CUDA`: p50 144.1311 ms, p95 156.2319 ms, mean 144.9096 ms, throughput 904508.47 tokens/s
- `TVM`: p50 10.5555 ms, p95 69.8519 ms, mean 25.4067 ms, throughput 5158946.83 tokens/s
- `XLA`: p50 0.1174 ms, p95 0.136 ms, mean 0.1196 ms, throughput 1095562041.03 tokens/s
- `TensorRT`: p50 0.1763 ms, p95 0.1889 ms, mean 0.1767 ms, throughput 741728274.35 tokens/s
- `Glow`: p50 0.1146 ms, p95 0.1262 ms, mean 0.1162 ms, throughput 1128199176.76 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
