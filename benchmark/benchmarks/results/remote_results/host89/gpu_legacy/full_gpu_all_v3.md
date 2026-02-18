# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:14:14.081940+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.1074 ms, p95 0.1207 ms, mean 0.1142 ms, throughput 1148051483.68 tokens/s
- `PyTorch Compile`: p50 0.1565 ms, p95 0.17 ms, mean 0.1588 ms, throughput 825641584.09 tokens/s
- `PyC CUDA`: p50 152.0273 ms, p95 170.1069 ms, mean 153.8173 ms, throughput 852127.69 tokens/s
- `TVM`: p50 11.5654 ms, p95 70.617 ms, mean 28.8425 ms, throughput 4544398.42 tokens/s
- `XLA`: p50 0.114 ms, p95 0.1386 ms, mean 0.1188 ms, throughput 1103033291.36 tokens/s
- `TensorRT`: p50 0.166 ms, p95 0.1907 ms, mean 0.1693 ms, throughput 774318447.61 tokens/s
- `Glow`: p50 0.1142 ms, p95 0.1286 ms, mean 0.1157 ms, throughput 1132491308.25 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
