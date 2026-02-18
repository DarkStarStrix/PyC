# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:22:44.480982+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 16.1717 ms, p95 77.3977 ms, mean 36.9826 ms, throughput 3544157.27 tokens/s
- `PyTorch Compile`: p50 9.6974 ms, p95 72.9653 ms, mean 24.7309 ms, throughput 5299933.5 tokens/s
- `PyC CUDA`: p50 140.245 ms, p95 148.7101 ms, mean 141.0577 ms, throughput 929208.52 tokens/s
- `TVM`: p50 13.5227 ms, p95 70.6271 ms, mean 29.3689 ms, throughput 4462948.94 tokens/s
- `XLA`: p50 16.6186 ms, p95 73.9017 ms, mean 36.9306 ms, throughput 3549140.25 tokens/s
- `TensorRT`: p50 9.6623 ms, p95 69.9137 ms, mean 24.5778 ms, throughput 5332936.81 tokens/s
- `Glow`: p50 15.1214 ms, p95 74.5309 ms, mean 34.8783 ms, throughput 3757982.89 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
