# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:13:43.755360+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 13.4279 ms, p95 72.9134 ms, mean 30.7592 ms, throughput 4261224.43 tokens/s
- `PyTorch Compile`: p50 14.7016 ms, p95 72.5172 ms, mean 33.2978 ms, throughput 3936360.59 tokens/s
- `PyC CUDA`: p50 143.3431 ms, p95 152.0857 ms, mean 144.0853 ms, throughput 909683.44 tokens/s
- `TVM`: p50 13.5731 ms, p95 77.2123 ms, mean 32.9258 ms, throughput 3980825.35 tokens/s
- `XLA`: p50 15.0325 ms, p95 73.7028 ms, mean 33.8499 ms, throughput 3872149.37 tokens/s
- `TensorRT`: error (external command must print JSON)
- `Glow`: p50 13.9688 ms, p95 86.5618 ms, mean 33.5014 ms, throughput 3912436.82 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
