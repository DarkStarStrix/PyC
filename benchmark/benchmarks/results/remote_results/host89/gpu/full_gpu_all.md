# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:09:12.663638+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.109 ms, p95 0.1181 ms, mean 0.11 ms, throughput 1191112859.45 tokens/s
- `PyTorch Compile`: p50 0.1645 ms, p95 0.1768 ms, mean 0.1665 ms, throughput 787025360.44 tokens/s
- `PyC CUDA`: p50 141.2184 ms, p95 146.2196 ms, mean 141.2764 ms, throughput 927769.7 tokens/s
- `TVM`: unavailable (TVM installed but CUDA device not available)
- `XLA`: unavailable (torch_xla not installed; install torch_xla or change XLA_BENCH_CMD)
- `TensorRT`: unavailable (torch_tensorrt not installed; install it or change TENSORRT_BENCH_CMD)
- `Glow`: p50 0.1095 ms, p95 0.1226 ms, mean 0.111 ms, throughput 1180886220.46 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
