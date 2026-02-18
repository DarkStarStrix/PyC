# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T00:58:42.597686+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.1158 ms, p95 0.1289 ms, mean 0.1178 ms, throughput 1112323333.17 tokens/s
- `PyTorch Compile`: p50 0.1553 ms, p95 0.1714 ms, mean 0.1586 ms, throughput 826273963.41 tokens/s
- `PyC CUDA`: unavailable (PyC CUDA backend is not implemented yet; current PyC benchmark command is CPU-only)
- `TVM`: unavailable (TVM installed but CUDA device not available)
- `XLA`: unavailable (torch_xla not installed; install torch_xla or change XLA_BENCH_CMD)
- `TensorRT`: unavailable (torch_tensorrt not installed; install it or change TENSORRT_BENCH_CMD)
- `Glow`: unavailable (Glow runtime benchmark integration is not wired in this environment yet)

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
