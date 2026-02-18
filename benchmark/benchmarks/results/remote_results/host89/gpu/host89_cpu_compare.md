# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T00:46:15.206732+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 15.3394 ms, p95 73.9333 ms, mean 33.9212 ms, throughput 3864010.47 tokens/s
- `PyTorch Compile`: p50 9.9002 ms, p95 71.4869 ms, mean 24.8649 ms, throughput 5271375.11 tokens/s
- `PyC CUDA`: p50 141.5733 ms, p95 148.8281 ms, mean 142.277 ms, throughput 921244.9 tokens/s
- `TVM`: unavailable (TVM not installed; install TVM or change TVM_BENCH_CMD)
- `XLA`: unavailable (torch_xla not installed; install torch_xla or change XLA_BENCH_CMD)
- `TensorRT`: unavailable (torch_tensorrt not installed; install it or change TENSORRT_BENCH_CMD)
- `Glow`: unavailable (Glow runtime benchmark integration is not wired in this environment yet)

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
