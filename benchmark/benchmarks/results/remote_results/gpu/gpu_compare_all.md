# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-17T23:34:56.960815+00:00
- Host: b5060ab798c6
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.1087 ms, p95 0.1179 ms, mean 0.11 ms, throughput 1191471592.77 tokens/s
- `PyTorch Compile`: p50 0.1446 ms, p95 0.1547 ms, mean 0.1463 ms, throughput 896015628.38 tokens/s
- `PyC CUDA`: unavailable (Set PYC_GPU_BENCH_CMD to a command that outputs benchmark JSON for PyC CUDA path)
- `TVM`: unavailable (Install TVM or set TVM_BENCH_CMD to external benchmark command)
- `XLA`: unavailable (Install torch_xla or set XLA_BENCH_CMD to external benchmark command)
- `TensorRT`: unavailable (Install TensorRT python bindings or set TENSORRT_BENCH_CMD)
- `Glow`: unavailable (Set GLOW_BENCH_CMD to external benchmark command for Glow)

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
