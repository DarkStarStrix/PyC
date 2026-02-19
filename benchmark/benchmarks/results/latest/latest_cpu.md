# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-19T01:21:33.573737+00:00
- Host: d9aff3ced886
- OS: Linux-6.8.0-83-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 3090 | driver 580.82.09 | mem 24576 MiB | cc 8.6

## Native Adapter Results

- `PyTorch Eager` [native]: p50 98.275 ms, p95 104.1338 ms, mean 92.913 ms, throughput 1410696.45 tokens/s, jitter 5.8588 ms, sample_cv 34.36%, repeat_cv 7.72%, est_tflops 0.0464, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB
- `PyTorch Compile` [native]: p50 91.9345 ms, p95 96.006 ms, mean 67.3705 ms, throughput 1945540.46 tokens/s, jitter 4.0715 ms, sample_cv 62.61%, repeat_cv 12.29%, est_tflops 0.0640, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB
- `PyC CUDA` [native]: p50 26.342 ms, p95 31.373 ms, mean 27.3495 ms, throughput 4792486.88 tokens/s, jitter 5.0310 ms, sample_cv 0.00%, repeat_cv 1.87%, est_tflops 0.1577, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0166 GiB
- `TVM` [native]: p50 93.2592 ms, p95 94.4789 ms, mean 70.1233 ms, throughput 1869164.22 tokens/s, jitter 1.2197 ms, sample_cv 51.16%, repeat_cv 4.93%, est_tflops 0.0615, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB
- `XLA` [native]: p50 15.377 ms, p95 53.956 ms, mean 24.6795 ms, throughput 5310958.0 tokens/s, jitter 38.5790 ms, sample_cv 64.67%, repeat_cv 3.33%, est_tflops 0.1747, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB

## Proxy Adapter Results

- `TensorRT` [proxy]: p50 100.5979 ms, p95 104.0894 ms, mean 102.7787 ms, throughput 1275283.89 tokens/s, jitter 3.4915 ms, sample_cv 0.00%, repeat_cv 1.05%, est_tflops 0.0420, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB
- `Glow` [proxy]: p50 97.8844 ms, p95 101.3468 ms, mean 87.9144 ms, throughput 1490905.14 tokens/s, jitter 3.4624 ms, sample_cv 0.00%, repeat_cv 11.29%, est_tflops 0.0490, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB

## Unavailable Adapters

- none

## Adapter Errors

- none

## Notes

- Native count: 5 | Proxy count: 2 | Unavailable: 0 | Errors: 0
- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
