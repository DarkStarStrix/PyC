# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-19T01:24:24.672845+00:00
- Host: d9aff3ced886
- OS: Linux-6.8.0-83-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 3090 | driver 580.82.09 | mem 24576 MiB | cc 8.6

## Native Adapter Results

- `PyTorch Eager` [native]: p50 0.1529 ms, p95 0.154 ms, mean 0.1529 ms, throughput 856966115.05 tokens/s, jitter 0.0011 ms, sample_cv 0.64%, repeat_cv 1.98%, est_tflops 28.1998, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0731 GiB
- `PyTorch Compile` [native]: p50 0.205 ms, p95 0.2126 ms, mean 0.2057 ms, throughput 637153918.18 tokens/s, jitter 0.0076 ms, sample_cv 2.69%, repeat_cv 3.55%, est_tflops 20.9613, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0710 GiB
- `PyC CUDA` [native]: p50 0.175 ms, p95 0.183 ms, mean 0.1753 ms, throughput 747807730.71 tokens/s, jitter 0.0080 ms, sample_cv 0.00%, repeat_cv 1.66%, est_tflops 24.5964, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0166 GiB
- `TVM` [native]: p50 0.3384 ms, p95 0.3397 ms, mean 0.3385 ms, throughput 387262987.34 tokens/s, jitter 0.0013 ms, sample_cv 0.25%, repeat_cv 0.64%, est_tflops 12.7378, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB
- `XLA` [native]: p50 0.3196 ms, p95 0.3261 ms, mean 0.3194 ms, throughput 410325519.42 tokens/s, jitter 0.0065 ms, sample_cv 3.67%, repeat_cv 3.60%, est_tflops 13.4995, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0000 GiB
- `TensorRT` [native]: p50 0.2071 ms, p95 0.2395 ms, mean 0.2137 ms, throughput 613229097.73 tokens/s, jitter 0.0324 ms, sample_cv 0.00%, repeat_cv 1.64%, est_tflops 20.1766, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0734 GiB

## Proxy Adapter Results

- `Glow` [proxy]: p50 0.1528 ms, p95 0.1587 ms, mean 0.1533 ms, throughput 854772290.62 tokens/s, jitter 0.0059 ms, sample_cv 0.00%, repeat_cv 0.43%, est_tflops 28.1262, startup 0.0000 ms, compile_est 0.0000 ms, peak 0.0731 GiB

## Unavailable Adapters

- none

## Adapter Errors

- none

## Notes

- Native count: 6 | Proxy count: 1 | Unavailable: 0 | Errors: 0
- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
