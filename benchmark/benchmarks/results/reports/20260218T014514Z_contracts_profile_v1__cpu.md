# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:45:14.804011+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 21.2666 ms, p95 80.5246 ms, mean 44.2254 ms, throughput 2963725.12 tokens/s
- `PyTorch Compile`: p50 9.8954 ms, p95 72.736 ms, mean 25.0023 ms, throughput 5242403.94 tokens/s
- `PyC CUDA`: error (CMake Error at CMakeLists.txt:77 (add_executable):
  Cannot find source file:

    /root/PyC/tests/compiler_next/test_deterministic_contracts.c

  Tried extensions .c .C .c++ .cc .cpp .cxx .cu .mpp .m .M .mm .ixx .cppm .h
  .hh .h++ .hm .hpp .hxx .in .txx .f .F .for .f77 .f90 .f95 .f03 .hip .ispc
Call Stack (most recent call first):
  CMakeLists.txt:107 (pyc_add_compiler_next_test)


CMake Error at CMakeLists.txt:77 (add_executable):
  No SOURCES given to target: pyc_compiler_next_test_deterministic_contracts
Call Stack (most recent call first):
  CMakeLists.txt:107 (pyc_add_compiler_next_test)


CMake Generate step failed.  Build files cannot be regenerated correctly.)
- `TVM`: p50 10.8319 ms, p95 70.3985 ms, mean 25.4081 ms, throughput 5158672.31 tokens/s
- `XLA`: p50 13.7354 ms, p95 74.7781 ms, mean 33.003 ms, throughput 3971512.23 tokens/s
- `TensorRT`: p50 8.7301 ms, p95 68.8152 ms, mean 22.3791 ms, throughput 5856894.58 tokens/s
- `Glow`: p50 14.7653 ms, p95 74.4527 ms, mean 34.2658 ms, throughput 3825151.09 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
