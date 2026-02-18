# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:45:48.273077+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.1209 ms, p95 0.1374 ms, mean 0.1237 ms, throughput 1059554024.41 tokens/s
- `PyTorch Compile`: p50 0.1563 ms, p95 0.1677 ms, mean 0.1574 ms, throughput 832923177.27 tokens/s
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
- `TVM`: p50 10.1429 ms, p95 69.3693 ms, mean 24.7732 ms, throughput 5290889.26 tokens/s
- `XLA`: p50 0.1096 ms, p95 0.1192 ms, mean 0.1111 ms, throughput 1179663297.56 tokens/s
- `TensorRT`: p50 0.1664 ms, p95 0.1786 ms, mean 0.1685 ms, throughput 778081141.34 tokens/s
- `Glow`: p50 0.1134 ms, p95 0.1223 ms, mean 0.1156 ms, throughput 1133584704.27 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
