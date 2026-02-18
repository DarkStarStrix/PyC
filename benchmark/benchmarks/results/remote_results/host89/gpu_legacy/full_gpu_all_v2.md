# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:12:03.818221+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 0.1091 ms, p95 0.1217 ms, mean 0.1112 ms, throughput 1179130083.69 tokens/s
- `PyTorch Compile`: p50 0.1599 ms, p95 0.1743 ms, mean 0.1602 ms, throughput 817952787.9 tokens/s
- `PyC CUDA`: p50 140.8507 ms, p95 148.2243 ms, mean 141.1747 ms, throughput 928438.01 tokens/s
- `TVM`: error (TVM build/run init failed: Traceback (most recent call last):
  14: tvm::relay::backend::RelayBuildModule::GetFunction(tvm::runtime::String const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
  13: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
  12: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
  11: tvm::transform::Pass::operator()(tvm::IRModule) const
  10: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  9: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  8: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  7: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  5: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
  4: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
  3: tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)
  2: tvm::relay::TypeSolver::Solve()
  1: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
  0: bool tvm::relay::BatchMatmulRel<tvm::relay::BatchMatmulAttrs>(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)
  File "/workspace/tvm/src/relay/op/nn/nn.h", line 212
InternalError: Check failed: (reporter->AssertEQ(xk, yk)) is false: BatchDot: shapes of x and y is inconsistent,  x shape=[64, 1, 2048], y shape=[1, 2048, 64])
- `XLA`: p50 0.1111 ms, p95 0.1287 ms, mean 0.1149 ms, throughput 1140711821.71 tokens/s
- `TensorRT`: p50 0.2049 ms, p95 0.2147 ms, mean 0.2057 ms, throughput 637328770.22 tokens/s
- `Glow`: p50 0.1093 ms, p95 0.121 ms, mean 0.1107 ms, throughput 1183536227.46 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
