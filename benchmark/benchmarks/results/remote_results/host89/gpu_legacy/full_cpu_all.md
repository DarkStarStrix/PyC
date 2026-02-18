# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:08:38.820905+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 16.8206 ms, p95 72.7831 ms, mean 32.5071 ms, throughput 4032106.5 tokens/s
- `PyTorch Compile`: p50 9.186 ms, p95 71.9963 ms, mean 24.6044 ms, throughput 5327180.72 tokens/s
- `PyC CUDA`: p50 140.6518 ms, p95 148.597 ms, mean 141.1542 ms, throughput 928573.43 tokens/s
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
- `XLA`: unavailable (torch_xla not installed; install torch_xla or change XLA_BENCH_CMD)
- `TensorRT`: unavailable (torch_tensorrt not installed; install it or change TENSORRT_BENCH_CMD)
- `Glow`: p50 15.1946 ms, p95 74.3603 ms, mean 34.5063 ms, throughput 3798495.25 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
