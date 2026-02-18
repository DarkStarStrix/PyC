# GPU Benchmark Suite

- Timestamp (UTC): 2026-02-18T01:11:32.964016+00:00
- Host: 90c892d22de7
- OS: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python: 3.11.11

## GPU

- GPU 1: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9
- GPU 2: NVIDIA GeForce RTX 4090 | driver 570.195.03 | mem 24564 MiB | cc 8.9

## Adapter Results

- `PyTorch Eager`: p50 12.9191 ms, p95 71.8259 ms, mean 30.2598 ms, throughput 4331556.32 tokens/s
- `PyTorch Compile`: p50 11.9115 ms, p95 71.6757 ms, mean 29.6697 ms, throughput 4417703.78 tokens/s
- `PyC CUDA`: p50 141.0531 ms, p95 146.8969 ms, mean 141.9295 ms, throughput 923500.56 tokens/s
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
- `XLA`: p50 12.2043 ms, p95 70.1422 ms, mean 27.5087 ms, throughput 4764742.16 tokens/s
- `TensorRT`: unavailable (TensorRT benchmark requires BENCH_DEVICE=cuda)
- `Glow`: p50 15.0033 ms, p95 73.9626 ms, mean 34.9326 ms, throughput 3752137.28 tokens/s

## Notes

- Adapters are normalized to a common JSON schema.
- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.
