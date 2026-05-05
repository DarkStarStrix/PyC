#include "Loader.h"

#include "glow/Base/Tensor.h"
#include "glow/Support/Error.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace glow;

namespace {

unsigned env_to_uint(const char *name, unsigned fallback) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    return fallback;
  }
  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end != '\0') {
    return fallback;
  }
  return static_cast<unsigned>(parsed);
}

std::string env_to_string(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    return fallback;
  }
  return std::string(value);
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto maxIndex = values.size() - 1;
  const auto idx = static_cast<size_t>(
      std::llround((p / 100.0) * static_cast<double>(maxIndex)));
  return values[std::min(idx, maxIndex)];
}

void randomize_tensor(Tensor *tensor, Module *module) {
  CHECK(tensor) << "Expected tensor allocation";
  switch (tensor->getElementType()) {
  case ElemKind::FloatTy:
    tensor->getHandle<float>().randomize(-1.0f, 1.0f, module->getPRNG());
    return;
  case ElemKind::Float16Ty:
    tensor->getHandle<float16_t>().randomize(-1.0f, 1.0f, module->getPRNG());
    return;
  case ElemKind::BFloat16Ty:
    tensor->getHandle<bfloat16_t>().randomize(-1.0f, 1.0f, module->getPRNG());
    return;
  default:
    LOG(FATAL) << "Unsupported input tensor element kind: "
               << static_cast<int>(tensor->getElementType());
  }
}

std::string json_escape(const std::string &value) {
  std::ostringstream out;
  for (char ch : value) {
    switch (ch) {
    case '\\':
      out << "\\\\";
      break;
    case '"':
      out << "\\\"";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      out << ch;
      break;
    }
  }
  return out.str();
}

} // namespace

int main(int argc, char **argv) {
  parseCommandLine(argc, argv);

  const unsigned warmup = env_to_uint("PYC_GLOW_WARMUP", 20);
  const unsigned iters = env_to_uint("PYC_GLOW_ITERS", 80);
  const unsigned m = env_to_uint("PYC_GLOW_M", 0);
  const unsigned k = env_to_uint("PYC_GLOW_K", 0);
  const unsigned n = env_to_uint("PYC_GLOW_N", 0);
  const std::string dtype = env_to_string("PYC_GLOW_DTYPE", "float32");

  Loader loader;
  PlaceholderBindings bindings;
  loader.loadModel(&bindings);

  const auto &inputs = loader.getInputPlaceholderMap();
  CHECK_EQ(inputs.size(), 1) << "Expected exactly one model input";
  auto inputIt = inputs.begin();
  Placeholder *input = inputIt->second;
  Tensor *inputTensor = bindings.get(input);
  if (!inputTensor) {
    inputTensor = bindings.allocate(input);
  }
  randomize_tensor(inputTensor, loader.getModule());

  const auto &outputs = loader.getOutputPlaceholderMap();
  for (const auto &entry : outputs) {
    if (!bindings.get(entry.second)) {
      bindings.allocate(entry.second);
    }
  }

  CompilationContext cctx = loader.getCompilationContext();
  cctx.bindings = &bindings;
  loader.compile(cctx);

  runtime::HostManager *hostManager = loader.getHostManager();
  const std::string functionName = loader.getFunctionName();

  for (unsigned i = 0; i < warmup; ++i) {
    EXIT_ON_ERR(hostManager->runNetworkBlocking(functionName, bindings));
  }

  std::vector<double> samplesMs;
  samplesMs.reserve(iters);
  for (unsigned i = 0; i < iters; ++i) {
    const auto start = std::chrono::steady_clock::now();
    EXIT_ON_ERR(hostManager->runNetworkBlocking(functionName, bindings));
    const auto end = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    samplesMs.push_back(ms);
  }

  const double meanMs =
      samplesMs.empty()
          ? 0.0
          : std::accumulate(samplesMs.begin(), samplesMs.end(), 0.0) /
                static_cast<double>(samplesMs.size());
  const double flopsPerIter =
      (m > 0 && k > 0 && n > 0) ? (2.0 * m * k * n) : 0.0;
  const double flopsPerSec =
      (meanMs > 0.0) ? (flopsPerIter / meanMs) * 1000.0 : 0.0;

  std::ostringstream out;
  out << std::fixed << std::setprecision(4);
  out << "{";
  out << "\"status\":\"ok\",";
  out << "\"backend\":\"glow_opencl\",";
  out << "\"mode\":\"native\",";
  out << "\"task\":\"gemm\",";
  out << "\"device\":\"opencl\",";
  out << "\"requested_device\":\"cuda\",";
  out << "\"dtype\":\"" << json_escape(dtype) << "\",";
  out << "\"m\":" << m << ",";
  out << "\"k\":" << k << ",";
  out << "\"n\":" << n << ",";
  out << "\"iters\":" << iters << ",";
  out << "\"warmup\":" << warmup << ",";
  out << "\"latency_ms\":{";
  out << "\"mean\":" << meanMs << ",";
  out << "\"p50\":" << percentile(samplesMs, 50.0) << ",";
  out << "\"p95\":" << percentile(samplesMs, 95.0) << ",";
  out << "\"min\":" << (samplesMs.empty() ? 0.0 : *std::min_element(samplesMs.begin(), samplesMs.end())) << ",";
  out << "\"max\":" << (samplesMs.empty() ? 0.0 : *std::max_element(samplesMs.begin(), samplesMs.end()));
  out << "},";
  out << "\"throughput_tokens_per_sec\":0.0,";
  out << "\"throughput_flops_per_sec\":" << flopsPerSec << ",";
  out << "\"throughput_tflops_per_sec\":" << (flopsPerSec / 1.0e12) << ",";
  out << "\"peak_memory_bytes\":0,";
  out << "\"note\":\"Glow runtime path via OpenCL HostManager\",";
  out << "\"shape\":{\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n << "},";
  out << "\"samples_ms\":[";
  for (size_t i = 0; i < samplesMs.size(); ++i) {
    if (i) {
      out << ",";
    }
    out << samplesMs[i];
  }
  out << "]";
  out << "}";

  llvm::outs() << out.str() << "\n";
  return 0;
}
