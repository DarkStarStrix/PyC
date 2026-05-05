#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using GemmFp16TensorCore = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3>;

using GemmBf16TensorCore = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3>;

using GemmF32Simt = cutlass::gemm::device::Gemm<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>>;

namespace {

struct Args {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int iters = 50;
    int warmup = 10;
    std::string dtype = "bfloat16";
};

bool starts_with(const char* arg, const char* prefix) {
    return std::strncmp(arg, prefix, std::strlen(prefix)) == 0;
}

bool parse_int_arg(const char* arg, const char* prefix, int* out) {
    if (!starts_with(arg, prefix)) {
        return false;
    }
    *out = std::atoi(arg + std::strlen(prefix));
    return true;
}

bool parse_str_arg(const char* arg, const char* prefix, std::string* out) {
    if (!starts_with(arg, prefix)) {
        return false;
    }
    *out = arg + std::strlen(prefix);
    return true;
}

template <typename T>
T make_value(float value);

template <>
cutlass::half_t make_value<cutlass::half_t>(float value) {
    return cutlass::half_t(value);
}

template <>
cutlass::bfloat16_t make_value<cutlass::bfloat16_t>(float value) {
    return cutlass::bfloat16_t(value);
}

template <>
float make_value<float>(float value) {
    return value;
}

template <typename T>
void fill_buffer(std::vector<T>& data, float seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = make_value<T>(seed + static_cast<float>((i % 13) - 6) * 0.03125f);
    }
}

void print_error(const char* reason) {
    std::printf("{\"status\":\"error\",\"error\":\"%s\"}\n", reason);
}

void print_unavailable(const char* reason) {
    std::printf("{\"status\":\"unavailable\",\"reason\":\"%s\"}\n", reason);
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>((p / 100.0) * static_cast<double>(values.size() - 1));
    if (idx >= values.size()) {
        idx = values.size() - 1;
    }
    return values[idx];
}

template <typename GemmOp, typename Element>
int run_tensorcore(const Args& args, const char* backend_name) {
    const size_t a_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.k);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t c_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);

    std::vector<Element> host_a(a_elems);
    std::vector<Element> host_b(b_elems);
    std::vector<Element> host_c(c_elems);
    fill_buffer(host_a, 0.25f);
    fill_buffer(host_b, -0.5f);
    fill_buffer(host_c, 0.0f);

    Element* dev_a = nullptr;
    Element* dev_b = nullptr;
    Element* dev_c = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    if (cudaMalloc(&dev_a, a_elems * sizeof(Element)) != cudaSuccess ||
        cudaMalloc(&dev_b, b_elems * sizeof(Element)) != cudaSuccess ||
        cudaMalloc(&dev_c, c_elems * sizeof(Element)) != cudaSuccess) {
        print_error("cudaMalloc failed");
        return 1;
    }

    cudaMemcpy(dev_a, host_a.data(), a_elems * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b.data(), b_elems * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c.data(), c_elems * sizeof(Element), cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    GemmOp gemm_op;
    typename GemmOp::Arguments gemm_args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {args.m, args.n, args.k},
        1,
        {1.0f, 0.0f},
        dev_a, dev_b, dev_c, dev_c,
        static_cast<int64_t>(args.m) * args.k,
        static_cast<int64_t>(args.k) * args.n,
        static_cast<int64_t>(args.m) * args.n,
        static_cast<int64_t>(args.m) * args.n,
        args.k, args.n, args.n, args.n
    );

    cutlass::Status status = gemm_op.can_implement(gemm_args);
    if (status != cutlass::Status::kSuccess) {
        print_unavailable("CUTLASS kernel cannot implement this problem");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        return 0;
    }

    for (int i = 0; i < args.warmup; ++i) {
        status = gemm_op(gemm_args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            print_error("CUTLASS warmup launch failed");
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_c);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            return 1;
        }
    }
    cudaStreamSynchronize(stream);

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(args.iters));
    for (int i = 0; i < args.iters; ++i) {
        cudaEventRecord(start, stream);
        status = gemm_op(gemm_args, nullptr, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        if (status != cutlass::Status::kSuccess) {
            print_error("CUTLASS timed launch failed");
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_c);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            return 1;
        }
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        samples_ms.push_back(static_cast<double>(elapsed_ms));
    }

    double mean_ms = 0.0;
    double min_ms = samples_ms.empty() ? 0.0 : *std::min_element(samples_ms.begin(), samples_ms.end());
    double max_ms = samples_ms.empty() ? 0.0 : *std::max_element(samples_ms.begin(), samples_ms.end());
    for (double sample : samples_ms) {
        mean_ms += sample;
    }
    if (!samples_ms.empty()) {
        mean_ms /= static_cast<double>(samples_ms.size());
    }

    const double flops = 2.0 * static_cast<double>(args.m) * static_cast<double>(args.k) * static_cast<double>(args.n);
    const double flops_per_sec = mean_ms > 0.0 ? (flops / (mean_ms / 1000.0)) : 0.0;

    std::printf(
        "{"
        "\"status\":\"ok\","
        "\"backend\":\"%s\","
        "\"mode\":\"native\","
        "\"task\":\"gemm\","
        "\"device\":\"cuda\","
        "\"requested_device\":\"cuda\","
        "\"dtype\":\"%s\","
        "\"m\":%d,"
        "\"k\":%d,"
        "\"n\":%d,"
        "\"iters\":%d,"
        "\"warmup\":%d,"
        "\"shape\":{\"m\":%d,\"k\":%d,\"n\":%d},"
        "\"latency_ms\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"min\":%.4f,\"max\":%.4f},"
        "\"throughput_tokens_per_sec\":0.0,"
        "\"throughput_flops_per_sec\":%.2f,"
        "\"throughput_tflops_per_sec\":%.6f,"
        "\"peak_memory_bytes\":%zu,"
        "\"note\":\"CUTLASS native harness result\""
        "}\n",
        backend_name,
        args.dtype.c_str(),
        args.m, args.k, args.n, args.iters, args.warmup,
        args.m, args.k, args.n,
        mean_ms,
        percentile(samples_ms, 50.0),
        percentile(samples_ms, 95.0),
        min_ms,
        max_ms,
        flops_per_sec,
        flops_per_sec / 1.0e12,
        (a_elems + b_elems + c_elems) * sizeof(Element)
    );

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return 0;
}

int run_f32(const Args& args) {
    const size_t a_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.k);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t c_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);

    std::vector<float> host_a(a_elems);
    std::vector<float> host_b(b_elems);
    std::vector<float> host_c(c_elems);
    fill_buffer(host_a, 0.25f);
    fill_buffer(host_b, -0.5f);
    fill_buffer(host_c, 0.0f);

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    if (cudaMalloc(&dev_a, a_elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_b, b_elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_c, c_elems * sizeof(float)) != cudaSuccess) {
        print_error("cudaMalloc failed");
        return 1;
    }

    cudaMemcpy(dev_a, host_a.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c.data(), c_elems * sizeof(float), cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    GemmF32Simt gemm_op;
    GemmF32Simt::Arguments gemm_args(
        {args.m, args.n, args.k},
        {dev_a, args.k},
        {dev_b, args.n},
        {dev_c, args.n},
        {dev_c, args.n},
        {1.0f, 0.0f}
    );

    cutlass::Status status = gemm_op.can_implement(gemm_args);
    if (status != cutlass::Status::kSuccess) {
        print_unavailable("CUTLASS SIMT kernel cannot implement this problem");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        return 0;
    }

    for (int i = 0; i < args.warmup; ++i) {
        status = gemm_op(gemm_args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            print_error("CUTLASS SIMT warmup launch failed");
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_c);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            return 1;
        }
    }
    cudaStreamSynchronize(stream);

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(args.iters));
    for (int i = 0; i < args.iters; ++i) {
        cudaEventRecord(start, stream);
        status = gemm_op(gemm_args, nullptr, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        if (status != cutlass::Status::kSuccess) {
            print_error("CUTLASS SIMT timed launch failed");
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_c);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            return 1;
        }
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        samples_ms.push_back(static_cast<double>(elapsed_ms));
    }

    double mean_ms = 0.0;
    double min_ms = samples_ms.empty() ? 0.0 : *std::min_element(samples_ms.begin(), samples_ms.end());
    double max_ms = samples_ms.empty() ? 0.0 : *std::max_element(samples_ms.begin(), samples_ms.end());
    for (double sample : samples_ms) {
        mean_ms += sample;
    }
    if (!samples_ms.empty()) {
        mean_ms /= static_cast<double>(samples_ms.size());
    }

    const double flops = 2.0 * static_cast<double>(args.m) * static_cast<double>(args.k) * static_cast<double>(args.n);
    const double flops_per_sec = mean_ms > 0.0 ? (flops / (mean_ms / 1000.0)) : 0.0;

    std::printf(
        "{"
        "\"status\":\"ok\","
        "\"backend\":\"cutlass_native_harness\","
        "\"mode\":\"native\","
        "\"task\":\"gemm\","
        "\"device\":\"cuda\","
        "\"requested_device\":\"cuda\","
        "\"dtype\":\"%s\","
        "\"m\":%d,"
        "\"k\":%d,"
        "\"n\":%d,"
        "\"iters\":%d,"
        "\"warmup\":%d,"
        "\"shape\":{\"m\":%d,\"k\":%d,\"n\":%d},"
        "\"latency_ms\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"min\":%.4f,\"max\":%.4f},"
        "\"throughput_tokens_per_sec\":0.0,"
        "\"throughput_flops_per_sec\":%.2f,"
        "\"throughput_tflops_per_sec\":%.6f,"
        "\"peak_memory_bytes\":%zu,"
        "\"note\":\"CUTLASS native harness result\""
        "}\n",
        args.dtype.c_str(),
        args.m, args.k, args.n, args.iters, args.warmup,
        args.m, args.k, args.n,
        mean_ms,
        percentile(samples_ms, 50.0),
        percentile(samples_ms, 95.0),
        min_ms,
        max_ms,
        flops_per_sec,
        flops_per_sec / 1.0e12,
        (a_elems + b_elems + c_elems) * sizeof(float)
    );

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (parse_int_arg(argv[i], "--m=", &args.m) ||
            parse_int_arg(argv[i], "--n=", &args.n) ||
            parse_int_arg(argv[i], "--k=", &args.k) ||
            parse_int_arg(argv[i], "--iters=", &args.iters) ||
            parse_int_arg(argv[i], "--warmup=", &args.warmup) ||
            parse_str_arg(argv[i], "--dtype=", &args.dtype)) {
            continue;
        }
    }

    if (args.m <= 0 || args.n <= 0 || args.k <= 0 || args.iters <= 0 || args.warmup < 0) {
        print_error("invalid arguments");
        return 1;
    }

    if (args.dtype == "float16" || args.dtype == "fp16" || args.dtype == "half") {
        return run_tensorcore<GemmFp16TensorCore, cutlass::half_t>(args, "cutlass_native_harness");
    }
    if (args.dtype == "bfloat16" || args.dtype == "bf16") {
        return run_tensorcore<GemmBf16TensorCore, cutlass::bfloat16_t>(args, "cutlass_native_harness");
    }
    if (args.dtype == "float32" || args.dtype == "fp32" || args.dtype == "float") {
        return run_f32(args);
    }

    print_unavailable("unsupported CUTLASS dtype");
    return 0;
}
