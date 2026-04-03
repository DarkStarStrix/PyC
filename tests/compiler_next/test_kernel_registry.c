#include <stdio.h>
#include <string.h>

#include "pyc/kernel_registry.h"

int main(void) {
    pyc_kernel_desc a;
    pyc_kernel_desc b;
    pyc_kernel_desc out;
    pyc_kernel_benchmark bench;
    pyc_kernel_selection_trace trace;
    pyc_kernel_coselect_context context;

    pyc_kernel_registry_reset();

    memset(&a, 0, sizeof(a));
    strcpy(a.op_key, "matmul_fused");
    a.backend = PYC_BACKEND_CPU;
    strcpy(a.symbol, "kernel_v1");
    a.priority = 5;
    a.estimated_occupancy = 0.50;
    a.shared_mem_bytes = 48 * 1024;
    a.reg_pressure_class = 3;

    memset(&b, 0, sizeof(b));
    strcpy(b.op_key, "matmul_fused");
    b.backend = PYC_BACKEND_CPU;
    strcpy(b.symbol, "kernel_v2");
    b.priority = 10;
    b.estimated_occupancy = 0.90;
    b.shared_mem_bytes = 4 * 1024;
    b.reg_pressure_class = 1;

    if (pyc_kernel_register(&a) != 0) return 1;
    if (pyc_kernel_register(&b) != 0) return 2;

    if (pyc_kernel_select("matmul_fused", PYC_BACKEND_CPU, &out) != 0) return 3;
    if (strcmp(out.symbol, "kernel_v2") != 0) return 4;

    if (pyc_kernel_benchmark_update("matmul_fused", PYC_BACKEND_CPU, 1.2) != 0) return 5;
    if (pyc_kernel_benchmark_update("matmul_fused", PYC_BACKEND_CPU, 0.9) != 0) return 6;
    if (pyc_kernel_select_with_policy("matmul_fused", PYC_BACKEND_CPU, PYC_MODE_MEMORY_FIRST, 4.0, &out, &trace) != 0) return 7;
    if (strcmp(out.symbol, "kernel_v2") != 0) return 8;
    if (trace.selected_score == 0.0) return 9;

    memset(&context, 0, sizeof(context));
    context.pressure_score = 2.5;
    context.pressure_events = 2;
    context.rematerialized_tensors = 3;
    context.reused_allocations = 1;
    context.total_requested_bytes = 4096;
    context.peak_bytes = 4096;
    context.memory_budget_bytes = 2048;
    a.priority = 20;
    strcpy(a.symbol, "kernel_co_high_pressure");
    a.shared_mem_bytes = 96 * 1024;
    a.reg_pressure_class = 4;
    a.estimated_occupancy = 0.98;
    if (pyc_kernel_register(&a) != 0) return 13;
    b.priority = 14;
    strcpy(b.symbol, "kernel_co_memory_friendly");
    b.shared_mem_bytes = 2 * 1024;
    b.reg_pressure_class = 1;
    b.estimated_occupancy = 0.72;
    if (pyc_kernel_register(&b) != 0) return 14;
    if (pyc_kernel_select_with_policy("matmul_fused", PYC_BACKEND_CPU, PYC_MODE_BALANCED, context.pressure_score, &out, NULL) != 0) return 15;
    if (strcmp(out.symbol, "kernel_co_high_pressure") != 0) return 16;
    if (pyc_kernel_coselect_with_context("matmul_fused", PYC_BACKEND_CPU, PYC_MODE_MEMORY_FIRST, &context, &out, &trace) != 0) return 17;
    if (strcmp(out.symbol, "kernel_co_memory_friendly") != 0) return 18;
    if (trace.allocator_penalty <= 0.0) return 19;
    if (trace.candidates_considered < 4) return 20;

    pyc_kernel_benchmark_read("matmul_fused", PYC_BACKEND_CPU, &bench);
    if (bench.considered < 2) return 10;
    if (bench.selected < 1) return 11;
    if (bench.best_time_ms <= 0.0) return 12;

    printf("test_kernel_registry: ok (considered=%d selected=%d best=%.3f)\n", bench.considered, bench.selected, bench.best_time_ms);
    return 0;
}
