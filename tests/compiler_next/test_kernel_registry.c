#include <stdio.h>
#include <string.h>

#include "pyc/kernel_registry.h"

int main(void) {
    pyc_kernel_desc a;
    pyc_kernel_desc b;
    pyc_kernel_desc out;
    pyc_kernel_benchmark bench;
    pyc_kernel_selection_trace trace;

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

    pyc_kernel_benchmark_read("matmul_fused", PYC_BACKEND_CPU, &bench);
    if (bench.considered < 2) return 10;
    if (bench.selected < 1) return 11;
    if (bench.best_time_ms <= 0.0) return 12;

    printf("test_kernel_registry: ok (considered=%d selected=%d best=%.3f)\n", bench.considered, bench.selected, bench.best_time_ms);
    return 0;
}
