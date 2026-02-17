#include <stdio.h>

#include "pyc/runtime_allocator.h"

int main(void) {
    pyc_alloc_plan plan;
    pyc_alloc_stats stats;
    pyc_alloc_request a;
    pyc_alloc_request b;
    pyc_alloc_request c;

    pyc_alloc_plan_init(&plan);

    a.tensor_id = 0; a.size_bytes = 128; a.alignment = 64; a.start_step = 0; a.end_step = 2;
    b.tensor_id = 1; b.size_bytes = 64;  b.alignment = 64; b.start_step = 3; b.end_step = 4;
    c.tensor_id = 2; c.size_bytes = 256; c.alignment = 64; c.start_step = 1; c.end_step = 3;

    if (pyc_alloc_plan_add_request(&plan, a) != 0) return 1;
    if (pyc_alloc_plan_add_request(&plan, b) != 0) return 2;
    if (pyc_alloc_plan_add_request(&plan, c) != 0) return 3;

    if (pyc_alloc_plan_build_with_mode(&plan, PYC_MODE_MEMORY_FIRST, 256) != 0) return 4;
    pyc_alloc_plan_stats(&plan, &stats);

    if (stats.total_requested_bytes != (128 + 64 + 256)) return 5;
    if (stats.reused_allocations == 0) return 6;
    if (stats.peak_bytes == 0) return 7;
    if (stats.largest_allocation_bytes != 256) return 8;
    if (stats.allocation_events == 0) return 9;
    if (stats.overlap_pairs_observed == 0) return 10;
    if (stats.pressure_events == 0) return 11;
    if (stats.rematerialized_tensors == 0) return 12;

    printf("test_runtime_allocator: ok (peak=%zu reuse=%zu alloc_events=%zu overlap=%zu largest=%zu)\n",
           stats.peak_bytes,
           stats.reused_allocations,
           stats.allocation_events,
           stats.overlap_pairs_observed,
           stats.largest_allocation_bytes);
    return 0;
}
