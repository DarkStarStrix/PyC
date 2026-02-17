#include "pyc/compiler_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pyc/kernel_registry.h"
#include "pyc/pass_manager.h"
#include "pyc/runtime_allocator.h"

typedef struct pyc_compiled_model {
    pyc_ir_module module;
    pyc_backend backend;
    pyc_compile_options options;
    pyc_alloc_plan alloc_plan;
    pyc_kernel_desc selected_kernel;
    pyc_kernel_selection_trace kernel_trace;
    int has_selected_kernel;
    double compile_ms;
    char decision_log[512];
} pyc_compiled_model;

static double elapsed_ms(clock_t start, clock_t end) {
    return ((double)(end - start) * 1000.0) / (double)CLOCKS_PER_SEC;
}

static pyc_compile_options default_compile_options(void) {
    pyc_compile_options o;
    memset(&o, 0, sizeof(o));
    o.enable_fusion = 1;
    o.enable_memory_reuse = 1;
    o.enable_autotune = 0;
    o.objective_mode = PYC_MODE_BALANCED;
    o.memory_budget_bytes = 0;
    o.target_utilization_floor = 0.70;
    o.deterministic_strict = 1;
    return o;
}

const char* pyc_status_string(pyc_status status) {
    switch (status) {
        case PYC_STATUS_OK: return "OK";
        case PYC_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case PYC_STATUS_VERIFY_FAILED: return "VERIFY_FAILED";
        case PYC_STATUS_COMPILE_FAILED: return "COMPILE_FAILED";
        case PYC_STATUS_RUNTIME_FAILED: return "RUNTIME_FAILED";
        default: return "UNKNOWN";
    }
}

pyc_status pyc_compile_model(const pyc_model_desc* desc, const pyc_compile_options* options, pyc_compiled_model** out_model) {
    pyc_ir_diagnostic diag;
    pyc_pass_pipeline pipeline;
    pyc_pass_report report;
    pyc_compiled_model* model;
    clock_t start;
    clock_t end;

    if (!desc || !desc->module || !out_model) {
        return PYC_STATUS_INVALID_ARGUMENT;
    }

    if (pyc_ir_verify(desc->module, &diag) != 0) {
        (void)diag;
        return PYC_STATUS_VERIFY_FAILED;
    }

    model = (pyc_compiled_model*)malloc(sizeof(*model));
    if (!model) {
        return PYC_STATUS_COMPILE_FAILED;
    }
    memset(model, 0, sizeof(*model));

    model->module = *desc->module;
    model->backend = desc->backend;
    model->options = default_compile_options();
    if (options) {
        model->options = *options;
        if (model->options.objective_mode < PYC_MODE_BALANCED ||
            model->options.objective_mode > PYC_MODE_UTILIZATION_FIRST) {
            model->options.objective_mode = PYC_MODE_BALANCED;
        }
    }

    pyc_alloc_plan_init(&model->alloc_plan);

    start = clock();

    pyc_pass_pipeline_default(&pipeline);
    if (!model->options.enable_fusion) {
        pipeline.config.fusion = 0;
    }

    if (pyc_pass_pipeline_run(&pipeline, &model->module, &report) != 0 || report.errors) {
        free(model);
        return PYC_STATUS_COMPILE_FAILED;
    }

    {
        size_t i;
        for (i = 0; i < model->module.op_count; ++i) {
            pyc_alloc_request req;
            size_t bytes = 4;
            size_t d;
            const pyc_ir_op* op = &model->module.ops[i];

            if (op->shape.rank == 0) {
                continue;
            }

            for (d = 0; d < op->shape.rank; ++d) {
                bytes *= (size_t)op->shape.dims[d];
            }

            req.tensor_id = (int)i;
            req.size_bytes = bytes;
            req.alignment = 64;
            req.start_step = (int)i;
            req.end_step = (int)(i + 2);
            pyc_alloc_plan_add_request(&model->alloc_plan, req);
        }

        if (model->options.enable_memory_reuse) {
            if (pyc_alloc_plan_build_with_mode(
                    &model->alloc_plan,
                    model->options.objective_mode,
                    model->options.memory_budget_bytes) != 0) {
                free(model);
                return PYC_STATUS_COMPILE_FAILED;
            }
        }
    }

    {
        pyc_kernel_desc selected;
        double pressure_score = model->alloc_plan.pressure_score;
        if (pyc_kernel_select_with_policy(
                "matmul_fused",
                model->backend,
                model->options.objective_mode,
                pressure_score,
                &selected,
                &model->kernel_trace) == 0) {
            model->selected_kernel = selected;
            model->has_selected_kernel = 1;
        }

        snprintf(
            model->decision_log,
            sizeof(model->decision_log),
            "mode=%d budget=%zu pressure=%.6f kernel=%s score=%.3f util=%.3f det=%d",
            (int)model->options.objective_mode,
            model->options.memory_budget_bytes,
            model->alloc_plan.pressure_score,
            model->has_selected_kernel ? model->selected_kernel.symbol : "none",
            model->kernel_trace.selected_score,
            model->kernel_trace.selected_estimated_utilization,
            model->options.deterministic_strict ? 1 : 0);
    }

    end = clock();
    model->compile_ms = elapsed_ms(start, end);

    *out_model = model;
    return PYC_STATUS_OK;
}

pyc_status pyc_run_model(pyc_compiled_model* model, const pyc_tensor* inputs, size_t input_count, pyc_tensor* outputs, size_t output_count, pyc_run_stats* out_stats) {
    clock_t start;
    clock_t end;
    pyc_alloc_stats stats;

    if (!model || !inputs || !outputs || input_count == 0 || output_count == 0) {
        return PYC_STATUS_INVALID_ARGUMENT;
    }

    start = clock();

    if (outputs[0].data && inputs[0].data && outputs[0].size_bytes >= inputs[0].size_bytes) {
        memcpy(outputs[0].data, inputs[0].data, inputs[0].size_bytes);
    }

    end = clock();

    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->compile_ms = model->compile_ms;
        out_stats->run_ms = elapsed_ms(start, end);
        pyc_alloc_plan_stats(&model->alloc_plan, &stats);
        out_stats->peak_bytes = stats.peak_bytes;
        out_stats->total_requested_bytes = stats.total_requested_bytes;
        out_stats->reused_allocations = stats.reused_allocations;
        out_stats->rematerialized_tensors = stats.rematerialized_tensors;
        out_stats->pressure_events = stats.pressure_events;
        out_stats->pressure_score = stats.pressure_score;
        out_stats->selected_kernel_count = model->has_selected_kernel ? 1 : 0;
        out_stats->selected_kernel_score = model->kernel_trace.selected_score;
        out_stats->estimated_utilization = model->kernel_trace.selected_estimated_utilization;
        if (model->has_selected_kernel) {
            strcpy(out_stats->selected_kernel_symbol, model->selected_kernel.symbol);
        }
    }

    return PYC_STATUS_OK;
}

const char* pyc_model_last_decision_log(const pyc_compiled_model* model) {
    if (!model) {
        return "";
    }
    return model->decision_log;
}

void pyc_destroy_model(pyc_compiled_model* model) {
    free(model);
}
