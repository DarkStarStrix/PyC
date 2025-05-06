#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#define MAX_WORKLOADS 3
#define MAX_BACKENDS 4
#define MAX_PATH 256
#define MAX_LINE 1024

typedef struct {
    char workload[MAX_PATH];
    char backend[MAX_PATH];
    double compile_time_s;
    double exec_time_ms;
    double mem_mb;
} Result;

void run_benchmark(const char *workload, const char *backend, Result *result) {
    char cmd[MAX_LINE];
    struct timeval start, end;
    struct rusage usage;

    // Initialize result
    strcpy(result->workload, workload);
    strcpy(result->backend, backend);
    result->compile_time_s = 0.0;
    result->exec_time_ms = 0.0;
    result->mem_mb = 0.0;

    // Measure compilation time
    snprintf(cmd, MAX_LINE, "./runners/run_%s.sh compile %s 2>/dev/null", backend, workload);
    gettimeofday(&start, NULL);
    system(cmd);
    gettimeofday(&end, NULL);
    result->compile_time_s = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    // Measure execution time and memory
    snprintf(cmd, MAX_LINE, "./runners/run_%s.sh run %s 2>/dev/null", backend, workload);
    gettimeofday(&start, NULL);
    system(cmd);
    gettimeofday(&end, NULL);
    result->exec_time_ms = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);

    // Get memory usage via measure_mem.py
    snprintf(cmd, MAX_LINE, "python3 ./tools/measure_mem.py ./runners/run_%s.sh run %s", backend, workload);
    FILE *fp = popen(cmd, "r");
    if (fp) {
        fscanf(fp, "%lf", &result->mem_mb);
        pclose(fp);
    }
}

void save_results(Result *results, int count) {
    FILE *fp = fopen("results/results.json", "w");
    if (!fp) {
        perror("Failed to open results.json");
        return;
    }

    fprintf(fp, "{\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "  \"%s\": {\n", results[i].workload);
        fprintf(fp, "    \"%s\": {\n", results[i].backend);
        fprintf(fp, "      \"compile_s\": %.2f,\n", results[i].compile_time_s);
        fprintf(fp, "      \"exec_ms\": %.2f,\n", results[i].exec_time_ms);
        fprintf(fp, "      \"mem_mb\": %.2f\n", results[i].mem_mb);
        fprintf(fp, "    }%s\n", (i % MAX_BACKENDS == MAX_BACKENDS - 1) ? "" : ",");
        fprintf(fp, "  }%s\n", (i == count - 1) ? "" : ",");
    }
    fprintf(fp, "}\n");
    fclose(fp);
}

int main() {
    const char *workloads[] = {"script.py", "ai_model.py", "cuda_kernel.cu"};
    const char *backends[] = {"pyc", "xla", "tvm", "glow"};
    Result results[MAX_WORKLOADS * MAX_BACKENDS];
    int result_count = 0;

    // Run benchmarks for each workload and backend
    for (int i = 0; i < MAX_WORKLOADS; i++) {
        for (int j = 0; j < MAX_BACKENDS; j++) {
            printf("Running %s on %s...\n", workloads[i], backends[j]);
            run_benchmark(workloads[i], backends[j], &results[result_count++]);
        }
    }

    // Save results to JSON
    save_results(results, result_count);
    printf("Results saved to results/results.json\n");

    // Generate visualization
    system("python3 ./tools/visualize.py");
    printf("Visualization generated at results/benchmark_plot.png\n");

    return 0;
}
