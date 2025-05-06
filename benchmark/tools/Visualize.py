import json
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmarks():
    with open("results/results.json", "r") as f:
        data = json.load(f)

    workloads = list(data.keys())
    backends = ["pyc", "xla", "tvm", "glow"]
    metrics = ["compile_s", "exec_ms", "mem_mb"]
    metric_labels = ["Compilation Time (s)", "Execution Time (ms)", "Memory Usage (MB)"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 8), sharex=True)
    bar_width = 0.2
    x = np.arange(len(workloads))

    for i, metric in enumerate(metrics):
        for j, backend in enumerate(backends):
            values = [data[w][backend][metric] for w in workloads]
            axes[i].bar(x + j * bar_width, values, bar_width, label=backend)
        axes[i].set_ylabel(metric_labels[i])
        axes[i].legend()
        axes[i].grid(True, axis="y", linestyle="--", alpha=0.7)

    axes[-1].set_xticks(x + bar_width * 1.5)
    axes[-1].set_xticklabels(workloads)
    plt.xlabel("Workloads")
    plt.tight_layout()
    plt.savefig("results/benchmark_plot.png")
    plt.close()

if __name__ == "__main__":
    plot_benchmarks()
    print("Benchmark results visualized and saved as 'results/benchmark_plot.png'.")
