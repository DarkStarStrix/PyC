#!/bin/bash

mkdir -p results
gcc benchmark.c -o benchmark
./benchmark
echo "Benchmarking complete. Results in results/results.json and visualization in results/benchmark_plot.png"
# shellcheck disable=SC1073
python3 -c "