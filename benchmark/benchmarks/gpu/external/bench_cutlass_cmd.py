#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if payload.get("status") in {"ok", "unavailable"} else 1


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def pick_field(row: dict[str, str], *candidates: str) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return ""


def read_cutlass_csv(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    successful = []
    for row in rows:
        disposition = pick_field(row, "Disposition", "status", "verification")
        if disposition and disposition.lower() not in {"passed", "success", "ok"}:
            continue
        provider = pick_field(row, "Provider", "provider", "Operation Provider")
        if provider and provider.lower() not in {"cutlass", "cutlass profiler"}:
            continue
        runtime_ms = to_float(pick_field(row, "Runtime", "runtime", "runtime_ms"), default=-1.0)
        gflops = to_float(pick_field(row, "GFLOPs", "gflops_per_sec", "math"), default=0.0)
        if runtime_ms <= 0:
            continue
        successful.append({"runtime_ms": runtime_ms, "gflops": gflops, "provider": provider, "row": row})
    if not successful:
        return None
    best = min(successful, key=lambda item: item["runtime_ms"])
    return best


def main() -> int:
    task = os.environ.get("BENCH_TASK", "mlp").strip().lower() or "mlp"
    if task != "gemm":
        return emit({"status": "unavailable", "reason": "CUTLASS harness currently supports GEMM-only benchmarking"})

    device = os.environ.get("BENCH_DEVICE", "cuda").strip().lower() or "cuda"
    if device != "cuda":
        return emit({"status": "unavailable", "reason": "CUTLASS profiler is CUDA-only"})

    m = int(os.environ.get("BENCH_M", os.environ.get("BENCH_BATCH", "64")))
    k = int(os.environ.get("BENCH_K", os.environ.get("BENCH_HIDDEN", "2048")))
    n = int(os.environ.get("BENCH_N", os.environ.get("BENCH_HIDDEN", "2048")))
    dtype = os.environ.get("BENCH_DTYPE", "float32").strip().lower() or "float32"
    native_harness = os.environ.get("CUTLASS_NATIVE_BENCH_BIN", "").strip() or shutil.which("cutlass_gemm_bench")
    if native_harness:
        cmd = [
            native_harness,
            f"--m={m}",
            f"--n={n}",
            f"--k={k}",
            f"--dtype={dtype}",
            f"--iters={max(1, int(os.environ.get('BENCH_ITERS', '80')))}",
            f"--warmup={max(0, int(os.environ.get('BENCH_WARMUP', '20')))}",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0 and not proc.stdout.strip():
            return emit(
                {
                    "status": "error",
                    "error": "CUTLASS native harness failed",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
        try:
            payload = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return emit(
                {
                    "status": "error",
                    "error": "invalid JSON from CUTLASS native harness",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
        return emit(payload)

    profiler = shutil.which("cutlass_profiler")
    if profiler is None:
        return emit({"status": "unavailable", "reason": "cutlass_profiler or cutlass_gemm_bench not found in PATH"})

    if dtype in {"float16", "fp16", "half"}:
        element_a = "f16"
        element_b = "f16"
        element_c = "f32"
        element_accumulator = "f32"
    elif dtype in {"bfloat16", "bf16"}:
        element_a = "bf16"
        element_b = "bf16"
        element_c = "f32"
        element_accumulator = "f32"
    elif dtype in {"float32", "fp32", "float"}:
        element_a = "f32"
        element_b = "f32"
        element_c = "f32"
        element_accumulator = "f32"
    else:
        return emit({"status": "unavailable", "reason": f"unsupported CUTLASS dtype: {dtype}"})

    with tempfile.TemporaryDirectory(prefix="pyc_cutlass_") as tmp:
        output_stem = Path(tmp) / "cutlass_report"
        cmd = [
            profiler,
            "--operation=gemm",
            f"--m={m}",
            f"--n={n}",
            f"--k={k}",
            f"--A={element_a}:row",
            f"--B={element_b}:row",
            f"--C={element_c}:row",
            f"--accum={element_accumulator}",
            "--profiling-iterations=5",
            "--warmup-iterations=2",
            "--verbose=false",
            f"--output={output_stem}",
            f"--tags=pyc_shape:{m}x{k}x{n},pyc_dtype:{dtype}",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        csv_path = Path(f"{output_stem}.gemm.csv")
        parsed = read_cutlass_csv(csv_path)
        if parsed is None:
            return emit(
                {
                    "status": "unavailable",
                    "reason": "CUTLASS profiler produced no successful GEMM rows",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )

        runtime_ms = parsed["runtime_ms"]
        gflops = parsed["gflops"]
        if gflops <= 0:
            flops_per_iter = float(2 * m * k * n)
            gflops = (flops_per_iter / runtime_ms) * 1.0e-6 if runtime_ms > 0 else 0.0
        samples = [runtime_ms]
        mean_ms = statistics.mean(samples)
        flops_per_sec = gflops * 1.0e9
        return emit(
            {
                "status": "ok",
                "backend": "cutlass_profiler",
                "mode": "native",
                "task": "gemm",
                "device": "cuda",
                "requested_device": device,
                "dtype": dtype,
                "m": m,
                "k": k,
                "n": n,
                "iters": 1,
                "warmup": 0,
                "shape": {"m": m, "k": k, "n": n},
                "latency_ms": {
                    "mean": round(mean_ms, 4),
                    "p50": round(percentile(samples, 50), 4),
                    "p95": round(percentile(samples, 95), 4),
                    "min": round(min(samples), 4),
                    "max": round(max(samples), 4),
                },
                "throughput_tokens_per_sec": 0.0,
                "throughput_flops_per_sec": round(flops_per_sec, 2),
                "throughput_tflops_per_sec": round(flops_per_sec / 1.0e12, 4),
                "peak_memory_bytes": 0,
                "note": "CUTLASS profiler harness result",
                "samples_ms": [round(v, 4) for v in samples],
                "cutlass_csv": str(csv_path),
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
