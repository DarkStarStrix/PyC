#!/usr/bin/env python3
"""Deterministic GPU workload for backend benchmarking.

Runs a fixed-shape MLP-style workload with optional backend mode and emits JSON
metrics on stdout so wrappers can aggregate results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def resolve_torch_dtype(torch, device: str, dtype_name: str):
    requested = (dtype_name or "auto").strip().lower()
    if requested in {"", "auto"}:
        requested = "float16" if device == "cuda" else "float32"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if requested not in mapping:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    return requested, mapping[requested]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def run_torch(device: str, mode: str, batch: int, hidden: int, iters: int, warmup: int, dtype_name: str) -> dict:
    import torch

    torch.manual_seed(7)
    if device == "cuda":
        torch.cuda.manual_seed_all(7)

    def full_device_sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    dtype_label, dtype = resolve_torch_dtype(torch, device, dtype_name)

    setup_start = time.perf_counter()
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 4),
        torch.nn.GELU(),
        torch.nn.Linear(hidden * 4, hidden),
    ).to(device=device, dtype=dtype)

    if mode == "torch_compile":
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            return {"status": "unavailable", "reason": "torch.compile not available"}
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    x = torch.randn(batch, hidden, device=device, dtype=dtype)

    times_ms: list[float] = []
    peak_mem = 0

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    first_step_ms = 0.0
    with torch.no_grad():
        full_device_sync()
        first_start = time.perf_counter()
        y = model(x)
        z = y @ y.transpose(-1, -2)
        _ = z.mean()
        full_device_sync()
        first_end = time.perf_counter()
        full_device_sync()
        first_step_ms = (first_end - first_start) * 1000.0

    total = warmup + iters
    with torch.no_grad():
        for i in range(total):
            full_device_sync()
            start = time.perf_counter()
            y = model(x)
            z = y @ y.transpose(-1, -2)
            _ = z.mean()
            full_device_sync()
            end = time.perf_counter()
            full_device_sync()
            elapsed = (end - start) * 1000.0
            if i >= warmup:
                times_ms.append(elapsed)

    if device == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated())

    mean_ms = statistics.mean(times_ms)
    p50_ms = percentile(times_ms, 50)
    p95_ms = percentile(times_ms, 95)
    tokens = batch * hidden
    throughput = (tokens / mean_ms) * 1000.0
    compile_overhead_ms = max(0.0, first_step_ms - p50_ms)

    return {
        "status": "ok",
        "backend": mode,
        "task": "mlp",
        "mode": "native",
        "device": device,
        "dtype": dtype_label,
        "batch": batch,
        "hidden": hidden,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "mean": round(mean_ms, 4),
            "p50": round(p50_ms, 4),
            "p95": round(p95_ms, 4),
            "min": round(min(times_ms), 4),
            "max": round(max(times_ms), 4),
        },
        "startup_ms": {
            "setup": round(setup_ms, 4),
            "first_step": round(first_step_ms, 4),
            "total": round(setup_ms + first_step_ms, 4),
            "compile_overhead_estimate": round(compile_overhead_ms, 4),
        },
        "throughput_tokens_per_sec": round(throughput, 2),
        "peak_memory_bytes": peak_mem,
        "samples_ms": [round(v, 4) for v in times_ms],
    }


def run_torch_gemm(
    device: str,
    mode: str,
    m: int,
    k: int,
    n: int,
    iters: int,
    warmup: int,
    dtype_name: str,
) -> dict:
    import torch

    torch.manual_seed(7)
    if device == "cuda":
        torch.cuda.manual_seed_all(7)

    def full_device_sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    dtype_label, dtype = resolve_torch_dtype(torch, device, dtype_name)

    setup_start = time.perf_counter()
    lhs = torch.randn(m, k, device=device, dtype=dtype)
    rhs = torch.randn(k, n, device=device, dtype=dtype)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    times_ms: list[float] = []
    peak_mem = 0

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    first_step_ms = 0.0
    with torch.no_grad():
        full_device_sync()
        first_start = time.perf_counter()
        out = lhs @ rhs
        _ = out.sum()
        full_device_sync()
        first_end = time.perf_counter()
        full_device_sync()
        first_step_ms = (first_end - first_start) * 1000.0

    total = warmup + iters
    with torch.no_grad():
        for i in range(total):
            full_device_sync()
            start = time.perf_counter()
            out = lhs @ rhs
            _ = out.sum()
            full_device_sync()
            end = time.perf_counter()
            full_device_sync()
            elapsed = (end - start) * 1000.0
            if i >= warmup:
                times_ms.append(elapsed)

    if device == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated())

    mean_ms = statistics.mean(times_ms)
    p50_ms = percentile(times_ms, 50)
    p95_ms = percentile(times_ms, 95)
    flops_per_iter = float(2 * m * k * n)
    throughput_flops_per_sec = (flops_per_iter / mean_ms) * 1000.0 if mean_ms > 0 else 0.0
    compile_overhead_ms = max(0.0, first_step_ms - p50_ms)

    return {
        "status": "ok",
        "backend": mode,
        "task": "gemm",
        "mode": "native",
        "device": device,
        "dtype": dtype_label,
        "m": m,
        "k": k,
        "n": n,
        "shape": {"m": m, "k": k, "n": n},
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "mean": round(mean_ms, 4),
            "p50": round(p50_ms, 4),
            "p95": round(p95_ms, 4),
            "min": round(min(times_ms), 4),
            "max": round(max(times_ms), 4),
        },
        "startup_ms": {
            "setup": round(setup_ms, 4),
            "first_step": round(first_step_ms, 4),
            "total": round(setup_ms + first_step_ms, 4),
            "compile_overhead_estimate": round(compile_overhead_ms, 4),
        },
        "throughput_tokens_per_sec": 0.0,
        "throughput_flops_per_sec": round(throughput_flops_per_sec, 2),
        "throughput_tflops_per_sec": round(throughput_flops_per_sec / 1.0e12, 4),
        "peak_memory_bytes": peak_mem,
        "samples_ms": [round(v, 4) for v in times_ms],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch_eager", "torch_compile"], required=True)
    parser.add_argument("--task", choices=["mlp", "gemm"], default="mlp")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    try:
        if args.task == "gemm":
            m = args.m if args.m > 0 else args.batch
            k = args.k if args.k > 0 else args.hidden
            n = args.n if args.n > 0 else args.hidden
            result = run_torch_gemm(args.device, args.backend, m, k, n, args.iters, args.warmup, args.dtype)
        else:
            result = run_torch(args.device, args.backend, args.batch, args.hidden, args.iters, args.warmup, args.dtype)
    except Exception as exc:  # noqa: BLE001
        result = {"status": "error", "backend": args.backend, "error": str(exc)}

    print(json.dumps(result))
    return 0 if result.get("status") in {"ok", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
