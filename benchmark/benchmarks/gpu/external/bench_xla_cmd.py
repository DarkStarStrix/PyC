#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import time


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if payload.get("status") in {"ok", "unavailable"} else 1


def run_jax_native(
    device_req: str,
    task: str,
    batch: int,
    hidden: int,
    m: int,
    k: int,
    n: int,
    iters: int,
    warmup: int,
) -> dict | None:
    try:
        import jax
        import jax.nn as jnn
        import jax.numpy as jnp
        import numpy as np
    except Exception:
        return None

    try:
        gpu_devices = list(jax.devices("gpu"))
    except Exception:
        gpu_devices = []
    try:
        cpu_devices = list(jax.devices("cpu"))
    except Exception:
        cpu_devices = []

    if device_req == "cuda":
        if not gpu_devices:
            return None
        target_device = gpu_devices[0]
    else:
        # Keep CPU benchmark honest: never silently route to GPU.
        if not cpu_devices:
            return {
                "status": "unavailable",
                "reason": "JAX CPU device is unavailable for BENCH_DEVICE=cpu",
            }
        target_device = cpu_devices[0]

    rng = np.random.default_rng(7)
    if task == "gemm":
        lhs = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32))
        rhs = jnp.asarray(rng.standard_normal((k, n), dtype=np.float32))
        lhs = jax.device_put(lhs, target_device)
        rhs = jax.device_put(rhs, target_device)

        def fn(a, b):
            return jnp.matmul(a, b)

        jit_fn = jax.jit(fn)
    else:
        x = jnp.asarray(rng.standard_normal((batch, hidden), dtype=np.float32))
        w1 = jnp.asarray(rng.standard_normal((hidden, hidden * 4), dtype=np.float32))
        w2 = jnp.asarray(rng.standard_normal((hidden * 4, hidden), dtype=np.float32))
        x = jax.device_put(x, target_device)
        w1 = jax.device_put(w1, target_device)
        w2 = jax.device_put(w2, target_device)

        def fn(lhs, a, b):
            y = jnp.matmul(lhs, a)
            y = jnn.gelu(y)
            y = jnp.matmul(y, b)
            z = jnp.matmul(y, jnp.transpose(y))
            return jnp.mean(z)

        jit_fn = jax.jit(fn)
    def full_device_sync() -> None:
        # Force a device barrier on the selected backend without mutating workload tensors.
        _ = jax.device_put(0.0, target_device)
        _.block_until_ready()

    # Trigger compilation before timing loop for deterministic warmup.
    if task == "gemm":
        _ = jit_fn(lhs, rhs).block_until_ready()
    else:
        _ = jit_fn(x, w1, w2).block_until_ready()

    samples = []
    total = warmup + iters
    for i in range(total):
        full_device_sync()
        start = time.perf_counter()
        out = jit_fn(lhs, rhs) if task == "gemm" else jit_fn(x, w1, w2)
        out.block_until_ready()
        end = time.perf_counter()
        full_device_sync()
        elapsed = (end - start) * 1000.0
        if i >= warmup:
            samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    flops_per_iter = 2 * m * k * n if task == "gemm" else 16 * batch * hidden * hidden + 2 * batch * batch * hidden
    flops_per_sec = (flops_per_iter / mean_ms) * 1000.0
    return {
        "status": "ok",
        "backend": "xla_jax",
        "mode": "native",
        "task": task,
        "device": target_device.platform,
        "requested_device": device_req,
        "batch": batch,
        "hidden": hidden,
        "m": m,
        "k": k,
        "n": n,
        "iters": iters,
        "warmup": warmup,
        "shape": {"m": m, "k": k, "n": n} if task == "gemm" else {"batch": batch, "hidden": hidden},
        "latency_ms": {
            "mean": round(mean_ms, 4),
            "p50": round(percentile(samples, 50), 4),
            "p95": round(percentile(samples, 95), 4),
            "min": round(min(samples), 4),
            "max": round(max(samples), 4),
        },
        "throughput_tokens_per_sec": round((tokens / mean_ms) * 1000.0, 2) if task != "gemm" else 0.0,
        "throughput_flops_per_sec": round(flops_per_sec, 2) if task == "gemm" else 0.0,
        "throughput_tflops_per_sec": round(flops_per_sec / 1.0e12, 4) if task == "gemm" else 0.0,
        "peak_memory_bytes": 0,
        "note": "Native XLA path via JAX/XLA backend." + (" GEMM task" if task == "gemm" else ""),
        "samples_ms": [round(v, 4) for v in samples],
    }


def run_torch_proxy(
    torch,
    device: str,
    task: str,
    batch: int,
    hidden: int,
    m: int,
    k: int,
    n: int,
    iters: int,
    warmup: int,
) -> dict:
    actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if actual_device == "cuda" else torch.float32
    torch.manual_seed(7)

    if task == "gemm":
        model = torch.nn.Linear(k, n, bias=False).eval().to(device=actual_device, dtype=dtype)
        x = torch.randn(m, k, device=actual_device, dtype=dtype)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden * 4, hidden),
        ).eval().to(device=actual_device, dtype=dtype)
        x = torch.randn(batch, hidden, device=actual_device, dtype=dtype)
    samples = []

    def full_device_sync() -> None:
        if actual_device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for i in range(warmup + iters):
            full_device_sync()
            start = time.perf_counter()
            y = model(x)
            _ = y.sum() if task == "gemm" else (y @ y.transpose(-1, -2)).mean()
            full_device_sync()
            end = time.perf_counter()
            full_device_sync()
            elapsed = (end - start) * 1000.0
            if i >= warmup:
                samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    flops_per_iter = 2 * m * k * n if task == "gemm" else 16 * batch * hidden * hidden + 2 * batch * batch * hidden
    flops_per_sec = (flops_per_iter / mean_ms) * 1000.0
    return {
        "status": "ok",
        "backend": "xla_proxy_torch",
        "mode": "proxy",
        "task": task,
        "device": actual_device,
        "requested_device": device,
        "batch": batch,
        "hidden": hidden,
        "m": m,
        "k": k,
        "n": n,
        "iters": iters,
        "warmup": warmup,
        "shape": {"m": m, "k": k, "n": n} if task == "gemm" else {"batch": batch, "hidden": hidden},
        "latency_ms": {
            "mean": round(mean_ms, 4),
            "p50": round(percentile(samples, 50), 4),
            "p95": round(percentile(samples, 95), 4),
            "min": round(min(samples), 4),
            "max": round(max(samples), 4),
        },
        "throughput_tokens_per_sec": round((tokens / mean_ms) * 1000.0, 2) if task != "gemm" else 0.0,
        "throughput_flops_per_sec": round(flops_per_sec, 2) if task == "gemm" else 0.0,
        "throughput_tflops_per_sec": round(flops_per_sec / 1.0e12, 4) if task == "gemm" else 0.0,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if actual_device == "cuda" else 0,
        "note": "torch_xla path unavailable; executed deterministic torch proxy workload." + (" GEMM task" if task == "gemm" else ""),
        "samples_ms": [round(v, 4) for v in samples],
    }


def main() -> int:
    device_req = os.environ.get("BENCH_DEVICE", "cuda")
    task = os.environ.get("BENCH_TASK", "mlp").strip() or "mlp"
    batch = int(os.environ.get("BENCH_BATCH", "64"))
    hidden = int(os.environ.get("BENCH_HIDDEN", "2048"))
    m = int(os.environ.get("BENCH_M", str(batch)))
    k = int(os.environ.get("BENCH_K", str(hidden)))
    n = int(os.environ.get("BENCH_N", str(hidden)))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))

    jax_payload = run_jax_native(device_req, task, batch, hidden, m, k, n, iters, warmup)
    if jax_payload is not None:
        return emit(jax_payload)

    try:
        import torch
    except Exception:
        return emit({"status": "unavailable", "reason": "PyTorch is not installed and JAX native path unavailable"})

    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        return emit(run_torch_proxy(torch, device_req, task, batch, hidden, m, k, n, iters, warmup))

    torch.manual_seed(7)
    device = xm.xla_device()
    if task == "gemm":
        model = torch.nn.Linear(k, n, bias=False).to(device=device, dtype=torch.float32)
        x = torch.randn(m, k, device=device, dtype=torch.float32)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden * 4, hidden),
        ).to(device=device, dtype=torch.float32)
        x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    samples = []
    def full_xla_sync() -> None:
        wait_fn = getattr(xm, "wait_device_ops", None)
        if callable(wait_fn):
            wait_fn()
        else:
            xm.mark_step()

    with torch.no_grad():
        for i in range(warmup + iters):
            full_xla_sync()
            start = time.perf_counter()
            y = model(x)
            _ = y.sum() if task == "gemm" else (y @ y.transpose(-1, -2)).mean()
            xm.mark_step()
            full_xla_sync()
            end = time.perf_counter()
            full_xla_sync()
            elapsed = (end - start) * 1000.0
            if i >= warmup:
                samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    flops_per_iter = 2 * m * k * n if task == "gemm" else 16 * batch * hidden * hidden + 2 * batch * batch * hidden
    flops_per_sec = (flops_per_iter / mean_ms) * 1000.0
    return emit(
        {
            "status": "ok",
            "backend": "xla",
            "mode": "native",
            "task": task,
            "device": str(device),
            "requested_device": device_req,
            "batch": batch,
            "hidden": hidden,
            "m": m,
            "k": k,
            "n": n,
            "iters": iters,
            "warmup": warmup,
            "shape": {"m": m, "k": k, "n": n} if task == "gemm" else {"batch": batch, "hidden": hidden},
            "latency_ms": {
                "mean": round(mean_ms, 4),
                "p50": round(percentile(samples, 50), 4),
                "p95": round(percentile(samples, 95), 4),
                "min": round(min(samples), 4),
                "max": round(max(samples), 4),
            },
            "throughput_tokens_per_sec": round((tokens / mean_ms) * 1000.0, 2) if task != "gemm" else 0.0,
            "throughput_flops_per_sec": round(flops_per_sec, 2) if task == "gemm" else 0.0,
            "throughput_tflops_per_sec": round(flops_per_sec / 1.0e12, 4) if task == "gemm" else 0.0,
            "peak_memory_bytes": 0,
            "note": "Native XLA path via JAX/XLA backend." + (" GEMM task" if task == "gemm" else ""),
            "samples_ms": [round(v, 4) for v in samples],
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
