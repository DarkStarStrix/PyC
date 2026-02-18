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


def run_torch_proxy(torch, device: str, batch: int, hidden: int, iters: int, warmup: int) -> dict:
    actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if actual_device == "cuda" else torch.float32
    torch.manual_seed(7)

    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 4),
        torch.nn.GELU(),
        torch.nn.Linear(hidden * 4, hidden),
    ).eval().to(device=actual_device, dtype=dtype)
    x = torch.randn(batch, hidden, device=actual_device, dtype=dtype)
    samples = []

    with torch.no_grad():
        for i in range(warmup + iters):
            if actual_device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            y = model(x)
            z = y @ y.transpose(-1, -2)
            _ = z.mean()
            if actual_device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= warmup:
                samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    return {
        "status": "ok",
        "backend": "xla_proxy_torch",
        "mode": "proxy",
        "device": actual_device,
        "requested_device": device,
        "batch": batch,
        "hidden": hidden,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "mean": round(mean_ms, 4),
            "p50": round(percentile(samples, 50), 4),
            "p95": round(percentile(samples, 95), 4),
            "min": round(min(samples), 4),
            "max": round(max(samples), 4),
        },
        "throughput_tokens_per_sec": round((tokens / mean_ms) * 1000.0, 2),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if actual_device == "cuda" else 0,
        "note": "torch_xla path unavailable; executed deterministic torch proxy workload.",
    }


def main() -> int:
    device_req = os.environ.get("BENCH_DEVICE", "cuda")
    batch = int(os.environ.get("BENCH_BATCH", "64"))
    hidden = int(os.environ.get("BENCH_HIDDEN", "2048"))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))

    try:
        import torch
    except Exception:
        return emit({"status": "unavailable", "reason": "PyTorch is not installed"})

    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        return emit(run_torch_proxy(torch, device_req, batch, hidden, iters, warmup))

    torch.manual_seed(7)
    device = xm.xla_device()
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 4),
        torch.nn.GELU(),
        torch.nn.Linear(hidden * 4, hidden),
    ).to(device=device, dtype=torch.float32)
    x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
    samples = []

    with torch.no_grad():
        for i in range(warmup + iters):
            start = time.perf_counter()
            y = model(x)
            z = y @ y.transpose(-1, -2)
            _ = z.mean()
            xm.mark_step()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= warmup:
                samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    return emit(
        {
            "status": "ok",
            "backend": "xla",
            "mode": "native",
            "device": str(device),
            "requested_device": device_req,
            "batch": batch,
            "hidden": hidden,
            "iters": iters,
            "warmup": warmup,
            "latency_ms": {
                "mean": round(mean_ms, 4),
                "p50": round(percentile(samples, 50), 4),
                "p95": round(percentile(samples, 95), 4),
                "min": round(min(samples), 4),
                "max": round(max(samples), 4),
            },
            "throughput_tokens_per_sec": round((tokens / mean_ms) * 1000.0, 2),
            "peak_memory_bytes": 0,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
