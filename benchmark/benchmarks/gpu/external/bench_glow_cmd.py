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


def main() -> int:
    try:
        import torch
    except Exception:
        return emit({"status": "unavailable", "reason": "PyTorch not installed"})

    device_req = os.environ.get("BENCH_DEVICE", "cuda")
    batch = int(os.environ.get("BENCH_BATCH", "64"))
    hidden = int(os.environ.get("BENCH_HIDDEN", "2048"))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))

    if device_req == "cuda" and not torch.cuda.is_available():
        return emit({"status": "unavailable", "reason": "CUDA not available for Glow proxy benchmark"})

    device = "cuda" if device_req == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(7)

    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 4),
        torch.nn.GELU(),
        torch.nn.Linear(hidden * 4, hidden),
    ).eval().to(device=device, dtype=dtype)
    x = torch.randn(batch, hidden, device=device, dtype=dtype)

    samples = []
    total = warmup + iters
    with torch.no_grad():
        for i in range(total):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            y = model(x)
            z = y @ y.transpose(-1, -2)
            _ = z.mean()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= warmup:
                samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    throughput = (tokens / mean_ms) * 1000.0
    peak_mem = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return emit(
        {
            "status": "ok",
            "backend": "glow_proxy",
            "mode": "proxy",
            "device": device,
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
            "throughput_tokens_per_sec": round(throughput, 2),
            "peak_memory_bytes": peak_mem,
            "note": "Glow runtime is not linked in this environment; this is a deterministic proxy workload.",
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
