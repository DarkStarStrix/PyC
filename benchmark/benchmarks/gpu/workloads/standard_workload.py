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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def run_torch(device: str, mode: str, batch: int, hidden: int, iters: int, warmup: int) -> dict:
    import torch

    torch.manual_seed(7)
    if device == "cuda":
        torch.cuda.manual_seed_all(7)

    dtype = torch.float16 if device == "cuda" else torch.float32

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

    x = torch.randn(batch, hidden, device=device, dtype=dtype)

    times_ms: list[float] = []
    peak_mem = 0

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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
                times_ms.append(elapsed)

    if device == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated())

    mean_ms = statistics.mean(times_ms)
    tokens = batch * hidden
    throughput = (tokens / mean_ms) * 1000.0

    return {
        "status": "ok",
        "backend": mode,
        "mode": "native",
        "device": device,
        "batch": batch,
        "hidden": hidden,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "mean": round(mean_ms, 4),
            "p50": round(percentile(times_ms, 50), 4),
            "p95": round(percentile(times_ms, 95), 4),
            "min": round(min(times_ms), 4),
            "max": round(max(times_ms), 4),
        },
        "throughput_tokens_per_sec": round(throughput, 2),
        "peak_memory_bytes": peak_mem,
        "samples_ms": [round(v, 4) for v in times_ms],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch_eager", "torch_compile"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    try:
        result = run_torch(args.device, args.backend, args.batch, args.hidden, args.iters, args.warmup)
    except Exception as exc:  # noqa: BLE001
        result = {"status": "error", "backend": args.backend, "error": str(exc)}

    print(json.dumps(result))
    return 0 if result.get("status") in {"ok", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
