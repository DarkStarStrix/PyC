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
    torch = None
    try:
        import torch
        import torch_tensorrt
    except Exception:
        try:
            import torch  # type: ignore[no-redef]
        except Exception:
            return emit({"status": "unavailable", "reason": "PyTorch is not installed"})
        torch_tensorrt = None

    device = os.environ.get("BENCH_DEVICE", "cuda")
    batch = int(os.environ.get("BENCH_BATCH", "64"))
    hidden = int(os.environ.get("BENCH_HIDDEN", "2048"))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))
    if device != "cuda":
        dtype = torch.float32
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden * 4, hidden),
        ).eval().to("cpu", dtype=dtype)
        proxy = torch.compile(model, backend="inductor")
        x = torch.randn(batch, hidden, device="cpu", dtype=dtype)
        samples = []
        total = warmup + iters
        with torch.no_grad():
            for i in range(total):
                start = time.perf_counter()
                y = proxy(x)
                z = y @ y.transpose(-1, -2)
                _ = z.mean()
                elapsed = (time.perf_counter() - start) * 1000.0
                if i >= warmup:
                    samples.append(elapsed)
        mean_ms = statistics.mean(samples)
        tokens = batch * hidden
        throughput = (tokens / mean_ms) * 1000.0
        return emit(
            {
                "status": "ok",
                "backend": "tensorrt_proxy_torch_compile",
                "mode": "proxy",
                "device": "cpu",
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
                "throughput_tokens_per_sec": round(throughput, 2),
                "peak_memory_bytes": 0,
                "note": "TensorRT is CUDA-only; executed deterministic torch.compile proxy on CPU.",
            }
        )

    torch.manual_seed(7)

    if not torch.cuda.is_available():
        return emit({"status": "unavailable", "reason": "CUDA not available"})

    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 4),
        torch.nn.GELU(),
        torch.nn.Linear(hidden * 4, hidden),
    ).eval().to("cuda", dtype=torch.float16)

    x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    if torch_tensorrt is None:
        trt_mod = torch.compile(model, backend="inductor")
        backend_name = "tensorrt_proxy_torch_compile"
        note = "torch_tensorrt unavailable; executed torch.compile proxy workload."
    else:
        try:
            trt_mod = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(x.shape, dtype=torch.half)],
                enabled_precisions={torch.half},
            )
            backend_name = "tensorrt"
            note = ""
        except Exception as exc:
            trt_mod = torch.compile(model, backend="inductor")
            backend_name = "tensorrt_proxy_torch_compile"
            note = f"TensorRT compile failed ({exc}); executed torch.compile proxy workload."

    samples = []
    total = warmup + iters
    with torch.no_grad():
        for i in range(total):
            torch.cuda.synchronize()
            start = time.perf_counter()
            y = trt_mod(x)
            z = y @ y.transpose(-1, -2)
            _ = z.mean()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= warmup:
                samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    throughput = (tokens / mean_ms) * 1000.0
    return emit(
        {
            "status": "ok",
            "backend": backend_name,
            "mode": "native" if backend_name == "tensorrt" else "proxy",
            "device": "cuda",
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
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "note": note,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
