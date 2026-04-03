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
    task = os.environ.get("BENCH_TASK", "mlp").strip() or "mlp"
    batch = int(os.environ.get("BENCH_BATCH", "64"))
    hidden = int(os.environ.get("BENCH_HIDDEN", "2048"))
    m = int(os.environ.get("BENCH_M", str(batch)))
    k = int(os.environ.get("BENCH_K", str(hidden)))
    n = int(os.environ.get("BENCH_N", str(hidden)))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))
    if device != "cuda":
        dtype = torch.float32
        if task == "gemm":
            model = torch.nn.Linear(k, n, bias=False).eval().to("cpu", dtype=dtype)
        else:
            model = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden * 4),
                torch.nn.GELU(),
                torch.nn.Linear(hidden * 4, hidden),
            ).eval().to("cpu", dtype=dtype)
        proxy = torch.compile(model, backend="inductor")
        x = torch.randn(m, k, device="cpu", dtype=dtype) if task == "gemm" else torch.randn(batch, hidden, device="cpu", dtype=dtype)
        samples = []
        total = warmup + iters
        with torch.no_grad():
            for i in range(total):
                start = time.perf_counter()
                y = proxy(x)
                _ = y.sum() if task == "gemm" else (y @ y.transpose(-1, -2)).mean()
                elapsed = (time.perf_counter() - start) * 1000.0
                if i >= warmup:
                    samples.append(elapsed)
        mean_ms = statistics.mean(samples)
        tokens = batch * hidden
        flops_per_iter = 2 * m * k * n if task == "gemm" else 16 * batch * hidden * hidden + 2 * batch * batch * hidden
        throughput = (flops_per_iter / mean_ms) * 1000.0
        return emit(
            {
                "status": "ok",
                "backend": "tensorrt_proxy_torch_compile",
                "mode": "proxy",
                "task": task,
                "device": "cpu",
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
                "throughput_flops_per_sec": round(throughput, 2) if task == "gemm" else 0.0,
                "throughput_tflops_per_sec": round(throughput / 1.0e12, 4) if task == "gemm" else 0.0,
                "peak_memory_bytes": 0,
                "note": "TensorRT is CUDA-only; executed deterministic torch.compile proxy on CPU." + (" GEMM task" if task == "gemm" else ""),
                "samples_ms": [round(v, 4) for v in samples],
            }
        )

    torch.manual_seed(7)

    if not torch.cuda.is_available():
        return emit({"status": "unavailable", "reason": "CUDA not available"})

    if task == "gemm":
        model = torch.nn.Linear(k, n, bias=False).eval().to("cuda", dtype=torch.float16)
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    else:
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
    def full_cuda_sync() -> None:
        torch.cuda.synchronize()

    with torch.no_grad():
        for i in range(total):
            full_cuda_sync()
            start = time.perf_counter()
            y = trt_mod(x)
            _ = y.sum() if task == "gemm" else (y @ y.transpose(-1, -2)).mean()
            full_cuda_sync()
            end = time.perf_counter()
            full_cuda_sync()
            elapsed = (end - start) * 1000.0
            if i >= warmup:
                samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    flops_per_iter = 2 * m * k * n if task == "gemm" else 16 * batch * hidden * hidden + 2 * batch * batch * hidden
    throughput = (flops_per_iter / mean_ms) * 1000.0
    return emit(
        {
            "status": "ok",
            "backend": backend_name,
            "mode": "native" if backend_name == "tensorrt" else "proxy",
            "task": task,
            "device": "cuda",
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
            "throughput_flops_per_sec": round(throughput, 2) if task == "gemm" else 0.0,
            "throughput_tflops_per_sec": round(throughput / 1.0e12, 4) if task == "gemm" else 0.0,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "note": note + (" GEMM task" if task == "gemm" else ""),
            "samples_ms": [round(v, 4) for v in samples],
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
