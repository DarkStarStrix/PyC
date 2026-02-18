#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import time

import numpy as np


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if payload.get("status") in {"ok", "unavailable"} else 1


def main() -> int:
    note = ""
    try:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor
    except Exception:
        return emit({"status": "unavailable", "reason": "TVM not installed; install TVM or change TVM_BENCH_CMD"})

    device = os.environ.get("BENCH_DEVICE", "cuda")
    batch = int(os.environ.get("BENCH_BATCH", "64"))
    hidden = int(os.environ.get("BENCH_HIDDEN", "2048"))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))
    np.random.seed(7)

    x_shape = (batch, hidden)
    w1_shape = (hidden * 4, hidden)
    w2_shape = (hidden, hidden * 4)

    x = relay.var("x", shape=x_shape, dtype="float32")
    w1 = relay.var("w1", shape=w1_shape, dtype="float32")
    w2 = relay.var("w2", shape=w2_shape, dtype="float32")
    y = relay.nn.dense(x, w1)
    # Relay API varies across TVM builds; use relu for portability in baseline harness.
    y = relay.nn.relu(y)
    y = relay.nn.dense(y, w2)
    out = relay.mean(y)
    mod = tvm.IRModule.from_expr(relay.Function([x, w1, w2], out))

    if device == "cuda":
        dev = tvm.cuda(0)
        if not dev.exist:
            dev = tvm.cpu(0)
            target = "llvm"
            note = "TVM CUDA target unavailable; benchmark executed on TVM CPU fallback."
        else:
            target = "cuda"
    else:
        dev = tvm.cpu(0)
        target = "llvm"

    params = {
        "w1": np.random.randn(*w1_shape).astype("float32"),
        "w2": np.random.randn(*w2_shape).astype("float32"),
    }
    x_data = np.random.randn(*x_shape).astype("float32")

    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input("x", x_data)
    except Exception as exc:
        return emit({"status": "error", "error": f"TVM build/run init failed: {exc}"})

    samples = []
    total = warmup + iters
    for i in range(total):
        start = time.perf_counter()
        module.run()
        if device == "cuda":
            dev.sync()
        elapsed = (time.perf_counter() - start) * 1000.0
        if i >= warmup:
            samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    throughput = (tokens / mean_ms) * 1000.0
    return emit(
        {
            "status": "ok",
            "backend": "tvm",
            "mode": "proxy" if target == "llvm" and device == "cuda" else "native",
            "device": "cpu" if target == "llvm" else "cuda",
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
            "note": note,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
