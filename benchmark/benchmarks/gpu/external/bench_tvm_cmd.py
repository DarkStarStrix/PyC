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
    note_parts: list[str] = []
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
    use_cublas = os.environ.get("TVM_USE_CUBLAS", "1").strip() not in {"0", "false", "False"}
    use_fp16_cuda = os.environ.get("TVM_CUDA_FP16", "1").strip() not in {"0", "false", "False"}
    np.random.seed(7)

    x_shape = (batch, hidden)
    w1_shape = (hidden * 4, hidden)
    w2_shape = (hidden, hidden * 4)

    if device == "cuda":
        cuda_enabled = bool(tvm.runtime.enabled("cuda"))
        dev = tvm.cuda(0)
        if not cuda_enabled or not dev.exist:
            return emit(
                {
                    "status": "unavailable",
                    "reason": (
                        "TVM CUDA backend not ready "
                        f"(runtime_enabled={cuda_enabled}, device_exist={bool(dev.exist)})"
                    ),
                }
            )
        target = "cuda -libs=cublas,cudnn" if use_cublas else "cuda"
        if use_cublas:
            try:
                from tvm.relay.op.contrib.cublas import partition_for_cublas

                mod = partition_for_cublas(mod)
                note_parts.append("cublas_partition=on")
            except Exception as exc:  # noqa: BLE001
                note_parts.append(f"cublas_partition=off:{exc}")
    else:
        dev = tvm.cpu(0)
        target = "llvm"

    def build_graph(active_dtype: str):
        x = relay.var("x", shape=x_shape, dtype=active_dtype)
        w1 = relay.var("w1", shape=w1_shape, dtype=active_dtype)
        w2 = relay.var("w2", shape=w2_shape, dtype=active_dtype)
        y = relay.nn.dense(x, w1)
        # Relay API varies across TVM builds; use relu for portability in baseline harness.
        y = relay.nn.relu(y)
        y = relay.nn.dense(y, w2)
        out = relay.mean(y)
        if active_dtype == "float16":
            out = relay.cast(out, "float32")
        mod_local = tvm.IRModule.from_expr(relay.Function([x, w1, w2], out))
        if device == "cuda" and use_cublas:
            try:
                from tvm.relay.op.contrib.cublas import partition_for_cublas

                mod_local = partition_for_cublas(mod_local)
                note_parts.append("cublas_partition=on")
            except Exception as exc:  # noqa: BLE001
                note_parts.append(f"cublas_partition=off:{exc}")

        params_local = {
            "w1": np.random.randn(*w1_shape).astype(active_dtype),
            "w2": np.random.randn(*w2_shape).astype(active_dtype),
        }
        x_local = np.random.randn(*x_shape).astype(active_dtype)
        return mod_local, params_local, x_local

    candidate_dtypes = ["float16", "float32"] if (device == "cuda" and use_fp16_cuda) else ["float32"]
    module = None
    selected_dtype = None
    last_error = None
    for active_dtype in candidate_dtypes:
        try:
            mod, params, x_data = build_graph(active_dtype)
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
            module = graph_executor.GraphModule(lib["default"](dev))
            module.set_input("x", x_data)
            selected_dtype = active_dtype
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            note_parts.append(f"build_failed_dtype={active_dtype}")
            continue

    if module is None:
        return emit({"status": "error", "error": f"TVM build/run init failed: {last_error}"})

    note_parts.insert(0, f"dtype={selected_dtype}")

    samples = []
    total = warmup + iters
    def full_device_sync() -> None:
        if device == "cuda":
            dev.sync()

    for i in range(total):
        full_device_sync()
        start = time.perf_counter()
        module.run()
        full_device_sync()
        end = time.perf_counter()
        full_device_sync()
        elapsed = (end - start) * 1000.0
        if i >= warmup:
            samples.append(elapsed)

    mean_ms = statistics.mean(samples)
    tokens = batch * hidden
    throughput = (tokens / mean_ms) * 1000.0
    return emit(
        {
            "status": "ok",
            "backend": "tvm",
            "mode": "native",
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
            "note": "; ".join(note_parts),
            "samples_ms": [round(v, 4) for v in samples],
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
