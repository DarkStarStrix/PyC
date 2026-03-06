#!/usr/bin/env python3
"""Ground-truth runtime validation for deterministic encoder-style inference.

This tool is intentionally focused on observability, not further optimization:
- strict device synchronization around timing boundaries
- compile/graph construction counters
- post-warmup memory stabilization checks
- shape bucket and batch scaling measurements
"""

from __future__ import annotations

import argparse
import json
import os
import time


def mean(values):
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


class Counters:
    def __init__(self):
        self.compile_calls = 0
        self.graph_build_calls = 0


def parse_csv_ints(value: str) -> list[int]:
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        items.append(int(raw))
    return items


def run() -> int:
    parser = argparse.ArgumentParser(description="Deterministic encoder runtime ground-truth validation")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--bucket-lengths", default="128,256,512")
    parser.add_argument("--batch-sizes", default="16,32,64,96")
    parser.add_argument("--single-pass-after-warmup", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--inductor-gemm-backends", default="")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--disable-inference-mode", action="store_true")
    parser.add_argument("--use-inplace-residual", action="store_true")
    parser.add_argument("--pad-hidden-to", type=int, default=0)
    parser.add_argument("--use-cuda-graph-arena", action="store_true")
    parser.add_argument("--single-shape-seq-len", type=int, default=0)
    parser.add_argument("--skip-batch-scaling", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "error", "error": f"PyTorch unavailable: {exc}"}
        print(json.dumps(payload))
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        payload = {"status": "error", "error": "CUDA requested but not available"}
        print(json.dumps(payload))
        return 1

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    if args.device == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    torch.manual_seed(7)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(7)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    hidden = args.hidden
    if args.pad_hidden_to and args.pad_hidden_to > 0:
        align = args.pad_hidden_to
        hidden = ((hidden + align - 1) // align) * align

    counters = Counters()
    compile_config = {
        "backend": args.compile_backend,
        "inductor_gemm_backends": args.inductor_gemm_backends,
        "inductor_gemm_backends_applied": False,
    }

    class EncoderBlock(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.lin1 = nn.Linear(hidden, hidden * 4)
            self.gelu = nn.GELU()
            self.lin2 = nn.Linear(hidden * 4, hidden)
            self.norm = nn.LayerNorm(hidden)

        def forward(self, x):  # type: ignore[override]
            y = self.lin1(x)
            y = self.gelu(y)
            y = self.lin2(y)
            if args.use_inplace_residual:
                y.add_(x)
                y = self.norm(y)
            else:
                y = self.norm(y + x)
            pooled = y.mean(dim=1)
            score = pooled @ pooled.transpose(-1, -2)
            return score

    def full_sync() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    def memory_snapshot() -> dict:
        if args.device != "cuda":
            return {
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "peak_allocated_bytes": 0,
                "allocation_events": 0,
                "segment_alloc_events": 0,
            }
        stats = torch.cuda.memory_stats()
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "allocation_events": int(stats.get("allocation.all.allocated", 0)),
            "segment_alloc_events": int(stats.get("segment.all.allocated", 0)),
        }

    def build_model():
        counters.graph_build_calls += 1
        model = EncoderBlock(hidden).eval().to(device=args.device, dtype=dtype)
        if args.use_torch_compile:
            counters.compile_calls += 1
            if args.compile_backend == "inductor" and args.inductor_gemm_backends:
                try:
                    # Route GEMM lowering through requested backend set (e.g. ATEN for cuBLAS/cuBLASLt path).
                    import torch._inductor.config as inductor_config  # type: ignore

                    if hasattr(inductor_config, "max_autotune_gemm_backends"):
                        inductor_config.max_autotune_gemm_backends = args.inductor_gemm_backends
                        compile_config["inductor_gemm_backends_applied"] = True
                    if hasattr(inductor_config, "max_autotune"):
                        inductor_config.max_autotune = True
                except Exception:
                    compile_config["inductor_gemm_backends_applied"] = False
            model = torch.compile(model, mode=args.compile_mode, backend=args.compile_backend)
        return model

    input_cache = {}

    def get_static_input(batch: int, seq_len: int) -> "torch.Tensor":
        key = (batch, seq_len)
        cached = input_cache.get(key)
        if cached is not None:
            return cached
        static_x = torch.empty((batch, seq_len, hidden), device=args.device, dtype=dtype)
        static_x.normal_(mean=0.0, std=1.0)
        input_cache[key] = static_x
        return static_x

    model = build_model()

    def run_phase(batch: int, seq_len: int, warmup: int, iters: int, single_pass_after_warmup: bool) -> dict:
        static_x = get_static_input(batch, seq_len)
        latencies_ms: list[float] = []

        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        compile_before_steady = counters.compile_calls
        graph_before_steady = counters.graph_build_calls

        grad_ctx = torch.no_grad if args.disable_inference_mode else torch.inference_mode
        with grad_ctx():
            for _ in range(warmup):
                full_sync()
                _ = model(static_x)
                full_sync()

            use_graph_arena = args.use_cuda_graph_arena and args.device == "cuda"
            graph = None
            if use_graph_arena:
                graph = torch.cuda.CUDAGraph()
                # Capture fixed forward-pass work with static buffers to eliminate per-iter temp allocations.
                with torch.cuda.graph(graph):
                    _ = model(static_x)
                full_sync()

            pre = memory_snapshot()
            alloc_events_pre = pre["allocation_events"]
            segment_events_pre = pre["segment_alloc_events"]

            pass_iters = 1 if single_pass_after_warmup else iters
            if args.cuda_profiler_range and args.device == "cuda":
                torch.cuda.cudart().cudaProfilerStart()

            for _ in range(pass_iters):
                full_sync()
                t0 = time.perf_counter()
                if use_graph_arena:
                    graph.replay()
                else:
                    _ = model(static_x)
                full_sync()
                t1 = time.perf_counter()
                # Extra post-stop sync to prevent async leakage across iterations.
                full_sync()
                latencies_ms.append((t1 - t0) * 1000.0)

            if args.cuda_profiler_range and args.device == "cuda":
                torch.cuda.cudart().cudaProfilerStop()

            post = memory_snapshot()

        compile_after_steady = counters.compile_calls
        graph_after_steady = counters.graph_build_calls
        tokens = batch * seq_len
        mean_ms = mean(latencies_ms)
        throughput = (tokens / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

        return {
            "batch": batch,
            "seq_len": seq_len,
            "samples_ms": [round(v, 4) for v in latencies_ms],
            "latency_ms": {
                "mean": round(mean_ms, 4),
                "p50": round(percentile(latencies_ms, 50), 4),
                "p95": round(percentile(latencies_ms, 95), 4),
                "min": round(min(latencies_ms), 4) if latencies_ms else 0.0,
                "max": round(max(latencies_ms), 4) if latencies_ms else 0.0,
            },
            "throughput_tokens_per_sec": round(throughput, 2),
            "compile_calls_during_steady_state": compile_after_steady - compile_before_steady,
            "graph_build_calls_during_steady_state": graph_after_steady - graph_before_steady,
            "memory_pre": pre,
            "memory_post": post,
            "allocation_event_delta": int(post["allocation_events"]) - int(alloc_events_pre),
            "segment_alloc_event_delta": int(post["segment_alloc_events"]) - int(segment_events_pre),
            "memory_stable": (
                (int(post["allocation_events"]) - int(alloc_events_pre)) == 0
                and (int(post["segment_alloc_events"]) - int(segment_events_pre)) == 0
            ),
        }

    buckets = parse_csv_ints(args.bucket_lengths)
    if args.single_shape_seq_len > 0:
        buckets = [int(args.single_shape_seq_len)]
    batches = parse_csv_ints(args.batch_sizes)
    if not buckets:
        buckets = [256]
    if not batches:
        batches = [args.batch]

    bucket_results = []
    for seq_len in buckets:
        bucket_results.append(
            run_phase(
                batch=args.batch,
                seq_len=seq_len,
                warmup=args.warmup,
                iters=args.iters,
                single_pass_after_warmup=args.single_pass_after_warmup,
            )
        )

    batch_scaling = []
    if not args.skip_batch_scaling:
        # Batch scaling on the median bucket.
        scale_seq = buckets[len(buckets) // 2]
        for batch in batches:
            batch_scaling.append(
                run_phase(
                    batch=batch,
                    seq_len=scale_seq,
                    warmup=max(5, args.warmup // 2),
                    iters=max(20, args.iters // 4),
                    single_pass_after_warmup=False,
                )
            )

    payload = {
        "status": "ok",
        "meta": {
            "device": args.device,
            "dtype": str(dtype),
            "hidden": args.hidden,
            "hidden_effective": hidden,
            "warmup": args.warmup,
            "iters": args.iters,
            "bucket_lengths": buckets,
            "batch_sizes": batches,
            "torch_compile": args.use_torch_compile,
            "compile_mode": args.compile_mode,
            "compile_backend": args.compile_backend,
            "inductor_gemm_backends": args.inductor_gemm_backends,
            "inductor_gemm_backends_applied": compile_config["inductor_gemm_backends_applied"],
            "allow_tf32": bool(args.allow_tf32),
            "inference_mode": not args.disable_inference_mode,
            "use_inplace_residual": bool(args.use_inplace_residual),
            "pad_hidden_to": args.pad_hidden_to,
            "use_cuda_graph_arena": bool(args.use_cuda_graph_arena),
            "single_shape_seq_len": args.single_shape_seq_len,
            "skip_batch_scaling": bool(args.skip_batch_scaling),
            "single_pass_after_warmup": args.single_pass_after_warmup,
        },
        "global_counters": {
            "compile_calls": counters.compile_calls,
            "graph_build_calls": counters.graph_build_calls,
        },
        "bucket_results": bucket_results,
        "batch_scaling": batch_scaling,
    }

    serialized = json.dumps(payload, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
