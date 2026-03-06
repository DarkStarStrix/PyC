#!/usr/bin/env python3
"""Concurrent inference operating-point benchmark (latency tails + memory stability)."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def parse_csv_ints(value: str) -> list[int]:
    out = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            out.append(int(raw))
    return out


def parse_csv_strs(value: str) -> list[str]:
    out = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            out.append(raw)
    return out


def run() -> int:
    parser = argparse.ArgumentParser(description="Concurrent inference benchmark")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--requests-per-worker", type=int, default=60)
    parser.add_argument("--concurrency-levels", default="4,8")
    parser.add_argument("--modes", default="eager,compiled_aten,arena")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--inductor-gemm-backends", default="ATEN")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--use-inplace-residual", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": f"PyTorch unavailable: {exc}"}))
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"status": "error", "error": "CUDA requested but not available"}))
        return 1

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    if args.device == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    torch.manual_seed(7)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(7)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

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
            return pooled @ pooled.transpose(-1, -2)

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

    def build_model(mode: str):
        model = EncoderBlock(args.hidden).eval().to(device=args.device, dtype=dtype)
        compile_applied = False
        if mode == "compiled_aten":
            try:
                import torch._inductor.config as inductor_config  # type: ignore

                if hasattr(inductor_config, "max_autotune_gemm_backends"):
                    inductor_config.max_autotune_gemm_backends = args.inductor_gemm_backends
                if hasattr(inductor_config, "max_autotune"):
                    inductor_config.max_autotune = True
                compile_applied = True
            except Exception:
                compile_applied = False
            model = torch.compile(model, mode=args.compile_mode, backend="inductor")
        return model, compile_applied

    class Worker:
        def __init__(self, mode: str, model):
            self.mode = mode
            self.model = model
            self.stream = torch.cuda.Stream() if args.device == "cuda" else None
            self.static_x = torch.empty(
                (args.batch, args.seq_len, args.hidden), device=args.device, dtype=dtype
            )
            self.static_x.normal_(mean=0.0, std=1.0)
            self.graph = None
            if self.mode == "arena" and args.device == "cuda":
                self.graph = torch.cuda.CUDAGraph()
                with torch.no_grad():
                    with torch.cuda.stream(self.stream):
                        with torch.cuda.graph(self.graph):
                            _ = self.model(self.static_x)
                self.stream.synchronize()

        def warm(self, n: int) -> None:
            with torch.no_grad():
                for _ in range(n):
                    if args.device == "cuda":
                        with torch.cuda.stream(self.stream):
                            if self.graph is not None:
                                self.graph.replay()
                            else:
                                _ = self.model(self.static_x)
                        self.stream.synchronize()
                    else:
                        _ = self.model(self.static_x)

        def infer_once_ms(self) -> float:
            t0 = time.perf_counter()
            with torch.no_grad():
                if args.device == "cuda":
                    with torch.cuda.stream(self.stream):
                        if self.graph is not None:
                            self.graph.replay()
                        else:
                            _ = self.model(self.static_x)
                    self.stream.synchronize()
                else:
                    _ = self.model(self.static_x)
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0

    modes = parse_csv_strs(args.modes)
    conc_levels = parse_csv_ints(args.concurrency_levels)
    valid_modes = {"eager", "compiled_aten", "arena"}
    for m in modes:
        if m not in valid_modes:
            print(json.dumps({"status": "error", "error": f"Unknown mode: {m}"}))
            return 1

    results = []
    tokens_per_request = args.batch * args.seq_len

    for mode in modes:
        for concurrency in conc_levels:
            model, compile_applied = build_model(mode)
            workers = [Worker(mode, model) for _ in range(concurrency)]
            for w in workers:
                w.warm(args.warmup)
            full_sync()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            pre = memory_snapshot()
            pre_alloc = pre["allocation_events"]
            pre_seg = pre["segment_alloc_events"]

            all_latencies: list[float] = []
            thread_errors: list[dict] = []
            lock = threading.Lock()
            go = threading.Event()

            def worker_fn(worker_idx: int, w: Worker) -> None:
                local = []
                go.wait()
                try:
                    for _ in range(args.requests_per_worker):
                        local.append(w.infer_once_ms())
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        thread_errors.append(
                            {
                                "worker_idx": worker_idx,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                    return
                with lock:
                    all_latencies.extend(local)

            threads = [
                threading.Thread(target=worker_fn, args=(idx, w))
                for idx, w in enumerate(workers)
            ]
            for t in threads:
                t.start()
            t0 = time.perf_counter()
            go.set()
            for t in threads:
                t.join()
            wall_s = time.perf_counter() - t0
            full_sync()
            post = memory_snapshot()

            if thread_errors:
                error_payload = {
                    "status": "error",
                    "error": "worker thread failure",
                    "mode": mode,
                    "concurrency": concurrency,
                    "thread_errors": thread_errors,
                }
                print(json.dumps(error_payload, indent=2))
                return 2

            total_requests = concurrency * args.requests_per_worker
            if len(all_latencies) != total_requests:
                error_payload = {
                    "status": "error",
                    "error": "latency sample count mismatch",
                    "mode": mode,
                    "concurrency": concurrency,
                    "expected_samples": total_requests,
                    "got_samples": len(all_latencies),
                }
                print(json.dumps(error_payload, indent=2))
                return 3

            req_per_sec = total_requests / wall_s if wall_s > 0 else 0.0
            tok_per_sec = req_per_sec * tokens_per_request

            results.append(
                {
                    "mode": mode,
                    "concurrency": concurrency,
                    "requests_per_worker": args.requests_per_worker,
                    "total_requests": total_requests,
                    "wall_time_sec": round(wall_s, 4),
                    "throughput_requests_per_sec": round(req_per_sec, 2),
                    "throughput_tokens_per_sec": round(tok_per_sec, 2),
                    "latency_ms": {
                        "p50": round(percentile(all_latencies, 50), 4),
                        "p95": round(percentile(all_latencies, 95), 4),
                        "p99": round(percentile(all_latencies, 99), 4),
                        "mean": round(sum(all_latencies) / len(all_latencies), 4) if all_latencies else 0.0,
                        "min": round(min(all_latencies), 4) if all_latencies else 0.0,
                        "max": round(max(all_latencies), 4) if all_latencies else 0.0,
                    },
                    "memory_pre": pre,
                    "memory_post": post,
                    "allocation_event_delta": int(post["allocation_events"]) - int(pre_alloc),
                    "segment_alloc_event_delta": int(post["segment_alloc_events"]) - int(pre_seg),
                    "memory_stable": (
                        (int(post["allocation_events"]) - int(pre_alloc)) == 0
                        and (int(post["segment_alloc_events"]) - int(pre_seg)) == 0
                    ),
                    "compile_applied": compile_applied,
                }
            )

    payload = {
        "status": "ok",
        "meta": {
            "device": args.device,
            "dtype": str(dtype),
            "hidden": args.hidden,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "warmup": args.warmup,
            "requests_per_worker": args.requests_per_worker,
            "concurrency_levels": conc_levels,
            "modes": modes,
            "compile_mode": args.compile_mode,
            "inductor_gemm_backends": args.inductor_gemm_backends,
            "allow_tf32": bool(args.allow_tf32),
            "use_inplace_residual": bool(args.use_inplace_residual),
        },
        "results": results,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
