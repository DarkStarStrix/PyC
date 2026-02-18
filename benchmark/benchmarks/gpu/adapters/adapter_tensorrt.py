#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
from common import apply_bench_env, emit, enrich, run_external_json_command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    apply_bench_env(args.device, args.batch, args.hidden, args.iters, args.warmup)
    if importlib.util.find_spec("tensorrt") is None and not os.environ.get("TENSORRT_BENCH_CMD"):
        payload = {
            "status": "unavailable",
            "reason": "Install TensorRT python bindings or set TENSORRT_BENCH_CMD",
        }
        return emit(enrich(payload, "tensorrt", "TensorRT"))

    command = os.environ.get("TENSORRT_BENCH_CMD", "").strip()
    if not command:
        payload = {
            "status": "unavailable",
            "reason": "TensorRT detected but no standardized benchmark command configured (set TENSORRT_BENCH_CMD)",
        }
        return emit(enrich(payload, "tensorrt", "TensorRT"))

    payload = run_external_json_command(command)
    return emit(enrich(payload, "tensorrt", "TensorRT"))


if __name__ == "__main__":
    raise SystemExit(main())
