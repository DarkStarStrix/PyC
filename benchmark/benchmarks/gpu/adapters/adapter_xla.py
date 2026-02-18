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
    if importlib.util.find_spec("torch_xla") is None and not os.environ.get("XLA_BENCH_CMD"):
        payload = {
            "status": "unavailable",
            "reason": "Install torch_xla or set XLA_BENCH_CMD to external benchmark command",
        }
        return emit(enrich(payload, "xla", "XLA"))

    command = os.environ.get("XLA_BENCH_CMD", "").strip()
    if not command:
        payload = {
            "status": "unavailable",
            "reason": "XLA detected but no standardized benchmark command configured (set XLA_BENCH_CMD)",
        }
        return emit(enrich(payload, "xla", "XLA"))

    payload = run_external_json_command(command)
    return emit(enrich(payload, "xla", "XLA"))


if __name__ == "__main__":
    raise SystemExit(main())
