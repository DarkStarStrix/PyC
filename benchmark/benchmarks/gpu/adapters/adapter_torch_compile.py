#!/usr/bin/env python3
from __future__ import annotations

import argparse
from common import emit, enrich, run_standard_workload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    payload = run_standard_workload(
        "torch_compile", args.device, args.batch, args.hidden, args.iters, args.warmup
    )
    return emit(enrich(payload, "torch_compile", "PyTorch Compile"))


if __name__ == "__main__":
    raise SystemExit(main())
