#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from common import apply_bench_env, current_python, emit, enrich, run_external_json_command


def helper_python() -> str:
    root = Path(__file__).resolve().parents[4]
    venv_python = root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return current_python()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    apply_bench_env(args.device, args.batch, args.hidden, args.iters, args.warmup)
    command = os.environ.get("GLOW_BENCH_CMD", "").strip()
    if not command:
        helper = Path(__file__).resolve().parents[1] / "external" / "bench_glow_cmd.py"
        if helper.exists():
            command = f"{helper_python()} {helper}"
            os.environ["GLOW_BENCH_CMD"] = command
    if not command:
        payload = {
            "status": "unavailable",
            "reason": "Set GLOW_BENCH_CMD to external benchmark command for Glow",
        }
        return emit(enrich(payload, "glow", "Glow"))

    payload = run_external_json_command(command)
    return emit(enrich(payload, "glow", "Glow"))


if __name__ == "__main__":
    raise SystemExit(main())
