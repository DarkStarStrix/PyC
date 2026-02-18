#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if payload.get("status") in {"ok", "unavailable"} else 1


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    build = root / "build"
    exe = build / "pyc_compiler_next_bench"

    device = os.environ.get("BENCH_DEVICE", "cuda")
    batch = int(os.environ.get("BENCH_BATCH", "64"))
    hidden = int(os.environ.get("BENCH_HIDDEN", "2048"))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))
    cfg = [
        "cmake",
        "-S",
        str(root),
        "-B",
        str(build),
        "-D",
        "PYC_BUILD_EXPERIMENTAL=OFF",
        "-D",
        "PYC_BUILD_BENCHMARKS=ON",
        "-D",
        "PYC_BUILD_COMPILER_NEXT=ON",
        "-D",
        "PYC_BUILD_COMPILER_NEXT_TESTS=OFF",
    ]
    bld = ["cmake", "--build", str(build), "--parallel", "--target", "pyc_compiler_next_bench"]
    for cmd in (cfg, bld):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return emit({"status": "error", "error": proc.stderr.strip() or "failed to build pyc_compiler_next_bench"})
    if not exe.exists():
        return emit({"status": "error", "error": f"missing benchmark executable: {exe}"})

    proc = subprocess.run(
        [str(exe), device, str(batch), str(hidden), str(iters), str(warmup)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        return emit({"status": "error", "error": proc.stderr.strip() or "pyc compiler-next bench failed"})
    try:
        payload = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return emit({"status": "error", "error": "invalid JSON from pyc_compiler_next_bench", "stdout": proc.stdout, "stderr": proc.stderr})
    if payload.get("status") == "ok":
        reliability = payload.get("reliability", {}) if isinstance(payload.get("reliability"), dict) else {}
        fallback_count = int(reliability.get("fallback_count", 0))
        if device == "cuda":
            payload["mode"] = "native" if fallback_count == 0 else "proxy"
            payload["note"] = (
                "PyC benchmark uses compiler-next API path; "
                "mode=native when CUDA executes without fallback, mode=proxy otherwise."
            )
        else:
            payload["mode"] = "native"
    return emit(payload)


if __name__ == "__main__":
    raise SystemExit(main())
