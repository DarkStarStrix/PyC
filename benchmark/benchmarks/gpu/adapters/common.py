#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORKLOAD = ROOT / "benchmark" / "benchmarks" / "gpu" / "workloads" / "standard_workload.py"


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if payload.get("status") in {"ok", "unavailable"} else 1


def run_standard_workload(backend: str, device: str, batch: int, hidden: int, iters: int, warmup: int) -> dict:
    task = os.environ.get("BENCH_TASK", "mlp").strip() or "mlp"
    m = os.environ.get("BENCH_M", "").strip()
    k = os.environ.get("BENCH_K", "").strip()
    n = os.environ.get("BENCH_N", "").strip()
    cmd = [
        "python3",
        str(WORKLOAD),
        "--backend",
        backend,
        "--task",
        task,
        "--device",
        device,
        "--batch",
        str(batch),
        "--hidden",
        str(hidden),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
    ]
    if task == "gemm" and m and k and n:
        cmd.extend(["--m", m, "--k", k, "--n", n])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"status": "error", "error": proc.stderr.strip() or "workload failed"}
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "invalid JSON from workload",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }


def run_external_json_command(command: str) -> dict:
    try:
        cmd = shlex.split(command)
    except ValueError as exc:
        return {"status": "error", "error": f"invalid command: {exc}"}

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"status": "error", "error": proc.stderr.strip() or "external command failed"}
    stdout = proc.stdout.strip()
    candidates = [stdout]
    if stdout:
        candidates.extend(line.strip() for line in stdout.splitlines()[::-1] if line.strip())
    payload = None
    for item in candidates:
        try:
            payload = json.loads(item)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        return {
            "status": "error",
            "error": "external command must print JSON",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return payload


def apply_bench_env(device: str, batch: int, hidden: int, iters: int, warmup: int) -> None:
    os.environ["BENCH_DEVICE"] = device
    os.environ["BENCH_BATCH"] = str(batch)
    os.environ["BENCH_HIDDEN"] = str(hidden)
    os.environ["BENCH_ITERS"] = str(iters)
    os.environ["BENCH_WARMUP"] = str(warmup)
    task = os.environ.get("PYC_BENCH_TASK", "").strip()
    if task:
        os.environ["BENCH_TASK"] = task
    for key in ("M", "K", "N"):
        value = os.environ.get(f"PYC_BENCH_{key}", "").strip()
        if value:
            os.environ[f"BENCH_{key}"] = value
    shape_name = os.environ.get("PYC_BENCH_SHAPE_NAME", "").strip()
    if shape_name:
        os.environ["BENCH_SHAPE_NAME"] = shape_name


def enrich(payload: dict, adapter: str, display_name: str) -> dict:
    payload = dict(payload)
    payload["adapter"] = adapter
    payload["display_name"] = display_name
    payload.setdefault("mode", "unknown")
    return payload
