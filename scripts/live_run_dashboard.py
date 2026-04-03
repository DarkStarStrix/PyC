#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


STEP_RE = re.compile(r"step=(\d+)/(\d+)\s+loss=([0-9.]+)\s+tokens=(\d+)")
EVAL_RE = re.compile(r"eval step=(\d+)\s+loss=([^\s]+)")
MAP_RE = re.compile(r"Map:\s+(\d+)%.*?([0-9]+(?:\.[0-9]+)?)\s+examples/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live training dashboard from log + telemetry files")
    parser.add_argument("--log", default="/root/qwen_smoke.log")
    parser.add_argument("--run-dir", default="/root/PyC/runs/smoke_qwen14b/smoke_qwen14b")
    parser.add_argument("--refresh-sec", type=float, default=2.0)
    return parser.parse_args()


def read_tail(path: Path, max_lines: int = 120) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def latest_step(lines: list[str]) -> dict[str, float | int | str] | None:
    for line in reversed(lines):
        match = STEP_RE.search(line)
        if match:
            return {
                "step": int(match.group(1)),
                "total": int(match.group(2)),
                "loss": float(match.group(3)),
                "tokens": int(match.group(4)),
                "line": line.strip(),
            }
    return None


def latest_eval(lines: list[str]) -> str | None:
    for line in reversed(lines):
        match = EVAL_RE.search(line)
        if match:
            return line.strip()
    return None


def latest_map(lines: list[str]) -> dict[str, float | int] | None:
    for line in reversed(lines):
        match = MAP_RE.search(line)
        if match:
            return {"pct": int(match.group(1)), "examples_per_sec": float(match.group(2))}
    return None


def read_gpu_telemetry(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"gpu_util_mean": 0.0}
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                values.append(float(row.get("util_gpu", 0.0)))
            except Exception:
                continue
    if not values:
        return {"gpu_util_mean": 0.0}
    return {"gpu_util_mean": sum(values[-64:]) / min(len(values), 64)}


def read_final_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def query_nvidia() -> list[tuple[int, int, int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    rows: list[tuple[int, int, int, int]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(tuple(int(part) for part in parts))
    return rows


def ema(prev: float | None, current: float, alpha: float = 0.12) -> float:
    if prev is None:
        return current
    return prev * (1.0 - alpha) + current * alpha


def clear() -> None:
    print("\033[2J\033[H", end="")


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    run_dir = Path(args.run_dir)
    gpu_csv = run_dir / "gpu_telemetry.csv"
    metrics_json = run_dir / "train_metrics.json"
    smooth_gpu = None
    smooth_examples = None

    while True:
        lines = read_tail(log_path)
        step = latest_step(lines)
        eval_line = latest_eval(lines)
        map_row = latest_map(lines)
        gpu = read_gpu_telemetry(gpu_csv)
        final = read_final_metrics(metrics_json)
        nvidia = query_nvidia()

        gpu_mean = float(gpu.get("gpu_util_mean", 0.0))
        smooth_gpu = ema(smooth_gpu, gpu_mean)
        map_eps = float(map_row["examples_per_sec"]) if map_row else 0.0
        smooth_examples = ema(smooth_examples, map_eps)

        width = shutil.get_terminal_size((120, 40)).columns
        clear()
        print("Live Run Dashboard".ljust(width))
        print(time.strftime("UTC %Y-%m-%d %H:%M:%S", time.gmtime()).ljust(width))
        print("-" * min(width, 120))

        if step:
            pct = int((step["step"] / step["total"]) * 100) if step["total"] else 0
            print(f"run progress   : {step['step']}/{step['total']} ({pct}%)")
            print(f"loss           : {step['loss']:.4f}")
            print(f"tokens         : {step['tokens']}")
            print(f"log line       : {step['line'][:max(20, width - 18)]}")
        else:
            print("run progress   : waiting for training step logs")

        if eval_line:
            print(f"eval           : {eval_line}")

        print(f"examples/s     : {map_eps:.2f} (smoothed {smooth_examples or 0.0:.2f})")
        print(f"gpu util mean  : {gpu_mean:.1f}% (smoothed {smooth_gpu or 0.0:.1f}%)")

        if final:
            print(f"final runtime  : {final.get('train_runtime_sec', 0.0)}s")
            print(f"final tok/s    : {final.get('tokens_per_sec', 0.0)}")
            print(f"final samples/s: {final.get('samples_per_sec', 0.0)}")

        print("-" * min(width, 120))
        if nvidia:
            for index, util, used, total in nvidia:
                print(f"GPU{index} util={util:3d}% mem={used:6d}/{total:6d}MB")
        else:
            print("nvidia-smi unavailable")

        print("-" * min(width, 120))
        print("Recent log tail:")
        for line in lines[-12:]:
            print(line[:width])

        time.sleep(args.refresh_sec)


if __name__ == "__main__":
    raise SystemExit(main())
