#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\\n"
            f"stdout:\n{proc.stdout}\\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def time_command(cmd, cwd=None):
    start = time.perf_counter()
    proc = run(cmd, cwd=cwd)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, proc


def resolve_executable(build_dir: Path, config: str, name: str) -> Path:
    if os.name == "nt":
        candidate = build_dir / config / f"{name}.exe"
        if candidate.exists():
            return candidate
        fallback = build_dir / f"{name}.exe"
        return fallback
    return build_dir / name


def resolve_library(build_dir: Path, config: str) -> Path:
    if os.name == "nt":
        candidate = build_dir / config / "pyc_core.lib"
        if candidate.exists():
            return candidate
        return build_dir / "pyc_core.lib"
    return build_dir / "libpyc_core.a"


def stats(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "stdev_ms": round(statistics.pstdev(values), 3),
        "samples": [round(v, 3) for v in values],
    }


def summarize_markdown(results):
    lines = []
    lines.append("# PyC Benchmark Results")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {results['meta']['timestamp_utc']}")
    lines.append(f"- Platform: {results['meta']['platform']}")
    lines.append(f"- CPU: {results['meta']['cpu']}")
    lines.append(f"- Python: {results['meta']['python']}")
    lines.append(f"- Build directory: `{results['meta']['build_dir']}`")
    lines.append("")
    lines.append("## Build")
    lines.append("")
    lines.append(f"- Configure: {results['configure_ms']:.3f} ms")
    lines.append(f"- Build (`pyc pyc_core pyc_foundation pyc_core_microbench`): {results['build_ms']:.3f} ms")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    smoke = results["smoke_pyc"]
    micro = results["microbench"]
    lines.append(
        f"- `pyc` smoke: mean {smoke['mean_ms']} ms (min {smoke['min_ms']}, max {smoke['max_ms']}, stdev {smoke['stdev_ms']})"
    )
    lines.append(
        f"- `pyc_core_microbench`: mean {micro['mean_ms']} ms (min {micro['min_ms']}, max {micro['max_ms']}, stdev {micro['stdev_ms']})"
    )
    lines.append("")
    lines.append("## Artifact Sizes")
    lines.append("")
    lines.append(f"- `pyc`: {results['artifacts']['pyc_bytes']} bytes")
    lines.append(f"- `pyc_core`: {results['artifacts']['pyc_core_bytes']} bytes")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- These numbers benchmark the stable core targets currently built in CI.")
    lines.append("- They do not benchmark the experimental full compiler pipeline.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run deterministic PyC core benchmarks")
    parser.add_argument("--build-dir", default="build", help="CMake build directory")
    parser.add_argument("--config", default="Release", help="Build config for multi-config generators")
    parser.add_argument("--repeats", type=int, default=7, help="Number of benchmark samples")
    parser.add_argument("--micro-rounds", type=int, default=4000, help="Rounds for core microbench workload")
    args = parser.parse_args()

    build_dir = (ROOT / args.build_dir).resolve()
    results_dir = (ROOT / "benchmark" / "benchmarks" / "results").resolve()
    json_dir = (results_dir / "json").resolve()
    reports_dir = (results_dir / "reports").resolve()
    docs_dir = (ROOT / "docs").resolve()
    json_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        "cmake", "-S", str(ROOT), "-B", str(build_dir),
        "-D", "PYC_BUILD_EXPERIMENTAL=OFF",
        "-D", "PYC_BUILD_BENCHMARKS=ON",
    ]
    build_cmd = [
        "cmake", "--build", str(build_dir), "--parallel",
        "--config", args.config,
        "--target", "pyc", "pyc_core", "pyc_foundation", "pyc_core_microbench",
    ]

    configure_ms, _ = time_command(configure_cmd)
    build_ms, _ = time_command(build_cmd)

    pyc_exe = resolve_executable(build_dir, args.config, "pyc")
    micro_exe = resolve_executable(build_dir, args.config, "pyc_core_microbench")
    pyc_core_lib = resolve_library(build_dir, args.config)

    smoke_samples = []
    micro_samples = []

    for _ in range(args.repeats):
        ms, _ = time_command([str(pyc_exe)])
        smoke_samples.append(ms)

    for _ in range(args.repeats):
        ms, proc = time_command([str(micro_exe), str(args.micro_rounds)])
        if "checksum=" not in proc.stdout:
            raise RuntimeError("microbench output missing checksum")
        micro_samples.append(ms)

    results = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "python": platform.python_version(),
            "build_dir": str(build_dir),
            "repeats": args.repeats,
            "micro_rounds": args.micro_rounds,
        },
        "configure_ms": round(configure_ms, 3),
        "build_ms": round(build_ms, 3),
        "smoke_pyc": stats(smoke_samples),
        "microbench": stats(micro_samples),
        "artifacts": {
            "pyc_bytes": pyc_exe.stat().st_size,
            "pyc_core_bytes": pyc_core_lib.stat().st_size,
        },
    }

    json_path = json_dir / "latest_core.json"
    md_path = reports_dir / "latest_core.md"
    docs_path = docs_dir / "performance-results.md"

    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    markdown = summarize_markdown(results)
    md_path.write_text(markdown, encoding="utf-8")
    docs_path.write_text(markdown, encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Updated {docs_path}")


if __name__ == "__main__":
    main()
