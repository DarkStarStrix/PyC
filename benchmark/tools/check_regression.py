#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"error: missing file: {path}")
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"error: invalid json in {path}: {exc}")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Check benchmark results against regression guardrail thresholds")
    parser.add_argument("--baseline", required=True, help="Path to baseline threshold json")
    parser.add_argument("--result", required=True, help="Path to benchmark result json")
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline))
    result = load_json(Path(args.result))

    checks = [
        ("configure_ms", float(result.get("configure_ms", 0.0)), float(baseline["max_configure_ms"])),
        ("build_ms", float(result.get("build_ms", 0.0)), float(baseline["max_build_ms"])),
        ("smoke_pyc.mean_ms", float(result.get("smoke_pyc", {}).get("mean_ms", 0.0)), float(baseline["max_smoke_mean_ms"])),
        ("microbench.mean_ms", float(result.get("microbench", {}).get("mean_ms", 0.0)), float(baseline["max_microbench_mean_ms"])),
    ]

    failed = False
    for name, val, max_allowed in checks:
        status = "ok" if val <= max_allowed else "fail"
        print(f"{name}: {val:.3f} (max {max_allowed:.3f}) -> {status}")
        if val > max_allowed:
            failed = True

    if failed:
        print("error: performance regression guardrail failed")
        return 1

    print("performance guardrail: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
