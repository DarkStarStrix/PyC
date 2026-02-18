#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STRUCTURED_RESULTS_DIR = ROOT / "benchmark" / "benchmarks" / "results"
ADAPTER_DIR = ROOT / "benchmark" / "benchmarks" / "gpu" / "adapters"

DEFAULT_ADAPTERS = [
    "torch_eager",
    "torch_compile",
    "pyc",
    "tvm",
    "xla",
    "tensorrt",
    "glow",
]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def git_head() -> str:
    override = (os.environ.get("PYC_BENCH_GIT_HEAD") or "").strip()
    if override:
        return override
    proc = run(["git", "rev-parse", "--short", "HEAD"])
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def git_dirty() -> int:
    override = (os.environ.get("PYC_BENCH_GIT_DIRTY") or "").strip()
    if override in {"0", "1"}:
        return int(override)
    proc = run(["git", "status", "--porcelain"])
    if proc.returncode != 0:
        return 0
    return 1 if proc.stdout.strip() else 0


def read_gpu_info() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    ]
    proc = run(query)
    if proc.returncode != 0:
        return {"available": False, "reason": proc.stderr.strip() or "nvidia-smi query failed"}
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total": parts[2],
                    "compute_capability": parts[3],
                }
            )
    return {"available": True, "gpus": gpus}


def run_adapter(adapter: str, device: str, batch: int, hidden: int, iters: int, warmup: int) -> dict:
    script = ADAPTER_DIR / f"adapter_{adapter}.py"
    if not script.exists():
        return {"status": "error", "adapter": adapter, "error": f"missing adapter script: {script}"}

    cmd = [
        "python3",
        str(script),
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
    proc = run(cmd)
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"status": "error", "adapter": adapter, "error": proc.stderr.strip() or "adapter failed"}
    try:
        payload = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {
            "status": "error",
            "adapter": adapter,
            "error": "invalid json from adapter",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return payload


def summarize_markdown(results: dict) -> str:
    lines = [
        "# GPU Benchmark Suite",
        "",
        f"- Timestamp (UTC): {results['meta']['timestamp_utc']}",
        f"- Host: {results['meta']['host']}",
        f"- OS: {results['meta']['os']}",
        f"- Python: {results['meta']['python']}",
        "",
        "## GPU",
        "",
    ]

    gpu = results["gpu"]
    if not gpu.get("available"):
        lines.append(f"- Not available: {gpu.get('reason', 'unknown')}")
    else:
        for idx, item in enumerate(gpu.get("gpus", []), start=1):
            lines.append(
                f"- GPU {idx}: {item['name']} | driver {item['driver_version']} | mem {item['memory_total']} | cc {item['compute_capability']}"
            )

    lines.extend(["", "## Adapter Results", ""])
    for name in results["meta"]["adapters"]:
        payload = results["adapters"].get(name, {})
        display = payload.get("display_name", name)
        mode = payload.get("mode", "unknown")
        if payload.get("status") == "ok":
            lat = payload["latency_ms"]
            lines.append(
                f"- `{display}` [{mode}]: p50 {lat['p50']} ms, p95 {lat['p95']} ms, mean {lat['mean']} ms, throughput {payload['throughput_tokens_per_sec']} tokens/s"
            )
        elif payload.get("status") == "unavailable":
            lines.append(f"- `{display}`: unavailable ({payload.get('reason', 'n/a')})")
        else:
            lines.append(f"- `{display}`: error ({payload.get('error', 'unknown')})")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Adapters are normalized to a common JSON schema.",
            "- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.",
        ]
    )

    return "\n".join(lines) + "\n"


def write_svg(results: dict, out_path: Path) -> None:
    order = results["meta"]["adapters"]
    ok_rows = []
    for name in order:
        payload = results["adapters"].get(name, {})
        if payload.get("status") == "ok":
            ok_rows.append((payload.get("display_name", name), float(payload["latency_ms"]["mean"])))

    width = 1200
    bar_h = 36
    gap = 16
    chart_x = 380
    chart_y = 110
    chart_w = 760
    rows = max(1, len(ok_rows))
    height = chart_y + rows * (bar_h + gap) + 120
    max_v = max((v for _, v in ok_rows), default=1.0)

    body = []
    for i, (label, value) in enumerate(ok_rows):
        y = chart_y + i * (bar_h + gap)
        bar_w = int((value / max_v) * chart_w) if max_v > 0 else 0
        body.append(
            f'<text x="30" y="{y + 24}" font-family="Arial, sans-serif" font-size="18" fill="#111">{label}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#2563eb" rx="6" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 24}" font-family="Arial, sans-serif" font-size="16" fill="#111">{value:.4f} ms</text>'
        )

    title = f"PyC Backend Suite ({results['meta']['device'].upper()})"
    subtitle = f"run_id={results['meta']['run_id']} | host={results['meta']['host']} | git={results['meta']['git_head']} | dirty={results['meta']['git_dirty']}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="42" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#0f172a">{title}</text>
  <text x="30" y="72" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle}</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standardized GPU backend comparisons")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--output-root",
        default=str(STRUCTURED_RESULTS_DIR),
        help="Structured output root (default: benchmark/benchmarks/results)",
    )
    parser.add_argument(
        "--adapters",
        default=",".join(DEFAULT_ADAPTERS),
        help="Comma-separated adapter ids (default: torch_eager,torch_compile,pyc,tvm,xla,tensorrt,glow)",
    )
    parser.add_argument(
        "--parity-strict",
        action="store_true",
        help="Fail if any adapter reports mode=proxy (native parity gate).",
    )
    parser.add_argument(
        "--require-native-adapter",
        default="",
        help="Comma-separated adapter ids that must report mode=native when status=ok.",
    )
    args = parser.parse_args()

    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    output_root = Path(args.output_root)
    (output_root / "json").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip()
    if not run_id:
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "host": platform.node(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "device": args.device,
            "batch": args.batch,
            "hidden": args.hidden,
            "iters": args.iters,
            "warmup": args.warmup,
            "tag": args.tag,
            "adapters": adapters,
            "run_id": run_id,
            "git_head": git_head(),
            "git_dirty": git_dirty(),
        },
        "gpu": read_gpu_info(),
        "adapters": {},
    }

    for adapter in adapters:
        results["adapters"][adapter] = run_adapter(
            adapter, args.device, args.batch, args.hidden, args.iters, args.warmup
        )

    stamp = f"{run_id}__{args.tag}"
    structured_json = output_root / "json" / f"{stamp}.json"
    structured_md = output_root / "reports" / f"{stamp}.md"
    structured_svg = output_root / "images" / f"{stamp}.svg"
    structured_meta = output_root / "json" / f"{stamp}.metadata.json"
    structured_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    structured_md.write_text(summarize_markdown(results), encoding="utf-8")
    write_svg(results, structured_svg)
    structured_meta.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tag": args.tag,
                "created_utc": results["meta"]["timestamp_utc"],
                "host": results["meta"]["host"],
                "os": results["meta"]["os"],
                "python": results["meta"]["python"],
                "git_head": results["meta"]["git_head"],
                "git_dirty": results["meta"]["git_dirty"],
                "device": results["meta"]["device"],
                "batch": results["meta"]["batch"],
                "hidden": results["meta"]["hidden"],
                "iters": results["meta"]["iters"],
                "warmup": results["meta"]["warmup"],
                "adapters": adapters,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {structured_json}")
    print(f"Wrote {structured_md}")
    print(f"Wrote {structured_svg}")
    print(f"Wrote {structured_meta}")
    if args.parity_strict:
        proxy_adapters = [
            name
            for name, payload in results["adapters"].items()
            if payload.get("status") == "ok" and payload.get("mode") == "proxy"
        ]
        if proxy_adapters:
            print(f"parity-strict failed: proxy adapters present: {','.join(proxy_adapters)}")
            return 2

    if args.require_native_adapter.strip():
        required = [item.strip() for item in args.require_native_adapter.split(",") if item.strip()]
        violations = []
        for name in required:
            payload = results["adapters"].get(name)
            if not payload:
                violations.append((name, "missing"))
                continue
            if payload.get("status") != "ok":
                violations.append((name, f"status={payload.get('status', 'unknown')}"))
                continue
            if payload.get("mode") != "native":
                violations.append((name, f"mode={payload.get('mode', 'unknown')}"))
        if violations:
            print("require-native-adapter failed:")
            for name, reason in violations:
                print(f"  - {name}: {reason}")
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
