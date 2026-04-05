#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

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


def progress_write(message: str) -> None:
    if tqdm is not None and sys.stderr.isatty():
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, flush=True)


SOLID_PROGRESS_CHARS = " ▏▎▍▌▋▊▉█"


def require_progress_support(enabled: bool) -> None:
    if enabled and tqdm is None:
        raise SystemExit("--progress requested but tqdm is not installed in this Python environment")


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


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile_jitter(p50: float, p95: float) -> tuple[float, float]:
    jitter_ms = max(0.0, p95 - p50)
    jitter_pct = (jitter_ms / p50 * 100.0) if p50 > 0 else 0.0
    return jitter_ms, jitter_pct


def estimate_workload_flops(batch: int, hidden: int) -> float:
    # Approximate FLOPs/iter for the common workload:
    # two dense layers (matmul+accumulate) + one BxH @ HxB projection.
    return float((16 * batch * hidden * hidden) + (2 * batch * batch * hidden))


def enrich_derived_metrics(payload: dict) -> dict:
    if payload.get("status") != "ok":
        return payload

    out = dict(payload)
    latency = out.get("latency_ms", {})
    mean_ms = to_float(latency.get("mean"))
    p50_ms = to_float(latency.get("p50"))
    p95_ms = to_float(latency.get("p95"))
    jitter_ms, jitter_pct = percentile_jitter(p50_ms, p95_ms)

    batch = int(out.get("batch", 0) or 0)
    hidden = int(out.get("hidden", 0) or 0)
    iters = int(out.get("iters", 0) or 0)
    tokens_per_iter = max(0, batch * hidden)
    tokens_total = max(0, tokens_per_iter * iters)

    throughput = to_float(out.get("throughput_tokens_per_sec"))
    throughput_recomputed = ((tokens_per_iter / mean_ms) * 1000.0) if mean_ms > 0 else 0.0
    throughput_error_pct = (
        abs(throughput - throughput_recomputed) / throughput_recomputed * 100.0
        if throughput_recomputed > 0
        else 0.0
    )

    peak_bytes = int(out.get("peak_memory_bytes", 0) or 0)
    peak_gib = peak_bytes / float(1024**3)
    throughput_per_gib = (throughput / peak_gib) if peak_gib > 0 else 0.0

    flops_per_iter = estimate_workload_flops(batch, hidden) if batch > 0 and hidden > 0 else 0.0
    est_tflops = (flops_per_iter / (mean_ms / 1000.0) / 1.0e12) if mean_ms > 0 and flops_per_iter > 0 else 0.0

    samples = out.get("samples_ms", [])
    sample_stdev_ms = 0.0
    sample_cv_pct = 0.0
    if isinstance(samples, list):
        sample_vals = [to_float(v, default=math.nan) for v in samples]
        sample_vals = [v for v in sample_vals if math.isfinite(v)]
        if len(sample_vals) >= 2:
            sample_stdev_ms = statistics.pstdev(sample_vals)
            mean_for_cv = statistics.mean(sample_vals)
            sample_cv_pct = (sample_stdev_ms / mean_for_cv * 100.0) if mean_for_cv > 0 else 0.0

    repeat_mean_vals = out.get("repeat_mean_ms_values", [])
    repeat_stdev_ms = 0.0
    repeat_cv_pct = 0.0
    if isinstance(repeat_mean_vals, list):
        repeat_vals = [to_float(v, default=math.nan) for v in repeat_mean_vals]
        repeat_vals = [v for v in repeat_vals if math.isfinite(v)]
        if len(repeat_vals) >= 2:
            repeat_stdev_ms = statistics.pstdev(repeat_vals)
            mean_for_cv = statistics.mean(repeat_vals)
            repeat_cv_pct = (repeat_stdev_ms / mean_for_cv * 100.0) if mean_for_cv > 0 else 0.0

    startup = out.get("startup_ms", {}) if isinstance(out.get("startup_ms"), dict) else {}
    compile_overhead_ms = to_float(startup.get("compile_overhead_estimate"))
    startup_total_ms = to_float(startup.get("total"))
    compile_share_pct = (compile_overhead_ms / mean_ms * 100.0) if mean_ms > 0 else 0.0
    compile_share_of_startup_pct = (
        (compile_overhead_ms / startup_total_ms * 100.0) if startup_total_ms > 0 else 0.0
    )

    out["derived"] = {
        "tokens_per_iter": tokens_per_iter,
        "tokens_total": tokens_total,
        "jitter_ms_p95_minus_p50": round(jitter_ms, 4),
        "jitter_pct_of_p50": round(jitter_pct, 4),
        "sample_stdev_ms": round(sample_stdev_ms, 4),
        "sample_cv_pct": round(sample_cv_pct, 4),
        "repeat_mean_stdev_ms": round(repeat_stdev_ms, 4),
        "repeat_mean_cv_pct": round(repeat_cv_pct, 4),
        "throughput_recomputed_tokens_per_sec": round(throughput_recomputed, 2),
        "throughput_calc_error_pct": round(throughput_error_pct, 6),
        "peak_memory_gib": round(peak_gib, 6),
        "throughput_tokens_per_sec_per_peak_gib": round(throughput_per_gib, 2),
        "estimated_flops_per_iter": round(flops_per_iter, 2),
        "estimated_tflops_per_sec": round(est_tflops, 4),
        "startup_total_ms": round(startup_total_ms, 4),
        "compile_overhead_ms_estimate": round(compile_overhead_ms, 4),
        "compile_overhead_pct_of_mean_latency": round(compile_share_pct, 4),
        "compile_overhead_pct_of_startup": round(compile_share_of_startup_pct, 4),
    }
    return out


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


def iter_with_progress(items, enabled: bool, desc: str):
    if enabled and tqdm is not None and sys.stderr.isatty():
        return tqdm(
            items,
            desc=desc,
            unit="adapter",
            file=sys.stderr,
            dynamic_ncols=True,
            ascii=SOLID_PROGRESS_CHARS,
            mininterval=0.1,
            leave=True,
            smoothing=0.05,
        )
    return items


def create_progress_bar(total: int, enabled: bool, desc: str):
    if enabled and tqdm is not None and sys.stderr.isatty() and total > 0:
        return tqdm(
            total=total,
            desc=desc,
            unit="run",
            file=sys.stderr,
            dynamic_ncols=True,
            ascii=SOLID_PROGRESS_CHARS,
            mininterval=0.1,
            leave=True,
            smoothing=0.05,
            miniters=1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
    return None


def summarize_artifact_write(scope: str, paths: list[Path]) -> str:
    labels = ", ".join(path.name for path in paths)
    return f"[gpu-suite] wrote {scope} artifacts: {labels}"


def aggregate_adapter_runs(adapter: str, runs: list[dict]) -> dict:
    if not runs:
        return {"status": "error", "adapter": adapter, "error": "no runs executed"}

    non_ok = [payload for payload in runs if payload.get("status") != "ok"]
    if non_ok:
        first = dict(non_ok[0])
        first["repeat_count"] = len(runs)
        return first

    modes = {payload.get("mode", "unknown") for payload in runs}
    backends = {payload.get("backend", "unknown") for payload in runs}
    devices = {payload.get("device", "unknown") for payload in runs}
    if len(modes) != 1 or len(backends) != 1 or len(devices) != 1:
        return {
            "status": "error",
            "adapter": adapter,
            "error": "inconsistent adapter mode/backend/device across repeats",
            "modes": sorted(modes),
            "backends": sorted(backends),
            "devices": sorted(devices),
        }

    means = [float(payload["latency_ms"]["mean"]) for payload in runs]
    p50s = [float(payload["latency_ms"]["p50"]) for payload in runs]
    p95s = [float(payload["latency_ms"]["p95"]) for payload in runs]
    mins = [float(payload["latency_ms"]["min"]) for payload in runs]
    maxes = [float(payload["latency_ms"]["max"]) for payload in runs]
    throughputs = [float(payload["throughput_tokens_per_sec"]) for payload in runs]
    peaks = [int(payload.get("peak_memory_bytes", 0)) for payload in runs]
    startup_setup = []
    startup_first = []
    startup_total = []
    startup_compile = []
    for payload in runs:
        startup = payload.get("startup_ms", {})
        if not isinstance(startup, dict):
            continue
        if "setup" in startup:
            startup_setup.append(to_float(startup.get("setup")))
        if "first_step" in startup:
            startup_first.append(to_float(startup.get("first_step")))
        if "total" in startup:
            startup_total.append(to_float(startup.get("total")))
        if "compile_overhead_estimate" in startup:
            startup_compile.append(to_float(startup.get("compile_overhead_estimate")))

    out = dict(runs[0])
    out["latency_ms"] = {
        "mean": round(statistics.median(means), 4),
        "p50": round(statistics.median(p50s), 4),
        "p95": round(statistics.median(p95s), 4),
        "min": round(statistics.median(mins), 4),
        "max": round(statistics.median(maxes), 4),
    }
    out["throughput_tokens_per_sec"] = round(statistics.median(throughputs), 2)
    out["peak_memory_bytes"] = max(peaks)
    out["repeat_count"] = len(runs)
    out["repeat_mean_ms_values"] = [round(v, 4) for v in means]
    out["repeat_throughput_values"] = [round(v, 2) for v in throughputs]
    if len(means) >= 2:
        out["repeat_mean_ms_stdev"] = round(statistics.pstdev(means), 4)
    if len(throughputs) >= 2:
        out["repeat_throughput_stdev"] = round(statistics.pstdev(throughputs), 2)
    if startup_setup or startup_first or startup_total or startup_compile:
        out["startup_ms"] = {
            "setup": round(statistics.median(startup_setup), 4) if startup_setup else 0.0,
            "first_step": round(statistics.median(startup_first), 4) if startup_first else 0.0,
            "total": round(statistics.median(startup_total), 4) if startup_total else 0.0,
            "compile_overhead_estimate": (
                round(statistics.median(startup_compile), 4) if startup_compile else 0.0
            ),
        }
    return out


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

    native_rows: list[str] = []
    proxy_rows: list[str] = []
    unavailable_rows: list[str] = []
    error_rows: list[str] = []

    for name in results["meta"]["adapters"]:
        payload = results["adapters"].get(name, {})
        display = payload.get("display_name", name)
        mode = payload.get("mode", "unknown")
        status = payload.get("status")
        if status == "ok":
            lat = payload["latency_ms"]
            derived = payload.get("derived", {}) if isinstance(payload.get("derived"), dict) else {}
            jitter = to_float(derived.get("jitter_ms_p95_minus_p50"))
            sample_cv = to_float(derived.get("sample_cv_pct"))
            repeat_cv = to_float(derived.get("repeat_mean_cv_pct"))
            tflops = to_float(derived.get("estimated_tflops_per_sec"))
            startup_total = to_float(derived.get("startup_total_ms"))
            compile_ms = to_float(derived.get("compile_overhead_ms_estimate"))
            peak_gib = to_float(derived.get("peak_memory_gib"))
            row = (
                f"- `{display}` [{mode}]: p50 {lat['p50']} ms, p95 {lat['p95']} ms, "
                f"mean {lat['mean']} ms, throughput {payload['throughput_tokens_per_sec']} tokens/s, "
                f"jitter {jitter:.4f} ms, sample_cv {sample_cv:.2f}%, repeat_cv {repeat_cv:.2f}%, "
                f"est_tflops {tflops:.4f}, startup {startup_total:.4f} ms, compile_est {compile_ms:.4f} ms, "
                f"peak {peak_gib:.4f} GiB"
            )
            if mode == "native":
                native_rows.append(row)
            else:
                proxy_rows.append(row)
        elif status == "unavailable":
            unavailable_rows.append(f"- `{display}`: unavailable ({payload.get('reason', 'n/a')})")
        else:
            error_rows.append(f"- `{display}`: error ({payload.get('error', 'unknown')})")

    lines.extend(["", "## Native Adapter Results", ""])
    if native_rows:
        lines.extend(native_rows)
    else:
        lines.append("- none")

    lines.extend(["", "## Proxy Adapter Results", ""])
    if proxy_rows:
        lines.extend(proxy_rows)
    else:
        lines.append("- none")

    lines.extend(["", "## Unavailable Adapters", ""])
    if unavailable_rows:
        lines.extend(unavailable_rows)
    else:
        lines.append("- none")

    lines.extend(["", "## Adapter Errors", ""])
    if error_rows:
        lines.extend(error_rows)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Native count: {len(native_rows)} | Proxy count: {len(proxy_rows)} | Unavailable: {len(unavailable_rows)} | Errors: {len(error_rows)}",
            "- Adapters are normalized to a common JSON schema.",
            "- For TVM/XLA/TensorRT/PyC custom paths, configure `*_BENCH_CMD` env vars in each adapter.",
        ]
    )

    return "\n".join(lines) + "\n"


def write_svg(results: dict, out_path: Path) -> None:
    order = results["meta"]["adapters"]
    native_rows = []
    proxy_rows = []
    for name in order:
        payload = results["adapters"].get(name, {})
        if payload.get("status") == "ok":
            row = (payload.get("display_name", name), float(payload["latency_ms"]["mean"]))
            if payload.get("mode") == "native":
                native_rows.append(row)
            else:
                proxy_rows.append(row)

    ok_rows = native_rows if native_rows else proxy_rows

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
    mode_hint = "native" if native_rows else "proxy_fallback"
    subtitle = f"run_id={results['meta']['run_id']} | host={results['meta']['host']} | git={results['meta']['git_head']} | dirty={results['meta']['git_dirty']}"
    subtitle_2 = f"chart_mode={mode_hint} | native={len(native_rows)} | proxy={len(proxy_rows)}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="42" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#0f172a">{title}</text>
  <text x="30" y="72" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle}</text>
  <text x="30" y="92" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle_2}</text>
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
    parser.add_argument("--repeats", type=int, default=1)
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
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars while running adapters and repeats.",
    )
    args = parser.parse_args()
    require_progress_support(args.progress)

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
            "repeats": args.repeats,
            "tag": args.tag,
            "adapters": adapters,
            "run_id": run_id,
            "git_head": git_head(),
            "git_dirty": git_dirty(),
            "required_native_adapters": [
                item.strip() for item in args.require_native_adapter.split(",") if item.strip()
            ],
        },
        "gpu": read_gpu_info(),
        "adapters": {},
    }

    progress_write(
        f"[gpu-suite] run_id={run_id} device={args.device} adapters={','.join(adapters)} repeats={args.repeats}"
    )
    progress = create_progress_bar(len(adapters) * args.repeats, args.progress, "gpu-suite")
    for index, adapter in enumerate(adapters, start=1):
        if progress is not None:
            progress.set_description_str(f"gpu-suite {index}/{len(adapters)}")
            progress.set_postfix_str(f"adapter={adapter}")
        runs = []
        for repeat_index in range(args.repeats):
            if progress is not None:
                progress.set_description_str(f"gpu-suite {index}/{len(adapters)}")
                progress.set_postfix_str(f"adapter={adapter} repeat={repeat_index + 1}/{args.repeats}")
            runs.append(run_adapter(adapter, args.device, args.batch, args.hidden, args.iters, args.warmup))
            if progress is not None:
                progress.update(1)
        aggregated = enrich_derived_metrics(aggregate_adapter_runs(adapter, runs))
        results["adapters"][adapter] = aggregated
        if aggregated.get("status") == "ok":
            progress_write(
                f"[gpu-suite] adapter done {adapter} mean_ms={to_float((aggregated.get('latency_ms') or {}).get('mean')):.4f} "
                f"mode={aggregated.get('mode', 'unknown')}"
            )
        else:
            progress_write(
                f"[gpu-suite] adapter done {adapter} status={aggregated.get('status')} "
                f"reason={aggregated.get('reason') or aggregated.get('error', 'n/a')}"
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
                "repeats": results["meta"]["repeats"],
                "adapters": adapters,
                "required_native_adapters": results["meta"]["required_native_adapters"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if progress is not None:
        progress.set_description_str("gpu-suite done")
        progress.set_postfix_str("finalizing artifacts")
        progress.close()
    progress_write(summarize_artifact_write("run", [structured_json, structured_md, structured_svg, structured_meta]))
    progress_write(f"[gpu-suite] complete run_id={run_id}")
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
