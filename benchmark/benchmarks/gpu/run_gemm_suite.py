#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

ROOT = Path(__file__).resolve().parents[3]
PROGRESS_STATE_DIR = ROOT / "infra" / "nexa_insight"
if str(PROGRESS_STATE_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRESS_STATE_DIR))
from progress_state import write_progress_state

from run_gpu_suite import ADAPTER_DIR, DEFAULT_ADAPTERS, git_dirty, git_head, read_gpu_info

RESULTS_DIR = ROOT / "benchmark" / "benchmarks" / "results"
DEFAULT_MATRIX_FILE = ROOT / "benchmark" / "benchmarks" / "gpu" / "configs" / "ada_fp32_gemm_shapes.json"
SOLID_PROGRESS_CHARS = " ▏▎▍▌▋▊▉█"


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def progress_write(message: str) -> None:
    if tqdm is not None and sys.stderr.isatty():
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, flush=True)


def require_progress_support(enabled: bool) -> None:
    if enabled and tqdm is None:
        raise SystemExit("--progress requested but tqdm is not installed in this Python environment")


def short_shape_name(name: str) -> str:
    return (
        name.replace("square-", "sq")
        .replace("tall-skinny-", "ts-")
        .replace("wide-skinny-", "ws-")
        .replace("x", "x")
    )


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile_jitter(p50: float, p95: float) -> tuple[float, float]:
    jitter_ms = max(0.0, p95 - p50)
    jitter_pct = (jitter_ms / p50 * 100.0) if p50 > 0 else 0.0
    return jitter_ms, jitter_pct


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "shape"


def load_matrix(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("matrix file must contain a JSON object")
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("matrix file must define a non-empty shapes array")
    return data


def build_env(
    task: str,
    shape_name: str,
    m: int,
    k: int,
    n: int,
    device: str,
    dtype: str,
    batch: int,
    hidden: int,
    iters: int,
    warmup: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYC_BENCH_TASK"] = task
    env["PYC_BENCH_SHAPE_NAME"] = shape_name
    env["PYC_BENCH_M"] = str(m)
    env["PYC_BENCH_K"] = str(k)
    env["PYC_BENCH_N"] = str(n)
    env["BENCH_TASK"] = task
    env["BENCH_SHAPE_NAME"] = shape_name
    env["BENCH_M"] = str(m)
    env["BENCH_K"] = str(k)
    env["BENCH_N"] = str(n)
    env["BENCH_DEVICE"] = device
    env["BENCH_DTYPE"] = dtype
    env["BENCH_BATCH"] = str(batch)
    env["BENCH_HIDDEN"] = str(hidden)
    env["BENCH_ITERS"] = str(iters)
    env["BENCH_WARMUP"] = str(warmup)
    return env


def run_adapter(adapter: str, device: str, shape: dict, iters: int, warmup: int) -> dict:
    script = ADAPTER_DIR / f"adapter_{adapter}.py"
    if not script.exists():
        return {"status": "error", "adapter": adapter, "error": f"missing adapter script: {script}"}

    m = int(shape["m"])
    k = int(shape["k"])
    n = int(shape["n"])
    batch = m
    hidden = n
    dtype = str(shape.get("dtype", "float32"))
    env = build_env("gemm", str(shape["name"]), m, k, n, device, dtype, batch, hidden, iters, warmup)
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
    proc = run(cmd, env=env)
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"status": "error", "adapter": adapter, "error": proc.stderr.strip() or "adapter failed"}
    try:
        payload = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {
            "status": "error",
            "adapter": adapter,
            "error": "invalid JSON from adapter",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return payload


def aggregate_runs(adapter: str, shape: dict, runs: list[dict]) -> dict:
    if not runs:
        return {"status": "error", "adapter": adapter, "error": "no runs executed"}

    non_ok = [payload for payload in runs if payload.get("status") != "ok"]
    if non_ok:
        first = dict(non_ok[0])
        first["repeat_count"] = len(runs)
        return first

    means = [to_float(payload["latency_ms"]["mean"]) for payload in runs]
    p50s = [to_float(payload["latency_ms"]["p50"]) for payload in runs]
    p95s = [to_float(payload["latency_ms"]["p95"]) for payload in runs]
    mins = [to_float(payload["latency_ms"]["min"]) for payload in runs]
    maxes = [to_float(payload["latency_ms"]["max"]) for payload in runs]
    flops = 2.0 * float(shape["m"]) * float(shape["k"]) * float(shape["n"])
    throughput_flops = [to_float(payload.get("throughput_flops_per_sec")) for payload in runs]
    throughput_tflops = [to_float(payload.get("throughput_tflops_per_sec")) for payload in runs]
    peaks = [int(payload.get("peak_memory_bytes", 0)) for payload in runs]

    out = dict(runs[0])
    out["shape"] = {
        "name": shape["name"],
        "m": int(shape["m"]),
        "k": int(shape["k"]),
        "n": int(shape["n"]),
    }
    out["latency_ms"] = {
        "mean": round(statistics.median(means), 4),
        "p50": round(statistics.median(p50s), 4),
        "p95": round(statistics.median(p95s), 4),
        "min": round(statistics.median(mins), 4),
        "max": round(statistics.median(maxes), 4),
    }
    out["throughput_flops_per_sec"] = round(statistics.median(throughput_flops), 2)
    out["throughput_tflops_per_sec"] = round(statistics.median(throughput_tflops), 6)
    out["peak_memory_bytes"] = max(peaks)
    out["repeat_count"] = len(runs)
    out["repeat_mean_ms_values"] = [round(v, 4) for v in means]
    out["repeat_throughput_flops_per_sec_values"] = [round(v, 2) for v in throughput_flops]
    out["repeat_throughput_tflops_per_sec_values"] = [round(v, 6) for v in throughput_tflops]
    if len(means) >= 2:
        out["repeat_mean_ms_stdev"] = round(statistics.pstdev(means), 4)
    if len(throughput_tflops) >= 2:
        out["repeat_throughput_tflops_per_sec_stdev"] = round(statistics.pstdev(throughput_tflops), 6)
    jitter_ms, jitter_pct = percentile_jitter(out["latency_ms"]["p50"], out["latency_ms"]["p95"])
    out["derived"] = {
        "flops_per_iter": round(flops, 2),
        "jitter_ms_p95_minus_p50": round(jitter_ms, 4),
        "jitter_pct_of_p50": round(jitter_pct, 4),
        "throughput_recomputed_tflops_per_sec": round((flops / (out["latency_ms"]["mean"] / 1000.0)) / 1.0e12, 6)
        if out["latency_ms"]["mean"] > 0
        else 0.0,
        "peak_memory_gib": round(out["peak_memory_bytes"] / float(1024**3), 6),
    }
    return out


def emit_shape_markdown(results: dict) -> str:
    shape = results["shape"]
    lines = [
        "# Ada FP32 GEMM Sweep",
        "",
        f"- Shape: `{shape['name']}`",
        f"- M/K/N: `{shape['m']} / {shape['k']} / {shape['n']}`",
        f"- Run ID: `{results['meta']['run_id']}`",
        f"- Tag: `{results['meta']['tag']}`",
        f"- Device: `{results['meta']['device']}`",
        f"- DType: `{results['meta']['dtype']}`",
        "",
        "| Adapter | Mode | Mean ms | P50 ms | P95 ms | TFLOPS | Peak GiB | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in results["meta"]["adapters"]:
        payload = results["adapters"].get(name, {})
        display = payload.get("display_name", name)
        if payload.get("status") != "ok":
            lines.append(f"| {display} | n/a | n/a | n/a | n/a | n/a | n/a | {payload.get('status', 'error')} |")
            continue
        lat = payload["latency_ms"]
        lines.append(
            f"| {display} | {payload.get('mode', 'unknown')} | {lat['mean']} | {lat['p50']} | {lat['p95']} | "
            f"{payload.get('throughput_tflops_per_sec', 0.0)} | {payload.get('derived', {}).get('peak_memory_gib', 0.0)} | ok |"
        )
    return "\n".join(lines) + "\n"


def emit_run_markdown(results: dict) -> str:
    lines = [
        "# Ada FP32 GEMM Sweep Summary",
        "",
        f"- Run ID: `{results['meta']['run_id']}`",
        f"- Tag: `{results['meta']['tag']}`",
        f"- Device: `{results['meta']['device']}`",
        f"- DType: `{results['meta']['dtype']}`",
        f"- Host: `{results['meta']['host']}`",
        f"- Git: `{results['meta']['git_head']}` dirty={results['meta']['git_dirty']}",
        "",
        "| Shape | Best Adapter | Best TFLOPS | Fastest Adapter | Fastest Mean ms |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for shape in results["shapes"]:
        adapters = shape["adapters"]
        ok = [payload for payload in adapters.values() if payload.get("status") == "ok"]
        if not ok:
            lines.append(f"| {shape['shape']['name']} | n/a | n/a | n/a | n/a |")
            continue
        best_tflops = max(ok, key=lambda p: to_float(p.get("throughput_tflops_per_sec")))
        fastest = min(ok, key=lambda p: to_float(p["latency_ms"]["mean"]))
        lines.append(
            f"| {shape['shape']['name']} | {best_tflops.get('display_name', best_tflops.get('adapter', 'n/a'))} | "
            f"{to_float(best_tflops.get('throughput_tflops_per_sec')):.6f} | "
            f"{fastest.get('display_name', fastest.get('adapter', 'n/a'))} | {fastest['latency_ms']['mean']} |"
        )
    return "\n".join(lines) + "\n"


def render_shape_svg(results: dict, out_path: Path) -> None:
    shape = results["shape"]
    ok_rows = []
    for name in results["meta"]["adapters"]:
        payload = results["adapters"].get(name, {})
        if payload.get("status") == "ok":
            ok_rows.append((payload.get("display_name", name), to_float(payload.get("throughput_tflops_per_sec"))))

    width = 1200
    bar_h = 36
    gap = 16
    chart_x = 380
    chart_y = 110
    chart_w = 760
    rows = max(1, len(ok_rows))
    height = chart_y + rows * (bar_h + gap) + 130
    max_v = max((v for _, v in ok_rows), default=1.0)
    max_v = max(max_v, 1e-9)

    body = []
    for i, (label, value) in enumerate(ok_rows):
        y = chart_y + i * (bar_h + gap)
        bar_w = int((value / max_v) * chart_w) if max_v > 0 else 0
        body.append(
            f'<text x="30" y="{y + 24}" font-family="Arial, sans-serif" font-size="18" fill="#111">{label}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#2563eb" rx="6" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 24}" font-family="Arial, sans-serif" font-size="16" fill="#111">{value:.6f} TFLOPS</text>'
        )

    title = f"Ada FP32 GEMM: {shape['name']}"
    subtitle = f"run_id={results['meta']['run_id']} | git={results['meta']['git_head']} | dirty={results['meta']['git_dirty']}"
    subtitle_2 = f"M={shape['m']} K={shape['k']} N={shape['n']} | dtype={results['meta']['dtype']} | adapters={len(ok_rows)}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="42" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#0f172a">{title}</text>
  <text x="30" y="72" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle}</text>
  <text x="30" y="92" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle_2}</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def render_run_svg(results: dict, out_path: Path) -> None:
    rows = []
    for shape in results["shapes"]:
        ok = [payload for payload in shape["adapters"].values() if payload.get("status") == "ok"]
        if not ok:
            rows.append((shape["shape"]["name"], 0.0))
            continue
        best = max(ok, key=lambda p: to_float(p.get("throughput_tflops_per_sec")))
        rows.append((shape["shape"]["name"], to_float(best.get("throughput_tflops_per_sec"))))

    width = 1200
    bar_h = 36
    gap = 16
    chart_x = 380
    chart_y = 110
    chart_w = 760
    rows_n = max(1, len(rows))
    height = chart_y + rows_n * (bar_h + gap) + 130
    max_v = max((v for _, v in rows), default=1.0)
    max_v = max(max_v, 1e-9)

    body = []
    for i, (label, value) in enumerate(rows):
        y = chart_y + i * (bar_h + gap)
        bar_w = int((value / max_v) * chart_w) if max_v > 0 else 0
        body.append(
            f'<text x="30" y="{y + 24}" font-family="Arial, sans-serif" font-size="18" fill="#111">{label}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#0ea5e9" rx="6" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 24}" font-family="Arial, sans-serif" font-size="16" fill="#111">{value:.6f} TFLOPS</text>'
        )

    title = "Ada FP32 GEMM Sweep Summary"
    subtitle = f"run_id={results['meta']['run_id']} | git={results['meta']['git_head']} | dirty={results['meta']['git_dirty']}"
    subtitle_2 = f"device={results['meta']['device']} | dtype={results['meta']['dtype']} | shapes={len(results['shapes'])}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="42" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#0f172a">{title}</text>
  <text x="30" y="72" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle}</text>
  <text x="30" y="92" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle_2}</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


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
    return f"[gemm-suite] wrote {scope} artifacts: {labels}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Ada FP32 GEMM benchmark matrix")
    parser.add_argument("--matrix-file", default=str(DEFAULT_MATRIX_FILE))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--adapters", default="")
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--tag", default="ada_fp32_gemm")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default=str(RESULTS_DIR))
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require_progress_support(args.progress)

    matrix_path = Path(args.matrix_file)
    matrix = load_matrix(matrix_path)
    adapters = [a.strip() for a in (args.adapters or ",".join(matrix.get("adapters", DEFAULT_ADAPTERS))).split(",") if a.strip()]
    defaults = matrix.get("defaults", {}) if isinstance(matrix.get("defaults"), dict) else {}
    base_iters = int(defaults.get("iters", args.iters))
    base_warmup = int(defaults.get("warmup", args.warmup))
    base_repeats = int(defaults.get("repeats", args.repeats))

    run_id = args.run_id.strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root)
    for folder in (output_root / "json", output_root / "reports", output_root / "images"):
        folder.mkdir(parents=True, exist_ok=True)
    progress_state_path = output_root / "json" / f"{run_id}__{args.tag}.progress.json"
    latest_progress_state_path = output_root / "json" / "latest_ada_fp32_gemm.progress.json"

    shapes = []
    for shape in matrix["shapes"]:
        if not isinstance(shape, dict):
            raise ValueError("every matrix shape must be a JSON object")
        name = str(shape.get("name", "")).strip()
        if not name:
            raise ValueError("every matrix shape must have a non-empty name")
        for key in ("m", "k", "n"):
            if int(shape.get(key, 0)) <= 0:
                raise ValueError(f"shape {name} must define positive {key}")
        shapes.append(
            {
                "name": name,
                "slug": slugify(name),
                "m": int(shape["m"]),
                "k": int(shape["k"]),
                "n": int(shape["n"]),
                "iters": int(shape.get("iters", base_iters)),
                "warmup": int(shape.get("warmup", base_warmup)),
                "repeats": int(shape.get("repeats", base_repeats)),
                "dtype": str(shape.get("dtype", args.dtype)).strip().lower() or args.dtype,
                "note": str(shape.get("note", "")).strip(),
            }
        )

    if args.dry_run:
        plan = {
            "status": "dry-run",
            "matrix_file": str(matrix_path),
            "run_id": run_id,
            "tag": args.tag,
            "device": args.device,
            "adapters": adapters,
            "shapes": shapes,
            "output_root": str(output_root),
            "outputs": {
                "json": str(output_root / "json"),
                "reports": str(output_root / "reports"),
                "images": str(output_root / "images"),
            },
        }
        print(json.dumps(plan, indent=2))
        return 0

    results = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "host": platform.node(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "device": args.device,
            "tag": args.tag,
            "run_id": run_id,
            "git_head": git_head(),
            "git_dirty": git_dirty(),
            "matrix_file": str(matrix_path),
            "matrix_name": matrix.get("name", "ada_fp32_gemm"),
            "task": matrix.get("task", "gemm"),
            "adapters": adapters,
            "dtype": args.dtype,
        },
        "gpu": read_gpu_info(),
        "matrix": matrix,
        "shapes": [],
    }

    total_runs = sum(shape["repeats"] * len(adapters) for shape in shapes)
    progress = create_progress_bar(total_runs, args.progress, "gemm-suite")
    progress_state = {
        "meta": {
            "run_id": run_id,
            "tag": args.tag,
            "device": args.device,
            "dtype": args.dtype,
            "host": results["meta"]["host"],
            "started_utc": results["meta"]["timestamp_utc"],
            "status": "running",
            "shape_count": len(shapes),
            "adapter_count": len(adapters),
            "total_runs": total_runs,
        },
        "progress": {
            "completed_runs": 0,
            "current_shape_index": 0,
            "current_shape_name": "",
            "current_adapter": "",
            "current_repeat": 0,
            "current_repeat_total": 0,
            "last_update_utc": results["meta"]["timestamp_utc"],
        },
        "recent_events": [],
    }
    write_progress_state(progress_state_path, progress_state)
    write_progress_state(latest_progress_state_path, progress_state)
    progress_write(
        f"[gemm-suite] run_id={run_id} device={args.device} dtype={args.dtype} adapters={','.join(adapters)} shapes={len(shapes)}"
    )
    for index, shape in enumerate(shapes, start=1):
        progress_state["progress"].update(
            {
                "current_shape_index": index,
                "current_shape_name": shape["name"],
                "current_adapter": "",
                "current_repeat": 0,
                "current_repeat_total": shape["repeats"],
                "last_update_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        write_progress_state(progress_state_path, progress_state)
        write_progress_state(latest_progress_state_path, progress_state)
        progress_write(
            f"[gemm-suite] shape {index}/{len(shapes)} {short_shape_name(shape['name'])} "
            f"{shape['m']}x{shape['k']}x{shape['n']} {shape['dtype']} "
            f"iters={shape['iters']} warmup={shape['warmup']} reps={shape['repeats']}"
        )
        shape_results = {
            "shape": {k: shape[k] for k in ("name", "m", "k", "n")},
            "meta": {
                "timestamp_utc": results["meta"]["timestamp_utc"],
                "host": results["meta"]["host"],
                "os": results["meta"]["os"],
                "python": results["meta"]["python"],
                "device": results["meta"]["device"],
                "tag": args.tag,
                "run_id": run_id,
                "git_head": results["meta"]["git_head"],
                "git_dirty": results["meta"]["git_dirty"],
                "matrix_name": results["meta"]["matrix_name"],
                "task": results["meta"]["task"],
                "adapters": adapters,
                "dtype": shape["dtype"],
            },
            "gpu": results["gpu"],
            "adapters": {},
            "note": shape.get("note", ""),
        }
        repeat_count = shape["repeats"]
        for adapter in adapters:
            progress_state["progress"].update(
                {
                    "current_adapter": adapter,
                    "current_repeat": 0,
                    "current_repeat_total": repeat_count,
                    "last_update_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            write_progress_state(progress_state_path, progress_state)
            write_progress_state(latest_progress_state_path, progress_state)
            if progress is not None:
                progress.set_postfix_str(f"{short_shape_name(shape['name'])} {adapter} 0/{repeat_count}")
            runs = []
            for repeat_index in range(repeat_count):
                progress_state["progress"].update(
                    {
                        "current_repeat": repeat_index + 1,
                        "last_update_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )
                write_progress_state(progress_state_path, progress_state)
                write_progress_state(latest_progress_state_path, progress_state)
                if progress is not None:
                    progress.set_description_str(f"gemm-suite {index}/{len(shapes)}")
                    progress.set_postfix_str(
                        f"{short_shape_name(shape['name'])} {adapter} {repeat_index + 1}/{repeat_count}"
                    )
                runs.append(run_adapter(adapter, args.device, shape, shape["iters"], shape["warmup"]))
                if progress is not None:
                    progress.update(1)
                progress_state["progress"]["completed_runs"] += 1
            aggregated = aggregate_runs(adapter, shape, runs)
            shape_results["adapters"][adapter] = aggregated
            progress_state["recent_events"].append(
                {
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "shape": shape["name"],
                    "adapter": adapter,
                    "status": aggregated.get("status"),
                    "mean_ms": to_float((aggregated.get("latency_ms") or {}).get("mean")),
                    "throughput_tflops": to_float(aggregated.get("throughput_tflops_per_sec")),
                }
            )
            progress_state["recent_events"] = progress_state["recent_events"][-12:]
            progress_state["progress"]["last_update_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_progress_state(progress_state_path, progress_state)
            write_progress_state(latest_progress_state_path, progress_state)
            if aggregated.get("status") != "ok":
                progress_write(
                    f"[gemm-suite] adapter done {short_shape_name(shape['name'])} {adapter} "
                    f"status={aggregated.get('status')} reason={aggregated.get('reason') or aggregated.get('error', 'n/a')}"
                )

        stamp = f"{run_id}__{args.tag}__{shape['slug']}"
        json_path = output_root / "json" / f"{stamp}.json"
        md_path = output_root / "reports" / f"{stamp}.md"
        svg_path = output_root / "images" / f"{stamp}.svg"
        meta_path = output_root / "json" / f"{stamp}.metadata.json"
        json_path.write_text(json.dumps(shape_results, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(emit_shape_markdown(shape_results), encoding="utf-8")
        render_shape_svg(shape_results, svg_path)
        meta_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "tag": args.tag,
                    "shape": shape_results["shape"],
                    "matrix_file": str(matrix_path),
                    "created_utc": results["meta"]["timestamp_utc"],
                    "host": results["meta"]["host"],
                    "os": results["meta"]["os"],
                    "python": results["meta"]["python"],
                    "git_head": results["meta"]["git_head"],
                    "git_dirty": results["meta"]["git_dirty"],
                    "device": results["meta"]["device"],
                    "dtype": shape["dtype"],
                    "adapters": adapters,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        ok_adapters = [payload for payload in shape_results["adapters"].values() if payload.get("status") == "ok"]
        if ok_adapters:
            best = max(ok_adapters, key=lambda payload: to_float(payload.get("throughput_tflops_per_sec")))
            pyc = shape_results["adapters"].get("pyc")
            pyc_note = ""
            if isinstance(pyc, dict) and pyc.get("status") == "ok":
                pyc_note = f" pyc={to_float(pyc.get('throughput_tflops_per_sec')):.4f}TF"
            progress_write(
                f"[gemm-suite] shape complete {shape['name']} best={best.get('display_name', best.get('adapter', 'n/a'))} "
                f"{to_float(best.get('throughput_tflops_per_sec')):.4f}TF{pyc_note}"
            )
        else:
            progress_write(f"[gemm-suite] shape complete {shape['name']} no successful adapters")
        results["shapes"].append(
            {
                "shape": shape_results["shape"],
                "note": shape_results["note"],
                "adapters": shape_results["adapters"],
            }
        )

    run_stamp = f"{run_id}__{args.tag}"
    run_json = output_root / "json" / f"{run_stamp}.json"
    run_md = output_root / "reports" / f"{run_stamp}.md"
    run_svg = output_root / "images" / f"{run_stamp}.svg"
    run_meta = output_root / "json" / f"{run_stamp}.metadata.json"
    run_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    run_md.write_text(emit_run_markdown(results), encoding="utf-8")
    render_run_svg(results, run_svg)
    run_meta.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tag": args.tag,
                "matrix_file": str(matrix_path),
                "created_utc": results["meta"]["timestamp_utc"],
                "host": results["meta"]["host"],
                "os": results["meta"]["os"],
                "python": results["meta"]["python"],
                "git_head": results["meta"]["git_head"],
                "git_dirty": results["meta"]["git_dirty"],
                "device": results["meta"]["device"],
                "dtype": results["meta"]["dtype"],
                "adapters": adapters,
                "shape_count": len(results["shapes"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    latest_json = output_root / "json" / "latest_ada_fp32_gemm.json"
    latest_md = output_root / "reports" / "latest_ada_fp32_gemm.md"
    latest_svg = output_root / "images" / "latest_ada_fp32_gemm.svg"
    latest_meta = output_root / "json" / "latest_ada_fp32_gemm.metadata.json"
    latest_json.write_text(run_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(run_md.read_text(encoding="utf-8"), encoding="utf-8")
    latest_svg.write_text(run_svg.read_text(encoding="utf-8"), encoding="utf-8")
    latest_meta.write_text(run_meta.read_text(encoding="utf-8"), encoding="utf-8")

    if progress is not None:
        progress.set_description_str("gemm-suite done")
        progress.set_postfix_str("finalizing artifacts")
        progress.close()
    progress_state["meta"]["status"] = "completed"
    progress_state["progress"]["current_adapter"] = ""
    progress_state["progress"]["current_repeat"] = 0
    progress_state["progress"]["last_update_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_progress_state(progress_state_path, progress_state)
    write_progress_state(latest_progress_state_path, progress_state)
    progress_write(summarize_artifact_write("run", [run_json, run_md, run_svg, run_meta]))
    progress_write(summarize_artifact_write("latest", [latest_json, latest_md, latest_svg, latest_meta]))
    progress_write(f"[gemm-suite] complete run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
