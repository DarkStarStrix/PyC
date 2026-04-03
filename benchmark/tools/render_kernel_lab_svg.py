#!/usr/bin/env python3
"""Render a normalized kernel-lab suite report as SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_ms(value) -> str:
    try:
        return f"{float(value):.3f} ms"
    except (TypeError, ValueError):
        return "n/a"


def _kernel_value(kernel: dict) -> float:
    result = kernel.get("result")
    if not isinstance(result, dict):
        return 0.0
    run = result.get("run")
    if isinstance(run, dict):
        stats = run.get("stats")
        if isinstance(stats, dict) and isinstance(stats.get("mean_ms"), (int, float)):
            return float(stats["mean_ms"])
    compile_ = result.get("compile")
    if isinstance(compile_, dict):
        stats = compile_.get("stats")
        if isinstance(stats, dict) and isinstance(stats.get("mean_ms"), (int, float)):
            return float(stats["mean_ms"])
    return 0.0


def render_svg(results: dict, out_path: Path) -> None:
    meta = results.get("meta", {}) if isinstance(results.get("meta"), dict) else {}
    summary = results.get("summary", {}) if isinstance(results.get("summary"), dict) else {}
    kernels = results.get("kernels", [])
    if not isinstance(kernels, list):
        kernels = []

    rows = []
    values = []
    for item in kernels:
        if not isinstance(item, dict):
            continue
        kernel = item.get("kernel", {})
        if not isinstance(kernel, dict):
            kernel = {}
        name = kernel.get("name", "kernel")
        tags = ",".join(kernel.get("tags", [])) if isinstance(kernel.get("tags"), list) else ""
        value = _kernel_value(item)
        values.append(value)
        status = "planned" if item.get("planned") else "ok" if item.get("result") else "n/a"
        detail = "planned only"
        result = item.get("result")
        if isinstance(result, dict):
            compile_ = result.get("compile", {})
            run = result.get("run", {})
            compile_mean = _fmt_ms(((compile_.get("stats") or {}).get("mean_ms")) if isinstance(compile_, dict) else None)
            run_mean = _fmt_ms(((run.get("stats") or {}).get("mean_ms")) if isinstance(run, dict) else None)
            detail = f"compile {compile_mean} | run {run_mean}"

        rows.append(
            {
                "name": name,
                "tags": tags,
                "value": value,
                "status": status,
                "detail": detail,
            }
        )

    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0

    width = 1180
    chart_x = 390
    chart_y = 150
    row_h = 52
    gap = 18
    chart_w = 700
    height = 260 + len(rows) * (row_h + gap)

    bar_colors = {
        "ok": "#2563eb",
        "planned": "#94a3b8",
        "n/a": "#f59e0b",
    }

    body = []
    for index, row in enumerate(rows):
        y = chart_y + index * (row_h + gap)
        bar_w = int((row["value"] / max_v) * chart_w) if row["value"] > 0 else 0
        color = bar_colors.get(row["status"], "#2563eb")
        body.append(
            f'<text x="30" y="{y + 22}" font-family="Arial, sans-serif" font-size="18" fill="#0f172a" font-weight="700">{row["name"]}</text>'
            f'<text x="30" y="{y + 42}" font-family="Arial, sans-serif" font-size="13" fill="#64748b">{row["tags"] or "untagged"}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{row_h}" fill="{color}" rx="8" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 22}" font-family="Arial, sans-serif" font-size="16" fill="#1f2937">{_fmt_ms(row["value"])}</text>'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 42}" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{row["detail"]}</text>'
        )

    toolchain = results.get("toolchain", {})
    nvcc = toolchain.get("path") or toolchain.get("nvcc") or "missing"
    nvcc_version = toolchain.get("version") or "unknown"
    subtitle = (
        f"run_id={meta.get('run_id', 'n/a')} | phase={meta.get('phase', 'n/a')} | "
        f"repeats={meta.get('repeats', 'n/a')} | warmup={meta.get('warmup', 'n/a')}"
    )
    subtitle_2 = f"filters={meta.get('filters', {})} | nvcc={nvcc} | nvcc_version={nvcc_version}"
    summary_line = (
        f"kernels={summary.get('kernels', len(rows))} | "
        f"planned={summary.get('planned_kernels', 0)} | "
        f"executed={summary.get('executed_kernels', 0)} | "
        f"compile_mean_ms={_fmt_ms(summary.get('compile_mean_ms'))} | "
        f"run_mean_ms={_fmt_ms(summary.get('run_mean_ms'))}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="42" font-family="Arial, sans-serif" font-size="32" font-weight="700" fill="#0f172a">Kernel Lab Suite Report</text>
  <text x="30" y="72" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle}</text>
  <text x="30" y="92" font-family="Arial, sans-serif" font-size="14" fill="#334155">{subtitle_2}</text>
  <text x="30" y="120" font-family="Arial, sans-serif" font-size="14" fill="#475569">{summary_line}</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a kernel-lab suite JSON file to SVG")
    parser.add_argument("--input", required=True, help="Path to normalized kernel-lab suite JSON")
    parser.add_argument("--output", required=True, help="Output SVG path")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    render_svg(payload, Path(args.output))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
