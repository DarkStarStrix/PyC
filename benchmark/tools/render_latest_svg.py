#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "benchmark" / "results" / "latest.json"
OUT_BENCH = ROOT / "benchmark" / "results" / "latest.svg"
OUT_DOCS = ROOT / "docs" / "assets" / "performance-latest.svg"


def fmt_ms(v):
    return f"{v:.3f} ms"


def fmt_bytes(v):
    if v >= 1024 * 1024:
        return f"{v / (1024 * 1024):.2f} MB"
    if v >= 1024:
        return f"{v / 1024:.2f} KB"
    return f"{v} B"


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))

    metrics = [
        ("Configure", data["configure_ms"]),
        ("Build", data["build_ms"]),
        ("pyc smoke mean", data["smoke_pyc"]["mean_ms"]),
        ("microbench mean", data["microbench"]["mean_ms"]),
    ]

    max_v = max(v for _, v in metrics) or 1.0

    width = 980
    height = 620
    chart_x = 280
    chart_y = 130
    bar_h = 52
    gap = 30
    chart_w = 620

    colors = ["#2f80ed", "#27ae60", "#f2994a", "#eb5757"]

    rows = []
    for i, (label, value) in enumerate(metrics):
        y = chart_y + i * (bar_h + gap)
        bar_w = int((value / max_v) * chart_w)
        rows.append(
            f'<text x="40" y="{y + 33}" font-family="Arial, sans-serif" font-size="24" fill="#111">{label}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{colors[i]}" rx="8" />'
            f'<text x="{chart_x + bar_w + 14}" y="{y + 34}" font-family="Arial, sans-serif" font-size="22" fill="#222">{fmt_ms(value)}</text>'
        )

    subtitle = (
        f"UTC {data['meta']['timestamp_utc']}  |  platform: {data['meta']['platform']}  |  "
        f"repeats: {data['meta']['repeats']}  |  rounds: {data['meta']['micro_rounds']}"
    )

    artifact_line = (
        f"Artifacts: pyc={fmt_bytes(data['artifacts']['pyc_bytes'])}, "
        f"pyc_core={fmt_bytes(data['artifacts']['pyc_core_bytes'])}"
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7fafc" />
  <text x="40" y="56" font-family="Arial, sans-serif" font-size="36" fill="#0f172a" font-weight="700">PyC Stable Core Benchmark Snapshot</text>
  <text x="40" y="92" font-family="Arial, sans-serif" font-size="16" fill="#334155">{subtitle}</text>
  {''.join(rows)}
  <text x="40" y="560" font-family="Arial, sans-serif" font-size="20" fill="#1f2937">{artifact_line}</text>
  <text x="40" y="590" font-family="Arial, sans-serif" font-size="16" fill="#64748b">Source: benchmark/results/latest.json</text>
</svg>
'''

    OUT_BENCH.write_text(svg, encoding="utf-8")
    OUT_DOCS.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT_BENCH}")
    print(f"wrote {OUT_DOCS}")


if __name__ == "__main__":
    main()
