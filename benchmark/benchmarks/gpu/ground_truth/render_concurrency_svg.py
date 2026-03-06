#!/usr/bin/env python3
"""Render a concise SVG for concurrent latency tails + memory stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mode_color(mode: str) -> str:
    colors = {
        "eager": "#3B82F6",
        "compiled_aten": "#F59E0B",
        "arena": "#10B981",
    }
    return colors.get(mode, "#6B7280")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render concurrency benchmark SVG")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Concurrent Inference Tails")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data.get("results", [])
    if not rows:
        raise SystemExit("No results in input JSON")

    # stable ordering
    rows = sorted(rows, key=lambda r: (r["concurrency"], r["mode"]))

    width = 1600
    row_h = 34
    left = 360
    top = 90
    chart_w = 1120
    h = top + len(rows) * row_h + 120

    max_p = max(float(r["latency_ms"]["p99"]) for r in rows) or 1.0
    max_scale = max_p * 1.1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="38" fill="#e8ecff" font-size="28" font-family="monospace">{args.title}</text>',
        '<text x="360" y="62" fill="#aeb8df" font-size="14" font-family="monospace">Bars: p50, p95, p99 latency (ms). Right labels include memory stability.</text>',
        '<text x="360" y="82" fill="#aeb8df" font-size="14" font-family="monospace">Mode colors: eager=blue, compiled+ATEN=amber, arena=green.</text>',
    ]

    # axis ticks
    for i in range(6):
        v = max_scale * i / 5.0
        x = left + (chart_w * i / 5.0)
        parts.append(f'<line x1="{x:.2f}" y1="{top-12}" x2="{x:.2f}" y2="{h-70}" stroke="#26314f" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{h-48}" fill="#9aa7d8" font-size="12" font-family="monospace" text-anchor="middle">{v:.1f} ms</text>')

    for idx, r in enumerate(rows):
        y = top + idx * row_h
        mode = r["mode"]
        conc = r["concurrency"]
        label = f"{mode} @ c={conc}"
        color = mode_color(mode)
        p50 = float(r["latency_ms"]["p50"])
        p95 = float(r["latency_ms"]["p95"])
        p99 = float(r["latency_ms"]["p99"])
        stable = bool(r.get("memory_stable", False))
        alloc_delta = int(r.get("allocation_event_delta", 0))

        def bar_x(v: float) -> float:
            return left + chart_w * (v / max_scale)

        parts.append(f'<text x="{left-12}" y="{y+12}" fill="#d7defc" font-size="13" font-family="monospace" text-anchor="end">{label}</text>')
        parts.append(f'<line x1="{left}" y1="{y+6}" x2="{bar_x(p50):.2f}" y2="{y+6}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<line x1="{left}" y1="{y+14}" x2="{bar_x(p95):.2f}" y2="{y+14}" stroke="{color}" stroke-width="3" opacity="0.82"/>')
        parts.append(f'<line x1="{left}" y1="{y+22}" x2="{bar_x(p99):.2f}" y2="{y+22}" stroke="{color}" stroke-width="2" opacity="0.62"/>')
        parts.append(
            f'<text x="{left+chart_w+12}" y="{y+14}" fill="#d7defc" font-size="12" font-family="monospace">'
            f"p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} | allocΔ={alloc_delta} | stable={str(stable).lower()}</text>"
        )

    # legend
    ly = h - 24
    parts.append(f'<line x1="{left}" y1="{ly}" x2="{left+70}" y2="{ly}" stroke="#ffffff" stroke-width="4"/>')
    parts.append(f'<text x="{left+78}" y="{ly+4}" fill="#d7defc" font-size="12" font-family="monospace">p50</text>')
    parts.append(f'<line x1="{left+150}" y1="{ly}" x2="{left+220}" y2="{ly}" stroke="#ffffff" stroke-width="3" opacity="0.82"/>')
    parts.append(f'<text x="{left+228}" y="{ly+4}" fill="#d7defc" font-size="12" font-family="monospace">p95</text>')
    parts.append(f'<line x1="{left+300}" y1="{ly}" x2="{left+370}" y2="{ly}" stroke="#ffffff" stroke-width="2" opacity="0.62"/>')
    parts.append(f'<text x="{left+378}" y="{ly+4}" fill="#d7defc" font-size="12" font-family="monospace">p99</text>')

    parts.append("</svg>")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
