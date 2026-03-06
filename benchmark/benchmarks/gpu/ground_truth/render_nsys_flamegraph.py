#!/usr/bin/env python3
"""Render a compact flame-style SVG from nsys kernel summary CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
from pathlib import Path


def color_for(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    r = 120 + int(digest[0:2], 16) % 100
    g = 80 + int(digest[2:4], 16) % 140
    b = 60 + int(digest[4:6], 16) % 160
    return f"rgb({r},{g},{b})"


def parse_kernel_table(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name"):
            header_idx = i
            break
    if header_idx < 0:
        return []

    rows: list[dict] = []
    for raw in lines[header_idx + 1 :]:
        if not raw.strip() or raw.startswith("Processing "):
            break
        cols = next(csv.reader([raw]))
        if len(cols) < 9:
            continue
        rows.append(
            {
                "pct": float(cols[0]),
                "time_ns": float(cols[1]),
                "instances": int(cols[2]),
                "name": cols[8],
            }
        )
    return rows


def render_svg(rows: list[dict], title: str) -> str:
    width = 1600
    flame_x = 40
    flame_y = 80
    flame_h = 70
    usable_w = width - (flame_x * 2)
    legend_row_h = 26
    legend_y0 = flame_y + flame_h + 28
    height = legend_y0 + max(1, len(rows)) * legend_row_h + 30

    total = sum(r["time_ns"] for r in rows) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f1320"/>',
        f'<text x="{flame_x}" y="38" fill="#f3f5ff" font-family="monospace" font-size="24">{html.escape(title)}</text>',
        f'<text x="{flame_x}" y="60" fill="#c2c8e0" font-family="monospace" font-size="14">Nsight Systems kernel-time composition (single captured pass)</text>',
    ]

    x = flame_x
    for r in rows:
        w = max(1.0, usable_w * (r["time_ns"] / total))
        color = color_for(r["name"])
        label = f"{r['pct']:.1f}%"
        parts.append(
            f'<rect x="{x:.2f}" y="{flame_y}" width="{w:.2f}" height="{flame_h}" fill="{color}" stroke="#121212" stroke-width="1"/>'
        )
        if w > 52:
            parts.append(
                f'<text x="{x + 6:.2f}" y="{flame_y + 40}" fill="#111" font-family="monospace" font-size="14">{html.escape(label)}</text>'
            )
        x += w

    for i, r in enumerate(rows):
        y = legend_y0 + i * legend_row_h
        color = color_for(r["name"])
        name = r["name"] if len(r["name"]) <= 116 else r["name"][:113] + "..."
        ms = r["time_ns"] / 1e6
        parts.append(f'<rect x="{flame_x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(
            f'<text x="{flame_x + 22}" y="{y}" fill="#e5e9ff" font-family="monospace" font-size="13">{html.escape(name)} | {ms:.4f} ms | {r["pct"]:.1f}% | instances={r["instances"]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render flame-style SVG from nsys kernel summary CSV")
    parser.add_argument("--input", required=True, help="Path to nsys_stats.csv")
    parser.add_argument("--output", required=True, help="Output SVG path")
    parser.add_argument("--title", default="PyC Ground Truth")
    parser.add_argument("--top", type=int, default=9)
    args = parser.parse_args()

    rows = parse_kernel_table(Path(args.input))
    if not rows:
        raise SystemExit(f"No kernel rows found in {args.input}")
    rows = rows[: max(1, args.top)]
    svg = render_svg(rows, args.title)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
