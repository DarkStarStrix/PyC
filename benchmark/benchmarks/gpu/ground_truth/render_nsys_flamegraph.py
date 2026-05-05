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


def render_legend(parts: list[str], rows: list[dict], flame_x: int, legend_y0: int, legend_row_h: int) -> None:
    for i, r in enumerate(rows):
        y = legend_y0 + i * legend_row_h
        color = color_for(r["name"])
        name = r["name"] if len(r["name"]) <= 116 else r["name"][:113] + "..."
        ms = r["time_ns"] / 1e6
        parts.append(f'<rect x="{flame_x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(
            f'<text x="{flame_x + 22}" y="{y}" fill="#e5e9ff" font-family="monospace" font-size="13">{html.escape(name)} | {ms:.4f} ms | {r["pct"]:.1f}% | instances={r["instances"]}</text>'
        )


def classic_group(x: float, y: float, w: float, h: float, name: str, subtitle: str, show_text: bool) -> str:
    color = color_for(name)
    label = html.escape(name)
    info = html.escape(f"{name} ({subtitle})", quote=True)
    title = html.escape(f"{name} ({subtitle})")
    text = ""
    if show_text:
        text = (
            f'\n<text text-anchor="" x="{x + 3:.1f}" y="{y + h - 4.5:.1f}" '
            f'font-size="12" font-family="Verdana" fill="rgb(0,0,0)">{label}</text>'
        )
    return (
        f'<g class="func_g" onmouseover="s(\'{info}\')" onmouseout="c()">\n'
        f'<title>{title}</title><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{color}" rx="2" ry="2" />{text}\n'
        f'</g>'
    )


def render_classic_flame(rows: list[dict], title: str) -> str:
    width = 1200
    frame_h = 15.0
    frame_pad = 16.0
    left = 10.0
    top = 50.0
    usable_w = width - 20.0
    levels = 2
    bottom_text_y = top + levels * frame_pad + 90.0
    height = int(bottom_text_y + 20.0)
    root_y = top + frame_pad
    child_y = root_y - frame_pad
    total = sum(r["time_ns"] for r in rows) or 1.0

    parts = [
        '<?xml version="1.0" standalone="no"?>',
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
        f'<svg version="1.1" width="{width}" height="{height}" onload="init(evt)" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">',
        '<defs >',
        '\t<linearGradient id="background" y1="0" y2="1" x1="0" x2="0" >',
        '\t\t<stop stop-color="#eeeeee" offset="5%" />',
        '\t\t<stop stop-color="#e0e0ff" offset="95%" />',
        '\t</linearGradient>',
        '</defs>',
        '<style type="text/css">',
        '\t.func_g:hover { stroke:black; stroke-width:0.5; }',
        '</style>',
        '<script type="text/ecmascript">',
        '<![CDATA[',
        '\tvar details;',
        '\tfunction init(evt) { details = document.getElementById("details").firstChild; }',
        '\tfunction s(info) { details.nodeValue = "Function: " + info; }',
        "\tfunction c() { details.nodeValue = ' '; }",
        ']]>',
        '</script>',
        f'<rect x="0.0" y="0" width="{width:.1f}" height="{height:.1f}" fill="url(#background)"  />',
        f'<text text-anchor="middle" x="{width / 2:.0f}" y="24" font-size="17" font-family="Verdana" fill="rgb(0,0,0)"  >{html.escape(title)}</text>',
        f'<text text-anchor="" x="10" y="{height - 17}" font-size="12" font-family="Verdana" fill="rgb(0,0,0)" id="details" > </text>',
    ]

    root_subtitle = f"{sum(r['pct'] for r in rows):.2f}% sampled stall budget"
    parts.append(classic_group(left, root_y, usable_w, frame_h, "Kernel Stall Budget", root_subtitle, True))

    x = left
    for r in rows:
        w = max(0.5, usable_w * (r["time_ns"] / total))
        subtitle = f"{r['pct']:.2f}% | {r['time_ns'] / 1e6:.4f} ms | instances={r['instances']}"
        parts.append(classic_group(x, child_y, w, frame_h, r["name"], subtitle, w > 42.0))
        x += w

    parts.append("</svg>")
    return "\n".join(parts)


def render_svg(rows: list[dict], title: str, style: str) -> str:
    if style == "classic":
        return render_classic_flame(rows, title)

    width = 1600
    flame_x = 40
    flame_y = 80
    flame_h = 70
    flame_gap = 8
    usable_w = width - (flame_x * 2)
    legend_row_h = 26
    flame_layers = max(1, len(rows))
    flame_area_h = flame_layers * flame_h + max(0, flame_layers - 1) * flame_gap
    legend_y0 = flame_y + flame_area_h + 28
    height = legend_y0 + max(1, len(rows)) * legend_row_h + 30

    total = sum(r["time_ns"] for r in rows) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f1320"/>',
        f'<text x="{flame_x}" y="38" fill="#f3f5ff" font-family="monospace" font-size="24">{html.escape(title)}</text>',
        f'<text x="{flame_x}" y="60" fill="#c2c8e0" font-family="monospace" font-size="14">Nsight Systems kernel-time composition (single captured pass)</text>',
    ]

    if style == "pyramid":
        center_x = width / 2.0
        for i, r in enumerate(rows):
            w = max(1.0, usable_w * (r["time_ns"] / total))
            x = center_x - (w / 2.0)
            y = flame_y + i * (flame_h + flame_gap)
            color = color_for(r["name"])
            label = f"{r['pct']:.1f}%"
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{flame_h}" rx="4" ry="4" fill="{color}" stroke="#121212" stroke-width="1"/>'
            )
            if w > 120:
                parts.append(
                    f'<text x="{center_x:.2f}" y="{y + 41:.2f}" text-anchor="middle" fill="#111" font-family="monospace" font-size="14">{html.escape(label)}</text>'
                )
    else:
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

    render_legend(parts, rows, flame_x, legend_y0, legend_row_h)

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render flame-style SVG from nsys kernel summary CSV")
    parser.add_argument("--input", required=True, help="Path to nsys_stats.csv")
    parser.add_argument("--output", required=True, help="Output SVG path")
    parser.add_argument("--title", default="PyC Ground Truth")
    parser.add_argument("--top", type=int, default=9)
    parser.add_argument("--style", choices=["bars", "pyramid", "classic"], default="bars")
    args = parser.parse_args()

    rows = parse_kernel_table(Path(args.input))
    if not rows:
        raise SystemExit(f"No kernel rows found in {args.input}")
    rows = rows[: max(1, args.top)]
    svg = render_svg(rows, args.title, args.style)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
