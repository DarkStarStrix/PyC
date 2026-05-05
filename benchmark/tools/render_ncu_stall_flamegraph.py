#!/usr/bin/env python3
"""Render a classic flamegraph-style SVG from NCU warp-stall metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
from dataclasses import dataclass, field
from pathlib import Path


STALL_CATEGORY_MAP = {
    "mio_throttle": "Memory Pipe",
    "lg_throttle": "Memory Pipe",
    "imc_miss": "Memory Pipe",
    "short_scoreboard": "Dependency Latency",
    "long_scoreboard": "Dependency Latency",
    "not_selected": "Scheduler",
    "dispatch_stall": "Scheduler",
    "no_instruction": "Scheduler",
    "wait": "Synchronization",
    "barrier": "Synchronization",
    "math_pipe_throttle": "Compute Pipe",
    "branch_resolving": "Control Flow",
    "drain": "Control Flow",
}


REASON_LABELS = {
    "mio_throttle": "MIO Throttle",
    "lg_throttle": "LG Throttle",
    "imc_miss": "IMC Miss",
    "short_scoreboard": "Short Scoreboard",
    "long_scoreboard": "Long Scoreboard",
    "not_selected": "Not Selected",
    "dispatch_stall": "Dispatch Stall",
    "no_instruction": "No Instruction",
    "wait": "Wait",
    "barrier": "Barrier",
    "math_pipe_throttle": "Math Pipe Throttle",
    "branch_resolving": "Branch Resolving",
    "drain": "Drain",
}


@dataclass
class Node:
    name: str
    value: float
    subtitle: str = ""
    children: list["Node"] = field(default_factory=list)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)


def color_for(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    r = 180 + int(digest[0:2], 16) % 60
    g = 180 + int(digest[2:4], 16) % 60
    b = 120 + int(digest[4:6], 16) % 80
    return f"rgb({r},{g},{b})"


def parse_stalls(path: Path) -> list[tuple[str, float]]:
    rows = list(csv.reader(path.open()))
    if len(rows) < 3:
        raise SystemExit(f"Unexpected NCU CSV layout in {path}")
    header, _, values = rows
    result = []
    for i, key in enumerate(header):
        prefix = "smsp__average_warps_issue_stalled_"
        suffix = "_per_issue_active.ratio"
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        reason = key[len(prefix) : -len(suffix)]
        if reason == "selected":
            continue
        value = float(values[i] or 0.0)
        if value <= 0.0:
            continue
        result.append((reason, value))
    result.sort(key=lambda item: item[1], reverse=True)
    return result


def build_tree(stalls: list[tuple[str, float]]) -> Node:
    total = sum(value for _, value in stalls) or 1.0
    categories: dict[str, list[tuple[str, float]]] = {}
    for reason, value in stalls:
        category = STALL_CATEGORY_MAP.get(reason, "Other")
        categories.setdefault(category, []).append((reason, value))

    category_nodes = []
    for category, items in sorted(categories.items(), key=lambda kv: sum(v for _, v in kv[1]), reverse=True):
        cat_total = sum(value for _, value in items)
        children = []
        for reason, value in sorted(items, key=lambda item: item[1], reverse=True):
            pct = value / total * 100.0
            children.append(
                Node(
                    name=REASON_LABELS.get(reason, reason.replace("_", " ").title()),
                    value=value,
                    subtitle=f"{pct:.2f}% | {value:.4f} stall-ratio units",
                )
            )
        category_nodes.append(
            Node(
                name=category,
                value=cat_total,
                subtitle=f"{cat_total / total * 100.0:.2f}% aggregated stall budget",
                children=children,
            )
        )

    return Node(
        name="Kernel Stall Budget",
        value=total,
        subtitle="100.00% full-pass sampled warp stall budget",
        children=category_nodes,
    )


def gather_levels(node: Node, depth: int = 0, levels: dict[int, list[Node]] | None = None) -> dict[int, list[Node]]:
    if levels is None:
        levels = {}
    levels.setdefault(depth, []).append(node)
    for child in node.children:
        gather_levels(child, depth + 1, levels)
    return levels


def render_group(x: float, y: float, w: float, h: float, name: str, subtitle: str, show_text: bool) -> str:
    info = html.escape(f"{name} ({subtitle})", quote=True)
    title = html.escape(f"{name} ({subtitle})")
    color = color_for(name)
    text = ""
    if show_text:
        text = (
            f'\n<text text-anchor="" x="{x + 3:.1f}" y="{y + h - 4.5:.1f}" '
            f'font-size="12" font-family="Verdana" fill="rgb(0,0,0)">{html.escape(name)}</text>'
        )
    return (
        f'<g class="func_g" onmouseover="s(\'{info}\')" onmouseout="c()">\n'
        f'<title>{title}</title><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{color}" rx="2" ry="2" />{text}\n'
        f'</g>'
    )


def render_tree(node: Node, x: float, y: float, width: float, frame_h: float, frame_pad: float, parts: list[str]) -> None:
    show_text = width >= 42.0
    parts.append(render_group(x, y, width, frame_h, node.name, node.subtitle, show_text))
    if not node.children:
        return
    child_x = x
    for child in node.children:
        child_w = width * (child.value / node.value) if node.value > 0 else 0.0
        render_tree(child, child_x, y - frame_pad, child_w, frame_h, frame_pad, parts)
        child_x += child_w


def render_svg(root: Node, title: str) -> str:
    width = 1200
    left = 10.0
    right = 10.0
    frame_h = 15.0
    frame_pad = 16.0
    top = 50.0
    depth = root.depth()
    root_y = top + (depth - 1) * frame_pad
    height = int(root_y + frame_h + 36.0 + 24.0)
    usable_w = width - left - right

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

    render_tree(root, left, root_y, usable_w, frame_h, frame_pad, parts)
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a classic flamegraph from NCU warp-stall metrics")
    parser.add_argument("--input", required=True, help="Path to NCU raw CSV export")
    parser.add_argument("--output", required=True, help="Output SVG path")
    parser.add_argument("--title", default="NCU Stall Flame Graph")
    args = parser.parse_args()

    stalls = parse_stalls(Path(args.input))
    if not stalls:
        raise SystemExit(f"No non-zero stall metrics found in {args.input}")
    root = build_tree(stalls)
    svg = render_svg(root, args.title)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
