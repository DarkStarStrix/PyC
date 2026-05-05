#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark" / "benchmarks" / "results" / "analysis" / "hopper"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def discover_json_dir(run_dir: Path) -> Path:
    candidate = run_dir / "json"
    return candidate if candidate.is_dir() else run_dir


def discover_artifacts(run_dir: Path) -> tuple[list[Path], Path | None, Path | None]:
    json_dir = discover_json_dir(run_dir)
    shape_paths = []
    progress_path = None
    aggregate_path = None
    for path in sorted(json_dir.glob("*.json")):
        name = path.name
        if name.endswith(".metadata.json"):
            continue
        if name.endswith(".progress.json"):
            progress_path = path
            continue
        payload = read_json(path)
        if "shape" in payload and "adapters" in payload:
            shape_paths.append(path)
        elif "shapes" in payload and "meta" in payload:
            aggregate_path = path
    return shape_paths, progress_path, aggregate_path


def geometric_size(shape: dict) -> float:
    m = to_float(shape.get("m"), 1.0)
    k = to_float(shape.get("k"), 1.0)
    n = to_float(shape.get("n"), 1.0)
    return m * k * n


def classify_scale(shape: dict) -> str:
    name = str(shape.get("name", ""))
    size = geometric_size(shape)
    if "control" in name or size <= 1024**3:
        return "control"
    if size >= 8192 * 4096 * 4096:
        return "transformer"
    return "throughput"


def best_native_adapter(adapters: dict) -> dict | None:
    rows = [row for row in adapters.values() if row.get("status") == "ok" and row.get("mode") == "native"]
    if not rows:
        return None
    return max(rows, key=lambda row: to_float(row.get("throughput_tflops_per_sec")))


def summarize_shape(path: Path) -> dict:
    payload = read_json(path)
    shape = payload["shape"]
    adapters = payload["adapters"]
    pyc = dict(adapters.get("pyc", {}))
    native_best = best_native_adapter(adapters)
    pyc_tflops = to_float(pyc.get("throughput_tflops_per_sec"))
    native_best_tflops = to_float(native_best.get("throughput_tflops_per_sec")) if native_best else 0.0
    ratio = pyc_tflops / native_best_tflops if native_best_tflops > 0 else 0.0
    return {
        "shape": shape,
        "scale": classify_scale(shape),
        "source": str(path),
        "pyc": pyc,
        "native_best": native_best,
        "pyc_to_best_native_ratio": ratio,
        "adapters": adapters,
    }


def build_adapter_health(shapes: list[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for shape in shapes:
        for adapter, payload in shape["adapters"].items():
            row = buckets.setdefault(
                adapter,
                {
                    "adapter": adapter,
                    "ok_native": 0,
                    "ok_proxy": 0,
                    "unavailable": 0,
                    "error": 0,
                },
            )
            status = payload.get("status")
            mode = payload.get("mode")
            if status == "ok" and mode == "native":
                row["ok_native"] += 1
            elif status == "ok" and mode == "proxy":
                row["ok_proxy"] += 1
            elif status == "unavailable":
                row["unavailable"] += 1
            else:
                row["error"] += 1
    return sorted(buckets.values(), key=lambda row: row["adapter"])


def build_rankings(shapes: list[dict]) -> dict:
    native_wins: dict[str, int] = {}
    for shape in shapes:
        winner = shape.get("native_best")
        if not winner:
            continue
        name = winner.get("adapter", "unknown")
        native_wins[name] = native_wins.get(name, 0) + 1
    return {
        "native_win_counts": [
            {"adapter": adapter, "wins": wins}
            for adapter, wins in sorted(native_wins.items(), key=lambda item: (-item[1], item[0]))
        ]
    }


def build_recommendations(shapes: list[dict], adapter_health: list[dict], progress: dict | None) -> list[dict]:
    recommendations: list[dict] = []

    health_by_adapter = {row["adapter"]: row for row in adapter_health}
    cutlass = health_by_adapter.get("cutlass", {})
    if cutlass.get("ok_native", 0) == 0:
        recommendations.append(
            {
                "priority": 1,
                "kind": "trust",
                "title": "Install CUTLASS profiler on the Hopper box",
                "why": "Tier-2 vendor reference is missing, so the arena cannot produce a full native trust ladder.",
            }
        )

    tvm = health_by_adapter.get("tvm", {})
    if tvm.get("ok_native", 0) == 0:
        recommendations.append(
            {
                "priority": 2,
                "kind": "trust",
                "title": "Install native TVM CUDA or remove TVM from trusted Hopper runs",
                "why": "TVM is currently unavailable, so challenger coverage is incomplete.",
            }
        )

    proxy_adapters = [row["adapter"] for row in adapter_health if row["ok_proxy"] > 0]
    if proxy_adapters:
        recommendations.append(
            {
                "priority": 3,
                "kind": "trust",
                "title": "Keep proxy adapters out of trust-grade Hopper runs",
                "why": f"These lanes fell back to proxies: {', '.join(proxy_adapters)}. Either provision native backends or mark them unavailable under strict-native mode.",
            }
        )

    large_shapes = [shape for shape in shapes if shape["scale"] != "control"]
    weak_large_shapes = [shape for shape in large_shapes if shape["pyc_to_best_native_ratio"] < 0.15]
    if weak_large_shapes:
        names = ", ".join(shape["shape"]["name"] for shape in weak_large_shapes[:3])
        recommendations.append(
            {
                "priority": 4,
                "kind": "performance",
                "title": "Route large BF16 Hopper GEMMs to a stronger tensor-core lane",
                "why": f"PyC is under 15% of the best native lane on these large shapes: {names}. The current production path is not competitive once the work is compute-dense.",
            }
        )

    if progress:
        prog = progress.get("progress", {})
        if prog.get("current_adapter") == "pyc" and progress.get("meta", {}).get("status") == "running":
            recommendations.append(
                {
                    "priority": 5,
                    "kind": "reliability",
                    "title": "Debug the PyC stall on the active Hopper GEMM shape",
                    "why": (
                        f"The latest run stopped advancing on shape `{prog.get('current_shape_name', 'unknown')}` "
                        f"while adapter `{prog.get('current_adapter', 'unknown')}` was active."
                    ),
                }
            )

    if not recommendations:
        recommendations.append(
            {
                "priority": 1,
                "kind": "steady-state",
                "title": "No immediate trust blockers detected",
                "why": "The current Hopper bundle is complete enough to drive the next tuning step directly.",
            }
        )
    return recommendations


def render_ratio_svg(shapes: list[dict], out_path: Path) -> None:
    width = 1200
    row_h = 56
    margin = 40
    chart_w = width - (margin * 2) - 280
    height = margin * 2 + max(1, len(shapes)) * row_h
    rows = []
    for index, shape in enumerate(shapes):
        y = margin + index * row_h
        ratio = max(0.0, min(1.0, shape["pyc_to_best_native_ratio"]))
        bar_w = chart_w * ratio
        label = shape["shape"]["name"]
        native_best = shape.get("native_best") or {}
        rows.append(
            f'<text x="{margin}" y="{y + 24}" font-family="Arial, sans-serif" font-size="16" fill="#111">{label}</text>'
            f'<rect x="{margin + 280}" y="{y + 8}" width="{chart_w}" height="22" fill="#eceff4" rx="4" ry="4"/>'
            f'<rect x="{margin + 280}" y="{y + 8}" width="{bar_w:.1f}" height="22" fill="#2563eb" rx="4" ry="4"/>'
            f'<text x="{margin + 280 + chart_w + 12}" y="{y + 24}" font-family="Arial, sans-serif" font-size="14" fill="#111">{ratio:.2%}</text>'
            f'<text x="{margin + 280}" y="{y + 48}" font-family="Arial, sans-serif" font-size="12" fill="#555">pyc={to_float(shape["pyc"].get("throughput_tflops_per_sec")):.4f} TF | native-best={to_float(native_best.get("throughput_tflops_per_sec")):.4f} TF</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<text x="{margin}" y="28" font-family="Arial, sans-serif" font-size="22" fill="#111">PyC vs Best Native Hopper GEMM</text>'
        + "".join(rows)
        + "</svg>"
    )
    out_path.write_text(svg, encoding="utf-8")


def render_markdown(analysis: dict) -> str:
    lines = [
        "# Hopper GEMM Analysis",
        "",
        f"- Run ID: `{analysis['meta'].get('run_id', 'unknown')}`",
        f"- Tag: `{analysis['meta'].get('tag', 'unknown')}`",
        f"- Shape count: `{len(analysis['shapes'])}`",
        f"- Aggregate artifact present: `{analysis['meta'].get('aggregate_present')}`",
        f"- Progress artifact present: `{analysis['meta'].get('progress_present')}`",
        "",
        "## Shape Summary",
        "",
        "| Shape | Scale | Best Native | Best Native TFLOPS | PyC TFLOPS | PyC / Native |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for shape in analysis["shapes"]:
        best = shape.get("native_best") or {}
        lines.append(
            f"| {shape['shape']['name']} | {shape['scale']} | "
            f"{best.get('display_name', best.get('adapter', 'n/a'))} | "
            f"{to_float(best.get('throughput_tflops_per_sec')):.4f} | "
            f"{to_float(shape['pyc'].get('throughput_tflops_per_sec')):.4f} | "
            f"{shape['pyc_to_best_native_ratio']:.2%} |"
        )

    lines.extend(["", "## Adapter Health", "", "| Adapter | Native OK | Proxy OK | Unavailable | Error |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in analysis["adapter_health"]:
        lines.append(
            f"| {row['adapter']} | {row['ok_native']} | {row['ok_proxy']} | {row['unavailable']} | {row['error']} |"
        )

    lines.extend(["", "## Ranked Actions", ""])
    for item in analysis["recommendations"]:
        lines.append(f"- P{item['priority']} `{item['kind']}`: {item['title']} - {item['why']}")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Hopper GEMM analysis bundle from run artifacts")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    shape_paths, progress_path, aggregate_path = discover_artifacts(run_dir)
    if not shape_paths:
        raise SystemExit(f"no Hopper shape JSON artifacts found under {run_dir}")

    shapes = [summarize_shape(path) for path in shape_paths]
    progress = read_json(progress_path) if progress_path and progress_path.exists() else None
    aggregate = read_json(aggregate_path) if aggregate_path and aggregate_path.exists() else None

    run_id = ""
    tag = ""
    if aggregate:
        run_id = str(aggregate.get("meta", {}).get("run_id", ""))
        tag = str(aggregate.get("meta", {}).get("tag", ""))
    if not run_id and progress:
        run_id = str(progress.get("meta", {}).get("run_id", ""))
        tag = str(progress.get("meta", {}).get("tag", ""))
    if not run_id:
        first = shape_paths[0].stem.split("__", 2)
        if len(first) >= 2:
            run_id, tag = first[0], first[1]
        else:
            run_id, tag = "unknown", "unknown"

    output_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_OUTPUT_ROOT / run_id / tag)
    graphs_dir = output_dir / "graphs"
    sheets_dir = output_dir / "sheets"
    rankings_dir = output_dir / "rankings"
    for folder in (graphs_dir, sheets_dir, rankings_dir):
        folder.mkdir(parents=True, exist_ok=True)

    adapter_health = build_adapter_health(shapes)
    recommendations = build_recommendations(shapes, adapter_health, progress)
    rankings = build_rankings(shapes)
    analysis = {
        "meta": {
            "run_id": run_id,
            "tag": tag,
            "run_dir": str(run_dir),
            "aggregate_present": aggregate is not None,
            "progress_present": progress is not None,
        },
        "shapes": shapes,
        "adapter_health": adapter_health,
        "recommendations": recommendations,
        "progress": progress or {},
    }

    ratio_svg_path = graphs_dir / "pyc_vs_best_native.svg"
    analysis_md_path = sheets_dir / "analysis.md"
    analysis_json_path = output_dir / "analysis.json"
    rankings_json_path = rankings_dir / "rankings.json"
    rankings_md_path = rankings_dir / "rankings.md"

    render_ratio_svg(shapes, ratio_svg_path)
    analysis_md_path.write_text(render_markdown(analysis), encoding="utf-8")
    write_json(analysis_json_path, analysis)
    write_json(rankings_json_path, rankings)
    rankings_md_path.write_text(
        "# Hopper GEMM Rankings\n\n"
        + "\n".join(
            f"- {row['adapter']}: {row['wins']} native-shape wins"
            for row in rankings["native_win_counts"]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {analysis_json_path}")
    print(f"wrote {analysis_md_path}")
    print(f"wrote {ratio_svg_path}")
    print(f"wrote {rankings_json_path}")
    print(f"wrote {rankings_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
