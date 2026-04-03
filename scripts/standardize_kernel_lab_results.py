#!/usr/bin/env python3
"""Normalize kernel-lab suite results into the benchmark reporting layout."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KERNEL_RESULTS = ROOT / "kernels" / "lab" / "results"
BENCH_RESULTS = ROOT / "benchmark" / "benchmarks" / "results"
RENDERER_PATH = ROOT / "benchmark" / "tools" / "render_kernel_lab_svg.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_kernel_lab_svg", RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load renderer: {RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    render_svg = getattr(module, "render_svg", None)
    if render_svg is None:
        raise RuntimeError("renderer module missing render_svg")
    return render_svg


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid json in {path}: {exc}") from exc


def pick_latest_input(input_path: Path | None) -> Path:
    if input_path:
        return input_path
    candidates = sorted(
        [p for p in KERNEL_RESULTS.glob("*.json") if not p.name.endswith(".metadata.json")],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"no kernel-lab result JSON found under {KERNEL_RESULTS}")
    return candidates[-1]


def classify_kernel_entry(entry: dict[str, Any]) -> str:
    if entry.get("planned"):
        return "planned"
    if isinstance(entry.get("result"), dict):
        return "executed"
    return "unknown"


def extract_phase_mean(result: dict[str, Any], phase: str) -> float | None:
    section = result.get(phase)
    if not isinstance(section, dict):
        return None
    stats = section.get("stats")
    if not isinstance(stats, dict):
        return None
    mean = stats.get("mean_ms")
    return float(mean) if isinstance(mean, (int, float)) else None


def normalize_suite(raw: dict[str, Any], source: Path) -> dict[str, Any]:
    try:
        source_path = str(source.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        source_path = str(source)

    meta = raw.get("meta", {}) if isinstance(raw.get("meta"), dict) else {}
    toolchain = raw.get("toolchain", {}) if isinstance(raw.get("toolchain"), dict) else {}
    kernels_in = raw.get("kernels", [])
    kernels: list[dict[str, Any]] = []
    compile_values: list[float] = []
    run_values: list[float] = []

    if not isinstance(kernels_in, list):
        kernels_in = []

    for item in kernels_in:
        if not isinstance(item, dict):
            continue
        kernel = item.get("kernel", {})
        if not isinstance(kernel, dict):
            kernel = {}
        result = item.get("result", {})
        if not isinstance(result, dict):
            result = {}
        compile_mean = extract_phase_mean(result, "compile")
        run_mean = extract_phase_mean(result, "run")
        if compile_mean is not None:
            compile_values.append(compile_mean)
        if run_mean is not None:
            run_values.append(run_mean)
        kernels.append(
            {
                "kernel": kernel,
                "status": classify_kernel_entry(item),
                "planned": bool(item.get("planned")),
                "result": result if result else None,
                "compile_mean_ms": compile_mean,
                "run_mean_ms": run_mean,
            }
        )

    planned_count = sum(1 for item in kernels if item["planned"])
    executed_count = sum(1 for item in kernels if item["status"] == "executed")
    timestamp = meta.get("timestamp_utc") or dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = meta.get("run_id") or source.stem
    tag = meta.get("label") or meta.get("phase") or "kernel_lab"

    summary = {
        "kernels": len(kernels),
        "planned_kernels": planned_count,
        "executed_kernels": executed_count,
        "compile_mean_ms": round(sum(compile_values) / len(compile_values), 3) if compile_values else None,
        "run_mean_ms": round(sum(run_values) / len(run_values), 3) if run_values else None,
    }

    return {
        "meta": {
            "artifact_kind": "kernel_lab_suite",
            "created_utc": timestamp,
            "timestamp_utc": timestamp,
            "source_path": source_path,
            "run_id": run_id,
            "label": tag,
            "phase": meta.get("phase", "both"),
            "repeats": meta.get("repeats"),
            "warmup": meta.get("warmup"),
            "platform": meta.get("platform"),
            "python": meta.get("python"),
            "manifest": meta.get("manifest"),
            "build_dir": meta.get("build_dir"),
            "results_dir": meta.get("results_dir"),
            "nvcc": meta.get("nvcc"),
            "filters": meta.get("filters", {}),
            "selected_kernels": meta.get("selected_kernels", []),
            "count": meta.get("count", len(kernels)),
        },
        "toolchain": toolchain,
        "summary": summary,
        "kernels": kernels,
    }


def write_markdown(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    summary = payload.get("summary", {})
    toolchain = payload.get("toolchain", {})
    kernels = payload.get("kernels", [])
    lines = [
        "# Kernel Lab Suite Report",
        "",
        f"- Timestamp (UTC): {meta.get('timestamp_utc', 'n/a')}",
        f"- Run ID: {meta.get('run_id', 'n/a')}",
        f"- Label: {meta.get('label', 'n/a')}",
        f"- Phase: {meta.get('phase', 'n/a')}",
        f"- Manifest: {meta.get('manifest', 'n/a')}",
        f"- Build dir: {meta.get('build_dir', 'n/a')}",
        f"- Results dir: {meta.get('results_dir', 'n/a')}",
        f"- NVCC: {toolchain.get('path') or toolchain.get('nvcc') or 'missing'}",
        "",
        "## Summary",
        "",
        f"- Kernels: {summary.get('kernels', 0)}",
        f"- Planned kernels: {summary.get('planned_kernels', 0)}",
        f"- Executed kernels: {summary.get('executed_kernels', 0)}",
        f"- Compile mean: {summary.get('compile_mean_ms', 'n/a')} ms",
        f"- Run mean: {summary.get('run_mean_ms', 'n/a')} ms",
        "",
        "## Kernels",
        "",
    ]
    if not kernels:
        lines.append("- none")
    for item in kernels:
        kernel = item.get("kernel", {})
        name = kernel.get("name", "kernel")
        tags = ",".join(kernel.get("tags", [])) if isinstance(kernel.get("tags"), list) else ""
        if item.get("planned"):
            status = "planned"
        elif item.get("status") == "executed":
            status = "executed"
        else:
            status = item.get("status", "unknown")
        compile_mean = item.get("compile_mean_ms")
        run_mean = item.get("run_mean_ms")
        lines.append(
            f"- `{name}` [{tags or 'untagged'}]: status={status}, "
            f"compile={compile_mean if compile_mean is not None else 'n/a'} ms, "
            f"run={run_mean if run_mean is not None else 'n/a'} ms"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report is normalized from the native kernel-lab JSON format.",
            "- It can be copied into `benchmark/benchmarks/results/` and rendered with the shared reporting path.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize kernel-lab results into benchmark report artifacts")
    parser.add_argument("--input", default="", help="Path to a kernel-lab result JSON file")
    parser.add_argument("--output-root", default=str(BENCH_RESULTS), help="Benchmark results root")
    parser.add_argument("--latest-only", action="store_true", help="Only write latest_* aliases")
    args = parser.parse_args()

    input_path = pick_latest_input(Path(args.input)) if args.input.strip() else pick_latest_input(None)
    raw = read_json(input_path)
    normalized = normalize_suite(raw, input_path)

    output_root = Path(args.output_root)
    json_root = output_root / "json"
    reports_root = output_root / "reports"
    images_root = output_root / "images"
    manifest_root = output_root / "manifest"
    for folder in (json_root, reports_root, images_root, manifest_root):
        folder.mkdir(parents=True, exist_ok=True)

    stamp = f"{normalized['meta']['run_id']}__kernel_lab"
    latest_json = json_root / "latest_kernel_lab.json"
    latest_md = reports_root / "latest_kernel_lab.md"
    latest_svg = images_root / "latest_kernel_lab.svg"
    latest_meta = json_root / "latest_kernel_lab.metadata.json"

    json_path = json_root / f"{stamp}.json"
    md_path = reports_root / f"{stamp}.md"
    svg_path = images_root / f"{stamp}.svg"
    meta_path = json_root / f"{stamp}.metadata.json"

    json_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(write_markdown(normalized), encoding="utf-8")
    render_svg = load_renderer()
    render_svg(normalized, svg_path)
    meta_path.write_text(
        json.dumps(
            {
                "artifact_kind": "kernel_lab_suite",
                "run_id": normalized["meta"]["run_id"],
                "label": normalized["meta"]["label"],
                "source": str(input_path),
                "json": str(json_path),
                "report": str(md_path),
                "svg": str(svg_path),
                "created_utc": normalized["meta"]["created_utc"],
                "kernels": normalized["summary"]["kernels"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Refresh canonical latest aliases.
    latest_json.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    latest_md.write_text(write_markdown(normalized), encoding="utf-8")
    render_svg(normalized, latest_svg)
    latest_meta.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")

    index = {
        "latest_run_id": normalized["meta"]["run_id"],
        "source": str(input_path),
        "json": str(json_path.relative_to(output_root)).replace("\\", "/"),
        "report": str(md_path.relative_to(output_root)).replace("\\", "/"),
        "svg": str(svg_path.relative_to(output_root)).replace("\\", "/"),
        "summary": normalized["summary"],
        "toolchain": normalized["toolchain"],
    }
    (manifest_root / "kernel_lab_results_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    if args.latest_only:
        # The canonical latest aliases already provide the reporting surface.
        pass

    print(f"input: {input_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {svg_path}")
    print(f"wrote {meta_path}")
    print(f"wrote {latest_json}")
    print(f"wrote {latest_md}")
    print(f"wrote {latest_svg}")
    print(f"wrote {latest_meta}")
    print(f"wrote {manifest_root / 'kernel_lab_results_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
