#!/usr/bin/env python3
"""Normalize benchmark artifacts into a canonical, reproducible layout.

This script consolidates run artifacts scattered across `results/` and
`results/remote_results/` into:

- flat canonical roots: `json/`, `reports/`, `images/`
- per-run canonical tree: `runs/<run_id>/<tag>/`
- latest aliases: `latest/latest_{cpu,gpu}.*`
- machine index: `manifest/results_index.json`

It also re-renders latest CPU/GPU SVG charts from JSON payloads using
`run_gpu_suite.py::write_svg` to keep chart format deterministic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "benchmark" / "benchmarks" / "results"
STEM_RE = re.compile(r"^(?P<run_id>.+)__(?P<tag>.+)$")


@dataclass(frozen=True)
class Artifact:
    stem: str
    kind: str
    path: Path


def parse_artifact(path: Path) -> Artifact | None:
    name = path.name
    kind = ""
    stem = ""
    if name.endswith(".metadata.json"):
        kind = "metadata"
        stem = name[: -len(".metadata.json")]
    elif name.endswith(".json"):
        kind = "json"
        stem = name[: -len(".json")]
    elif name.endswith(".md"):
        kind = "report"
        stem = name[: -len(".md")]
    elif name.endswith(".svg"):
        kind = "image"
        stem = name[: -len(".svg")]
    else:
        return None

    if "__" not in stem:
        return None
    if stem.startswith("latest_"):
        return None
    if stem == "latest_core":
        return None
    return Artifact(stem=stem, kind=kind, path=path)


def parse_canonical_run_artifact(path: Path, results_root: Path) -> Artifact | None:
    try:
        rel = path.relative_to(results_root)
    except ValueError:
        return None
    parts = rel.parts
    # runs/<run_id>/<tag>/{result.json,metadata.json,report.md,chart.svg}
    if len(parts) != 4 or parts[0] != "runs":
        return None
    run_id, tag, name = parts[1], parts[2], parts[3]
    stem = f"{run_id}__{tag}"
    if name == "result.json":
        return Artifact(stem=stem, kind="json", path=path)
    if name == "metadata.json":
        return Artifact(stem=stem, kind="metadata", path=path)
    if name == "report.md":
        return Artifact(stem=stem, kind="report", path=path)
    if name == "chart.svg":
        return Artifact(stem=stem, kind="image", path=path)
    return None


def kind_to_flat_path(results_root: Path, stem: str, kind: str) -> Path:
    if kind == "json":
        return results_root / "json" / f"{stem}.json"
    if kind == "metadata":
        return results_root / "json" / f"{stem}.metadata.json"
    if kind == "report":
        return results_root / "reports" / f"{stem}.md"
    if kind == "image":
        return results_root / "images" / f"{stem}.svg"
    raise ValueError(f"unsupported kind: {kind}")


def choose_best(candidates: list[Path], results_root: Path) -> Path:
    def score(path: Path) -> tuple[int, float]:
        score_base = 0
        rel = path.relative_to(results_root)
        parts = rel.parts
        if len(parts) >= 2 and parts[0] in {"json", "reports", "images"}:
            score_base = 30
        elif "remote_results" in parts:
            score_base = 20
        elif "runs" in parts:
            score_base = 10
        return (score_base, path.stat().st_mtime)

    return max(candidates, key=score)


def copy_if_needed(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and src.resolve() == dst.resolve():
        return False
    if dst.exists():
        if src.stat().st_size == dst.stat().st_size and src.read_bytes() == dst.read_bytes():
            return False
    shutil.copy2(src, dst)
    return True


def load_write_svg(results_root: Path) -> Callable[[dict, Path], None]:
    runner = results_root.parent / "gpu" / "run_gpu_suite.py"
    spec = importlib.util.spec_from_file_location("pyc_run_gpu_suite", runner)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark renderer from {runner}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    write_svg = getattr(module, "write_svg", None)
    if write_svg is None:
        raise RuntimeError("run_gpu_suite.py does not export write_svg")
    return write_svg


def extract_rows(result_payload: dict) -> list[dict]:
    rows = []
    adapters = result_payload.get("adapters", {})
    if not isinstance(adapters, dict):
        return rows
    for key, entry in adapters.items():
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        latency = entry.get("latency_ms", {})
        rows.append(
            {
                "adapter": key,
                "display_name": entry.get("display_name", key),
                "mode": entry.get("mode", "unknown"),
                "mean_ms": latency.get("mean"),
                "p50_ms": latency.get("p50"),
                "p95_ms": latency.get("p95"),
                "throughput_tokens_per_sec": entry.get("throughput_tokens_per_sec"),
            }
        )
    rows.sort(key=lambda row: (row["mean_ms"] is None, row["mean_ms"]))
    return rows


def metadata_created(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("created_utc", ""))


def run() -> int:
    parser = argparse.ArgumentParser(description="Standardize benchmark artifacts layout")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Benchmark results root (default: benchmark/benchmarks/results)",
    )
    parser.add_argument(
        "--prune-flat-history",
        action="store_true",
        help="Keep only latest cpu/gpu run artifacts in flat json/reports/images roots (history remains in runs/).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        raise SystemExit(f"results root not found: {results_root}")

    runs_root = results_root / "runs"
    latest_root = results_root / "latest"
    manifest_root = results_root / "manifest"
    for folder in [results_root / "json", results_root / "reports", results_root / "images", runs_root, latest_root, manifest_root]:
        folder.mkdir(parents=True, exist_ok=True)

    # Collect all possible artifacts except current canonical outputs.
    skip_prefixes = {
        latest_root.resolve(),
        manifest_root.resolve(),
    }
    collected: dict[tuple[str, str], list[Path]] = {}
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(str(resolved).startswith(str(prefix)) for prefix in skip_prefixes):
            continue
        parsed = parse_canonical_run_artifact(path, results_root)
        if parsed is None:
            parsed = parse_artifact(path)
        if parsed is None:
            continue
        collected.setdefault((parsed.stem, parsed.kind), []).append(parsed.path)

    selected: dict[str, dict[str, Path]] = {}
    for (stem, kind), candidates in collected.items():
        selected.setdefault(stem, {})[kind] = choose_best(candidates, results_root)

    copied_count = 0
    run_index: dict[str, dict[str, dict[str, str]]] = {}
    for stem, kinds in sorted(selected.items()):
        match = STEM_RE.match(stem)
        if not match:
            continue
        run_id = match.group("run_id")
        tag = match.group("tag")

        # Ensure canonical flat roots are populated (imports remote-only runs).
        for kind, src in kinds.items():
            flat_dst = kind_to_flat_path(results_root, stem, kind)
            if copy_if_needed(src, flat_dst):
                copied_count += 1

        # Canonical per-run structure.
        run_dir = runs_root / run_id / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        if "json" in kinds:
            copy_if_needed(kinds["json"], run_dir / "result.json")
        if "metadata" in kinds:
            copy_if_needed(kinds["metadata"], run_dir / "metadata.json")
        if "report" in kinds:
            copy_if_needed(kinds["report"], run_dir / "report.md")
        if "image" in kinds:
            copy_if_needed(kinds["image"], run_dir / "chart.svg")

        run_index.setdefault(run_id, {})[tag] = {
            kind: str(kind_to_flat_path(results_root, stem, kind).relative_to(results_root)).replace("\\", "/")
            for kind in kinds.keys()
        }

    # Latest canonical CPU/GPU pair.
    cpu_runs: list[tuple[str, Path]] = []
    gpu_runs: list[tuple[str, Path]] = []
    for stem, kinds in selected.items():
        match = STEM_RE.match(stem)
        if not match or "metadata" not in kinds:
            continue
        run_id = match.group("run_id")
        tag = match.group("tag")
        if tag == "cpu":
            cpu_runs.append((run_id, kinds["metadata"]))
        elif tag == "gpu":
            gpu_runs.append((run_id, kinds["metadata"]))

    cpu_ids = {run_id for run_id, _ in cpu_runs}
    gpu_ids = {run_id for run_id, _ in gpu_runs}
    shared_ids = sorted(cpu_ids & gpu_ids)
    latest_run_id = ""
    if shared_ids:
        meta_by_id_cpu = {run_id: meta for run_id, meta in cpu_runs}
        latest_run_id = max(shared_ids, key=lambda rid: (metadata_created(meta_by_id_cpu[rid]), rid))

    latest_summary: dict[str, object] = {"run_id": latest_run_id}
    if latest_run_id:
        write_svg = load_write_svg(results_root)
        for tag in ["cpu", "gpu"]:
            stem = f"{latest_run_id}__{tag}"
            kinds = selected.get(stem, {})
            if not kinds:
                continue
            json_src = kinds.get("json")
            meta_src = kinds.get("metadata")
            md_src = kinds.get("report")

            if json_src:
                copy_if_needed(json_src, latest_root / f"latest_{tag}.json")
                copy_if_needed(json_src, results_root / "json" / f"latest_{tag}.json")
            if meta_src:
                copy_if_needed(meta_src, latest_root / f"latest_{tag}.metadata.json")
                copy_if_needed(meta_src, results_root / "json" / f"latest_{tag}.metadata.json")
            if md_src:
                copy_if_needed(md_src, latest_root / f"latest_{tag}.md")
                copy_if_needed(md_src, results_root / "reports" / f"latest_{tag}.md")

            # Re-render latest chart from canonical JSON to keep visual format deterministic.
            if json_src:
                payload = json.loads(json_src.read_text(encoding="utf-8"))
                write_svg(payload, latest_root / f"latest_{tag}.svg")
                write_svg(payload, results_root / "images" / f"latest_{tag}.svg")
                # Also refresh run-scoped svg in canonical flat root.
                write_svg(payload, results_root / "images" / f"{stem}.svg")

                latest_summary[tag] = {
                    "stem": stem,
                    "json": str((latest_root / f"latest_{tag}.json").relative_to(results_root)).replace("\\", "/"),
                    "metadata": str((latest_root / f"latest_{tag}.metadata.json").relative_to(results_root)).replace("\\", "/"),
                    "report": str((latest_root / f"latest_{tag}.md").relative_to(results_root)).replace("\\", "/"),
                    "svg": str((latest_root / f"latest_{tag}.svg").relative_to(results_root)).replace("\\", "/"),
                    "rows": extract_rows(payload),
                }

    # Write index manifests.
    results_index = {
        "results_root": str(results_root.relative_to(ROOT)).replace("\\", "/"),
        "run_count": len(run_index),
        "runs": run_index,
        "latest_run_id": latest_run_id,
    }
    (manifest_root / "results_index.json").write_text(json.dumps(results_index, indent=2) + "\n", encoding="utf-8")
    (latest_root / "latest_summary.json").write_text(json.dumps(latest_summary, indent=2) + "\n", encoding="utf-8")

    pruned_count = 0
    if args.prune_flat_history and latest_run_id:
        keep_stems = {f"{latest_run_id}__cpu", f"{latest_run_id}__gpu"}
        for folder in [results_root / "json", results_root / "reports", results_root / "images"]:
            for path in folder.iterdir():
                if not path.is_file():
                    continue
                parsed = parse_artifact(path)
                if parsed is None:
                    continue
                if parsed.stem in keep_stems:
                    continue
                path.unlink()
                pruned_count += 1

    print(f"Standardized runs: {len(run_index)}")
    print(f"Imported/copied artifacts: {copied_count}")
    if latest_run_id:
        print(f"Latest canonical pair: {latest_run_id} (cpu/gpu)")
    else:
        print("No canonical latest cpu/gpu pair found")
    if args.prune_flat_history:
        print(f"Pruned flat historical artifacts: {pruned_count}")
    print(f"Wrote {manifest_root / 'results_index.json'}")
    print(f"Wrote {latest_root / 'latest_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
