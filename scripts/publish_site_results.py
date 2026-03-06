#!/usr/bin/env python3
"""Publish benchmark artifacts for the static website.

Copies all SVG charts and metadata JSON files from benchmark results into a
single canonical website dataset and emits machine-readable manifests.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmark" / "benchmarks" / "results"
REMOTE_SRC = ROOT / "benchmark" / "remote_results" / "runpod_h100_8x"
DST = ROOT / "website" / "results"
ARTIFACTS = DST / "artifacts"


@dataclass(frozen=True)
class PublishedArtifact:
    kind: str
    source: str
    published: str
    bytes: int


def _clean_destination() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _copy_artifacts() -> list[PublishedArtifact]:
    published: list[PublishedArtifact] = []

    for src in sorted(SRC.rglob("*.svg")):
        rel = src.relative_to(SRC)
        if "archive" in rel.parts:
            continue
        out = ARTIFACTS / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        published.append(
            PublishedArtifact(
                kind="image_svg",
                source=str(rel).replace("\\", "/"),
                published=str(out.relative_to(ROOT)).replace("\\", "/"),
                bytes=src.stat().st_size,
            )
        )

    for src in sorted(SRC.rglob("*.metadata.json")):
        rel = src.relative_to(SRC)
        if "archive" in rel.parts:
            continue
        out = ARTIFACTS / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        published.append(
            PublishedArtifact(
                kind="metadata_json",
                source=str(rel).replace("\\", "/"),
                published=str(out.relative_to(ROOT)).replace("\\", "/"),
                bytes=src.stat().st_size,
            )
        )

    return published


def _extract_adapter_table(result_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    adapters = payload.get("adapters", {})
    rows: list[dict[str, Any]] = []

    for key, entry in adapters.items():
        if entry.get("status") != "ok":
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


def _metadata_created_utc(meta_path: Path) -> str:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(payload.get("created_utc", ""))


def _latest_by_timestamp(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    return max(candidates, key=_metadata_created_utc)


def _find_latest_pair() -> tuple[Path | None, Path | None]:
    cpu_candidates = list(SRC.rglob("*__cpu.metadata.json"))
    gpu_candidates = list(SRC.rglob("*__gpu.metadata.json"))
    return (_latest_by_timestamp(cpu_candidates), _latest_by_timestamp(gpu_candidates))


def _result_json_from_meta(meta: Path) -> Path:
    return meta.with_name(meta.name.replace(".metadata.json", ".json"))


def _emit_latest_summary(cpu_meta: Path | None, gpu_meta: Path | None) -> dict[str, Any]:
    latest: dict[str, Any] = {}

    def add_entry(label: str, meta_path: Path) -> None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        result_json = _result_json_from_meta(meta_path)
        rows = _extract_adapter_table(result_json) if result_json.exists() else []
        rel_meta = str((ARTIFACTS / meta_path.relative_to(SRC)).relative_to(ROOT)).replace("\\", "/")
        latest[label] = {
            "run_id": meta.get("run_id"),
            "created_utc": meta.get("created_utc"),
            "host": meta.get("host"),
            "device": meta.get("device"),
            "batch": meta.get("batch"),
            "hidden": meta.get("hidden"),
            "iters": meta.get("iters"),
            "warmup": meta.get("warmup"),
            "metadata_path": rel_meta,
            "adapters": rows,
        }

    if cpu_meta:
        add_entry("cpu", cpu_meta)
    if gpu_meta:
        add_entry("gpu", gpu_meta)

    return latest


def _distributed_latest_payload() -> dict[str, Any]:
    candidates = list(REMOTE_SRC.glob("campaign_v*/campaign_*/train_metrics.json"))
    if not candidates:
        return {"latest": None, "runs": [], "notes": "No distributed training artifacts found."}

    def sort_key(path: Path) -> str:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("timestamp_utc", ""))

    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(candidates, key=sort_key, reverse=True):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        rel_run_dir = str(run_dir.relative_to(ROOT)).replace("\\", "/")
        run = {
            "run_id": payload.get("run_id"),
            "timestamp_utc": payload.get("timestamp_utc"),
            "mode": payload.get("mode"),
            "dist": payload.get("dist"),
            "backend": payload.get("backend"),
            "world_size": payload.get("world_size"),
            "model_name": payload.get("model_name"),
            "dataset_name": payload.get("dataset_name"),
            "train_runtime_sec": payload.get("train_runtime_sec"),
            "samples_per_sec": payload.get("samples_per_sec"),
            "steps_per_sec": payload.get("steps_per_sec"),
            "tokens_per_sec": payload.get("tokens_per_sec"),
            "eval_loss": payload.get("eval_loss"),
            "loss_final": payload.get("loss_final"),
            "gpu_util_mean": payload.get("gpu_util_mean"),
            "gpu_util_p95": payload.get("gpu_util_p95"),
            "h2d_time_ms_mean": payload.get("h2d_time_ms_mean"),
            "compute_time_ms_mean": payload.get("compute_time_ms_mean"),
            "comm_time_ms_mean": payload.get("comm_time_ms_mean"),
            "idle_gap_ms_mean": payload.get("idle_gap_ms_mean"),
            "idle_gap_ms_p95": payload.get("idle_gap_ms_p95"),
            "paths": {
                "run_dir": rel_run_dir,
                "summary_md": f"{rel_run_dir}/summary.md",
                "summary_svg": f"{rel_run_dir}/summary.svg",
                "metrics_json": f"{rel_run_dir}/train_metrics.json",
                "run_config_json": f"{rel_run_dir}/run_config.json",
                "gpu_telemetry_csv": f"{rel_run_dir}/gpu_telemetry.csv",
            },
        }
        runs.append(run)

    latest_distributed = next((run for run in runs if int(run.get("world_size") or 0) > 1), None)
    latest = latest_distributed or runs[0]
    return {
        "latest": latest,
        "runs": runs[:10],
        "notes": "Latest successful distributed runs from RunPod H100 campaign artifacts.",
    }


def _copy_distributed_run_artifacts(payload: dict[str, Any]) -> list[PublishedArtifact]:
    published: list[PublishedArtifact] = []
    dist_dir = ARTIFACTS / "distributed"
    dist_dir.mkdir(parents=True, exist_ok=True)

    def copy_if_exists(src_path: str, run_id: str, suffix: str) -> str | None:
        src = ROOT / src_path
        if not src.exists():
            return None
        out = dist_dir / f"{run_id}{suffix}"
        shutil.copy2(src, out)
        published.append(
            PublishedArtifact(
                kind="distributed_artifact",
                source=str(src_path).replace("\\", "/"),
                published=str(out.relative_to(ROOT)).replace("\\", "/"),
                bytes=src.stat().st_size,
            )
        )
        return str(out.relative_to(ROOT)).replace("\\", "/")

    for run in payload.get("runs", []):
        run_id = str(run.get("run_id", "unknown"))
        paths = run.get("paths", {})
        run["published"] = {
            "summary_svg": copy_if_exists(str(paths.get("summary_svg", "")), run_id, "__summary.svg"),
            "metrics_json": copy_if_exists(str(paths.get("metrics_json", "")), run_id, "__metrics.json"),
        }

    latest = payload.get("latest", {})
    if latest:
        latest["published"] = next(
            (r.get("published", {}) for r in payload.get("runs", []) if r.get("run_id") == latest.get("run_id")),
            {},
        )
    return published


def _render_distributed_svgs(payload: dict[str, Any]) -> list[PublishedArtifact]:
    images_dir = ARTIFACTS / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    generated: list[PublishedArtifact] = []

    runs = list(payload.get("runs") or [])
    latest = payload.get("latest") or {}

    throughput_svg = images_dir / "latest_distributed_throughput.svg"
    pipeline_svg = images_dir / "latest_distributed_pipeline.svg"

    # Throughput chart from the most recent campaign runs.
    run_rows = [
        run for run in runs
        if isinstance(run.get("tokens_per_sec"), (int, float)) and run.get("tokens_per_sec", 0) > 0
    ]
    run_rows = sorted(run_rows, key=lambda r: float(r.get("tokens_per_sec", 0.0)), reverse=True)[:6]
    if run_rows:
        max_tokens = max(float(run.get("tokens_per_sec", 0.0)) for run in run_rows)
        bar_width = 560
        row_h = 52
        height = 140 + row_h * len(run_rows)
        lines = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="' + str(height) + '" viewBox="0 0 980 ' + str(height) + '">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            '<text x="28" y="38" font-size="24" font-family="Arial, sans-serif" fill="#0f172a">Latest Distributed Throughput</text>',
            '<text x="28" y="62" font-size="12" font-family="Arial, sans-serif" fill="#475569">Source: campaign_v4/v5 run metrics synced from the latest GPU runs.</text>',
        ]
        for idx, run in enumerate(run_rows):
            y = 96 + idx * row_h
            tokens = float(run.get("tokens_per_sec", 0.0))
            bar = 0 if max_tokens <= 0 else int((tokens / max_tokens) * bar_width)
            run_id = str(run.get("run_id", "unknown"))
            lines.append(f'<text x="28" y="{y + 15}" font-size="13" font-family="Arial, sans-serif" fill="#0f172a">{run_id}</text>')
            lines.append(f'<rect x="320" y="{y}" width="{bar}" height="18" rx="4" fill="#0f4f8a"/>')
            lines.append(
                f'<text x="{330 + bar}" y="{y + 14}" font-size="12" font-family="Arial, sans-serif" fill="#1f2937">'
                f'{tokens:.2f} tok/s | {float(run.get("samples_per_sec", 0.0)):.2f} samples/s | ws={int(run.get("world_size", 0) or 0)}</text>'
            )
        lines.append("</svg>")
        throughput_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        throughput_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="220" viewBox="0 0 980 220"><rect width="100%" height="100%" fill="#f8fafc"/><text x="28" y="40" font-size="24" font-family="Arial, sans-serif" fill="#0f172a">Latest Distributed Throughput</text><text x="28" y="84" font-size="14" font-family="Arial, sans-serif" fill="#475569">No throughput rows available.</text></svg>\n',
            encoding="utf-8",
        )

    # Pipeline breakdown chart for the latest selected run.
    metrics = [
        ("compute_time_ms_mean", "Compute", "#0f4f8a"),
        ("comm_time_ms_mean", "Comm", "#d97706"),
        ("idle_gap_ms_mean", "Idle", "#b91c1c"),
        ("h2d_time_ms_mean", "H2D", "#0f766e"),
    ]
    vals = [(label, float(latest.get(key, 0.0) or 0.0), color) for key, label, color in metrics]
    max_v = max((v for _, v, _ in vals), default=0.0)
    bw = 620
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="320" viewBox="0 0 980 320">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="28" y="38" font-size="24" font-family="Arial, sans-serif" fill="#0f172a">Latest Distributed Pipeline Breakdown</text>',
        f'<text x="28" y="62" font-size="12" font-family="Arial, sans-serif" fill="#475569">run_id={latest.get("run_id", "unknown")} model={latest.get("model_name", "-")} dataset={latest.get("dataset_name", "-")}</text>',
    ]
    for idx, (label, value, color) in enumerate(vals):
        y = 96 + idx * 48
        bar = 0 if max_v <= 0 else int((value / max_v) * bw)
        lines.append(f'<text x="28" y="{y + 14}" font-size="13" font-family="Arial, sans-serif" fill="#0f172a">{label}</text>')
        lines.append(f'<rect x="220" y="{y}" width="{bar}" height="18" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{230 + bar}" y="{y + 14}" font-size="12" font-family="Arial, sans-serif" fill="#1f2937">{value:.4f} ms</text>')
    lines.append("</svg>")
    pipeline_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    throughput_pub = str(throughput_svg.relative_to(ROOT)).replace("\\", "/")
    pipeline_pub = str(pipeline_svg.relative_to(ROOT)).replace("\\", "/")
    payload["visuals"] = {
        "throughput_svg": throughput_pub,
        "pipeline_svg": pipeline_pub,
    }

    for src in (throughput_svg, pipeline_svg):
        generated.append(
            PublishedArtifact(
                kind="image_svg_generated",
                source=f"generated/{src.name}",
                published=str(src.relative_to(ROOT)).replace("\\", "/"),
                bytes=src.stat().st_size,
            )
        )

    return generated


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"Source results directory not found: {SRC}")

    _clean_destination()
    artifacts = _copy_artifacts()
    cpu_meta, gpu_meta = _find_latest_pair()
    latest_summary = _emit_latest_summary(cpu_meta, gpu_meta)
    distributed_latest = _distributed_latest_payload()
    artifacts.extend(_copy_distributed_run_artifacts(distributed_latest))
    artifacts.extend(_render_distributed_svgs(distributed_latest))

    manifest = {
        "source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "published_root": str(DST.relative_to(ROOT)).replace("\\", "/"),
        "counts": {
            "total": len(artifacts),
            "images": sum(1 for a in artifacts if a.kind.startswith("image_svg")),
            "metadata": sum(1 for a in artifacts if a.kind == "metadata_json"),
        },
        "artifacts": [a.__dict__ for a in artifacts],
        "latest": latest_summary,
    }

    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DST / "latest-summary.json").write_text(json.dumps(latest_summary, indent=2), encoding="utf-8")
    (DST / "distributed-latest.json").write_text(json.dumps(distributed_latest, indent=2), encoding="utf-8")

    print(f"Published {manifest['counts']['total']} artifacts to {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
