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


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"Source results directory not found: {SRC}")

    _clean_destination()
    artifacts = _copy_artifacts()
    cpu_meta, gpu_meta = _find_latest_pair()
    latest_summary = _emit_latest_summary(cpu_meta, gpu_meta)

    manifest = {
        "source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "published_root": str(DST.relative_to(ROOT)).replace("\\", "/"),
        "counts": {
            "total": len(artifacts),
            "images": sum(1 for a in artifacts if a.kind == "image_svg"),
            "metadata": sum(1 for a in artifacts if a.kind == "metadata_json"),
        },
        "artifacts": [a.__dict__ for a in artifacts],
        "latest": latest_summary,
    }

    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DST / "latest-summary.json").write_text(json.dumps(latest_summary, indent=2), encoding="utf-8")

    print(f"Published {manifest['counts']['total']} artifacts to {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
