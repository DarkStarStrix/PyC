#!/usr/bin/env python3
"""Normalize benchmark remote_results layout into host/run canonical structure."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_ROOT = ROOT / "benchmark" / "benchmarks" / "results" / "remote_results"


@dataclass(frozen=True)
class ParsedArtifact:
    run_id: str
    tag: str
    kind: str


def parse_run_artifact(path: Path) -> ParsedArtifact | None:
    name = path.name
    stem = ""
    kind = ""
    if name.endswith(".metadata.json"):
        stem = name[: -len(".metadata.json")]
        kind = "metadata"
    elif name.endswith(".json"):
        stem = name[: -len(".json")]
        kind = "json"
    elif name.endswith(".md"):
        stem = name[: -len(".md")]
        kind = "report"
    elif name.endswith(".svg"):
        stem = name[: -len(".svg")]
        kind = "image"
    else:
        return None

    if "__" not in stem:
        return None
    run_id, tag = stem.rsplit("__", 1)
    return ParsedArtifact(run_id=run_id, tag=tag, kind=kind)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return
    if src.stat().st_size == dst.stat().st_size and file_sha256(src) == file_sha256(dst):
        return
    if src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)


def merge_tree(src_dir: Path, dst_dir: Path) -> None:
    for item in sorted(src_dir.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src_dir)
        copy_or_replace(item, dst_dir / rel)


def run() -> int:
    parser = argparse.ArgumentParser(description="Normalize remote_results host/run layout")
    parser.add_argument(
        "--remote-root",
        default=str(DEFAULT_REMOTE_ROOT),
        help="remote_results directory (default: benchmark/benchmarks/results/remote_results)",
    )
    args = parser.parse_args()

    remote_root = Path(args.remote_root).resolve()
    if not remote_root.exists():
        raise SystemExit(f"remote results root missing: {remote_root}")

    hosts_root = remote_root / "hosts"
    archive_root = remote_root / "archive" / "root_legacy"
    hosts_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    host_dirs = sorted([p for p in remote_root.iterdir() if p.is_dir() and p.name.startswith("host") and p.name != "hosts"])
    moved_artifacts = 0
    moved_logs = 0
    moved_legacy = 0

    for host_src in host_dirs:
        host_dst = hosts_root / host_src.name
        runs_dst = host_dst / "runs"
        logs_dst = host_dst / "logs"
        legacy_dst = host_dst / "legacy"
        runs_dst.mkdir(parents=True, exist_ok=True)
        logs_dst.mkdir(parents=True, exist_ok=True)
        legacy_dst.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in host_src.rglob("*") if p.is_file()])
        for src in files:
            parsed = parse_run_artifact(src)
            if parsed is not None:
                run_dir = runs_dst / parsed.run_id / parsed.tag
                if parsed.kind == "json":
                    dst = run_dir / "result.json"
                elif parsed.kind == "metadata":
                    dst = run_dir / "metadata.json"
                elif parsed.kind == "report":
                    dst = run_dir / "report.md"
                else:
                    dst = run_dir / "chart.svg"
                copy_or_replace(src, dst)
                moved_artifacts += 1
                continue

            if src.suffix == ".log":
                copy_or_replace(src, logs_dst / src.name)
                moved_logs += 1
                continue

            rel = src.relative_to(host_src)
            copy_or_replace(src, legacy_dst / rel)
            moved_legacy += 1

        shutil.rmtree(host_src)

    # If host folders were already cleaned or unavailable, hydrate host runs from canonical results/runs.
    if not host_dirs:
        runs_root = remote_root.parent / "runs"
        if runs_root.exists():
            for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
                run_id = run_dir.name
                run_id_lower = run_id.lower()
                if "host89" in run_id_lower:
                    host = "host89"
                elif "3090" in run_id_lower or run_id in {"20260218T231940Z", "20260218T_fix2"}:
                    host = "host3090"
                else:
                    host = "misc"

                for tag_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
                    dst_run = hosts_root / host / "runs" / run_id / tag_dir.name
                    dst_run.mkdir(parents=True, exist_ok=True)
                    if (tag_dir / "result.json").exists():
                        copy_or_replace(tag_dir / "result.json", dst_run / "result.json")
                        moved_artifacts += 1
                    if (tag_dir / "metadata.json").exists():
                        copy_or_replace(tag_dir / "metadata.json", dst_run / "metadata.json")
                        moved_artifacts += 1
                    if (tag_dir / "report.md").exists():
                        copy_or_replace(tag_dir / "report.md", dst_run / "report.md")
                        moved_artifacts += 1
                    if (tag_dir / "chart.svg").exists():
                        copy_or_replace(tag_dir / "chart.svg", dst_run / "chart.svg")
                        moved_artifacts += 1

    # Move remaining legacy root-level files/dirs into archive.
    for item in sorted(remote_root.iterdir()):
        if item.name in {"hosts", "archive"}:
            continue
        if item.is_file():
            copy_or_replace(item, archive_root / "files" / item.name)
            item.unlink()
            continue
        if item.is_dir():
            dst = archive_root / item.name
            merge_tree(item, dst)
            shutil.rmtree(item)

    print(f"Normalized hosts: {len(host_dirs)}")
    print(f"Copied run artifacts: {moved_artifacts}")
    print(f"Copied logs: {moved_logs}")
    print(f"Copied legacy files: {moved_legacy}")
    print(f"Wrote hosts tree: {hosts_root}")
    print(f"Wrote root legacy archive: {archive_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
