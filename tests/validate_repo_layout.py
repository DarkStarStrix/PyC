#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ALLOWED_ROOT_MARKDOWN = {
    "AGENTS.md",
    "README.md",
    "LICENSE",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "SUPPORT.md",
}

REQUIRED_DOC_DIRS = (
    "architecture",
    "compiler-next",
    "contracts",
    "milestones",
    "plans",
    "reference",
    "reports",
    "roadmap",
)

SHARED_CUDA_BUCKETS = (
    Path("kernels"),
    Path("src/compiler/cutlass_kernels"),
    Path("benchmark/workloads"),
)


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def validate_root_markdown(root: Path, failures: list[str]) -> None:
    for path in sorted(root.glob("*.md")):
        if path.name not in ALLOWED_ROOT_MARKDOWN:
            fail(f"unexpected root markdown file: {path.name}", failures)


def validate_docs_layout(root: Path, failures: list[str]) -> None:
    docs_root = root / "docs"
    if not docs_root.exists():
        fail("missing docs/ directory", failures)
        return

    for dirname in REQUIRED_DOC_DIRS:
        if not (docs_root / dirname).is_dir():
            fail(f"missing docs/{dirname}/ directory", failures)

    for path in sorted(docs_root.glob("*.md")):
        if path.name != "README.md":
            fail(f"stray top-level docs markdown: {path.relative_to(root).as_posix()}", failures)


def validate_cuda_layout(root: Path, failures: list[str]) -> None:
    for bucket in SHARED_CUDA_BUCKETS:
        bucket_path = root / bucket
        if not bucket_path.exists():
            continue
        for path in sorted(bucket_path.glob("*.cu")):
            fail(f"free-floating CUDA file in shared bucket: {path.relative_to(root).as_posix()}", failures)


def validate_required_paths(root: Path, failures: list[str]) -> None:
    expected_paths = (
        root / "web" / "site",
        root / "kernels" / "lab" / "manifests" / "kernels.json",
        root / "kernels" / "prototypes",
    )
    for path in expected_paths:
        if not path.exists():
            fail(f"missing expected path: {path.relative_to(root).as_posix()}", failures)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_repo_layout.py <repo_root>")
        return 2

    root = Path(sys.argv[1]).resolve()
    failures: list[str] = []

    validate_root_markdown(root, failures)
    validate_docs_layout(root, failures)
    validate_cuda_layout(root, failures)
    validate_required_paths(root, failures)

    if failures:
        print("ERROR: repository layout validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("validate_repo_layout: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
