#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_cmake_sources.py <cmake_lists> <repo_root>")
        return 2

    cmake_path = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()
    cmake_text = cmake_path.read_text(encoding="utf-8").replace('\\\\', '/')

    source_roots = [
        root / "Core" / "C_Files",
        root / "compiler",
        root / "AI",
        root / "tests" / "compiler_next",
    ]

    missing: list[str] = []
    for src_root in source_roots:
        if not src_root.exists():
            continue
        for c_file in sorted(src_root.rglob("*.c")):
            rel = c_file.relative_to(root).as_posix()
            name = c_file.name
            if rel not in cmake_text and name not in cmake_text:
                missing.append(rel)

    if missing:
        print("ERROR: .c files not referenced by CMakeLists.txt:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("validate_cmake_sources: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
