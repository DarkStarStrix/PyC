#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "benchmark" / "benchmarks" / "results"


def run_backend(bin_path: Path, backend: str, backend_path: Path, iters: int, count: int, repeats: int) -> dict:
    runs = []
    for _ in range(repeats):
        cmd = [
            str(bin_path),
            "--backend",
            backend,
            "--backend-path",
            str(backend_path),
            "--iters",
            str(iters),
            "--count",
            str(count),
            "--config-json",
            "{\"strict\":false}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return {"status": "error", "error": proc.stderr.strip() or proc.stdout.strip(), "returncode": proc.returncode}
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        runs.append(payload)

    per_iter = [r["per_iter_us"] for r in runs]
    ok = sum(r["ok"] for r in runs)
    hardware = sum(r["hardware"] for r in runs)
    invalid = sum(r["invalid"] for r in runs)
    timeout = sum(r["timeout"] for r in runs)
    return {
        "status": "ok",
        "repeats": repeats,
        "mean_per_iter_us": round(sum(per_iter) / len(per_iter), 4),
        "min_per_iter_us": round(min(per_iter), 4),
        "max_per_iter_us": round(max(per_iter), 4),
        "ok": ok,
        "hardware": hardware,
        "invalid": invalid,
        "timeout": timeout,
        "path": str(backend_path),
    }


def render_svg(results: dict, out_path: Path) -> None:
    backends = list(results["backends"].keys())
    width = 980
    height = 170 + (len(backends) * 80)
    left = 220
    bar_max = max((results["backends"][b].get("mean_per_iter_us", 0.0) for b in backends), default=1.0)
    bar_max = max(bar_max, 1.0)
    row_h = 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="28" y="36" font-size="22" font-family="Arial, sans-serif" fill="#0f172a">Distributed Comm Backend Bench</text>',
        f'<text x="28" y="58" font-size="12" font-family="Arial, sans-serif" fill="#475569">run_id={results["meta"]["run_id"]} host={results["meta"]["host"]} git={results["meta"]["git_head"]}</text>',
    ]
    for i, backend in enumerate(backends):
        y = 92 + i * row_h
        payload = results["backends"][backend]
        mean = float(payload.get("mean_per_iter_us", 0.0))
        w = int((mean / bar_max) * (width - left - 80))
        status_fill = "#0ea5e9" if payload.get("ok", 0) > 0 else "#f97316"
        parts.append(f'<text x="28" y="{y+24}" font-size="14" font-family="Arial, sans-serif" fill="#0f172a">{backend}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w}" height="26" rx="4" fill="{status_fill}"/>')
        parts.append(
            f'<text x="{left + max(w + 8, 8)}" y="{y+18}" font-size="12" font-family="Arial, sans-serif" fill="#334155">'
            f'mean={mean:.3f}us ok={payload.get("ok",0)} hw={payload.get("hardware",0)} invalid={payload.get("invalid",0)}</text>'
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run distributed communication backend microbenchmarks")
    parser.add_argument("--build-dir", default=str(ROOT / "build-distributed"))
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tag", default="distributed_comm")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    bench_bin = build_dir / "pyc_distributed_comm_bench"
    if not bench_bin.exists():
        raise SystemExit(f"missing benchmark binary: {bench_bin}")

    ext = ".dll" if platform.system().lower().startswith("win") else (".dylib" if platform.system() == "Darwin" else ".so")
    backends = {
        "nccl": build_dir / f"libpyc_comm_backend_nccl{ext}",
        "rccl": build_dir / f"libpyc_comm_backend_rccl{ext}",
        "mpi": build_dir / f"libpyc_comm_backend_mpi{ext}",
        "stub": build_dir / f"libpyc_comm_backend_stub{ext}",
    }

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamp = f"{run_id}__{args.tag}"

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "tag": args.tag,
            "host": socket.gethostname(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip(),
            "git_dirty": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip() != "",
            "iters": args.iters,
            "count": args.count,
            "repeats": args.repeats,
        },
        "backends": {},
    }

    for name, path in backends.items():
        if not path.exists():
            payload["backends"][name] = {"status": "unavailable", "error": f"missing backend binary: {path}"}
            continue
        payload["backends"][name] = run_backend(bench_bin, name, path, args.iters, args.count, args.repeats)

    json_path = RESULTS_ROOT / "json" / f"{stamp}.json"
    md_path = RESULTS_ROOT / "reports" / f"{stamp}.md"
    svg_path = RESULTS_ROOT / "images" / f"{stamp}.svg"
    for p in (json_path.parent, md_path.parent, svg_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Distributed Comm Backend Bench",
        "",
        f"- Run ID: `{run_id}`",
        f"- Host: `{payload['meta']['host']}`",
        f"- Iters: `{args.iters}`",
        f"- Count: `{args.count}`",
        "",
        "| Backend | Mean us | OK | HW | Invalid | Timeout |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in payload["backends"].items():
        if r.get("status") != "ok":
            lines.append(f"| {name} | n/a | 0 | 0 | 0 | 0 |")
            continue
        lines.append(
            f"| {name} | {r['mean_per_iter_us']} | {r['ok']} | {r['hardware']} | {r['invalid']} | {r['timeout']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_svg(payload, svg_path)

    latest_json = RESULTS_ROOT / "json" / "latest_distributed_comm.json"
    latest_md = RESULTS_ROOT / "reports" / "latest_distributed_comm.md"
    latest_svg = RESULTS_ROOT / "images" / "latest_distributed_comm.svg"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_svg.write_text(svg_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
