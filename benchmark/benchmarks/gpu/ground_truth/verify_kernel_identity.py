#!/usr/bin/env python3
"""Compare eager vs compiled Nsight kernel summaries for GEMM identity."""

from __future__ import annotations

import argparse
import csv
import json
import os


def parse_kernel_rows(path: str) -> list[dict]:
    lines = open(path, encoding="utf-8").read().splitlines()
    start = -1
    header = "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name"
    for i, line in enumerate(lines):
        if line.startswith(header):
            start = i + 1
            break
    if start < 0:
        return []
    rows = []
    for line in lines[start:]:
        if not line.strip() or line.startswith("Processing "):
            break
        cols = next(csv.reader([line]))
        if len(cols) < 9:
            continue
        rows.append(
            {
                "time_pct": float(cols[0]),
                "time_ns": float(cols[1]),
                "instances": int(cols[2]),
                "name": cols[8],
            }
        )
    return rows


def parse_api_rows(path: str) -> list[dict]:
    lines = open(path, encoding="utf-8").read().splitlines()
    header = "Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name"
    starts = [i + 1 for i, line in enumerate(lines) if line.startswith(header)]
    if not starts:
        return []
    start = starts[-1]
    rows = []
    for line in lines[start:]:
        if not line.strip() or line.startswith("Processing "):
            break
        cols = next(csv.reader([line]))
        if len(cols) < 9:
            continue
        rows.append(
            {
                "time_pct": float(cols[0]),
                "time_ns": float(cols[1]),
                "calls": int(cols[2]),
                "name": cols[8],
            }
        )
    return rows


def gemm_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        name = r["name"].lower()
        if "gemm" in name or "xmma" in name or "cublas" in name or "cutlass" in name:
            out.append(r)
    return out


def top_names(rows: list[dict], k: int) -> list[str]:
    return [r["name"] for r in sorted(rows, key=lambda x: x["time_ns"], reverse=True)[:k]]


def overlap_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    sa = set(a)
    sb = set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / float(len(sa))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify eager vs compiled GEMM kernel identity from nsys stats")
    parser.add_argument("--eager-stats", required=True)
    parser.add_argument("--compiled-stats", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    eager_all = parse_kernel_rows(args.eager_stats)
    comp_all = parse_kernel_rows(args.compiled_stats)
    eager_api = parse_api_rows(args.eager_stats)
    comp_api = parse_api_rows(args.compiled_stats)
    eager_gemm = gemm_rows(eager_all)
    comp_gemm = gemm_rows(comp_all)
    eager_top = top_names(eager_gemm, args.top_k)
    comp_top = top_names(comp_gemm, args.top_k)

    eager_graph_launch = next((r for r in eager_api if r["name"] == "cudaGraphLaunch_v10000"), None)
    comp_graph_launch = next((r for r in comp_api if r["name"] == "cudaGraphLaunch_v10000"), None)

    not_comparable = (not comp_all and comp_graph_launch is not None) or (not eager_all and eager_graph_launch is not None)
    identity_match = (eager_top == comp_top) if not not_comparable else False
    ratio = overlap_ratio(eager_top, comp_top)
    result = {
        "status": "ok",
        "top_k": args.top_k,
        "eager_stats": args.eager_stats,
        "compiled_stats": args.compiled_stats,
        "eager_top_gemm_names": eager_top,
        "compiled_top_gemm_names": comp_top,
        "exact_top_k_match": identity_match,
        "top_k_overlap_ratio": round(ratio, 4),
        "not_comparable_graph_replay_only": not_comparable,
        "cuda_graph_launch_calls": {
            "eager": eager_graph_launch["calls"] if eager_graph_launch else 0,
            "compiled": comp_graph_launch["calls"] if comp_graph_launch else 0,
        },
        "recommendation": (
            "Kernel-level identity unavailable because capture contains graph replay API without kernel rows. Compare against non-graph capture."
            if not_comparable
            else (
                "Kernels match. Keep lowering path and focus on memory stability."
                if identity_match
                else "Kernels differ. Route compiled GEMMs through ATEN/cuBLASLt path and re-verify."
            )
        ),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
