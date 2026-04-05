#!/usr/bin/env python3
"""Build a local Ada GEMM analysis bundle from sweep + kernel-lab artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_RUN_DIR = (
    ROOT
    / "benchmark"
    / "benchmarks"
    / "results"
    / "remote_results"
    / "hosts"
    / "host0356_kci2_ty6k_prxmx100056"
    / "runs"
    / "20260404T191958Z"
    / "ada-sm89-fp32-baseline-sweep"
)
DEFAULT_SWEEP_JSON = DEFAULT_REMOTE_RUN_DIR / "20260404T191958Z__ada-sm89-fp32-baseline-sweep.json"
DEFAULT_KERNEL_JSON = DEFAULT_REMOTE_RUN_DIR / "kernel_lab_ada_gemm_result.json"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "benchmark"
    / "benchmarks"
    / "results"
    / "analysis"
    / "ada"
    / "20260404T191958Z"
    / "ada-sm89-fp32-baseline-sweep"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    absolute = path.resolve() if not path.is_absolute() else path
    try:
        return str(absolute.relative_to(ROOT))
    except ValueError:
        return str(absolute)


def slugify(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "item"


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def geometric_mean(values: list[float]) -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def adapter_color(adapter: str) -> str:
    palette = {
        "torch_compile": "#2563eb",
        "torch_eager": "#0891b2",
        "pyc": "#dc2626",
        "ada_gemm": "#7c3aed",
    }
    return palette.get(adapter, "#475569")


def copy_raw_inputs(run_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(run_dir.glob("*.json")):
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst.name)
    return copied


def normalize_adapter_payload(adapter: str, payload: dict) -> dict:
    return {
        "adapter": adapter,
        "display_name": payload.get("display_name", adapter),
        "status": payload.get("status", "unknown"),
        "mean_ms": to_float((payload.get("latency_ms") or {}).get("mean")),
        "p50_ms": to_float((payload.get("latency_ms") or {}).get("p50")),
        "p95_ms": to_float((payload.get("latency_ms") or {}).get("p95")),
        "throughput_tflops": to_float(payload.get("throughput_tflops_per_sec")),
        "peak_memory_bytes": int(payload.get("peak_memory_bytes", 0) or 0),
        "mode": payload.get("mode", "unknown"),
        "dtype": payload.get("dtype", "unknown"),
        "reason": payload.get("reason"),
        "error": payload.get("error"),
        "note": payload.get("note"),
    }


def summarize_issue(item: dict) -> str:
    for key in ("error", "reason", "note"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.strip().split())
            return text[:220] + ("..." if len(text) > 220 else "")
    return ""


def build_shape_rankings(sweep: dict) -> list[dict]:
    rows = []
    for shape_doc in sweep.get("shapes", []):
        shape = shape_doc.get("shape", {})
        normalized = []
        for adapter, payload in (shape_doc.get("adapters") or {}).items():
            normalized.append(normalize_adapter_payload(adapter, payload))
        ok = [item for item in normalized if item["status"] == "ok"]
        ok_latency = sorted(ok, key=lambda item: item["mean_ms"])
        ok_tflops = sorted(ok, key=lambda item: item["throughput_tflops"], reverse=True)
        rows.append(
            {
                "shape": shape,
                "available_adapters": [item["adapter"] for item in ok],
                "fastest_latency": ok_latency,
                "highest_throughput": ok_tflops,
                "unavailable": [item for item in normalized if item["status"] != "ok"],
            }
        )
    return rows


def build_adapter_summary(shape_rankings: list[dict]) -> list[dict]:
    aggregate = {}
    for row in shape_rankings:
        for category in ("fastest_latency", "highest_throughput"):
            ordered = row.get(category, [])
            if not ordered:
                continue
            winner = ordered[0]["adapter"]
            aggregate.setdefault(winner, {"latency_wins": 0, "throughput_wins": 0, "latency_values": [], "throughput_values": []})
            if category == "fastest_latency":
                aggregate[winner]["latency_wins"] += 1
            else:
                aggregate[winner]["throughput_wins"] += 1
        for item in row.get("fastest_latency", []):
            bucket = aggregate.setdefault(
                item["adapter"],
                {"latency_wins": 0, "throughput_wins": 0, "latency_values": [], "throughput_values": []},
            )
            if item["mean_ms"] > 0:
                bucket["latency_values"].append(item["mean_ms"])
            if item["throughput_tflops"] > 0:
                bucket["throughput_values"].append(item["throughput_tflops"])

    summary = []
    for adapter, stats in sorted(aggregate.items()):
        summary.append(
            {
                "adapter": adapter,
                "latency_wins": stats["latency_wins"],
                "throughput_wins": stats["throughput_wins"],
                "geomean_latency_ms": round(geometric_mean(stats["latency_values"]), 6),
                "geomean_throughput_tflops": round(geometric_mean(stats["throughput_values"]), 6),
                "shape_count": len(stats["latency_values"]),
            }
        )
    return sorted(summary, key=lambda item: (item["throughput_wins"], -item["geomean_latency_ms"]), reverse=True)


def build_adapter_health(shape_rankings: list[dict]) -> list[dict]:
    aggregate = {}
    for row in shape_rankings:
        shape_name = row.get("shape", {}).get("name", "unknown")
        for item in row.get("fastest_latency", []):
            bucket = aggregate.setdefault(
                item["adapter"],
                {"adapter": item["adapter"], "display_name": item["display_name"], "ok_shapes": 0, "error_shapes": 0, "unavailable_shapes": 0, "issue_examples": []},
            )
            bucket["ok_shapes"] += 1
        for item in row.get("unavailable", []):
            bucket = aggregate.setdefault(
                item["adapter"],
                {"adapter": item["adapter"], "display_name": item["display_name"], "ok_shapes": 0, "error_shapes": 0, "unavailable_shapes": 0, "issue_examples": []},
            )
            status = item.get("status", "unknown")
            if status == "unavailable":
                bucket["unavailable_shapes"] += 1
            else:
                bucket["error_shapes"] += 1
            issue = summarize_issue(item)
            if issue and len(bucket["issue_examples"]) < 3:
                bucket["issue_examples"].append({"shape": shape_name, "status": status, "message": issue})
    return sorted(aggregate.values(), key=lambda item: item["adapter"])


def build_pyc_phantom_summary(sweep: dict) -> dict:
    summary = {
        "enabled_shapes": 0,
        "matched_shapes": 0,
        "mismatched_shapes": 0,
        "reshape_events": 0,
        "mean_confidence": None,
        "mean_match_score": None,
        "examples": [],
    }
    confidence_values = []
    match_score_values = []
    for shape_doc in sweep.get("shapes", []):
        shape_name = (shape_doc.get("shape") or {}).get("name", "unknown")
        pyc = ((shape_doc.get("adapters") or {}).get("pyc") or {})
        phantom = pyc.get("phantom_graph") if isinstance(pyc.get("phantom_graph"), dict) else {}
        if not phantom or not phantom.get("enabled"):
            continue
        summary["enabled_shapes"] += 1
        if phantom.get("match"):
            summary["matched_shapes"] += 1
        else:
            summary["mismatched_shapes"] += 1
        summary["reshape_events"] += int(phantom.get("reshape_count", 0) or 0)
        confidence = to_float(phantom.get("confidence"), default=-1.0)
        match_score = to_float(phantom.get("match_score"), default=-1.0)
        if confidence >= 0.0:
            confidence_values.append(confidence)
        if match_score >= 0.0:
            match_score_values.append(match_score)
        if len(summary["examples"]) < 3:
            summary["examples"].append(
                {
                    "shape": shape_name,
                    "match": bool(phantom.get("match")),
                    "expected_signature": phantom.get("expected_signature", ""),
                    "observed_signature": phantom.get("observed_signature", ""),
                    "match_score": round(match_score, 4) if match_score >= 0.0 else None,
                }
            )
    if confidence_values:
        summary["mean_confidence"] = round(sum(confidence_values) / len(confidence_values), 6)
    if match_score_values:
        summary["mean_match_score"] = round(sum(match_score_values) / len(match_score_values), 6)
    return summary


def is_gemm_sequence_payload(doc: dict) -> bool:
    if not isinstance(doc, dict):
        return False
    if doc.get("task") == "gemm_sequence":
        return True
    return isinstance(doc.get("sequence"), dict) and isinstance(doc.get("sequence", {}).get("steps"), list)


def normalize_sequence_step(step: dict, index: int | None = None) -> dict:
    shape = {
        "name": step.get("name")
        or step.get("shape_name")
        or f"{int(step.get('m', 0) or 0)}x{int(step.get('k', 0) or 0)}x{int(step.get('n', 0) or 0)}",
        "m": int(step.get("m", 0) or 0),
        "k": int(step.get("k", 0) or 0),
        "n": int(step.get("n", 0) or 0),
    }
    reliability = step.get("reliability") if isinstance(step.get("reliability"), dict) else {}
    phantom = step.get("phantom_graph") if isinstance(step.get("phantom_graph"), dict) else {}
    remat_tensors = int(reliability.get("rematerialized_tensors", 0) or 0)
    remat_bytes = int(reliability.get("rematerialized_bytes", 0) or 0)
    return {
        "index": int(step.get("index", index if index is not None else 0) or 0),
        "shape": shape,
        "profile": step.get("profile", "unknown"),
        "latency_ms": to_float(step.get("latency_ms"), default=0.0),
        "throughput_tflops": to_float(step.get("throughput_tflops_per_sec"), default=0.0),
        "reliability": {
            "compile_cache_hit": bool(reliability.get("compile_cache_hit")),
            "speculative_plan_hit": bool(reliability.get("speculative_plan_hit")),
            "rematerialized_tensors": remat_tensors,
            "rematerialized_bytes": remat_bytes,
        },
        "phantom_graph": {
            "enabled": bool(phantom.get("enabled")),
            "match": bool(phantom.get("match")),
            "match_count": int(phantom.get("match_count", 0) or 0),
            "mismatch_count": int(phantom.get("mismatch_count", 0) or 0),
            "reshape_count": int(phantom.get("reshape_count", 0) or 0),
            "confidence": to_float(phantom.get("confidence"), default=-1.0),
            "match_score": to_float(phantom.get("match_score"), default=-1.0),
            "expected_signature": phantom.get("expected_signature", ""),
            "observed_signature": phantom.get("observed_signature", ""),
        },
    }


def build_sequence_steps(sequence: dict) -> list[dict]:
    return [normalize_sequence_step(step, index=index) for index, step in enumerate(sequence.get("steps", []), start=1)]


def build_sequence_phantom_summary(sequence_steps: list[dict]) -> dict:
    summary = {
        "enabled_steps": 0,
        "matched_steps": 0,
        "mismatched_steps": 0,
        "reshape_events": 0,
        "mean_confidence": None,
        "mean_match_score": None,
        "examples": [],
    }
    confidence_values = []
    match_score_values = []
    for step in sequence_steps:
        phantom = step.get("phantom_graph") if isinstance(step.get("phantom_graph"), dict) else {}
        if not phantom or not phantom.get("enabled"):
            continue
        summary["enabled_steps"] += 1
        if phantom.get("match"):
            summary["matched_steps"] += 1
        else:
            summary["mismatched_steps"] += 1
        summary["reshape_events"] += int(phantom.get("reshape_count", 0) or 0)
        confidence = to_float(phantom.get("confidence"), default=-1.0)
        match_score = to_float(phantom.get("match_score"), default=-1.0)
        if confidence >= 0.0:
            confidence_values.append(confidence)
        if match_score >= 0.0:
            match_score_values.append(match_score)
        if len(summary["examples"]) < 3:
            summary["examples"].append(
                {
                    "step": step.get("index"),
                    "shape": step.get("shape", {}).get("name", "unknown"),
                    "match": bool(phantom.get("match")),
                    "expected_signature": phantom.get("expected_signature", ""),
                    "observed_signature": phantom.get("observed_signature", ""),
                    "match_score": round(match_score, 4) if match_score >= 0.0 else None,
                }
            )
    if confidence_values:
        summary["mean_confidence"] = round(sum(confidence_values) / len(confidence_values), 6)
    if match_score_values:
        summary["mean_match_score"] = round(sum(match_score_values) / len(match_score_values), 6)
    return summary


def build_sequence_rematerialization_summary(sequence_steps: list[dict]) -> dict:
    summary = {
        "rematerialized_steps": 0,
        "rematerialized_tensors": 0,
        "rematerialized_bytes": 0,
        "max_step": None,
    }
    heaviest = {"bytes": 0, "step": None}
    for step in sequence_steps:
        reliability = step.get("reliability") if isinstance(step.get("reliability"), dict) else {}
        tensors = int(reliability.get("rematerialized_tensors", 0) or 0)
        bytes_ = int(reliability.get("rematerialized_bytes", 0) or 0)
        if tensors > 0 or bytes_ > 0:
            summary["rematerialized_steps"] += 1
        summary["rematerialized_tensors"] += tensors
        summary["rematerialized_bytes"] += bytes_
        if bytes_ > heaviest["bytes"]:
            heaviest = {"bytes": bytes_, "step": step.get("index")}
    if heaviest["step"] is not None:
        summary["max_step"] = heaviest["step"]
    return summary


def build_sequence_summary(sequence: dict, sequence_steps: list[dict]) -> dict:
    raw_summary = sequence.get("summary") if isinstance(sequence.get("summary"), dict) else {}
    shapes = [step.get("shape", {}).get("name", "unknown") for step in sequence_steps]
    transitions = sum(1 for left, right in zip(shapes, shapes[1:]) if left != right)
    latencies = [to_float(step.get("latency_ms")) for step in sequence_steps if to_float(step.get("latency_ms")) > 0]
    throughputs = [to_float(step.get("throughput_tflops")) for step in sequence_steps if to_float(step.get("throughput_tflops")) > 0]
    summary = dict(raw_summary)
    summary.update(
        {
            "step_count": len(sequence_steps),
            "unique_shape_count": len(dict.fromkeys(shapes)),
            "shape_transition_count": transitions,
            "shape_sequence": shapes,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 6) if latencies else None,
            "mean_throughput_tflops": round(sum(throughputs) / len(throughputs), 6) if throughputs else None,
        }
    )
    return summary


def build_sequence_judgment(sequence_summary: dict, sequence_phantom: dict, sequence_remat: dict, focus: dict) -> dict:
    guard_miss_count = int(sequence_summary.get("guard_miss_count", 0) or 0)
    fallback_count = int(sequence_summary.get("fallback_count", 0) or 0)
    transition_count = int(sequence_summary.get("shape_transition_count", 0) or 0)
    rematerialized_steps = int(sequence_remat.get("rematerialized_steps", 0) or 0)
    reshape_events = int(sequence_phantom.get("reshape_events", 0) or 0)
    status = "mixed_shape_hot_path_low_signal"
    reasons = []
    if guard_miss_count or fallback_count:
        status = "mixed_shape_hot_path_needs_attention"
        reasons.append(
            "The mixed-shape hot path recorded guard misses or fallbacks, so the runtime is not yet staying inside the intended adaptive envelope."
        )
    elif transition_count > 0 and (reshape_events > 0 or rematerialized_steps > 0):
        status = "mixed_shape_hot_path_validated"
        reasons.append(
            "The mixed-shape hot path stayed on the intended CUDA path while phantom reshapes and rematerialization reacted to shape drift as expected."
        )
    elif transition_count > 0:
        status = "mixed_shape_hot_path_observed"
        reasons.append(
            "The mixed-shape hot path completed without guard misses, but the run did not exercise enough drift-sensitive behavior to validate phantom or rematerialization policy changes."
        )
    else:
        reasons.append("The sequence did not include meaningful shape transitions, so it is not a strong mixed-shape runtime signal.")
    if sequence_phantom.get("enabled_steps", 0):
        reasons.append(
            f"Phantom graph tracked {sequence_phantom.get('enabled_steps', 0)} enabled steps with mean match score {sequence_phantom.get('mean_match_score')}."
        )
    if rematerialized_steps:
        reasons.append(
            f"Rematerialization triggered on {rematerialized_steps} step(s) and is visible in the sequence report instead of being hidden in raw JSON."
        )
    peak = focus.get("peak_step") or {}
    next_task = {
        "name": "ada-sm89-mixed-shape-hot-path-followup",
        "objective": "Use the mixed-shape Ada hot path to decide whether phantom graph and rematerialization should be tightened before kernel promotion.",
        "required_changes": [
            "Compare phantom reshapes and rematerialization across additional mixed-shape sequences with different drift patterns.",
            "Keep the runtime on the CUDA path and watch for guard misses, fallback churn, or allocator oscillation.",
            "Use the report to decide whether the current control behavior is stable enough to return to kernel tuning.",
        ],
        "proof_point": peak.get("shape", {}).get("name", "sequence"),
    }
    return {
        "status": status,
        "current_sequence_peak": peak,
        "reasons": reasons,
        "next_task": next_task,
    }


def build_sequence_bundle(sweep: dict, kernel_doc: dict, args) -> tuple[dict, dict]:
    sequence = sweep.get("sequence") if isinstance(sweep.get("sequence"), dict) else {}
    sequence_steps = build_sequence_steps(sequence)
    sequence_summary = build_sequence_summary(sequence, sequence_steps)
    sequence_phantom = build_sequence_phantom_summary(sequence_steps)
    sequence_remat = build_sequence_rematerialization_summary(sequence_steps)
    direct_metrics = resolve_direct_kernel_metrics(kernel_doc, args)
    focus = {
        "peak_step": max(sequence_steps, key=lambda step: to_float(step.get("throughput_tflops")), default={}) if sequence_steps else {},
        "direct_kernel": direct_metrics,
        "sequence_summary": sequence_summary,
    }
    rankings = {
        "meta": {
            "run_id": sweep["meta"]["run_id"],
            "tag": sweep["meta"]["tag"],
        },
        "task": "gemm_sequence",
        "sequence_steps": sequence_steps,
        "sequence_summary": sequence_summary,
        "sequence_phantom": sequence_phantom,
        "sequence_rematerialization": sequence_remat,
    }
    analysis = {
        "meta": {
            "run_id": sweep["meta"]["run_id"],
            "tag": sweep["meta"]["tag"],
            "host": sweep["meta"]["host"],
            "gpu_name": (((sweep.get("gpu") or {}).get("gpus") or [{}])[0]).get("name", "unknown"),
            "git_head": sweep["meta"]["git_head"],
            "git_dirty": sweep["meta"]["git_dirty"],
        },
        "sources": {
            "sweep_json": str(args.sweep_json),
            "kernel_json": str(args.kernel_json),
            "run_dir": str(args.run_dir),
        },
        "task": "gemm_sequence",
        "sequence": {
            "summary": sequence_summary,
            "steps": sequence_steps,
            "phantom": sequence_phantom,
            "rematerialization": sequence_remat,
            "direct_kernel": direct_metrics,
        },
        "judgment": build_sequence_judgment(sequence_summary, sequence_phantom, sequence_remat, focus),
    }
    return analysis, rankings


def infer_sweep_dtype(shape_rankings: list[dict]) -> str:
    for row in shape_rankings:
        for item in row.get("highest_throughput", []):
            dtype = str(item.get("dtype", "")).strip().lower()
            if dtype and dtype != "unknown":
                return dtype
    return "unknown"


def resolve_direct_kernel_metrics(kernel_doc: dict, args) -> dict:
    observed = ((kernel_doc.get("run") or {}).get("observed_metrics") or {}) if isinstance(kernel_doc, dict) else {}
    best_ms = to_float(observed.get("best_ms") or args.kernel_best_ms)
    gflops = to_float(observed.get("gflops") or args.kernel_gflops)
    tflops = round(gflops / 1000.0, 6) if gflops > 0 else 0.0
    max_abs_diff = to_float(observed.get("max_abs_diff") or args.kernel_max_abs_diff)
    process_mean_ms = to_float((((kernel_doc.get("run") or {}).get("stats") or {}).get("mean_ms")))
    ratio = round(process_mean_ms / best_ms, 3) if best_ms > 0 and process_mean_ms > 0 else None
    return {
        "kernel": (kernel_doc.get("kernel") or {}).get("name", "ada_gemm"),
        "shape": observed.get("shape") or args.kernel_shape,
        "device": observed.get("device") or args.kernel_device,
        "best_ms": round(best_ms, 6) if best_ms > 0 else 0.0,
        "gflops": round(gflops, 6) if gflops > 0 else 0.0,
        "tflops": tflops,
        "max_abs_diff": round(max_abs_diff, 6) if max_abs_diff else 0.0,
        "process_bench_mean_ms": round(process_mean_ms, 6) if process_mean_ms > 0 else 0.0,
        "process_to_kernel_ratio": ratio,
        "measurement_surface": "standalone kernel stdout / in-kernel timer" if best_ms > 0 else "unavailable",
        "process_surface": "kernel_lab process-level elapsed timing",
    }


def focus_square_1024(shape_rankings: list[dict], direct_metrics: dict, sweep_dtype: str) -> dict:
    target = next((row for row in shape_rankings if row.get("shape", {}).get("name") == "square-1024"), None)
    if target is None:
        return {}
    best_tflops = target.get("highest_throughput", [])
    best_latency = target.get("fastest_latency", [])
    top_tflops = best_tflops[0] if best_tflops else None
    top_latency = best_latency[0] if best_latency else None
    direct_tflops = to_float(direct_metrics.get("tflops"))
    gap = round((to_float(top_tflops.get("throughput_tflops")) / direct_tflops), 4) if top_tflops and direct_tflops > 0 else None
    return {
        "shape": target.get("shape"),
        "sweep_best_throughput": top_tflops,
        "sweep_fastest_latency": top_latency,
        "all_adapters": best_tflops,
        "direct_kernel": direct_metrics,
        "apparent_throughput_gap_vs_sweep_winner": gap,
        "comparison_blockers": [
            f"The sweep adapters use benchmark/benchmarks/gpu/workloads/standard_workload.py with dtype={sweep_dtype}.",
            "The direct ada_gemm prototype is an FP32 shared-memory kernel per kernels/lab/manifests/kernels.json.",
            "The kernel-lab JSON measures whole-process elapsed time, while the direct kernel metrics come from in-kernel stdout timing.",
        ],
    }


def build_judgment(
    sweep: dict,
    shape_rankings: list[dict],
    adapter_summary: list[dict],
    adapter_health: list[dict],
    focus: dict,
) -> dict:
    pyc_health = next((row for row in adapter_health if row.get("adapter") == "pyc"), {})
    pyc_available = bool(pyc_health.get("ok_shapes"))
    pyc_error_shapes = int(pyc_health.get("error_shapes", 0) or 0)
    pyc_issue_examples = pyc_health.get("issue_examples", [])
    winner = adapter_summary[0]["adapter"] if adapter_summary else None
    status = "blocked_on_comparable_pyc_measurement"
    reasons = []
    sweep_dtype = str(focus.get("sweep_dtype", "unknown")).lower()
    if pyc_error_shapes:
        status = "blocked_on_pyc_build_integration"
        reasons.append("PyC adapter failed during the sweep, so the compiler-next CUDA benchmark path is not currently buildable on the GPU box.")
        if pyc_issue_examples:
            reasons.append(f"First PyC failure: {pyc_issue_examples[0]['message']}")
    elif not pyc_available:
        reasons.append("PyC adapter was unavailable in the sweep, so the canonical GPU suite did not measure the custom kernel path.")
    if not pyc_error_shapes and pyc_available and sweep_dtype == "float32":
        status = "performance_gap_to_close"
        reasons.append("The sweep is now FP32-comparable across PyTorch and PyC, so the remaining blocker is throughput, not measurement validity.")
        reasons.append("Promotion should wait until PyC materially closes the gap to the FP32 PyTorch baselines on the same shape family.")
        next_task_name = "ada-sm89-fp32-performance-tuning"
        next_task_objective = "Tune the PyC FP32 GEMM path on SM89 until it is competitive on the standardized FP32 sweep."
        required_changes = [
            "Profile the PyC CUDA path on the worst-gap FP32 shapes and separate kernel time from host/runtime overhead.",
            "Tune kernel selection, graph replay, and matmul pipeline choices against the standardized FP32 sweep.",
            "Rerun the same FP32 shape matrix and track whether PyC closes the gap on square-1024 and square-2048 first.",
        ]
    else:
        reasons.append("The sweep compared CUDA float16 PyTorch GEMM against an FP32 custom kernel, so the current top-line ranking is not apples-to-apples.")
        reasons.append("Promotion should wait for a single measurement surface that emits comparable JSON for both baseline and challenger.")
        next_task_name = "ada-sm89-fp32-comparable-pyc-sweep"
        next_task_objective = "Benchmark the PyC/Ada GEMM path on the same JSON surface and dtype as the baseline so promotion decisions are defensible."
        required_changes = [
            "Add an FP32 CUDA GEMM sweep mode in benchmark/benchmarks/gpu/workloads/standard_workload.py or a parallel FP32 workload path.",
            "Rerun the shape matrix with pyc active and compare against a forced-FP32 baseline.",
        ]
    if pyc_error_shapes:
        next_task_name = "ada-sm89-pyc-build-bridge"
        next_task_objective = "Repair the compiler-next CUDA benchmark build so PyC can participate in the standardized Ada sweep."
        required_changes.insert(0, "Fix the pyc_compiler_next_bench link/build path so pyc_cutlass_gemm_dispatch resolves or cleanly stubs when CUTLASS is absent.")
    return {
        "status": status,
        "current_sweep_winner": winner,
        "reasons": reasons,
        "next_task": {
            "name": next_task_name,
            "objective": next_task_objective,
            "required_changes": required_changes,
            "proof_point": focus.get("shape", {}).get("name", "square-1024"),
        },
    }


def render_shape_winners_svg(shape_rankings: list[dict], out_path: Path) -> None:
    rows = []
    for row in shape_rankings:
        leader = row.get("highest_throughput", [])
        if not leader:
            continue
        top = leader[0]
        rows.append(
            {
                "label": row["shape"]["name"],
                "adapter": top["adapter"],
                "display_name": top["display_name"],
                "value": top["throughput_tflops"],
            }
        )
    width = 1240
    bar_h = 34
    gap = 18
    chart_x = 360
    chart_y = 130
    chart_w = 820
    height = chart_y + max(1, len(rows)) * (bar_h + gap) + 120
    max_v = max((row["value"] for row in rows), default=1.0)
    body = []
    for index, row in enumerate(rows):
        y = chart_y + index * (bar_h + gap)
        bar_w = int((row["value"] / max_v) * chart_w) if max_v > 0 else 0
        color = adapter_color(row["adapter"])
        body.append(
            f'<text x="30" y="{y + 22}" font-family="Arial, sans-serif" font-size="18" fill="#0f172a">{row["label"]}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="7" fill="{color}" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 22}" font-family="Arial, sans-serif" font-size="15" fill="#1f2937">{row["display_name"]} {row["value"]:.3f} TFLOPS</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="40" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#0f172a">Ada GEMM Sweep Throughput Winners</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="14" fill="#475569">Each bar shows the best throughput result per shape from the consolidated sweep JSON.</text>
  <text x="30" y="88" font-family="Arial, sans-serif" font-size="14" fill="#475569">This chart is descriptive only; it does not resolve the FP16-vs-FP32 comparability gap.</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def render_square_1024_focus_svg(focus: dict, out_path: Path) -> None:
    direct = focus.get("direct_kernel") or {}
    rows = []
    for item in focus.get("all_adapters", []):
        rows.append(
            {
                "label": item.get("display_name", item.get("adapter", "unknown")),
                "adapter": item.get("adapter", "unknown"),
                "value": to_float(item.get("throughput_tflops")),
                "detail": f"{item.get('mode', 'unknown')} sweep path",
            }
        )
    rows.append(
        {
            "label": direct.get("kernel", "ada_gemm"),
            "adapter": direct.get("kernel", "ada_gemm"),
            "value": to_float(direct.get("tflops")),
            "detail": "direct kernel stdout",
        }
    )
    width = 1100
    bar_h = 44
    gap = 28
    chart_x = 340
    chart_y = 150
    chart_w = 680
    height = chart_y + max(1, len(rows)) * (bar_h + gap) + 120
    max_v = max((row["value"] for row in rows), default=1.0)
    body = []
    for index, row in enumerate(rows):
        y = chart_y + index * (bar_h + gap)
        bar_w = int((row["value"] / max_v) * chart_w) if max_v > 0 else 0
        body.append(
            f'<text x="30" y="{y + 28}" font-family="Arial, sans-serif" font-size="20" fill="#0f172a">{row["label"]}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="8" fill="{adapter_color(row["adapter"])}" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 28}" font-family="Arial, sans-serif" font-size="16" fill="#1f2937">{row["value"]:.3f} TFLOPS | {row["detail"]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff7ed"/>
  <text x="30" y="42" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#9a3412">Square-1024 Focus View</text>
  <text x="30" y="70" font-family="Arial, sans-serif" font-size="14" fill="#7c2d12">This focus view renders every sweep adapter plus the direct ada_gemm reading, so PyC stays visible even when it does not win the shape.</text>
  <text x="30" y="92" font-family="Arial, sans-serif" font-size="14" fill="#7c2d12">The direct kernel metric comes from stdout best_ms, while kernel-lab process timing measures a larger host-side envelope.</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def render_adapter_summary_svg(rankings: dict, out_path: Path) -> None:
    rows = []
    for row in rankings.get("adapter_summary", []):
        rows.append(
            {
                "label": row["adapter"],
                "adapter": row["adapter"],
                "value": to_float(row.get("geomean_throughput_tflops")),
                "detail": f"wins={row.get('throughput_wins', 0)} latency_wins={row.get('latency_wins', 0)}",
            }
        )
    width = 1180
    bar_h = 34
    gap = 18
    chart_x = 320
    chart_y = 130
    chart_w = 780
    height = chart_y + max(1, len(rows)) * (bar_h + gap) + 100
    max_v = max((row["value"] for row in rows), default=1.0)
    body = []
    for index, row in enumerate(rows):
        y = chart_y + index * (bar_h + gap)
        bar_w = int((row["value"] / max_v) * chart_w) if max_v > 0 else 0
        color = adapter_color(row["adapter"])
        body.append(
            f'<text x="30" y="{y + 22}" font-family="Arial, sans-serif" font-size="18" fill="#0f172a">{row["label"]}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="7" fill="{color}" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 22}" font-family="Arial, sans-serif" font-size="15" fill="#1f2937">{row["value"]:.3f} TFLOPS | {row["detail"]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="40" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#0f172a">Adapter Geomean Throughput</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="14" fill="#475569">This chart keeps every adapter visible in the bundle, including PyC when it wins no shapes.</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def render_sequence_steps_svg(sequence: dict, out_path: Path) -> None:
    steps = sequence.get("steps", [])
    rows = []
    for step in steps:
        phantom = step.get("phantom_graph", {})
        remat = step.get("reliability", {})
        if phantom.get("match") and int(remat.get("rematerialized_tensors", 0) or 0) == 0:
            color = "#16a34a"
        elif int(remat.get("rematerialized_tensors", 0) or 0) > 0:
            color = "#d97706"
        elif phantom.get("enabled"):
            color = "#2563eb"
        else:
            color = "#6b7280"
        rows.append(
            {
                "label": f"{step.get('index', 0)} {step.get('shape', {}).get('name', 'n/a')}",
                "value": to_float(step.get("throughput_tflops")),
                "detail": f"{to_float(step.get('latency_ms')):.4f} ms | match={'yes' if phantom.get('match') else 'no'} | remat={int(remat.get('rematerialized_tensors', 0) or 0)}",
                "color": color,
            }
        )
    width = 1240
    bar_h = 34
    gap = 18
    chart_x = 390
    chart_y = 130
    chart_w = 760
    height = chart_y + max(1, len(rows)) * (bar_h + gap) + 120
    max_v = max((row["value"] for row in rows), default=1.0)
    body = []
    for index, row in enumerate(rows):
        y = chart_y + index * (bar_h + gap)
        bar_w = int((row["value"] / max_v) * chart_w) if max_v > 0 else 0
        body.append(
            f'<text x="30" y="{y + 22}" font-family="Arial, sans-serif" font-size="18" fill="#0f172a">{row["label"]}</text>'
            f'<rect x="{chart_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="7" fill="{row["color"]}" />'
            f'<text x="{chart_x + bar_w + 12}" y="{y + 22}" font-family="Arial, sans-serif" font-size="15" fill="#1f2937">{row["value"]:.3f} TFLOPS | {row["detail"]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="30" y="40" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#0f172a">Ada Mixed-Shape Sequence Throughput</text>
  <text x="30" y="68" font-family="Arial, sans-serif" font-size="14" fill="#475569">Bars show each sequence step. Green means phantom matched with no rematerialization, amber marks rematerialization, blue marks enabled phantom coverage.</text>
  <text x="30" y="88" font-family="Arial, sans-serif" font-size="14" fill="#475569">This view is meant for hot-path judgment, not just final ranking.</text>
  {''.join(body)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def render_rankings_markdown(rankings: dict) -> str:
    if rankings.get("task") == "gemm_sequence":
        lines = [
            "# Ada GEMM Sequence Rankings",
            "",
            "## Sequence Summary",
            "",
            "| Steps | Unique Shapes | Transitions | Mean Latency ms | Mean TFLOPS |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        summary = rankings.get("sequence_summary", {})
        lines.append(
            f"| {summary.get('step_count', 0)} | {summary.get('unique_shape_count', 0)} | {summary.get('shape_transition_count', 0)} | "
            f"{to_float(summary.get('mean_latency_ms')):.6f} | {to_float(summary.get('mean_throughput_tflops')):.6f} |"
        )
        lines.extend(
            [
                "",
                "## Per-Step Summary",
                "",
                "| Step | Shape | Profile | Latency ms | TFLOPS | Phantom Match | Match Score | Remat Tensors | Remat Bytes |",
                "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for step in rankings.get("sequence_steps", []):
            phantom = step.get("phantom_graph", {})
            remat = step.get("reliability", {})
            lines.append(
                f"| {step.get('index', 0)} | {step.get('shape', {}).get('name', 'n/a')} | {step.get('profile', 'unknown')} | "
                f"{to_float(step.get('latency_ms')):.6f} | {to_float(step.get('throughput_tflops')):.6f} | "
                f"{'yes' if phantom.get('match') else 'no'} | "
                f"{to_float(phantom.get('match_score'), default=0.0):.4f} | "
                f"{int(remat.get('rematerialized_tensors', 0) or 0)} | {int(remat.get('rematerialized_bytes', 0) or 0)} |"
            )
        lines.extend(
            [
                "",
                "## Phantom Summary",
                "",
                "| Enabled Steps | Matched Steps | Mismatched Steps | Reshape Events | Mean Confidence | Mean Match Score |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        phantom = rankings.get("sequence_phantom", {})
        lines.append(
            f"| {phantom.get('enabled_steps', 0)} | {phantom.get('matched_steps', 0)} | {phantom.get('mismatched_steps', 0)} | "
            f"{phantom.get('reshape_events', 0)} | {phantom.get('mean_confidence')} | {phantom.get('mean_match_score')} |"
        )
        lines.extend(
            [
                "",
                "## Rematerialization Summary",
                "",
                "| Rematerialized Steps | Rematerialized Tensors | Rematerialized Bytes | Max Step |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        remat = rankings.get("sequence_rematerialization", {})
        lines.append(
            f"| {remat.get('rematerialized_steps', 0)} | {remat.get('rematerialized_tensors', 0)} | {remat.get('rematerialized_bytes', 0)} | {remat.get('max_step', 'n/a')} |"
        )
        return "\n".join(lines) + "\n"
    lines = [
        "# Ada GEMM Rankings",
        "",
        "## Adapter Summary",
        "",
        "| Adapter | Throughput Wins | Latency Wins | Geomean TFLOPS | Geomean ms | Shapes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rankings.get("adapter_summary", []):
        lines.append(
            f"| {row['adapter']} | {row['throughput_wins']} | {row['latency_wins']} | "
            f"{row['geomean_throughput_tflops']:.6f} | {row['geomean_latency_ms']:.6f} | {row['shape_count']} |"
        )
    lines.extend(["", "## Adapter Health", "", "| Adapter | OK Shapes | Error Shapes | Unavailable Shapes | First Issue |", "| --- | ---: | ---: | ---: | --- |"])
    for row in rankings.get("adapter_health", []):
        first_issue = ""
        if row.get("issue_examples"):
            first_issue = row["issue_examples"][0]["message"].replace("|", "/")
        lines.append(
            f"| {row['display_name']} | {row['ok_shapes']} | {row['error_shapes']} | {row['unavailable_shapes']} | {first_issue or 'n/a'} |"
        )
    lines.extend(["", "## Shape Winners", "", "| Shape | Fastest Adapter | Fastest ms | Top Throughput Adapter | Top TFLOPS |", "| --- | --- | ---: | --- | ---: |"])
    for row in rankings.get("shape_rankings", []):
        fastest = (row.get("fastest_latency") or [{}])[0]
        top = (row.get("highest_throughput") or [{}])[0]
        lines.append(
            f"| {row['shape']['name']} | {fastest.get('display_name', 'n/a')} | {to_float(fastest.get('mean_ms')):.4f} | "
            f"{top.get('display_name', 'n/a')} | {to_float(top.get('throughput_tflops')):.6f} |"
        )
    return "\n".join(lines) + "\n"


def render_analysis_markdown(analysis: dict) -> str:
    if analysis.get("task") == "gemm_sequence":
        sequence = analysis.get("sequence", {})
        summary = sequence.get("summary", {})
        phantom = sequence.get("phantom", {})
        remat = sequence.get("rematerialization", {})
        direct = sequence.get("direct_kernel", {})
        judgment = analysis.get("judgment", {})
        lines = [
            "# Ada GEMM Sequence Analysis Sheet",
            "",
            "## Run Context",
            "",
            f"- Run ID: `{analysis['meta']['run_id']}`",
            f"- Tag: `{analysis['meta']['tag']}`",
            f"- Host: `{analysis['meta']['host']}`",
            f"- GPU: `{analysis['meta']['gpu_name']}`",
            f"- Sweep source: `{analysis['sources']['sweep_json']}`",
            f"- Kernel source: `{analysis['sources']['kernel_json']}`",
            "",
            "## Sequence Context",
            "",
            f"- Steps: `{summary.get('step_count', 0)}`",
            f"- Unique shapes: `{summary.get('unique_shape_count', 0)}`",
            f"- Shape transitions: `{summary.get('shape_transition_count', 0)}`",
            f"- Mean throughput: `{summary.get('mean_throughput_tflops')}` TFLOPS",
            f"- Mean latency: `{summary.get('mean_latency_ms')}` ms",
            "",
            "## Per-Step Summary",
            "",
            "| Step | Shape | Profile | Latency ms | TFLOPS | Phantom Match | Match Score | Remat Tensors | Remat Bytes |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for step in sequence.get("steps", []):
            phantom_step = step.get("phantom_graph", {})
            remat_step = step.get("reliability", {})
            lines.append(
                f"| {step.get('index', 0)} | {step.get('shape', {}).get('name', 'n/a')} | {step.get('profile', 'unknown')} | "
                f"{to_float(step.get('latency_ms')):.6f} | {to_float(step.get('throughput_tflops')):.6f} | "
                f"{'yes' if phantom_step.get('match') else 'no'} | {to_float(phantom_step.get('match_score'), default=0.0):.4f} | "
                f"{int(remat_step.get('rematerialized_tensors', 0) or 0)} | {int(remat_step.get('rematerialized_bytes', 0) or 0)} |"
            )
        lines.extend(
            [
                "",
                "## Phantom Summary",
                "",
                f"- Enabled steps: `{phantom.get('enabled_steps', 0)}`",
                f"- Matched steps: `{phantom.get('matched_steps', 0)}`",
                f"- Mismatched steps: `{phantom.get('mismatched_steps', 0)}`",
                f"- Reshape events: `{phantom.get('reshape_events', 0)}`",
                f"- Mean confidence: `{phantom.get('mean_confidence')}`",
                f"- Mean match score: `{phantom.get('mean_match_score')}`",
                "",
                "## Rematerialization Summary",
                "",
                f"- Rematerialized steps: `{remat.get('rematerialized_steps', 0)}`",
                f"- Rematerialized tensors: `{remat.get('rematerialized_tensors', 0)}`",
                f"- Rematerialized bytes: `{remat.get('rematerialized_bytes', 0)}`",
                f"- Max rematerialization step: `{remat.get('max_step', 'n/a')}`",
                "",
                "## Direct Kernel Reference",
                "",
                f"- `ada_gemm` direct kernel: `{direct.get('tflops', 0.0):.6f}` TFLOPS, `best_ms={direct.get('best_ms', 0.0):.6f}`.",
                "",
                "## Judgment",
                "",
                f"- Status: `{judgment.get('status', 'unknown')}`",
            ]
        )
        for reason in judgment.get("reasons", []):
            lines.append(f"- {reason}")
        next_task = judgment.get("next_task", {})
        lines.extend(
            [
                "",
                "## Next Task",
                "",
                f"- Name: `{next_task.get('name', 'n/a')}`",
                f"- Objective: {next_task.get('objective', 'n/a')}",
            ]
        )
        for item in next_task.get("required_changes", []):
            lines.append(f"- {item}")
        return "\n".join(lines) + "\n"
    focus = analysis.get("square_1024_focus", {})
    direct = focus.get("direct_kernel", {})
    sweep = focus.get("sweep_best_throughput", {})
    judgment = analysis.get("judgment", {})
    pyc_health = next((row for row in analysis.get("adapter_health", []) if row.get("adapter") == "pyc"), {})
    phantom = analysis.get("pyc_phantom", {})
    lines = [
        "# Ada GEMM Analysis Sheet",
        "",
        "## Run Context",
        "",
        f"- Run ID: `{analysis['meta']['run_id']}`",
        f"- Tag: `{analysis['meta']['tag']}`",
        f"- Host: `{analysis['meta']['host']}`",
        f"- GPU: `{analysis['meta']['gpu_name']}`",
        f"- Sweep source: `{analysis['sources']['sweep_json']}`",
        f"- Kernel source: `{analysis['sources']['kernel_json']}`",
        "",
        "## What The Sweep Really Measured",
        "",
        f"- PyC sweep health: `{pyc_health.get('ok_shapes', 0)}` ok shapes, `{pyc_health.get('error_shapes', 0)}` error shapes, `{pyc_health.get('unavailable_shapes', 0)}` unavailable shapes.",
        f"- Phantom graph: `{phantom.get('enabled_shapes', 0)}` enabled shapes, `{phantom.get('matched_shapes', 0)}` matches, `{phantom.get('mismatched_shapes', 0)}` mismatches, `{phantom.get('reshape_events', 0)}` reshape events.",
        f"- Sweep dtype: `{focus.get('sweep_dtype', 'unknown')}` from `benchmark/benchmarks/gpu/workloads/standard_workload.py`.",
        "- The standalone `ada_gemm` prototype is an FP32 kernel, so its direct number is not directly comparable to the sweep winner.",
        "- The focus graph now includes every sweep adapter, so PyC remains visible even when it loses every shape.",
        "",
        "## Square-1024 Focus",
        "",
        f"- Sweep throughput winner: `{sweep.get('display_name', 'n/a')}` at `{to_float(sweep.get('throughput_tflops')):.6f}` TFLOPS and `{to_float(sweep.get('mean_ms')):.4f}` ms.",
        f"- Direct `ada_gemm` result: `{direct.get('tflops', 0.0):.6f}` TFLOPS, `best_ms={direct.get('best_ms', 0.0):.6f}`, `max_abs_diff={direct.get('max_abs_diff', 0.0):.6f}`.",
        f"- Kernel-lab process envelope: `{direct.get('process_bench_mean_ms', 0.0):.6f}` ms mean.",
        f"- Process-to-kernel timing ratio: `{direct.get('process_to_kernel_ratio')}`.",
        "",
        "## Judgment",
        "",
        f"- Status: `{judgment.get('status', 'unknown')}`",
    ]
    if phantom.get("enabled_shapes", 0):
        lines.append(
            f"- Phantom summary: mean confidence `{phantom.get('mean_confidence')}`, mean match score `{phantom.get('mean_match_score')}`."
        )
    for example in pyc_health.get("issue_examples", []):
        lines.append(f"- PyC issue on `{example['shape']}` [{example['status']}]: {example['message']}")
    for example in phantom.get("examples", []):
        lines.append(
            f"- Phantom example `{example['shape']}`: match=`{example['match']}` score=`{example.get('match_score')}` expected=`{example['expected_signature']}` observed=`{example['observed_signature']}`"
        )
    for reason in judgment.get("reasons", []):
        lines.append(f"- {reason}")
    next_task = judgment.get("next_task", {})
    lines.extend(
        [
            "",
            "## Next Task",
            "",
            f"- Name: `{next_task.get('name', 'n/a')}`",
            f"- Objective: {next_task.get('objective', 'n/a')}",
        ]
    )
    for item in next_task.get("required_changes", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_bundle(sweep: dict, kernel_doc: dict, args) -> tuple[dict, dict]:
    if is_gemm_sequence_payload(sweep):
        return build_sequence_bundle(sweep, kernel_doc, args)
    shape_rankings = build_shape_rankings(sweep)
    adapter_summary = build_adapter_summary(shape_rankings)
    adapter_health = build_adapter_health(shape_rankings)
    pyc_phantom = build_pyc_phantom_summary(sweep)
    sweep_dtype = infer_sweep_dtype(shape_rankings)
    direct_metrics = resolve_direct_kernel_metrics(kernel_doc, args)
    focus = focus_square_1024(shape_rankings, direct_metrics, sweep_dtype)
    focus["sweep_dtype"] = sweep_dtype
    rankings = {
        "meta": {
            "run_id": sweep["meta"]["run_id"],
            "tag": sweep["meta"]["tag"],
        },
        "shape_rankings": shape_rankings,
        "adapter_summary": adapter_summary,
        "adapter_health": adapter_health,
        "pyc_phantom": pyc_phantom,
    }
    analysis = {
        "meta": {
            "run_id": sweep["meta"]["run_id"],
            "tag": sweep["meta"]["tag"],
            "host": sweep["meta"]["host"],
            "gpu_name": (((sweep.get("gpu") or {}).get("gpus") or [{}])[0]).get("name", "unknown"),
            "git_head": sweep["meta"]["git_head"],
            "git_dirty": sweep["meta"]["git_dirty"],
        },
        "sources": {
            "sweep_json": str(args.sweep_json),
            "kernel_json": str(args.kernel_json),
            "run_dir": str(args.run_dir),
        },
        "adapter_health": adapter_health,
        "pyc_phantom": pyc_phantom,
        "square_1024_focus": focus,
        "judgment": build_judgment(sweep, shape_rankings, adapter_summary, adapter_health, focus),
    }
    return analysis, rankings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an Ada GEMM analysis bundle from local benchmark artifacts")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_REMOTE_RUN_DIR)
    parser.add_argument("--sweep-json", type=Path, default=DEFAULT_SWEEP_JSON)
    parser.add_argument("--kernel-json", type=Path, default=DEFAULT_KERNEL_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kernel-best-ms", type=float, default=0.091)
    parser.add_argument("--kernel-gflops", type=float, default=23563.505)
    parser.add_argument("--kernel-max-abs-diff", type=float, default=0.0)
    parser.add_argument("--kernel-shape", default="1024x1024x1024")
    parser.add_argument("--kernel-device", default="NVIDIA RTX 6000 Ada Generation cc=8.9")
    args = parser.parse_args()

    sweep = read_json(args.sweep_json)
    kernel_doc = read_json(args.kernel_json)
    analysis, rankings = build_bundle(sweep, kernel_doc, args)

    raw_dir = args.output_dir / "raw"
    graphs_dir = args.output_dir / "graphs"
    sheets_dir = args.output_dir / "sheets"
    rankings_dir = args.output_dir / "rankings"
    for folder in (raw_dir, graphs_dir, sheets_dir, rankings_dir):
        folder.mkdir(parents=True, exist_ok=True)

    copied_raw = copy_raw_inputs(args.run_dir, raw_dir)

    analysis_path = sheets_dir / "analysis.md"
    rankings_json_path = rankings_dir / "rankings.json"
    rankings_md_path = rankings_dir / "rankings.md"
    manifest_path = args.output_dir / "manifest.json"
    analysis_json_path = args.output_dir / "analysis.json"
    analysis_path.write_text(render_analysis_markdown(analysis), encoding="utf-8")
    rankings_md_path.write_text(render_rankings_markdown(rankings), encoding="utf-8")
    write_json(rankings_json_path, rankings)
    write_json(analysis_json_path, analysis)
    graphs = []
    if analysis.get("task") == "gemm_sequence":
        sequence_svg_path = graphs_dir / "sequence_steps.svg"
        render_sequence_steps_svg(analysis["sequence"], sequence_svg_path)
        graphs.append(display_path(sequence_svg_path))
    else:
        winners_svg_path = graphs_dir / "throughput_winners.svg"
        focus_svg_path = graphs_dir / "square_1024_focus.svg"
        adapter_summary_svg_path = graphs_dir / "adapter_summary.svg"
        render_shape_winners_svg(rankings["shape_rankings"], winners_svg_path)
        render_square_1024_focus_svg(analysis["square_1024_focus"], focus_svg_path)
        render_adapter_summary_svg(rankings, adapter_summary_svg_path)
        graphs.extend(
            [
                display_path(winners_svg_path),
                display_path(focus_svg_path),
                display_path(adapter_summary_svg_path),
            ]
        )

    manifest = {
        "artifact_kind": "ada_gemm_analysis_bundle",
        "run_id": analysis["meta"]["run_id"],
        "tag": analysis["meta"]["tag"],
        "raw_inputs": copied_raw,
        "generated": {
            "analysis_json": display_path(analysis_json_path),
            "analysis_markdown": display_path(analysis_path),
            "rankings_json": display_path(rankings_json_path),
            "rankings_markdown": display_path(rankings_md_path),
            "graphs": graphs,
        },
    }
    write_json(manifest_path, manifest)

    print(f"wrote {analysis_json_path}")
    print(f"wrote {analysis_path}")
    print(f"wrote {rankings_json_path}")
    print(f"wrote {rankings_md_path}")
    print(f"wrote {winners_svg_path}")
    print(f"wrote {focus_svg_path}")
    print(f"wrote {adapter_summary_svg_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
