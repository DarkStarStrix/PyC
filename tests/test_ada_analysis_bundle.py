from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kernel_lab = _load_module("kernel_lab_testable", ROOT / "kernels" / "lab" / "kernel_lab.py")
ada_analysis = _load_module("ada_analysis_testable", ROOT / "benchmark" / "tools" / "analyze_ada_gemm_results.py")


def test_extract_key_value_metrics_parses_kernel_stdout():
    payload = kernel_lab.extract_key_value_metrics(
        "\n".join(
            [
                "device=NVIDIA RTX 6000 Ada Generation cc=8.9",
                "shape=1024x1024x1024",
                "best_ms=0.091",
                "max_abs_diff=0.000000",
                "gflops=23563.505",
            ]
        )
    )

    assert payload["shape"] == "1024x1024x1024"
    assert payload["best_ms"] == 0.091
    assert payload["gflops"] == 23563.505
    assert payload["device"] == "NVIDIA RTX 6000 Ada Generation cc=8.9"


def test_build_bundle_flags_incomparability_and_next_task():
    sweep = {
        "meta": {
            "run_id": "20260404T191958Z",
            "tag": "ada-sm89-fp32-baseline-sweep",
            "host": "host89",
            "git_head": "491ddc6",
            "git_dirty": 1,
        },
        "gpu": {"gpus": [{"name": "NVIDIA RTX 6000 Ada Generation"}]},
        "shapes": [
            {
                "shape": {"name": "square-1024", "m": 1024, "k": 1024, "n": 1024},
                "adapters": {
                    "torch_eager": {
                        "status": "ok",
                        "display_name": "PyTorch Eager",
                        "latency_ms": {"mean": 0.0366, "p50": 0.0360, "p95": 0.0400},
                        "throughput_tflops_per_sec": 58.6477,
                        "mode": "native",
                    },
                    "torch_compile": {
                        "status": "ok",
                        "display_name": "PyTorch Compile",
                        "latency_ms": {"mean": 0.0358, "p50": 0.0350, "p95": 0.0390},
                        "throughput_tflops_per_sec": 59.9069,
                        "mode": "native",
                    },
                    "pyc": {"status": "unavailable", "reason": "missing command"},
                },
            }
        ],
    }
    kernel_doc = {
        "kernel": {"name": "ada_gemm"},
        "run": {
            "stats": {"mean_ms": 3752.409},
            "observed_metrics": {
                "shape": "1024x1024x1024",
                "device": "NVIDIA RTX 6000 Ada Generation cc=8.9",
                "best_ms": 0.091,
                "max_abs_diff": 0.0,
                "gflops": 23563.505,
            },
        },
    }

    class Args:
        sweep_json = Path("sweep.json")
        kernel_json = Path("kernel.json")
        run_dir = Path("run_dir")
        kernel_best_ms = 0.0
        kernel_gflops = 0.0
        kernel_max_abs_diff = 0.0
        kernel_shape = ""
        kernel_device = ""

    analysis, rankings = ada_analysis.build_bundle(sweep, kernel_doc, Args())

    assert rankings["adapter_summary"][0]["adapter"] == "torch_compile"
    assert analysis["judgment"]["status"] == "blocked_on_comparable_pyc_measurement"
    assert analysis["square_1024_focus"]["direct_kernel"]["tflops"] == 23.563505
    assert analysis["judgment"]["next_task"]["name"] == "ada-sm89-fp32-comparable-pyc-sweep"


def test_build_bundle_surfaces_pyc_build_errors():
    sweep = {
        "meta": {
            "run_id": "20260404T194427Z",
            "tag": "ada-sm89-fp32-comparable-pyc-sweep",
            "host": "host89",
            "git_head": "491ddc6",
            "git_dirty": 0,
        },
        "gpu": {"gpus": [{"name": "NVIDIA RTX 6000 Ada Generation"}]},
        "shapes": [
            {
                "shape": {"name": "square-1024", "m": 1024, "k": 1024, "n": 1024},
                "adapters": {
                    "torch_compile": {
                        "status": "ok",
                        "display_name": "PyTorch Compile",
                        "latency_ms": {"mean": 0.0358, "p50": 0.0350, "p95": 0.0390},
                        "throughput_tflops_per_sec": 59.9069,
                        "mode": "native",
                    },
                    "pyc": {
                        "status": "error",
                        "display_name": "PyC CUDA",
                        "error": "undefined reference to pyc_cutlass_gemm_dispatch",
                    },
                },
            }
        ],
    }
    kernel_doc = {
        "kernel": {"name": "ada_gemm"},
        "run": {
            "stats": {"mean_ms": 3752.409},
            "observed_metrics": {
                "shape": "1024x1024x1024",
                "device": "NVIDIA RTX 6000 Ada Generation cc=8.9",
                "best_ms": 0.091,
                "max_abs_diff": 0.0,
                "gflops": 23563.505,
            },
        },
    }

    class Args:
        sweep_json = Path("sweep.json")
        kernel_json = Path("kernel.json")
        run_dir = Path("run_dir")
        kernel_best_ms = 0.0
        kernel_gflops = 0.0
        kernel_max_abs_diff = 0.0
        kernel_shape = ""
        kernel_device = ""

    analysis, rankings = ada_analysis.build_bundle(sweep, kernel_doc, Args())

    pyc_health = next(row for row in rankings["adapter_health"] if row["adapter"] == "pyc")
    assert pyc_health["error_shapes"] == 1
    assert analysis["judgment"]["status"] == "blocked_on_pyc_build_integration"
    assert analysis["judgment"]["next_task"]["name"] == "ada-sm89-pyc-build-bridge"
    assert "pyc_cutlass_gemm_dispatch" in analysis["judgment"]["reasons"][1]


def test_build_bundle_includes_pyc_phantom_summary():
    sweep = {
        "meta": {
            "run_id": "20260404T230000Z",
            "tag": "ada-sm89-phantom-shadow",
            "host": "host89",
            "git_head": "491ddc6",
            "git_dirty": 0,
        },
        "gpu": {"gpus": [{"name": "NVIDIA RTX 6000 Ada Generation"}]},
        "shapes": [
            {
                "shape": {"name": "square-1024", "m": 1024, "k": 1024, "n": 1024},
                "adapters": {
                    "torch_compile": {
                        "status": "ok",
                        "display_name": "PyTorch Compile",
                        "latency_ms": {"mean": 0.0800, "p50": 0.0790, "p95": 0.0820},
                        "throughput_tflops_per_sec": 26.0,
                        "mode": "native",
                        "dtype": "float32",
                    },
                    "pyc": {
                        "status": "ok",
                        "display_name": "PyC CUDA",
                        "latency_ms": {"mean": 0.0810, "p50": 0.0800, "p95": 0.0830},
                        "throughput_tflops_per_sec": 25.7,
                        "mode": "native",
                        "dtype": "float32",
                        "phantom_graph": {
                            "enabled": True,
                            "match": False,
                            "match_count": 3,
                            "mismatch_count": 1,
                            "reshape_count": 1,
                            "confidence": 0.92,
                            "match_score": 0.75,
                            "expected_signature": "1:r2x2048x2048;1:r2x2048x2048",
                            "observed_signature": "1:r2x1024x1024;1:r2x1024x1024",
                        },
                    },
                },
            }
        ],
    }
    kernel_doc = {"kernel": {"name": "ada_gemm"}, "run": {"observed_metrics": {"gflops": 23563.505, "best_ms": 0.091}}}

    class Args:
        sweep_json = Path("sweep.json")
        kernel_json = Path("kernel.json")
        run_dir = Path("run_dir")
        kernel_best_ms = 0.0
        kernel_gflops = 0.0
        kernel_max_abs_diff = 0.0
        kernel_shape = ""
        kernel_device = ""

    analysis, rankings = ada_analysis.build_bundle(sweep, kernel_doc, Args())
    markdown = ada_analysis.render_analysis_markdown(analysis)

    assert rankings["pyc_phantom"]["enabled_shapes"] == 1
    assert rankings["pyc_phantom"]["mismatched_shapes"] == 1
    assert rankings["pyc_phantom"]["reshape_events"] == 1
    assert analysis["pyc_phantom"]["mean_match_score"] == 0.75
    assert "Phantom graph" in markdown
    assert "expected=`1:r2x2048x2048;1:r2x2048x2048`" in markdown


def test_build_bundle_recognizes_gemm_sequence_and_summarizes_steps():
    sweep = {
        "task": "gemm_sequence",
        "meta": {
            "run_id": "20260404T230013Z",
            "tag": "ada-sm89-fp32-phantom-sequence-v2",
            "host": "host89",
            "git_head": "491ddc6",
            "git_dirty": 0,
        },
        "gpu": {"gpus": [{"name": "NVIDIA RTX 6000 Ada Generation"}]},
        "sequence": {
            "summary": {
                "guard_miss_count": 0,
                "fallback_count": 0,
            },
            "steps": [
                {
                    "index": 1,
                    "m": 512,
                    "k": 512,
                    "n": 512,
                    "latency_ms": 0.055,
                    "throughput_tflops_per_sec": 9.31,
                    "profile": "pyc-fp32-speculative",
                    "reliability": {
                        "compile_cache_hit": True,
                        "speculative_plan_hit": True,
                        "rematerialized_tensors": 0,
                        "rematerialized_bytes": 0,
                    },
                    "phantom_graph": {
                        "enabled": True,
                        "match": True,
                        "match_count": 4,
                        "mismatch_count": 0,
                        "reshape_count": 0,
                        "confidence": 1.0,
                        "match_score": 1.0,
                        "expected_signature": "1:r2x512x512;1:r2x512x512",
                        "observed_signature": "1:r2x512x512;1:r2x512x512",
                    },
                },
                {
                    "index": 2,
                    "m": 1024,
                    "k": 1024,
                    "n": 1024,
                    "latency_ms": 0.074,
                    "throughput_tflops_per_sec": 26.6,
                    "profile": "pyc-fp32-speculative",
                    "reliability": {
                        "compile_cache_hit": True,
                        "speculative_plan_hit": True,
                        "rematerialized_tensors": 1,
                        "rematerialized_bytes": 4096,
                    },
                    "phantom_graph": {
                        "enabled": True,
                        "match": False,
                        "match_count": 2,
                        "mismatch_count": 1,
                        "reshape_count": 1,
                        "confidence": 0.94,
                        "match_score": 0.81,
                        "expected_signature": "1:r2x512x512;1:r2x512x512",
                        "observed_signature": "1:r2x1024x1024;1:r2x1024x1024",
                    },
                },
                {
                    "index": 3,
                    "m": 2048,
                    "k": 2048,
                    "n": 2048,
                    "latency_ms": 0.089,
                    "throughput_tflops_per_sec": 47.26,
                    "profile": "pyc-fp32-speculative",
                    "reliability": {
                        "compile_cache_hit": True,
                        "speculative_plan_hit": True,
                        "rematerialized_tensors": 0,
                        "rematerialized_bytes": 0,
                    },
                    "phantom_graph": {
                        "enabled": True,
                        "match": True,
                        "match_count": 3,
                        "mismatch_count": 1,
                        "reshape_count": 1,
                        "confidence": 0.95,
                        "match_score": 0.91,
                        "expected_signature": "1:r2x1024x1024;1:r2x1024x1024",
                        "observed_signature": "1:r2x2048x2048;1:r2x2048x2048",
                    },
                },
            ],
        },
    }
    kernel_doc = {"kernel": {"name": "ada_gemm"}, "run": {"observed_metrics": {"gflops": 23563.505, "best_ms": 0.091}}}

    class Args:
        sweep_json = Path("sweep.json")
        kernel_json = Path("kernel.json")
        run_dir = Path("run_dir")
        kernel_best_ms = 0.0
        kernel_gflops = 0.0
        kernel_max_abs_diff = 0.0
        kernel_shape = ""
        kernel_device = ""

    analysis, rankings = ada_analysis.build_bundle(sweep, kernel_doc, Args())
    markdown = ada_analysis.render_analysis_markdown(analysis)
    rankings_md = ada_analysis.render_rankings_markdown(rankings)

    assert rankings["task"] == "gemm_sequence"
    assert rankings["sequence_summary"]["step_count"] == 3
    assert rankings["sequence_summary"]["shape_transition_count"] == 2
    assert rankings["sequence_phantom"]["enabled_steps"] == 3
    assert rankings["sequence_phantom"]["mismatched_steps"] == 1
    assert rankings["sequence_rematerialization"]["rematerialized_steps"] == 1
    assert analysis["judgment"]["status"] == "mixed_shape_hot_path_validated"
    assert analysis["judgment"]["next_task"]["name"] == "ada-sm89-mixed-shape-hot-path-followup"
    assert "Per-Step Summary" in markdown
    assert "Rematerialization Summary" in markdown
    assert "mixed_shape_hot_path_validated" in markdown
    assert "Ada GEMM Sequence Rankings" in rankings_md
    assert "2048x2048x2048" in rankings_md
