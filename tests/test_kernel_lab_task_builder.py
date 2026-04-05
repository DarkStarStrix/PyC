from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "kernels" / "lab" / "kernel_lab.py"
SPEC = importlib.util.spec_from_file_location("kernel_lab", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
kernel_lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kernel_lab
SPEC.loader.exec_module(kernel_lab)


def test_resolve_task_baseline_prefers_arch_specific_entry():
    kernels = {
        "matrix_mult": {"name": "matrix_mult"},
        "ada_gemm": {"name": "ada_gemm"},
    }
    manifest = {
        "baselines": [
            {"task_kind": "gemm", "backend": "cuda", "arch": "generic", "kernel": "matrix_mult"},
            {"task_kind": "gemm", "backend": "cuda", "arch": "sm89", "kernel": "ada_gemm"},
        ]
    }

    kernel, meta = kernel_lab.resolve_task_baseline(kernels, manifest, "gemm", "cuda", "sm89")

    assert kernel["name"] == "ada_gemm"
    assert meta["resolution"] == "manifest"
    assert meta["entry"]["arch"] == "sm89"


def test_promote_task_baseline_updates_matching_slot():
    manifest = {
        "baselines": [
            {
                "task_kind": "gemm",
                "backend": "cuda",
                "arch": "sm89",
                "kernel": "ada_gemm",
                "source": "seed",
                "updated_utc": None,
                "notes": "seed",
            }
        ]
    }

    entry = kernel_lab.promote_task_baseline(
        manifest,
        task_kind="gemm",
        backend="cuda",
        arch="sm89",
        kernel_name="ada_tensor_core_fp16",
        notes="Task winner",
        source="task:ada-tc",
    )

    assert entry["kernel"] == "ada_tensor_core_fp16"
    assert entry["source"] == "task:ada-tc"
    assert entry["notes"] == "Task winner"
    assert manifest["baselines"][0]["kernel"] == "ada_tensor_core_fp16"


def test_default_baseline_manifest_tracks_current_sm89_winner():
    manifest = kernel_lab.default_baseline_manifest()

    sm89_gemm = next(
        entry
        for entry in manifest["baselines"]
        if entry["task_kind"] == "gemm" and entry["backend"] == "cuda" and entry["arch"] == "sm89"
    )

    assert sm89_gemm["kernel"] == "ada_gemm_k64_warp32_async"
    assert "winner" in sm89_gemm["notes"].lower()


def test_task_record_embeds_baseline_and_benchmark_plan():
    class Args:
        candidate_tag = ["ada"]
        candidate_name = ["ada_gemm"]
        pyc_feature_profile = ["pyc-fp32-speculative"]
        matrix_file = "benchmark/benchmarks/gpu/configs/ada_fp32_gemm_shapes.json"
        repeats = 7
        warmup = 3
        nvcc = "nvcc"

    hardware = {
        "backend": "cuda",
        "arch": "sm89",
        "capacity": {"tier": "large"},
    }
    baseline_kernel = {
        "name": "ada_gemm",
        "source": "kernels/prototypes/ada/gemm/kernel.cu",
        "description": "",
        "tags": ["ada"],
        "compile_cmd": "nvcc ...",
        "run_cmd": "run ...",
    }

    record = kernel_lab.task_record(
        "Ada GEMM Loop",
        "gemm",
        "beat the previous Ada GEMM baseline",
        hardware,
        baseline_kernel,
        {"resolution": "manifest"},
        Args(),
    )

    assert record["task"]["baseline_kernel"] == "ada_gemm"
    assert record["hardware"]["arch"] == "sm89"
    assert record["task"]["pyc_feature_profiles"] == ["pyc-fp32-speculative"]
    assert record["benchmark_plan"]["hardware_constraints"]["capacity_tier"] == "large"
    assert any("run_gemm_suite.py" in cmd for cmd in record["benchmark_plan"]["profile_protocol"])
    assert any("PYC_BENCH_ENABLE_SPECULATIVE_PLANS=1" in cmd for cmd in record["benchmark_plan"]["profile_protocol"])


def test_cmd_task_run_dry_run_writes_execution_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        task_dir = root / "tasks"
        run_dir = root / "runs"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / "ada-sm89-gemm.json"
        kernel_lab.write_json(
            task_path,
            {
                "meta": {"name": "ada-sm89-gemm", "slug": "ada-sm89-gemm"},
                "task": {"kind": "gemm"},
                "hardware": {"backend": "cuda", "arch": "sm89"},
                "benchmark_plan": {
                    "profile_protocol": [
                        "python3 -c 'print(\"wrote fake.json\")'",
                        "python3 -c 'print(\"best_ms=1.25\")'",
                    ]
                },
            },
        )

        args = type(
            "Args",
            (),
            {
                "name": "ada-sm89-gemm",
                "task_dir": str(task_dir),
                "task_run_dir": str(run_dir),
                "keep_going": False,
                "dry_run": True,
                "progress": False,
            },
        )()

        rc = kernel_lab.cmd_task_run(args)

        assert rc == kernel_lab.EXIT_OK
        records = sorted(run_dir.glob("ada-sm89-gemm-*.json"))
        assert len(records) == 1
        doc = kernel_lab.read_json(records[0])
        assert doc["meta"]["status"] == "planned"
        assert len(doc["steps"]) == 2


def test_cmd_task_run_captures_observed_metrics_and_written_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        task_dir = root / "tasks"
        run_dir = root / "runs"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / "probe-task.json"
        kernel_lab.write_json(
            task_path,
            {
                "meta": {"name": "probe-task", "slug": "probe-task"},
                "task": {"kind": "gemm"},
                "hardware": {"backend": "cuda", "arch": "sm89"},
                "benchmark_plan": {
                    "profile_protocol": [
                        "python3 -c 'print(\"wrote fake.json\"); print(\"best_ms=1.25\")'",
                    ]
                },
            },
        )

        args = type(
            "Args",
            (),
            {
                "name": "probe-task",
                "task_dir": str(task_dir),
                "task_run_dir": str(run_dir),
                "keep_going": False,
                "dry_run": False,
                "progress": False,
            },
        )()

        rc = kernel_lab.cmd_task_run(args)

        assert rc == kernel_lab.EXIT_OK
        records = sorted(run_dir.glob("probe-task-*.json"))
        assert len(records) == 1
        doc = kernel_lab.read_json(records[0])
        step = doc["steps"][0]
        assert step["status"] == "ok"
        assert step["observed_metrics"]["best_ms"] == 1.25
        assert "fake.json" in step["written_paths"][0]


def test_parse_command_with_env_prefix():
    env, argv = kernel_lab.parse_command_with_env(
        "PYC_BENCH_ENABLE_SPECULATIVE_PLANS=1 PYC_BENCH_OBJECTIVE_MODE=balanced python3 script.py --flag"
    )

    assert env["PYC_BENCH_ENABLE_SPECULATIVE_PLANS"] == "1"
    assert env["PYC_BENCH_OBJECTIVE_MODE"] == "balanced"
    assert argv == ["python3", "script.py", "--flag"]
