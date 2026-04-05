from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_pyc_compiler_next_bench_supports_mixed_shape_sequence():
    build_dir = ROOT / "build-bench-sequence-test"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    cfg = _run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-D",
            "PYC_BUILD_EXPERIMENTAL=OFF",
            "-D",
            "PYC_BUILD_BENCHMARKS=ON",
            "-D",
            "PYC_BUILD_COMPILER_NEXT=ON",
            "-D",
            "PYC_BUILD_COMPILER_NEXT_TESTS=OFF",
        ],
        cwd=ROOT,
    )
    assert cfg.returncode == 0, cfg.stderr

    bld = _run(
        ["cmake", "--build", str(build_dir), "--parallel", "--target", "pyc_compiler_next_bench"],
        cwd=ROOT,
    )
    assert bld.returncode == 0, bld.stderr

    exe = build_dir / "pyc_compiler_next_bench"
    env = os.environ.copy()
    env.update(
        {
            "BENCH_TASK": "gemm",
            "BENCH_SEQUENCE": "128x128x128;256x256x256;512x512x512;256x256x256;128x128x128",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "1",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "3",
            "PYC_BENCH_ENABLE_PHANTOM_GRAPH": "1",
            "PYC_BENCH_PHANTOM_HORIZON_STEPS": "1",
            "PYC_BENCH_CACHE_IN_MEMORY": "1",
            "PYC_BENCH_MEMORY_BUDGET_BYTES": "65536",
        }
    )
    proc = _run([str(exe), "cpu", "128", "128", "4", "1"], cwd=ROOT, env=env)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["task"] == "gemm_sequence"
    assert payload["sequence"]["count"] == 5
    assert payload["compile_options"]["enable_phantom_graph"] is True
    assert payload["compile_options"]["enable_speculative_plans"] is True
    assert payload["controller"]["objective_mode"] == "balanced"
    assert payload["controller"]["shadow_mode"] == "memory_first"
    assert payload["controller"]["shadow_reason"] == "pressure"
    assert payload["controller"]["rollback_reason"] == "none"
    assert payload["controller"]["rollback_count"] == 0

    steps = payload["sequence"]["steps"]
    assert steps[0]["m"] == 128
    assert steps[1]["m"] == 256
    assert steps[2]["m"] == 512
    assert steps[1]["phantom_graph"]["reshape_delta"] >= 1
    assert steps[1]["phantom_graph"]["mismatch_delta"] >= 1
    assert payload["sequence"]["summary"]["phantom_reshape_count"] >= 1
    assert steps[0]["reliability"]["rematerialized_tensors"] >= 1
    assert steps[0]["reliability"]["rematerialized_bytes"] >= 1
    assert payload["reliability"]["rematerialized_bytes"] >= steps[0]["reliability"]["rematerialized_bytes"]

    env_single = env.copy()
    env_single.pop("BENCH_SEQUENCE", None)
    proc_single = _run([str(exe), "cpu", "128", "128", "4", "1"], cwd=ROOT, env=env_single)
    assert proc_single.returncode == 0, proc_single.stderr
    payload_single = json.loads(proc_single.stdout)
    assert payload_single["status"] == "ok"
    assert payload_single["task"] == "gemm"
    assert payload_single["controller"]["objective_mode"] == "balanced"
    assert payload_single["controller"]["shadow_mode"] == "memory_first"
    assert payload_single["controller"]["shadow_reason"] == "pressure"
    assert payload_single["controller"]["rollback_reason"] == "none"
    assert payload_single["controller"]["rollback_count"] == 0
