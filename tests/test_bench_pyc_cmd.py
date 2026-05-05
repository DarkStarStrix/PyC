from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bench_pyc_cmd_times_out_cleanly(tmp_path: Path):
    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True)
    exe = build_dir / "pyc_compiler_next_bench"
    exe.write_text(
        "#!/usr/bin/env bash\n"
        "sleep 2\n",
        encoding="utf-8",
    )
    exe.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PYC_GPU_BENCH_BUILD_DIR": str(build_dir),
            "PYC_GPU_BENCH_EXE": "pyc_compiler_next_bench",
            "PYC_GPU_BENCH_SKIP_BUILD_IF_PRESENT": "1",
            "PYC_GPU_BENCH_TIMEOUT_SEC": "1",
            "BENCH_TASK": "gemm",
            "BENCH_DEVICE": "cuda",
            "BENCH_M": "4096",
            "BENCH_K": "4096",
            "BENCH_N": "16384",
            "BENCH_DTYPE": "bfloat16",
        }
    )

    proc = subprocess.run(
        ["python3", str(ROOT / "benchmark" / "benchmarks" / "gpu" / "external" / "bench_pyc_cmd.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["status"] == "error"
    assert "timed out" in payload["error"]
    assert payload["shape"] == {"m": 4096, "k": 4096, "n": 16384}
    assert payload["timeout_sec"] == 1
