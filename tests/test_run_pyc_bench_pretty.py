from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_pyc_bench_pretty_prints_summary_and_keeps_json(tmp_path: Path):
    build_dir = tmp_path / "build"
    build_dir.mkdir(parents=True)
    exe = build_dir / "pyc_compiler_next_bench"
    json_out = tmp_path / "raw.json"
    exe.write_text(
        "#!/usr/bin/env bash\n"
        "cat <<'JSON'\n"
        '{"status":"ok","task":"gemm","device":"cuda","m":1024,"k":1024,"n":1024,'
        '"latency_ms":{"mean":0.1234},"throughput_tflops_per_sec":12.3456,'
        '"profile":{"dispatch_ms_mean":0.1111,"kernel_select_ms_mean":0.2222},'
        '"execution_path":"cuda_promoted_gemm:[kernel]","kernel_selection":{"symbol":"k[0]"},'
        '"reliability":{"fallback_count":0}}\n'
        "JSON\n",
        encoding="utf-8",
    )
    exe.chmod(0o755)

    env = os.environ.copy()
    env["PYC_GPU_BENCH_BUILD_DIR"] = str(build_dir)
    env["PYC_GPU_BENCH_EXE"] = "pyc_compiler_next_bench"
    env["PYC_BENCH_JSON_OUT"] = str(json_out)

    proc = subprocess.run(
        [str(ROOT / "scripts" / "run_pyc_bench_pretty.sh"), "cuda", "64", "1024", "5", "2"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert json_out.exists()

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["task"] == "gemm"

    assert "[pyc-bench] task=gemm device=cuda shape=1024x1024x1024" in proc.stdout
    assert "[pyc-bench] mean_ms=0.1234 tflops=12.3456" in proc.stdout
    assert "[pyc-bench] path=cuda_promoted_gemm:[kernel] kernel=k[0] fallback=0" in proc.stdout
    assert f"[pyc-bench] json={json_out}" in proc.stdout
