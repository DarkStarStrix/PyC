from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_shape(path: Path, shape_name: str, adapters: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "shape": {
                    "name": shape_name,
                    "m": 4096,
                    "k": 4096,
                    "n": 4096,
                },
                "adapters": adapters,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_hopper_analysis_emits_ranked_actions(tmp_path: Path):
    run_dir = tmp_path / "run"
    json_dir = run_dir / "json"
    json_dir.mkdir(parents=True)
    output_dir = tmp_path / "analysis"

    _write_shape(
        json_dir / "20260421T000000Z__hopper_loop__square-4096.json",
        "square-4096",
        {
            "torch_eager": {
                "adapter": "torch_eager",
                "display_name": "PyTorch Eager",
                "status": "ok",
                "mode": "native",
                "throughput_tflops_per_sec": 600.0,
            },
            "xla": {
                "adapter": "xla",
                "display_name": "XLA",
                "status": "ok",
                "mode": "proxy",
                "throughput_tflops_per_sec": 550.0,
            },
            "cutlass": {
                "adapter": "cutlass",
                "display_name": "CUTLASS",
                "status": "unavailable",
                "mode": "unknown",
                "reason": "cutlass_profiler not found in PATH",
            },
            "tvm": {
                "adapter": "tvm",
                "display_name": "TVM",
                "status": "unavailable",
                "mode": "unknown",
            },
            "pyc": {
                "adapter": "pyc",
                "display_name": "PyC CUDA",
                "status": "ok",
                "mode": "native",
                "throughput_tflops_per_sec": 50.0,
            },
        },
    )

    (json_dir / "20260421T000000Z__hopper_loop.progress.json").write_text(
        json.dumps(
            {
                "meta": {"run_id": "20260421T000000Z", "tag": "hopper_loop", "status": "running"},
                "progress": {
                    "completed_runs": 120,
                    "current_shape_name": "ff-up-4096x4096x16384",
                    "current_adapter": "pyc",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(ROOT / "benchmark" / "tools" / "analyze_hopper_gemm_results.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    analysis = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    titles = [item["title"] for item in analysis["recommendations"]]
    assert "Install CUTLASS profiler on the Hopper box" in titles
    assert "Install native TVM CUDA or remove TVM from trusted Hopper runs" in titles
    assert "Keep proxy adapters out of trust-grade Hopper runs" in titles
    assert "Debug the PyC stall on the active Hopper GEMM shape" in titles
