#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
GLOW_RUNTIME_DIR = Path(__file__).resolve().parent / "glow_runtime"
RUNNER_SRC = GLOW_RUNTIME_DIR / "glow_gemm_runner.cpp"
RUNNER_TARGET = "pyc_glow_gemm_runner"
RUNNER_FILE = "PyCGemmRunner.cpp"
RUNNER_MARKER_START = "# >>> PYC_GLOW_GEMM_RUNNER"
RUNNER_MARKER_END = "# <<< PYC_GLOW_GEMM_RUNNER"


def emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0 if payload.get("status") in {"ok", "unavailable"} else 1


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def resolve_paths() -> tuple[Path, Path]:
    glow_root = Path(os.environ.get("GLOW_ROOT", "/root/work/glow")).resolve()
    glow_build = Path(
        os.environ.get("GLOW_BUILD_DIR", str(glow_root / "build_Release"))
    ).resolve()
    return glow_root, glow_build


def ensure_glow_install(glow_root: Path, glow_build: Path) -> tuple[bool, str]:
    if not RUNNER_SRC.exists():
        return False, f"missing runner source: {RUNNER_SRC}"
    model_compiler = glow_build / "bin" / "model-compiler"
    if not model_compiler.exists():
        return False, f"missing Glow model-compiler: {model_compiler}"
    help_proc = run([str(model_compiler), "-help"])
    help_text = f"{help_proc.stdout}\n{help_proc.stderr}"
    if "OpenCL" not in help_text:
        return False, "Glow model-compiler was built without OpenCL backend support"
    if help_proc.returncode != 0:
        return False, help_proc.stderr.strip() or "Glow model-compiler help failed"
    if shutil.which("ninja") is None:
        return False, "ninja is not installed"
    return True, ""


def sync_runner_source(glow_root: Path) -> tuple[bool, str]:
    tools_loader = glow_root / "tools" / "loader"
    cmake_path = tools_loader / "CMakeLists.txt"
    dst_src = tools_loader / RUNNER_FILE
    if not cmake_path.exists():
        return False, f"missing Glow loader CMakeLists: {cmake_path}"

    desired = RUNNER_SRC.read_text(encoding="utf-8")
    if not dst_src.exists() or dst_src.read_text(encoding="utf-8") != desired:
        dst_src.write_text(desired, encoding="utf-8")

    cmake_text = cmake_path.read_text(encoding="utf-8")
    if RUNNER_MARKER_START not in cmake_text:
        block = f"""
{RUNNER_MARKER_START}
add_executable({RUNNER_TARGET}
  {RUNNER_FILE}
  Loader.cpp)

target_link_libraries({RUNNER_TARGET}
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)
{RUNNER_MARKER_END}
"""
        cmake_text = cmake_text.rstrip() + "\n\n" + block.strip("\n") + "\n"
        cmake_path.write_text(cmake_text, encoding="utf-8")
    return True, ""


def ensure_runner_built(glow_root: Path, glow_build: Path) -> tuple[Path | None, str]:
    ok, reason = sync_runner_source(glow_root)
    if not ok:
        return None, reason
    binary = glow_build / "bin" / RUNNER_TARGET
    source_digest = hashlib.sha256(RUNNER_SRC.read_bytes()).hexdigest()
    stamp = glow_build / ".pyc_glow_runner.sha256"
    needs_build = not binary.exists() or not stamp.exists() or stamp.read_text(encoding="utf-8").strip() != source_digest
    if needs_build:
        proc = run(["ninja", "-C", str(glow_build), RUNNER_TARGET])
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "runner build failed"
            return None, stderr
        stamp.write_text(source_digest, encoding="utf-8")
    return binary, ""


def export_onnx_model(workdir: Path, m: int, k: int, n: int, dtype_label: str) -> tuple[Path | None, str]:
    try:
        import torch
    except Exception:
        return None, "PyTorch not installed"
    try:
        import onnx  # noqa: F401
    except Exception:
        return None, "onnx not installed"

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    key = (dtype_label or "float32").strip().lower()
    if key not in dtype_map:
        return None, f"unsupported Glow dtype: {dtype_label}"
    torch_dtype = dtype_map[key]

    class Gemm(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(in_features, out_features, dtype=dtype))

        def forward(self, x):
            return x @ self.weight

    model = Gemm(k, n, torch_dtype).eval()
    x = torch.randn(m, k, dtype=torch_dtype)
    out_path = workdir / f"glow_gemm_{m}x{k}x{n}_{key}.onnx"
    torch.onnx.export(
        model,
        x,
        str(out_path),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        do_constant_folding=True,
    )
    return out_path, ""


def main() -> int:
    task = os.environ.get("BENCH_TASK", "mlp").strip().lower() or "mlp"
    if task != "gemm":
        return emit(
            {
                "status": "unavailable",
                "reason": "Glow tier-2 path currently supports GEMM-only benchmarking",
            }
        )

    requested_device = os.environ.get("BENCH_DEVICE", "cuda").strip() or "cuda"
    if requested_device != "cuda":
        return emit({"status": "unavailable", "reason": "Glow OpenCL path is configured only for GPU runs"})

    m = int(os.environ.get("BENCH_M", os.environ.get("BENCH_BATCH", "64")))
    k = int(os.environ.get("BENCH_K", os.environ.get("BENCH_HIDDEN", "2048")))
    n = int(os.environ.get("BENCH_N", os.environ.get("BENCH_HIDDEN", "2048")))
    iters = int(os.environ.get("BENCH_ITERS", "80"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))
    dtype_label = os.environ.get("BENCH_DTYPE", "float32").strip().lower() or "float32"

    glow_root, glow_build = resolve_paths()
    ok, reason = ensure_glow_install(glow_root, glow_build)
    if not ok:
        return emit({"status": "unavailable", "reason": reason})

    runner_bin, reason = ensure_runner_built(glow_root, glow_build)
    if runner_bin is None:
        return emit({"status": "unavailable", "reason": f"Glow runtime runner build failed: {reason}"})

    with tempfile.TemporaryDirectory(prefix="pyc_glow_") as tmp:
        tmpdir = Path(tmp)
        model_path, reason = export_onnx_model(tmpdir, m, k, n, dtype_label)
        if model_path is None:
            return emit({"status": "unavailable", "reason": reason})

        env = os.environ.copy()
        env["PYC_GLOW_WARMUP"] = str(warmup)
        env["PYC_GLOW_ITERS"] = str(iters)
        env["PYC_GLOW_M"] = str(m)
        env["PYC_GLOW_K"] = str(k)
        env["PYC_GLOW_N"] = str(n)
        env["PYC_GLOW_DTYPE"] = dtype_label
        proc = run(
            [str(runner_bin), f"-model={model_path}", "-backend=OpenCL"],
            env=env,
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            return emit(
                {
                    "status": "error",
                    "error": proc.stderr.strip() or "Glow runtime runner failed",
                }
            )
        payload = None
        stdout = proc.stdout.strip()
        candidates = [stdout]
        if stdout:
            candidates.extend(line.strip() for line in stdout.splitlines()[::-1] if line.strip())
        for item in candidates:
            try:
                payload = json.loads(item)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            return emit(
                {
                    "status": "error",
                    "error": "Glow runtime runner did not emit JSON",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
        payload.setdefault("status", "ok")
        payload.setdefault("backend", "glow_opencl")
        payload.setdefault("mode", "native")
        payload.setdefault("task", "gemm")
        payload.setdefault("device", "opencl")
        payload.setdefault("requested_device", requested_device)
        payload.setdefault("m", m)
        payload.setdefault("k", k)
        payload.setdefault("n", n)
        payload.setdefault("iters", iters)
        payload.setdefault("warmup", warmup)
        payload.setdefault("shape", {"m": m, "k": k, "n": n})
        payload.setdefault("peak_memory_bytes", 0)
        payload.setdefault("note", "Glow runtime path via OpenCL HostManager")
        samples = payload.get("samples_ms")
        if isinstance(samples, list) and samples:
            numeric = [float(v) for v in samples]
            mean_ms = statistics.mean(numeric)
            payload["latency_ms"] = {
                "mean": round(mean_ms, 4),
                "p50": round(percentile(numeric, 50), 4),
                "p95": round(percentile(numeric, 95), 4),
                "min": round(min(numeric), 4),
                "max": round(max(numeric), 4),
            }
            flops_per_iter = float(2 * m * k * n)
            flops_per_sec = (flops_per_iter / mean_ms) * 1000.0 if mean_ms > 0 else 0.0
            payload["throughput_flops_per_sec"] = round(flops_per_sec, 2)
            payload["throughput_tflops_per_sec"] = round(flops_per_sec / 1.0e12, 4)
            payload["throughput_tokens_per_sec"] = 0.0
        return emit(payload)


if __name__ == "__main__":
    raise SystemExit(main())
