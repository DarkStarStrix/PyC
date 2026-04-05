#!/usr/bin/env python3
"""Kernel Lab CLI: prototype, test, and benchmark kernel commands.

Deterministic behavior goals:
- No uncaught tracebacks for expected operational failures.
- Stable exit codes.
- Explicit preflight checks for required toolchains (e.g., nvcc).
- Manifest schema validation before execution.
"""

import argparse
import datetime as dt
import json
import os
import platform
import sys
import shlex
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
PROGRESS_STATE_DIR = ROOT / "infra" / "nexa_insight"
if str(PROGRESS_STATE_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRESS_STATE_DIR))
from progress_state import write_progress_state

DEFAULT_MANIFEST = ROOT / "kernels" / "lab" / "manifests" / "kernels.json"
DEFAULT_BASELINE_MANIFEST = ROOT / "kernels" / "lab" / "manifests" / "task_baselines.json"
DEFAULT_RESULTS_DIR = ROOT / "kernels" / "lab" / "results"
DEFAULT_TASK_DIR = ROOT / "kernels" / "lab" / "tasks"
DEFAULT_TASK_RUN_DIR = ROOT / "kernels" / "lab" / "tasks" / "runs"
DEFAULT_TASK_PROGRESS_FILE = DEFAULT_TASK_RUN_DIR / "latest_task_run.progress.json"

EXIT_OK = 0
EXIT_USER_ERROR = 2
EXIT_MANIFEST_ERROR = 3
EXIT_TOOLCHAIN_MISSING = 4
EXIT_COMMAND_FAILED = 5

REQUIRED_KERNEL_KEYS = {"name", "source", "compile_cmd", "run_cmd"}
SOLID_PROGRESS_CHARS = " ▏▎▍▌▋▊▉█"
VALID_ENV_PREFIX_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")

PYC_FEATURE_PROFILES = {
    "pyc-fp32-baseline": {
        "description": "Current tuned FP32 CUDA path with cuBLASLt + TF32 and no speculative plans.",
        "env": {
            "PYC_CUDA_ENABLE_CUBLASLT": "1",
            "PYC_CUDA_ALLOW_TF32": "1",
            "PYC_CUDA_ENABLE_GRAPH_REPLAY": "1",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "0",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "0",
            "PYC_BENCH_OBJECTIVE_MODE": "balanced",
        },
    },
    "pyc-fp32-speculative": {
        "description": "Enable speculative plans on the tuned FP32 CUDA path.",
        "env": {
            "PYC_CUDA_ENABLE_CUBLASLT": "1",
            "PYC_CUDA_ALLOW_TF32": "1",
            "PYC_CUDA_ENABLE_GRAPH_REPLAY": "1",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "1",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "4",
            "PYC_BENCH_OBJECTIVE_MODE": "balanced",
        },
    },
    "pyc-fp32-speculative-memory": {
        "description": "Speculative plans with memory-first objective and a bounded memory budget.",
        "env": {
            "PYC_CUDA_ENABLE_CUBLASLT": "1",
            "PYC_CUDA_ALLOW_TF32": "1",
            "PYC_CUDA_ENABLE_GRAPH_REPLAY": "1",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "1",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "4",
            "PYC_BENCH_OBJECTIVE_MODE": "memory_first",
            "PYC_BENCH_MEMORY_BUDGET_BYTES": str(8 * 1024 * 1024 * 1024),
        },
    },
    "pyc-fp32-speculative-util": {
        "description": "Speculative plans with utilization-first objective.",
        "env": {
            "PYC_CUDA_ENABLE_CUBLASLT": "1",
            "PYC_CUDA_ALLOW_TF32": "1",
            "PYC_CUDA_ENABLE_GRAPH_REPLAY": "1",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "1",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "4",
            "PYC_BENCH_OBJECTIVE_MODE": "utilization_first",
            "PYC_BENCH_TARGET_UTILIZATION_FLOOR": "0.85",
        },
    },
    "pyc-fp32-phantom-shadow": {
        "description": "Tuned FP32 CUDA path with phantom graph enabled in shadow/reshape mode.",
        "env": {
            "PYC_CUDA_ENABLE_CUBLASLT": "1",
            "PYC_CUDA_ALLOW_TF32": "1",
            "PYC_CUDA_ENABLE_GRAPH_REPLAY": "1",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "1",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "4",
            "PYC_BENCH_ENABLE_PHANTOM_GRAPH": "1",
            "PYC_BENCH_PHANTOM_HORIZON_STEPS": "1",
            "PYC_BENCH_OBJECTIVE_MODE": "balanced",
        },
    },
    "pyc-fp32-phantom-shadow-util": {
        "description": "Phantom graph plus utilization-first objective for runtime shape-family experiments.",
        "env": {
            "PYC_CUDA_ENABLE_CUBLASLT": "1",
            "PYC_CUDA_ALLOW_TF32": "1",
            "PYC_CUDA_ENABLE_GRAPH_REPLAY": "1",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "1",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "4",
            "PYC_BENCH_ENABLE_PHANTOM_GRAPH": "1",
            "PYC_BENCH_PHANTOM_HORIZON_STEPS": "2",
            "PYC_BENCH_OBJECTIVE_MODE": "utilization_first",
            "PYC_BENCH_TARGET_UTILIZATION_FLOOR": "0.85",
        },
    },
    "pyc-fp32-no-graph-replay": {
        "description": "Tuned FP32 CUDA path with graph replay disabled to isolate replay value.",
        "env": {
            "PYC_CUDA_ENABLE_CUBLASLT": "1",
            "PYC_CUDA_ALLOW_TF32": "1",
            "PYC_CUDA_ENABLE_GRAPH_REPLAY": "0",
            "PYC_BENCH_ENABLE_SPECULATIVE_PLANS": "0",
            "PYC_BENCH_MAX_SPECULATIVE_PLANS": "0",
            "PYC_BENCH_OBJECTIVE_MODE": "balanced",
        },
    },
}


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


class LabError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def ensure_results_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def summarize(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "stdev_ms": round(statistics.pstdev(values), 3),
        "samples_ms": [round(v, 3) for v in values],
    }


def iter_with_progress(items, enabled: bool, desc: str, unit: str):
    if enabled and tqdm is not None and sys.stderr.isatty():
        return tqdm(
            items,
            desc=desc,
            unit=unit,
            file=sys.stderr,
            dynamic_ncols=True,
            ascii=SOLID_PROGRESS_CHARS,
            mininterval=0.1,
            leave=True,
            smoothing=0.05,
        )
    return items


def require_progress_support(enabled: bool):
    if enabled and tqdm is None:
        raise LabError("--progress requested but tqdm is not installed in this Python environment", EXIT_USER_ERROR)


def coerce_metric_value(raw: str):
    text = raw.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def extract_key_value_metrics(text: str):
    metrics = {}
    for line in str(text or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            continue
        metrics[key] = coerce_metric_value(value)
    return metrics


def trim_output(text: str, max_lines: int = 20, max_chars: int = 4000):
    lines = str(text or "").splitlines()
    trimmed = "\n".join(lines[-max_lines:])
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[-max_chars:]


def print_status(message: str):
    print(message, flush=True)


def is_env_assignment_token(token: str):
    if "=" not in token:
        return False
    key, _ = token.split("=", 1)
    return bool(key) and all(ch in VALID_ENV_PREFIX_CHARS for ch in key)


def parse_command_with_env(cmd: str):
    try:
        parts = shlex.split(cmd)
    except ValueError as exc:
        raise LabError(f"Invalid command quoting: {exc}", EXIT_MANIFEST_ERROR)

    if not parts:
        raise LabError("Empty command after template expansion", EXIT_MANIFEST_ERROR)

    env = {}
    index = 0
    while index < len(parts) and is_env_assignment_token(parts[index]):
        key, value = parts[index].split("=", 1)
        env[key] = value
        index += 1

    argv = parts[index:]
    if not argv:
        raise LabError("Command contains environment assignments but no executable", EXIT_MANIFEST_ERROR)
    return env, argv


def feature_profile_names():
    return sorted(PYC_FEATURE_PROFILES.keys())


def resolve_feature_profiles(names):
    resolved = []
    for name in names or []:
        profile = PYC_FEATURE_PROFILES.get(name)
        if not profile:
            choices = ", ".join(feature_profile_names())
            raise LabError(f"Unknown feature profile '{name}'. Choices: {choices}", EXIT_USER_ERROR)
        resolved.append(
            {
                "name": name,
                "description": profile["description"],
                "env": dict(profile["env"]),
            }
        )
    return resolved


def env_prefix_from_mapping(env_map: dict):
    if not env_map:
        return ""
    parts = [f"{key}={shlex.quote(str(value))}" for key, value in env_map.items()]
    return " ".join(parts) + " "


def extract_written_paths(text: str):
    paths = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("wrote "):
            paths.append(stripped[len("wrote ") :].strip())
        elif stripped.startswith("Wrote "):
            paths.append(stripped[len("Wrote ") :].strip())
        elif stripped.startswith("updated "):
            paths.append(stripped[len("updated ") :].strip())
    return paths


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LabError(f"File not found: {path}", EXIT_USER_ERROR)
    except json.JSONDecodeError as exc:
        raise LabError(f"Invalid JSON in {path}: {exc}", EXIT_MANIFEST_ERROR)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path):
    data = read_json(path)
    kernels = data.get("kernels")
    if not isinstance(kernels, list):
        raise LabError("Manifest must contain a 'kernels' array", EXIT_MANIFEST_ERROR)

    out = {}
    for idx, entry in enumerate(kernels):
        if not isinstance(entry, dict):
            raise LabError(f"Manifest kernels[{idx}] must be an object", EXIT_MANIFEST_ERROR)

        missing = [k for k in REQUIRED_KERNEL_KEYS if k not in entry]
        if missing:
            raise LabError(f"Kernel entry {idx} missing required keys: {', '.join(missing)}", EXIT_MANIFEST_ERROR)

        name = str(entry.get("name", "")).strip()
        if not name:
            raise LabError(f"Kernel entry {idx} has empty name", EXIT_MANIFEST_ERROR)
        if name in out:
            raise LabError(f"Duplicate kernel name in manifest: {name}", EXIT_MANIFEST_ERROR)

        source = str(entry.get("source", "")).strip()
        if not source:
            raise LabError(f"Kernel '{name}' has empty source", EXIT_MANIFEST_ERROR)

        out[name] = {
            "name": name,
            "source": source,
            "description": str(entry.get("description", "")).strip(),
            "tags": entry.get("tags", []),
            "compile_cmd": str(entry.get("compile_cmd", "")),
            "run_cmd": str(entry.get("run_cmd", "")),
        }

    return out


def kernel_context(kernel: dict, build_dir: Path, nvcc: str):
    source = ROOT / kernel["source"]
    return {
        "name": kernel["name"],
        "source": str(source),
        "source_rel": kernel["source"],
        "root": str(ROOT),
        "build_dir": str(build_dir),
        "nvcc": nvcc,
    }


def task_slug(value: str):
    slug = "".join(c.lower() if c.isalnum() else "-" for c in value.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "task"


def kernel_profile(kernel: dict):
    return {
        "name": kernel["name"],
        "source": kernel["source"],
        "description": kernel.get("description", ""),
        "tags": list(kernel.get("tags", [])),
        "compile_cmd": kernel.get("compile_cmd", ""),
        "run_cmd": kernel.get("run_cmd", ""),
    }


def probe_nvcc_toolchain(nvcc: str):
    toolchain = {
        "nvcc": nvcc,
        "found": False,
        "path": None,
        "version": None,
    }
    resolved = shutil.which(nvcc)
    if not resolved:
        return toolchain

    toolchain["found"] = True
    toolchain["path"] = resolved
    proc = subprocess.run(
        [resolved, "--version"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    if output:
        toolchain["version"] = output.splitlines()[-1].strip()
    return toolchain


def parse_mib_string(value: str):
    digits = "".join(c for c in str(value) if c.isdigit())
    return int(digits) if digits else 0


def capacity_tier_from_memory_mib(memory_mib: int):
    if memory_mib >= 32768:
        return "large"
    if memory_mib >= 16384:
        return "medium"
    if memory_mib > 0:
        return "small"
    return "unknown"


def arch_from_hardware(compute_capability: str, gpu_name: str):
    cap = str(compute_capability or "").strip()
    if cap:
        digits = cap.replace(".", "")
        if digits.isdigit() and len(digits) >= 2:
            return f"sm{digits[:2]}"
    lowered = str(gpu_name or "").lower()
    if "ada" in lowered:
        return "sm89"
    return "generic"


def collect_hardware_profile(nvcc: str):
    profile = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend": "cpu",
        "nvcc": probe_nvcc_toolchain(nvcc),
        "gpu": None,
        "capacity": {
            "memory_total_mib": 0,
            "tier": "unknown",
        },
        "arch": "generic",
    }

    if not shutil.which("nvidia-smi"):
        return profile

    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    if proc.returncode != 0 or not proc.stdout.strip():
        return profile

    first = proc.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    gpu_name = parts[0] if len(parts) >= 1 else ""
    memory_total = parts[1] if len(parts) >= 2 else ""
    driver_version = parts[2] if len(parts) >= 3 else ""
    compute_capability = parts[3] if len(parts) >= 4 else ""
    memory_mib = parse_mib_string(memory_total)
    profile["backend"] = "cuda"
    profile["gpu"] = {
        "name": gpu_name,
        "memory_total": memory_total,
        "memory_total_mib": memory_mib,
        "driver_version": driver_version,
        "compute_capability": compute_capability,
    }
    profile["capacity"] = {
        "memory_total_mib": memory_mib,
        "tier": capacity_tier_from_memory_mib(memory_mib),
    }
    profile["arch"] = arch_from_hardware(compute_capability, gpu_name)
    return profile


def load_baseline_manifest(path: Path):
    data = read_json(path)
    baselines = data.get("baselines")
    if not isinstance(baselines, list):
        raise LabError("Baseline manifest must contain a 'baselines' array", EXIT_MANIFEST_ERROR)
    return data


def default_baseline_manifest():
    return {
        "version": 1,
        "baselines": [
            {
                "task_kind": "gemm",
                "backend": "cuda",
                "arch": "sm89",
                "kernel": "ada_gemm_k64_warp32_async",
                "source": "seed",
                "updated_utc": None,
                "notes": "Current Ada FP32 GEMM winner for SM89 GPUs.",
            },
            {
                "task_kind": "gemm",
                "backend": "cuda",
                "arch": "generic",
                "kernel": "matrix_mult",
                "source": "seed",
                "updated_utc": None,
                "notes": "Generic CUDA GEMM baseline when no architecture-specific winner exists.",
            },
        ],
    }


def ensure_baseline_manifest(path: Path):
    if not path.exists():
        write_json(path, default_baseline_manifest())
    return load_baseline_manifest(path)


def resolve_task_baseline(kernels: dict, baseline_manifest: dict, task_kind: str, backend: str, arch: str, override: str = ""):
    if override:
        kernel = kernels.get(override)
        if not kernel:
            raise LabError(f"Baseline kernel not found: {override}", EXIT_USER_ERROR)
        return kernel, {"resolution": "explicit"}

    baselines = baseline_manifest.get("baselines", [])
    ranked = []
    for entry in baselines:
        if str(entry.get("task_kind", "")).strip() != task_kind:
            continue
        if str(entry.get("backend", "")).strip() != backend:
            continue
        entry_arch = str(entry.get("arch", "generic")).strip() or "generic"
        score = 0
        if entry_arch == arch:
            score = 3
        elif entry_arch == "generic":
            score = 2
        else:
            continue
        ranked.append((score, entry))

    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[0][1]
        kernel_name = str(selected.get("kernel", "")).strip()
        kernel = kernels.get(kernel_name)
        if not kernel:
            raise LabError(f"Baseline manifest references missing kernel: {kernel_name}", EXIT_MANIFEST_ERROR)
        return kernel, {"resolution": "manifest", "entry": selected}

    fallback = kernels.get("matrix_mult")
    if fallback:
        return fallback, {"resolution": "fallback", "entry": None}
    raise LabError("No baseline kernel could be resolved for task", EXIT_USER_ERROR)


def benchmark_plan_for_task(task_name: str, task_kind: str, baseline_kernel: dict, hardware: dict, args):
    baseline_phase = "run" if str(baseline_kernel.get("run_cmd", "")).strip() else "compile"
    progress_flag = " --progress" if getattr(args, "progress", False) else ""
    feature_profiles = resolve_feature_profiles(getattr(args, "pyc_feature_profile", []))
    candidate_filters = (
        "".join(f" --tag {shlex.quote(tag)}" for tag in (args.candidate_tag or []))
        + "".join(f" --name {shlex.quote(name)}" for name in (args.candidate_name or []))
    )
    plan = {
        "task_kind": task_kind,
        "profile_protocol": [],
        "metrics": ["mean_ms", "p50_ms", "p95_ms", "stdev_ms"],
        "promotion_rule": "winner becomes next baseline for matching task_kind/backend/arch",
        "hardware_constraints": {
            "backend": hardware.get("backend"),
            "arch": hardware.get("arch"),
            "capacity_tier": hardware.get("capacity", {}).get("tier"),
        },
        "feature_profiles": feature_profiles,
    }

    if baseline_phase == "run":
        plan["profile_protocol"].append(
            f"python3 kernels/lab/kernel_lab.py --nvcc {shlex.quote(args.nvcc)} bench {baseline_kernel['name']} --phase compile --repeats 1 --warmup 0{progress_flag}"
        )
    plan["profile_protocol"].append(
        f"python3 kernels/lab/kernel_lab.py --nvcc {shlex.quote(args.nvcc)} bench {baseline_kernel['name']} --phase {baseline_phase} --repeats {args.repeats} --warmup {args.warmup}{progress_flag}"
    )

    if candidate_filters:
        plan["profile_protocol"].append(
            f"python3 kernels/lab/kernel_lab.py --nvcc {shlex.quote(args.nvcc)} bench-suite --phase compile --repeats 1 --warmup 0 --label {task_slug(task_name)}-candidates-compile{progress_flag}{candidate_filters}"
        )
        plan["profile_protocol"].append(
            f"python3 kernels/lab/kernel_lab.py --nvcc {shlex.quote(args.nvcc)} bench-suite --phase run --repeats {args.repeats} --warmup {args.warmup} --label {task_slug(task_name)}-candidates{progress_flag}{candidate_filters}"
        )

    if task_kind == "gemm" and args.matrix_file:
        if feature_profiles:
            for profile in feature_profiles:
                env_prefix = env_prefix_from_mapping(profile["env"])
                plan["profile_protocol"].append(
                    env_prefix
                    + f"python3 benchmark/benchmarks/gpu/run_gemm_suite.py --matrix-file {shlex.quote(args.matrix_file)} --device {hardware.get('backend', 'cuda')} --dtype float32 --tag {task_slug(task_name)}-{task_slug(profile['name'])}{progress_flag}"
                )
        else:
            plan["profile_protocol"].append(
                f"python3 benchmark/benchmarks/gpu/run_gemm_suite.py --matrix-file {shlex.quote(args.matrix_file)} --device {hardware.get('backend', 'cuda')} --dtype float32 --tag {task_slug(task_name)}{progress_flag}"
            )
        plan["matrix_file"] = args.matrix_file
    return plan


def task_record(task_name: str, task_kind: str, objective: str, hardware: dict, baseline_kernel: dict, baseline_meta: dict, args):
    feature_profiles = resolve_feature_profiles(getattr(args, "pyc_feature_profile", []))
    return {
        "meta": {
            "name": task_name,
            "slug": task_slug(task_name),
            "status": "planned",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "hardware": hardware,
        "task": {
            "kind": task_kind,
            "objective": objective,
            "baseline_kernel": baseline_kernel["name"],
            "candidate_tags": list(args.candidate_tag or []),
            "candidate_names": list(args.candidate_name or []),
            "pyc_feature_profiles": [profile["name"] for profile in feature_profiles],
            "baseline_resolution": baseline_meta,
        },
        "baseline": kernel_profile(baseline_kernel),
        "benchmark_plan": benchmark_plan_for_task(task_name, task_kind, baseline_kernel, hardware, args),
        "result": {
            "winner_kernel": None,
            "result_json": None,
            "promoted": False,
            "notes": "",
            "completed_utc": None,
        },
    }


def task_path(task_dir: Path, name: str):
    return task_dir / f"{task_slug(name)}.json"


def load_task(path: Path):
    return read_json(path)


def promote_task_baseline(baseline_manifest: dict, task_kind: str, backend: str, arch: str, kernel_name: str, notes: str, source: str):
    baselines = baseline_manifest.setdefault("baselines", [])
    for entry in baselines:
        if (
            str(entry.get("task_kind", "")).strip() == task_kind
            and str(entry.get("backend", "")).strip() == backend
            and str(entry.get("arch", "")).strip() == arch
        ):
            entry["kernel"] = kernel_name
            entry["notes"] = notes
            entry["source"] = source
            entry["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            return entry

    entry = {
        "task_kind": task_kind,
        "backend": backend,
        "arch": arch,
        "kernel": kernel_name,
        "source": source,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": notes,
    }
    baselines.append(entry)
    return entry


def kernel_matches_filters(kernel: dict, include_tags, exclude_tags, include_names, exclude_names):
    tags = set(str(tag).strip() for tag in kernel.get("tags", []) if str(tag).strip())
    name = kernel["name"]

    if include_names and name not in include_names:
        return False
    if exclude_names and name in exclude_names:
        return False
    if include_tags and not any(tag in tags for tag in include_tags):
        return False
    if exclude_tags and any(tag in tags for tag in exclude_tags):
        return False
    return True


def collect_filtered_kernels(kernels: dict, include_tags, exclude_tags, include_names, exclude_names):
    selected = []
    for _, kernel in kernels.items():
        if kernel_matches_filters(kernel, include_tags, exclude_tags, include_names, exclude_names):
            selected.append(kernel)
    return selected


def expand_template(template: str, context: dict):
    try:
        return template.format(**context)
    except KeyError as exc:
        raise LabError(f"Template variable missing: {exc}", EXIT_MANIFEST_ERROR)


def _resolve_executable(token0: str):
    if token0.startswith("/") or token0.startswith("./") or token0.startswith("../"):
        return Path(token0).exists()
    return shutil.which(token0) is not None


def check_command_preflight(cmd: str):
    _, parts = parse_command_with_env(cmd)
    token0 = parts[0]
    if not _resolve_executable(token0):
        raise LabError(f"Required executable not found: {token0}", EXIT_TOOLCHAIN_MISSING)


def run_command(cmd: str, cwd: Path):
    check_command_preflight(cmd)
    env_overrides, argv = parse_command_with_env(cmd)
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


def timed_run(cmd: str, cwd: Path):
    start = time.perf_counter()
    proc = run_command(cmd, cwd)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, proc


def task_run_path(task_run_dir: Path, task_name: str):
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return task_run_dir / f"{task_slug(task_name)}-{ts}.json"


def task_progress_path(task_run_dir: Path, task_name: str):
    return Path(task_run_dir) / f"{task_slug(task_name)}.progress.json"


def write_task_progress(task_run_dir: Path, task_name: str, run_doc: dict, current_step: dict | None = None):
    payload = {
        "source": "kernel_lab_task_run",
        "meta": {
            "task_name": run_doc.get("meta", {}).get("task_name", task_name),
            "task_slug": run_doc.get("meta", {}).get("task_slug", task_slug(task_name)),
            "task_kind": run_doc.get("meta", {}).get("task_kind", "unknown"),
            "status": run_doc.get("meta", {}).get("status", "running"),
            "started_utc": run_doc.get("meta", {}).get("started_utc"),
            "completed_utc": run_doc.get("meta", {}).get("completed_utc"),
            "command_count": run_doc.get("meta", {}).get("command_count", 0),
            "host": run_doc.get("meta", {}).get("host"),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "progress": {
            "completed_runs": sum(1 for step in run_doc.get("steps", []) if step.get("status") == "ok"),
            "total_runs": run_doc.get("meta", {}).get("command_count", 0),
            "current_shape_name": current_step.get("label", "-") if current_step else "-",
            "current_adapter": current_step.get("status", "idle") if current_step else "idle",
            "current_repeat": current_step.get("index", 0) if current_step else 0,
            "current_repeat_total": run_doc.get("meta", {}).get("command_count", 0),
        },
        "recent_events": [
            {
                "shape": step.get("label", f"step-{step.get('index', 0)}"),
                "adapter": step.get("status", "unknown"),
                "status": step.get("status", "unknown"),
                "throughput_tflops": float(step.get("observed_metrics", {}).get("gflops", 0.0) or 0.0) / 1000.0,
                "mean_ms": float(step.get("elapsed_ms", 0.0) or 0.0),
            }
            for step in run_doc.get("steps", [])[-8:]
        ],
    }
    specific_path = task_progress_path(task_run_dir, task_name)
    write_progress_state(specific_path, payload)
    write_progress_state(DEFAULT_TASK_PROGRESS_FILE, payload)


def execute_task_command(command: str, stream_live: bool):
    check_command_preflight(command)
    env_overrides, argv = parse_command_with_env(command)
    env = os.environ.copy()
    env.update(env_overrides)
    start = time.perf_counter()
    if stream_live:
        proc = subprocess.run(
            argv,
            cwd=str(ROOT),
            text=True,
            env=env,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms, CmdResult(proc.returncode, "", "")

    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, CmdResult(proc.returncode, proc.stdout, proc.stderr)


def write_result(results_dir: Path, payload: dict, label: str):
    ensure_results_dir(results_dir)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    path = results_dir / f"{safe_label}-{ts}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "artifact_kind": "kernel_lab_suite" if isinstance(payload.get("kernels"), list) else "kernel_lab_result",
        "artifact_id": path.stem,
        "label": label,
        "source_json": path.name,
        "timestamp_utc": payload.get("meta", {}).get("timestamp_utc"),
        "phase": payload.get("meta", {}).get("phase"),
        "repeats": payload.get("meta", {}).get("repeats"),
        "warmup": payload.get("meta", {}).get("warmup"),
        "kernel_count": len(payload.get("kernels", [])) if isinstance(payload.get("kernels"), list) else 0,
        "selected_kernels": payload.get("meta", {}).get("selected_kernels", []),
        "filters": payload.get("meta", {}).get("filters", {}),
        "toolchain": payload.get("toolchain", {}),
    }
    metadata_path = results_dir / f"{safe_label}-{ts}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def get_kernel_or_fail(args, name):
    kernels = load_manifest(Path(args.manifest))
    kernel = kernels.get(name)
    if not kernel:
        raise LabError(f"Kernel not found: {name}", EXIT_USER_ERROR)
    return kernel, kernels


def require_positive_int(name: str, value: int):
    if value <= 0:
        raise LabError(f"{name} must be > 0", EXIT_USER_ERROR)


def maybe_bench_phase(template: str, context: dict, repeats: int, warmup: int, progress: bool, phase_label: str):
    cmd = expand_template(template, context)

    for _ in iter_with_progress(range(warmup), progress, f"{context['name']}:{phase_label}:warmup", "run"):
        proc = run_command(cmd, ROOT)
        if proc.returncode != 0:
            return cmd, None, None, proc

    samples = []
    last_proc = None
    for _ in iter_with_progress(range(repeats), progress, f"{context['name']}:{phase_label}:bench", "run"):
        elapsed_ms, proc = timed_run(cmd, ROOT)
        if proc.returncode != 0:
            return cmd, None, None, proc
        samples.append(elapsed_ms)
        last_proc = proc

    details = None
    if last_proc is not None:
        details = {
            "stdout_tail": trim_output(last_proc.stdout),
            "stderr_tail": trim_output(last_proc.stderr),
            "observed_metrics": extract_key_value_metrics(last_proc.stdout),
        }

    return cmd, summarize(samples), details, None


def run_kernel_bench(kernel: dict, args):
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    context = kernel_context(kernel, build_dir, args.nvcc)

    result = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "kernel": kernel["name"],
            "manifest": str(Path(args.manifest)),
            "build_dir": str(Path(args.build_dir)),
            "results_dir": str(Path(args.results_dir)),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "phase": args.phase,
            "nvcc": args.nvcc,
        },
        "kernel": kernel_profile(kernel),
        "toolchain": probe_nvcc_toolchain(args.nvcc),
        "compile": None,
        "run": None,
    }

    if args.phase in ("compile", "both"):
        compile_t = kernel.get("compile_cmd", "").strip()
        if not compile_t:
            raise LabError(f"Kernel {kernel['name']} missing compile_cmd for phase {args.phase}.", EXIT_USER_ERROR)
        cmd, stats, details, err_proc = maybe_bench_phase(
            compile_t,
            context,
            args.repeats,
            args.warmup,
            args.progress,
            "compile",
        )
        if err_proc:
            return result, cmd, err_proc
        result["compile"] = {"cmd": cmd, "stats": stats}
        if details:
            result["compile"].update(details)

    if args.phase in ("run", "both"):
        run_t = kernel.get("run_cmd", "").strip()
        if not run_t:
            raise LabError(f"Kernel {kernel['name']} missing run_cmd for phase {args.phase}.", EXIT_USER_ERROR)
        cmd, stats, details, err_proc = maybe_bench_phase(
            run_t,
            context,
            args.repeats,
            args.warmup,
            args.progress,
            "run",
        )
        if err_proc:
            return result, cmd, err_proc
        result["run"] = {"cmd": cmd, "stats": stats}
        if details:
            result["run"].update(details)

    return result, None, None


def cmd_doctor(args):
    kernels = load_manifest(Path(args.manifest))
    print(f"manifest: {args.manifest}")
    print(f"kernels: {len(kernels)}")

    nvcc_found = shutil.which(args.nvcc) is not None
    print(f"nvcc ({args.nvcc}): {'found' if nvcc_found else 'missing'}")

    bad_sources = 0
    for name, k in sorted(kernels.items()):
        src = ROOT / k["source"]
        if not src.exists():
            bad_sources += 1
            print(f"[source-missing] {name}: {src}")

    if bad_sources:
        raise LabError(f"doctor failed: {bad_sources} kernel source file(s) missing", EXIT_MANIFEST_ERROR)

    if not nvcc_found:
        print("note: CUDA compile/run commands requiring nvcc will fail until CUDA toolkit is installed.")

    print("doctor: ok")
    return EXIT_OK


def cmd_list(args):
    kernels = load_manifest(Path(args.manifest))
    if not kernels:
        print("No kernels found in manifest.")
        return EXIT_USER_ERROR
    for name, k in sorted(kernels.items()):
        tags = ",".join(k.get("tags", []))
        print(f"{name:20} source={k.get('source','')} tags=[{tags}]")
    return EXIT_OK


def cmd_show(args):
    kernel, _ = get_kernel_or_fail(args, args.name)
    print(json.dumps(kernel, indent=2))
    return EXIT_OK


def cmd_compile(args):
    kernel, _ = get_kernel_or_fail(args, args.name)
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    template = kernel.get("compile_cmd", "").strip()
    if not template:
        raise LabError(f"Kernel {args.name} has no compile_cmd.", EXIT_USER_ERROR)

    context = kernel_context(kernel, build_dir, args.nvcc)
    cmd = expand_template(template, context)
    elapsed_ms, proc = timed_run(cmd, ROOT)

    print(f"compile_cmd: {cmd}")
    print(f"elapsed_ms: {elapsed_ms:.3f}")
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip())
        return EXIT_COMMAND_FAILED

    print("compile: ok")
    return EXIT_OK


def cmd_run(args):
    kernel, _ = get_kernel_or_fail(args, args.name)
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    template = kernel.get("run_cmd", "").strip()
    if not template:
        raise LabError(f"Kernel {args.name} has no run_cmd configured.", EXIT_USER_ERROR)

    context = kernel_context(kernel, build_dir, args.nvcc)
    cmd = expand_template(template, context)
    elapsed_ms, proc = timed_run(cmd, ROOT)

    print(f"run_cmd: {cmd}")
    print(f"elapsed_ms: {elapsed_ms:.3f}")
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip())
        return EXIT_COMMAND_FAILED

    print("run: ok")
    return EXIT_OK


def cmd_bench(args):
    require_positive_int("repeats", args.repeats)
    require_progress_support(args.progress)
    if args.warmup < 0:
        raise LabError("warmup must be >= 0", EXIT_USER_ERROR)

    kernel, _ = get_kernel_or_fail(args, args.name)
    result, failed_cmd, err_proc = run_kernel_bench(kernel, args)
    if err_proc:
        phase = "compile" if args.phase in ("compile", "both") and result.get("compile") is None else "run"
        print(f"{phase} phase failed: {failed_cmd}")
        if err_proc.stderr.strip():
            print(err_proc.stderr.strip())
        return EXIT_COMMAND_FAILED
    out_path = write_result(Path(args.results_dir), result, args.name)
    print(f"wrote {out_path}")
    return EXIT_OK


def cmd_bench_cmd(args):
    require_positive_int("repeats", args.repeats)
    require_progress_support(args.progress)
    if args.warmup < 0:
        raise LabError("warmup must be >= 0", EXIT_USER_ERROR)

    check_command_preflight(args.cmd)

    for _ in iter_with_progress(range(args.warmup), args.progress, f"{args.label}:warmup", "run"):
        proc = run_command(args.cmd, ROOT)
        if proc.returncode != 0:
            if proc.stderr.strip():
                print(proc.stderr.strip())
            return EXIT_COMMAND_FAILED

    samples = []
    last_proc = None
    for _ in iter_with_progress(range(args.repeats), args.progress, f"{args.label}:bench", "run"):
        elapsed_ms, proc = timed_run(args.cmd, ROOT)
        if proc.returncode != 0:
            if proc.stderr.strip():
                print(proc.stderr.strip())
            return EXIT_COMMAND_FAILED
        samples.append(elapsed_ms)
        last_proc = proc

    result = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "label": args.label,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "results_dir": str(Path(args.results_dir)),
        },
        "command": args.cmd,
        "stats": summarize(samples),
    }
    if last_proc is not None:
        result["stdout_tail"] = trim_output(last_proc.stdout)
        result["stderr_tail"] = trim_output(last_proc.stderr)
        result["observed_metrics"] = extract_key_value_metrics(last_proc.stdout)
    out_path = write_result(Path(args.results_dir), result, args.label)
    print(f"wrote {out_path}")
    return EXIT_OK


def cmd_bench_suite(args):
    require_positive_int("repeats", args.repeats)
    require_progress_support(args.progress)
    if args.warmup < 0:
        raise LabError("warmup must be >= 0", EXIT_USER_ERROR)

    kernels = load_manifest(Path(args.manifest))
    selected = collect_filtered_kernels(
        kernels,
        set(args.tag or []),
        set(args.exclude_tag or []),
        set(args.name or []),
        set(args.exclude_name or []),
    )
    if not selected:
        raise LabError("No kernels matched the requested filters", EXIT_USER_ERROR)

    suite = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest": str(Path(args.manifest)),
            "build_dir": str(Path(args.build_dir)),
            "results_dir": str(Path(args.results_dir)),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "phase": args.phase,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "dry_run": bool(args.dry_run),
            "nvcc": args.nvcc,
            "filters": {
                "tag": list(args.tag or []),
                "exclude_tag": list(args.exclude_tag or []),
                "name": list(args.name or []),
                "exclude_name": list(args.exclude_name or []),
            },
            "selected_kernels": [k["name"] for k in selected],
            "count": len(selected),
        },
        "toolchain": probe_nvcc_toolchain(args.nvcc),
        "kernels": [],
    }

    if args.dry_run:
        for kernel in selected:
            kernel_meta = kernel_profile(kernel)
            kernel_meta["planned"] = True
            suite["kernels"].append(
                {
                    "kernel": kernel_meta,
                    "planned": True,
                    "context": kernel_context(kernel, Path(args.build_dir), args.nvcc),
                }
            )
        out_path = write_result(Path(args.results_dir), suite, f"{args.label}-plan")
        print(f"wrote {out_path}")
        return EXIT_OK

    for kernel in iter_with_progress(selected, args.progress, "kernel-suite", "kernel"):
        result, failed_cmd, err_proc = run_kernel_bench(kernel, args)
        if err_proc:
            phase = "compile" if args.phase in ("compile", "both") and result.get("compile") is None else "run"
            print(f"{kernel['name']} {phase} phase failed: {failed_cmd}")
            if err_proc.stderr.strip():
                print(err_proc.stderr.strip())
            return EXIT_COMMAND_FAILED
        suite["kernels"].append(
            {
                "kernel": kernel_profile(kernel),
                "result": result,
            }
        )

    suite["summary"] = suite.get("summary") or {}
    suite["summary"] = {
        "kernels": len(suite["kernels"]),
        "planned_kernels": sum(1 for item in suite["kernels"] if item.get("planned")),
        "executed_kernels": sum(1 for item in suite["kernels"] if item.get("result")),
        "compile_mean_ms": round(
            statistics.mean(
                item["result"]["compile"]["stats"]["mean_ms"]
                for item in suite["kernels"]
                if item["result"].get("compile")
            ),
            3,
        )
        if any(item["result"].get("compile") for item in suite["kernels"])
        else None,
        "run_mean_ms": round(
            statistics.mean(
                item["result"]["run"]["stats"]["mean_ms"]
                for item in suite["kernels"]
                if item["result"].get("run")
            ),
            3,
        )
        if any(item["result"].get("run") for item in suite["kernels"])
        else None,
    }

    out_path = write_result(Path(args.results_dir), suite, args.label)
    print(f"wrote {out_path}")
    return EXIT_OK


def _extract_mean(doc, phase):
    section = doc.get(phase)
    if not isinstance(section, dict):
        return None
    stats = section.get("stats")
    if not isinstance(stats, dict):
        return None
    val = stats.get("mean_ms")
    return float(val) if isinstance(val, (int, float)) else None


def cmd_compare(args):
    a = read_json(Path(args.a))
    b = read_json(Path(args.b))

    printed = False
    for phase in ("compile", "run"):
        ma = _extract_mean(a, phase)
        mb = _extract_mean(b, phase)
        if ma is None or mb is None:
            continue
        delta = mb - ma
        pct = (delta / ma * 100.0) if ma else 0.0
        print(f"{phase:8} a={ma:.3f} ms b={mb:.3f} ms delta={delta:.3f} ms ({pct:+.2f}%)")
        printed = True

    if isinstance(a.get("stats"), dict) and isinstance(b.get("stats"), dict):
        ma = a["stats"].get("mean_ms")
        mb = b["stats"].get("mean_ms")
        if isinstance(ma, (int, float)) and isinstance(mb, (int, float)):
            delta = mb - ma
            pct = (delta / ma * 100.0) if ma else 0.0
            print(f"command  a={ma:.3f} ms b={mb:.3f} ms delta={delta:.3f} ms ({pct:+.2f}%)")
            printed = True

    if not printed:
        raise LabError("No comparable mean_ms metrics found in provided files", EXIT_USER_ERROR)

    return EXIT_OK


def cmd_task_hardware(args):
    hardware = collect_hardware_profile(args.nvcc)
    print(json.dumps(hardware, indent=2))
    return EXIT_OK


def cmd_task_create(args):
    kernels = load_manifest(Path(args.manifest))
    baseline_manifest = ensure_baseline_manifest(Path(args.baseline_manifest))
    hardware = collect_hardware_profile(args.nvcc)
    objective = args.objective.strip() or "beat the current baseline on target hardware"
    baseline_kernel, baseline_meta = resolve_task_baseline(
        kernels,
        baseline_manifest,
        args.task_kind,
        hardware.get("backend", "cpu"),
        hardware.get("arch", "generic"),
        args.baseline,
    )
    record = task_record(args.name, args.task_kind, objective, hardware, baseline_kernel, baseline_meta, args)
    out_path = task_path(Path(args.task_dir), args.name)
    write_json(out_path, record)
    print(f"wrote {out_path}")
    return EXIT_OK


def cmd_task_list(args):
    task_dir = Path(args.task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(task_dir.glob("*.json"))
    if not tasks:
        print("No task records found.")
        return EXIT_OK
    for path in tasks:
        doc = load_task(path)
        meta = doc.get("meta", {})
        task = doc.get("task", {})
        print(
            f"{meta.get('name', path.stem):24} status={meta.get('status', 'unknown'):10} "
            f"kind={task.get('kind', 'unknown'):8} baseline={task.get('baseline_kernel', 'n/a')}"
        )
    return EXIT_OK


def cmd_task_show(args):
    path = task_path(Path(args.task_dir), args.name)
    print(json.dumps(load_task(path), indent=2))
    return EXIT_OK


def cmd_task_run(args):
    require_progress_support(args.progress)
    path = task_path(Path(args.task_dir), args.name)
    doc = load_task(path)
    meta = doc.setdefault("meta", {})
    task = doc.get("task", {})
    hardware = doc.get("hardware", {})
    plan = doc.get("benchmark_plan", {})
    commands = list(plan.get("profile_protocol", []))
    if not commands:
        raise LabError(f"Task {args.name} has no profile_protocol commands", EXIT_USER_ERROR)

    run_doc = {
        "meta": {
            "task_name": meta.get("name", args.name),
            "task_slug": meta.get("slug", task_slug(args.name)),
            "task_kind": task.get("kind", "unknown"),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "running",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "command_count": len(commands),
            "stop_on_error": not args.keep_going,
            "dry_run": bool(args.dry_run),
        },
        "hardware": hardware,
        "steps": [],
    }
    task_run_dir = Path(args.task_run_dir)

    if args.dry_run:
        for index, command in enumerate(commands, start=1):
            run_doc["steps"].append(
                {
                    "index": index,
                    "status": "planned",
                    "command": command,
                }
            )
        run_doc["meta"]["status"] = "planned"
        run_doc["meta"]["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        out_path = task_run_path(task_run_dir, args.name)
        write_json(out_path, run_doc)
        write_task_progress(task_run_dir, args.name, run_doc)
        print(f"wrote {out_path}")
        return EXIT_OK

    overall_ok = True
    live_child_output = bool(args.progress and sys.stdout.isatty())
    if live_child_output:
        print_status("[task-run] live child output enabled")
    task_progress_enabled = bool(args.progress and not live_child_output)
    for index, command in enumerate(iter_with_progress(commands, task_progress_enabled, f"task:{args.name}", "step"), start=1):
        print_status(f"[task-run] step {index}/{len(commands)}: {command}")
        current_step = {
            "index": index,
            "label": f"step-{index}",
            "status": "running",
        }
        run_doc["meta"]["status"] = "running"
        write_task_progress(task_run_dir, args.name, run_doc, current_step)
        elapsed_ms, proc = execute_task_command(command, stream_live=live_child_output)
        status = "ok" if proc.returncode == 0 else "failed"
        step = {
            "index": index,
            "label": f"step-{index}",
            "status": status,
            "command": command,
            "returncode": proc.returncode,
            "elapsed_ms": round(elapsed_ms, 3),
            "stdout_tail": trim_output(proc.stdout),
            "stderr_tail": trim_output(proc.stderr),
            "observed_metrics": extract_key_value_metrics(proc.stdout),
            "written_paths": extract_written_paths(proc.stdout),
        }
        run_doc["steps"].append(step)
        write_task_progress(task_run_dir, args.name, run_doc, step)
        if proc.stdout.strip():
            print_status(trim_output(proc.stdout, max_lines=12, max_chars=2000))
        print_status(f"[task-run] step {index} status={status} elapsed_ms={step['elapsed_ms']}")
        if proc.returncode != 0:
            overall_ok = False
            if proc.stderr.strip():
                print_status(proc.stderr.strip())
            if not args.keep_going:
                break

    run_doc["meta"]["status"] = "ok" if overall_ok else "failed"
    run_doc["meta"]["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out_path = task_run_path(task_run_dir, args.name)
    write_json(out_path, run_doc)
    write_task_progress(task_run_dir, args.name, run_doc)

    meta["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    doc["last_run"] = {
        "status": run_doc["meta"]["status"],
        "run_record": str(out_path),
        "completed_utc": run_doc["meta"]["completed_utc"],
    }
    write_json(path, doc)
    print(f"wrote {out_path}")
    print(f"updated {path}")
    return EXIT_OK if overall_ok else EXIT_COMMAND_FAILED


def cmd_task_complete(args):
    kernels = load_manifest(Path(args.manifest))
    winner = kernels.get(args.winner)
    if not winner:
        raise LabError(f"Winner kernel not found: {args.winner}", EXIT_USER_ERROR)

    path = task_path(Path(args.task_dir), args.name)
    doc = load_task(path)
    meta = doc.setdefault("meta", {})
    task = doc.setdefault("task", {})
    result = doc.setdefault("result", {})
    hardware = doc.get("hardware", {})

    result["winner_kernel"] = args.winner
    result["result_json"] = args.result_json
    result["notes"] = args.notes
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["promoted"] = True
    meta["status"] = "completed"
    meta["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    doc["winner"] = kernel_profile(winner)

    baseline_manifest_path = Path(args.baseline_manifest)
    baseline_manifest = ensure_baseline_manifest(baseline_manifest_path)
    promoted_entry = promote_task_baseline(
        baseline_manifest,
        task.get("kind", "gemm"),
        hardware.get("backend", "cpu"),
        hardware.get("arch", "generic"),
        args.winner,
        args.notes or f"Promoted from task {meta.get('name', path.stem)}",
        source=f"task:{meta.get('name', path.stem)}",
    )
    doc["promoted_baseline"] = promoted_entry

    write_json(path, doc)
    write_json(baseline_manifest_path, baseline_manifest)
    print(f"updated {path}")
    print(f"updated {baseline_manifest_path}")
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(description="Kernel Lab CLI for kernel prototyping/testing/benchmarking")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to kernels manifest JSON")
    parser.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST), help="Path to task baseline manifest JSON")
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR), help="Path to persistent task record directory")
    parser.add_argument("--task-run-dir", default=str(DEFAULT_TASK_RUN_DIR), help="Path to task execution record directory")
    parser.add_argument("--build-dir", default=str(ROOT / "kernels" / "lab" / "build"), help="Build output dir")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Benchmark results output dir")
    parser.add_argument("--nvcc", default="nvcc", help="Path to nvcc executable")

    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Validate manifest, sources, and toolchain availability")
    p_doctor.set_defaults(func=cmd_doctor)

    p_list = sub.add_parser("list", help="List kernels from manifest")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show kernel entry")
    p_show.add_argument("name")
    p_show.set_defaults(func=cmd_show)

    p_compile = sub.add_parser("compile", help="Compile a kernel using manifest compile_cmd")
    p_compile.add_argument("name")
    p_compile.set_defaults(func=cmd_compile)

    p_run = sub.add_parser("run", help="Run a kernel using manifest run_cmd")
    p_run.add_argument("name")
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser("bench", help="Benchmark compile/run phases for a kernel")
    p_bench.add_argument("name")
    p_bench.add_argument("--phase", choices=["compile", "run", "both"], default="compile")
    p_bench.add_argument("--repeats", type=int, default=10)
    p_bench.add_argument("--warmup", type=int, default=2)
    p_bench.add_argument("--progress", action="store_true", help="Show tqdm progress bars during warmup/repeat loops")
    p_bench.set_defaults(func=cmd_bench)

    p_bench_cmd = sub.add_parser("bench-cmd", help="Benchmark any prototype shell command")
    p_bench_cmd.add_argument("label", help="Label used in output filename")
    p_bench_cmd.add_argument("cmd", help="Command to execute and benchmark")
    p_bench_cmd.add_argument("--repeats", type=int, default=10)
    p_bench_cmd.add_argument("--warmup", type=int, default=2)
    p_bench_cmd.add_argument("--progress", action="store_true", help="Show tqdm progress bars during warmup/repeat loops")
    p_bench_cmd.set_defaults(func=cmd_bench_cmd)

    p_bench_suite = sub.add_parser("bench-suite", help="Benchmark all manifest kernels matching filters")
    p_bench_suite.add_argument("--phase", choices=["compile", "run", "both"], default="both")
    p_bench_suite.add_argument("--repeats", type=int, default=10)
    p_bench_suite.add_argument("--warmup", type=int, default=2)
    p_bench_suite.add_argument("--tag", action="append", default=[], help="Include kernels with any of these tags")
    p_bench_suite.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="Exclude kernels with any of these tags",
    )
    p_bench_suite.add_argument("--name", action="append", default=[], help="Include only these kernel names")
    p_bench_suite.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Exclude these kernel names",
    )
    p_bench_suite.add_argument("--dry-run", action="store_true", help="List the selected kernels without executing them")
    p_bench_suite.add_argument("--label", default="kernel-suite", help="Label used in output filename")
    p_bench_suite.add_argument("--progress", action="store_true", help="Show tqdm progress bars during suite execution")
    p_bench_suite.set_defaults(func=cmd_bench_suite)

    p_compare = sub.add_parser("compare", help="Compare two benchmark result JSON files")
    p_compare.add_argument("a", help="Baseline result JSON")
    p_compare.add_argument("b", help="Candidate result JSON")
    p_compare.set_defaults(func=cmd_compare)

    p_task_hardware = sub.add_parser("task-hardware", help="Capture current hardware/capacity profile for task planning")
    p_task_hardware.set_defaults(func=cmd_task_hardware)

    p_task_create = sub.add_parser("task-create", help="Create a kernel optimization task with hardware-aware baseline selection")
    p_task_create.add_argument("name")
    p_task_create.add_argument("--task-kind", default="gemm")
    p_task_create.add_argument("--objective", default="")
    p_task_create.add_argument("--baseline", default="", help="Explicit baseline kernel override")
    p_task_create.add_argument("--candidate-tag", action="append", default=[], help="Candidate kernel tag filter")
    p_task_create.add_argument("--candidate-name", action="append", default=[], help="Candidate kernel name filter")
    p_task_create.add_argument(
        "--pyc-feature-profile",
        action="append",
        default=[],
        help="Named PyC runtime feature profile to benchmark during the task",
    )
    p_task_create.add_argument("--matrix-file", default=str(ROOT / "benchmark" / "benchmarks" / "gpu" / "configs" / "ada_fp32_gemm_shapes.json"))
    p_task_create.add_argument("--repeats", type=int, default=10)
    p_task_create.add_argument("--warmup", type=int, default=2)
    p_task_create.add_argument("--progress", action="store_true", help="Include progress-enabled commands in the saved task plan")
    p_task_create.set_defaults(func=cmd_task_create)

    p_task_list = sub.add_parser("task-list", help="List saved kernel optimization tasks")
    p_task_list.set_defaults(func=cmd_task_list)

    p_task_show = sub.add_parser("task-show", help="Show a saved kernel optimization task")
    p_task_show.add_argument("name")
    p_task_show.set_defaults(func=cmd_task_show)

    p_task_run = sub.add_parser("task-run", help="Execute the saved benchmark/profile commands for a task")
    p_task_run.add_argument("name")
    p_task_run.add_argument("--keep-going", action="store_true", help="Continue running later commands if one step fails")
    p_task_run.add_argument("--dry-run", action="store_true", help="Write a planned execution record without running commands")
    p_task_run.add_argument("--progress", action="store_true", help="Show tqdm progress across task steps")
    p_task_run.set_defaults(func=cmd_task_run)

    p_task_complete = sub.add_parser("task-complete", help="Close a task and promote its winner as the next baseline")
    p_task_complete.add_argument("name")
    p_task_complete.add_argument("--winner", required=True, help="Kernel that won this task")
    p_task_complete.add_argument("--result-json", default="", help="Result artifact proving the winner")
    p_task_complete.add_argument("--notes", default="", help="Promotion notes")
    p_task_complete.set_defaults(func=cmd_task_complete)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        return args.func(args)
    except LabError as exc:
        print(f"error: {exc}")
        return exc.code
    except Exception as exc:  # defensive boundary for deterministic failure mode
        print(f"error: unexpected failure: {exc}")
        return EXIT_COMMAND_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
