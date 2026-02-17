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
import platform
import shlex
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "Kernel" / "lab" / "kernels.json"
DEFAULT_RESULTS_DIR = ROOT / "Kernel" / "lab" / "results"

EXIT_OK = 0
EXIT_USER_ERROR = 2
EXIT_MANIFEST_ERROR = 3
EXIT_TOOLCHAIN_MISSING = 4
EXIT_COMMAND_FAILED = 5

REQUIRED_KERNEL_KEYS = {"name", "source", "compile_cmd", "run_cmd"}


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


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LabError(f"File not found: {path}", EXIT_USER_ERROR)
    except json.JSONDecodeError as exc:
        raise LabError(f"Invalid JSON in {path}: {exc}", EXIT_MANIFEST_ERROR)


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
        "root": str(ROOT),
        "build_dir": str(build_dir),
        "nvcc": nvcc,
    }


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
    try:
        parts = shlex.split(cmd)
    except ValueError as exc:
        raise LabError(f"Invalid command quoting: {exc}", EXIT_MANIFEST_ERROR)

    if not parts:
        raise LabError("Empty command after template expansion", EXIT_MANIFEST_ERROR)

    token0 = parts[0]
    if not _resolve_executable(token0):
        raise LabError(f"Required executable not found: {token0}", EXIT_TOOLCHAIN_MISSING)


def run_command(cmd: str, cwd: Path):
    check_command_preflight(cmd)
    proc = subprocess.run(
        shlex.split(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


def timed_run(cmd: str, cwd: Path):
    start = time.perf_counter()
    proc = run_command(cmd, cwd)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, proc


def write_result(results_dir: Path, payload: dict, label: str):
    ensure_results_dir(results_dir)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    path = results_dir / f"{safe_label}-{ts}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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


def maybe_bench_phase(template: str, context: dict, repeats: int, warmup: int):
    cmd = expand_template(template, context)

    for _ in range(warmup):
        proc = run_command(cmd, ROOT)
        if proc.returncode != 0:
            return cmd, None, proc

    samples = []
    for _ in range(repeats):
        elapsed_ms, proc = timed_run(cmd, ROOT)
        if proc.returncode != 0:
            return cmd, None, proc
        samples.append(elapsed_ms)

    return cmd, summarize(samples), None


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
    if args.warmup < 0:
        raise LabError("warmup must be >= 0", EXIT_USER_ERROR)

    kernel, _ = get_kernel_or_fail(args, args.name)
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    context = kernel_context(kernel, build_dir, args.nvcc)
    result = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "kernel": args.name,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "phase": args.phase,
        },
        "compile": None,
        "run": None,
    }

    if args.phase in ("compile", "both"):
        compile_t = kernel.get("compile_cmd", "").strip()
        if not compile_t:
            raise LabError(f"Kernel {args.name} missing compile_cmd for phase {args.phase}.", EXIT_USER_ERROR)
        cmd, stats, err_proc = maybe_bench_phase(compile_t, context, args.repeats, args.warmup)
        if err_proc:
            print(f"compile phase failed: {cmd}")
            if err_proc.stderr.strip():
                print(err_proc.stderr.strip())
            return EXIT_COMMAND_FAILED
        result["compile"] = {"cmd": cmd, "stats": stats}

    if args.phase in ("run", "both"):
        run_t = kernel.get("run_cmd", "").strip()
        if not run_t:
            raise LabError(f"Kernel {args.name} missing run_cmd for phase {args.phase}.", EXIT_USER_ERROR)
        cmd, stats, err_proc = maybe_bench_phase(run_t, context, args.repeats, args.warmup)
        if err_proc:
            print(f"run phase failed: {cmd}")
            if err_proc.stderr.strip():
                print(err_proc.stderr.strip())
            return EXIT_COMMAND_FAILED
        result["run"] = {"cmd": cmd, "stats": stats}

    out_path = write_result(Path(args.results_dir), result, args.name)
    print(f"wrote {out_path}")
    return EXIT_OK


def cmd_bench_cmd(args):
    require_positive_int("repeats", args.repeats)
    if args.warmup < 0:
        raise LabError("warmup must be >= 0", EXIT_USER_ERROR)

    check_command_preflight(args.cmd)

    samples = []
    for _ in range(args.warmup):
        proc = run_command(args.cmd, ROOT)
        if proc.returncode != 0:
            if proc.stderr.strip():
                print(proc.stderr.strip())
            return EXIT_COMMAND_FAILED

    for _ in range(args.repeats):
        elapsed_ms, proc = timed_run(args.cmd, ROOT)
        if proc.returncode != 0:
            if proc.stderr.strip():
                print(proc.stderr.strip())
            return EXIT_COMMAND_FAILED
        samples.append(elapsed_ms)

    result = {
        "meta": {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "label": args.label,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "command": args.cmd,
        "stats": summarize(samples),
    }
    out_path = write_result(Path(args.results_dir), result, args.label)
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


def build_parser():
    parser = argparse.ArgumentParser(description="Kernel Lab CLI for kernel prototyping/testing/benchmarking")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to kernels manifest JSON")
    parser.add_argument("--build-dir", default=str(ROOT / "Kernel" / "lab" / "build"), help="Build output dir")
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
    p_bench.set_defaults(func=cmd_bench)

    p_bench_cmd = sub.add_parser("bench-cmd", help="Benchmark any prototype shell command")
    p_bench_cmd.add_argument("label", help="Label used in output filename")
    p_bench_cmd.add_argument("cmd", help="Command to execute and benchmark")
    p_bench_cmd.add_argument("--repeats", type=int, default=10)
    p_bench_cmd.add_argument("--warmup", type=int, default=2)
    p_bench_cmd.set_defaults(func=cmd_bench_cmd)

    p_compare = sub.add_parser("compare", help="Compare two benchmark result JSON files")
    p_compare.add_argument("a", help="Baseline result JSON")
    p_compare.add_argument("b", help="Candidate result JSON")
    p_compare.set_defaults(func=cmd_compare)

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
