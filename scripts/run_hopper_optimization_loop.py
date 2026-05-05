#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "benchmark" / "benchmarks" / "gpu" / "configs" / "hopper_bf16_gemm_shapes.json"
DEFAULT_REMOTE_ROOT = "/root/work/PyC/repo"
DEFAULT_SESSION = "pyc-hopper"
DEFAULT_HOST_SLUG = "excellent-wonderful-trout"

SYNC_PATHS = [
    Path("benchmark/benchmarks/gpu"),
    Path("benchmark/tools/analyze_hopper_gemm_results.py"),
    Path("scripts/setup_benchmark_env_locked.sh"),
    Path("scripts/setup_tvm_cuda_remote_ubuntu.sh"),
    Path("scripts/setup_hopper_benchmark_remote.sh"),
]


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(shlex.quote(part) for part in cmd)}\n{proc.stderr or proc.stdout}")
    return proc


def ssh_cmd(identity: str, target: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["ssh", "-i", identity, "-o", "StrictHostKeyChecking=no", target, command], cwd=ROOT, check=check)


def scp_to(identity: str, source: str, target: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["scp", "-i", identity, "-o", "StrictHostKeyChecking=no", source, target], cwd=ROOT, check=check)


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_sync_archive() -> Path:
    tmp = Path(tempfile.gettempdir()) / f"pyc_hopper_loop_sync_{utc_stamp()}.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in SYNC_PATHS:
            tar.add(ROOT / rel, arcname=str(rel))
    return tmp


def ensure_remote_session(identity: str, target: str, session: str) -> None:
    ssh_cmd(
        identity,
        target,
        f"tmux has-session -t {shlex.quote(session)} 2>/dev/null || tmux new-session -d -s {shlex.quote(session)} -n main bash",
    )


def launch_remote_run(
    identity: str,
    target: str,
    *,
    session: str,
    remote_repo_root: str,
    remote_archive: str,
    matrix_file: str,
    run_id: str,
    tag: str,
    strict_native: bool,
    pyc_timeout_sec: int,
) -> None:
    env_exports = [
        f"export PYC_GPU_BENCH_TIMEOUT_SEC={pyc_timeout_sec}",
    ]
    if strict_native:
        env_exports.append("export BENCH_STRICT_NATIVE=1")
    remote_cmd = " && ".join(
        [
            f"cd {shlex.quote(remote_repo_root)}",
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_repo_root)}",
            "source .venv/bin/activate",
            "source benchmark/benchmarks/gpu/configure_adapter_cmds.sh",
            *env_exports,
            (
                "python benchmark/benchmarks/gpu/run_gemm_suite.py "
                f"--device cuda --matrix-file {shlex.quote(matrix_file)} "
                f"--arena-mode --parity-strict --run-id {shlex.quote(run_id)} --tag {shlex.quote(tag)} --progress"
            ),
        ]
    )
    ssh_cmd(
        identity,
        target,
        f"tmux send-keys -t {shlex.quote(session)}:0.0 {shlex.quote(remote_cmd)} C-m",
    )


def read_remote_json(identity: str, target: str, remote_path: str) -> dict | None:
    cmd = (
        "python3 - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        f"path = Path({remote_path!r})\n"
        "if not path.exists():\n"
        "    raise SystemExit(2)\n"
        "print(path.read_text(encoding='utf-8'))\n"
        "PY"
    )
    proc = ssh_cmd(identity, target, cmd, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def pull_matches(identity: str, target: str, remote_glob: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    scp_to(identity, f"{target}:{remote_glob}", str(local_dir), check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Hopper benchmark and analysis loop end-to-end")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--identity", default=str(Path.home() / ".ssh" / "prime_next"))
    parser.add_argument("--remote-repo-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--matrix-file", default=str(DEFAULT_MATRIX.relative_to(ROOT)))
    parser.add_argument("--tag", default=f"hopper_loop_{utc_stamp().lower()}")
    parser.add_argument("--local-host-slug", default=DEFAULT_HOST_SLUG)
    parser.add_argument("--progress-poll-sec", type=int, default=20)
    parser.add_argument("--stall-sec", type=int, default=120)
    parser.add_argument("--pyc-timeout-sec", type=int, default=120)
    parser.add_argument("--enable-cutlass-profiler", action="store_true")
    parser.add_argument("--enable-tvm-cuda-build", action="store_true")
    parser.add_argument("--no-provision", action="store_true")
    parser.add_argument("--allow-proxy", action="store_true")
    args = parser.parse_args()

    target = f"{args.user}@{args.host}"
    run_id = utc_stamp()
    archive = make_sync_archive()
    remote_archive = f"/tmp/{archive.name}"

    print(f"[hopper-loop] sync archive: {archive}")
    scp_to(args.identity, str(archive), f"{target}:{remote_archive}")

    if not args.no_provision:
        provision_cmd = " && ".join(
            [
                f"cd {shlex.quote(args.remote_repo_root)}",
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(args.remote_repo_root)}",
                f"ENABLE_CUTLASS_PROFILER={'1' if args.enable_cutlass_profiler else '0'} "
                f"ENABLE_TVM_CUDA_BUILD={'1' if args.enable_tvm_cuda_build else '0'} "
                "bash scripts/setup_hopper_benchmark_remote.sh",
            ]
        )
        print("[hopper-loop] provisioning remote benchmark env")
        ssh_cmd(args.identity, target, provision_cmd)

    ensure_remote_session(args.identity, target, args.session)
    launch_remote_run(
        args.identity,
        target,
        session=args.session,
        remote_repo_root=args.remote_repo_root,
        remote_archive=remote_archive,
        matrix_file=args.matrix_file,
        run_id=run_id,
        tag=args.tag,
        strict_native=not args.allow_proxy,
        pyc_timeout_sec=args.pyc_timeout_sec,
    )

    remote_progress = f"{args.remote_repo_root}/benchmark/benchmarks/results/json/{run_id}__{args.tag}.progress.json"
    remote_aggregate = f"{args.remote_repo_root}/benchmark/benchmarks/results/json/{run_id}__{args.tag}.json"
    last_completed = -1
    last_progress_change = time.monotonic()
    stalled = False

    print(f"[hopper-loop] run_id={run_id} tag={args.tag}")
    while True:
        progress = read_remote_json(args.identity, target, remote_progress)
        aggregate = read_remote_json(args.identity, target, remote_aggregate)

        if aggregate is not None:
            print("[hopper-loop] aggregate artifact detected")
            break

        if progress is not None:
            prog = progress.get("progress", {})
            completed = int(prog.get("completed_runs", 0))
            shape_name = prog.get("current_shape_name", "")
            adapter = prog.get("current_adapter", "")
            repeat = prog.get("current_repeat", 0)
            repeat_total = prog.get("current_repeat_total", 0)
            print(
                f"[hopper-loop] progress completed={completed}/{progress.get('meta', {}).get('total_runs', '?')} "
                f"shape={shape_name} adapter={adapter} repeat={repeat}/{repeat_total}"
            )
            if completed != last_completed:
                last_completed = completed
                last_progress_change = time.monotonic()
            elif time.monotonic() - last_progress_change > args.stall_sec:
                stalled = True
                print("[hopper-loop] stall detected, terminating remote run")
                ssh_cmd(
                    args.identity,
                    target,
                    (
                        f"pkill -f {shlex.quote(run_id)} || true; "
                        "pkill -f 'benchmark/benchmarks/gpu/run_gemm_suite.py' || true; "
                        "pkill -f 'benchmark/benchmarks/gpu/external/bench_pyc_cmd.py' || true"
                    ),
                    check=False,
                )
                break

        time.sleep(max(5, args.progress_poll_sec))

    local_run_root = (
        ROOT
        / "benchmark"
        / "benchmarks"
        / "results"
        / "remote_results"
        / "hosts"
        / args.local_host_slug
        / "runs"
        / run_id
        / args.tag
    )
    for folder in ("json", "reports", "images"):
        (local_run_root / folder).mkdir(parents=True, exist_ok=True)

    pull_matches(
        args.identity,
        target,
        f"{args.remote_repo_root}/benchmark/benchmarks/results/json/{run_id}__{args.tag}*",
        local_run_root / "json",
    )
    pull_matches(
        args.identity,
        target,
        f"{args.remote_repo_root}/benchmark/benchmarks/results/reports/{run_id}__{args.tag}*",
        local_run_root / "reports",
    )
    pull_matches(
        args.identity,
        target,
        f"{args.remote_repo_root}/benchmark/benchmarks/results/images/{run_id}__{args.tag}*",
        local_run_root / "images",
    )

    analysis_dir = (
        ROOT
        / "benchmark"
        / "benchmarks"
        / "results"
        / "analysis"
        / "hopper"
        / run_id
        / args.tag
    )
    run(
        [
            "python3",
            str(ROOT / "benchmark" / "tools" / "analyze_hopper_gemm_results.py"),
            "--run-dir",
            str(local_run_root),
            "--output-dir",
            str(analysis_dir),
        ],
        cwd=ROOT,
    )

    report = {
        "run_id": run_id,
        "tag": args.tag,
        "host": args.host,
        "user": args.user,
        "remote_repo_root": args.remote_repo_root,
        "remote_session": args.session,
        "local_run_root": str(local_run_root),
        "analysis_dir": str(analysis_dir),
        "stalled": stalled,
        "strict_native": not args.allow_proxy,
        "pyc_timeout_sec": args.pyc_timeout_sec,
        "provisioned": not args.no_provision,
    }
    report_path = analysis_dir / "loop_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[hopper-loop] local run dir: {local_run_root}")
    print(f"[hopper-loop] analysis dir: {analysis_dir}")
    print(f"[hopper-loop] report: {report_path}")
    if stalled:
        print("[hopper-loop] run ended as partial due to stall detection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
