#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_STATE_DIR = Path(__file__).resolve().parent
if str(PROGRESS_STATE_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRESS_STATE_DIR))
from progress_state import choose_progress_state, progress_updated_utc


REMOTE_REPO = "/home/ubuntu/work/PyC"
REMOTE_PROGRESS_FILE = (
    "/home/ubuntu/work/PyC/benchmark/benchmarks/results/json/latest_ada_fp32_gemm.progress.json"
)
REMOTE_TASK_PROGRESS_FILE = (
    "/home/ubuntu/work/PyC/kernels/lab/tasks/runs/latest_task_run.progress.json"
)
REMOTE_RESULTS_DIR = "/home/ubuntu/work/PyC/benchmark/benchmarks/results/json"
REMOTE_PROFILES_DIR = "/home/ubuntu/work/PyC/benchmark/benchmarks/results/profiles"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_lines(text: str, columns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = next(csv.reader([line], skipinitialspace=True), [])
        if len(parsed) < len(columns):
            parsed += [""] * (len(columns) - len(parsed))
        rows.append({column: parsed[index].strip() for index, column in enumerate(columns)})
    return rows


def safe_float(text: str, default: float = 0.0) -> float:
    cleaned = str(text or "").strip()
    if not cleaned or cleaned in {"[Not Supported]", "N/A"}:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def safe_int(text: str, default: int = 0) -> int:
    cleaned = str(text or "").strip()
    if not cleaned or cleaned in {"[Not Supported]", "N/A"}:
        return default
    try:
        return int(float(cleaned))
    except ValueError:
        return default


def parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    rows = parse_csv_lines(
        text,
        ["index", "name", "gpu_util", "mem_util", "mem_used", "mem_total", "temp_c", "power_w"],
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "index": safe_int(row["index"]),
                "name": row["name"],
                "gpu_util": safe_float(row["gpu_util"]),
                "mem_util": safe_float(row["mem_util"]),
                "mem_used_mib": safe_float(row["mem_used"]),
                "mem_total_mib": safe_float(row["mem_total"]),
                "temp_c": safe_float(row["temp_c"]),
                "power_w": safe_float(row["power_w"]),
            }
        )
    return out


def parse_process_rows(text: str) -> list[dict[str, Any]]:
    rows = parse_csv_lines(text, ["pid", "name", "used_gpu_memory"])
    out: list[dict[str, Any]] = []
    for row in rows:
        pid = safe_int(row["pid"], -1)
        if pid < 0:
            continue
        out.append(
            {
                "pid": pid,
                "name": row["name"],
                "used_gpu_memory_mib": safe_float(row["used_gpu_memory"]),
                "kind": "Compute",
            }
        )
    return out


def parse_tmux_windows(text: str) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split("|", 4)
        if len(parts) != 5:
            continue
        windows.append(
            {
                "index": safe_int(parts[0]),
                "name": parts[1],
                "active": parts[2] == "1",
                "pane_command": parts[3],
                "pane_title": parts[4],
            }
        )
    return windows


@dataclass
class ObserverConfig:
    host: str
    user: str
    identity: str
    session: str
    ssh_port: int
    repo_path: str
    progress_file: str
    task_progress_file: str


class RemoteObserver:
    def __init__(self, config: ObserverConfig) -> None:
        self.config = config

    def _ssh_base(self) -> list[str]:
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if self.config.identity:
            cmd += ["-i", self.config.identity]
        if self.config.ssh_port:
            cmd += ["-p", str(self.config.ssh_port)]
        cmd.append(f"{self.config.user}@{self.config.host}")
        return cmd

    def run_remote(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self._ssh_base() + [script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def collect_snapshot(self) -> dict[str, Any]:
        remote_script = f"""
set -euo pipefail
cd {self.config.repo_path}
latest_match() {{
    pattern="$1"
    ls -1t $pattern 2>/dev/null | head -n 1 || true
}}
TASK_PROGRESS="$(latest_match '{self.config.repo_path}/kernels/lab/tasks/runs/*.progress.json')"
GEMM_PROGRESS="$(latest_match '{self.config.repo_path}/benchmark/benchmarks/results/json/*.progress.json')"
LATEST_RESULT="$(ls -1t {self.config.repo_path}/benchmark/benchmarks/results/json/*.json 2>/dev/null | grep -v '\\.progress\\.json$' | head -n 1 || true)"
LATEST_PROFILE="$(latest_match '{self.config.repo_path}/benchmark/benchmarks/results/profiles/*_nsys.summary.txt')"
if [ -z "$TASK_PROGRESS" ] && [ -f {self.config.task_progress_file} ]; then
    TASK_PROGRESS="{self.config.task_progress_file}"
fi
if [ -z "$GEMM_PROGRESS" ] && [ -f {self.config.progress_file} ]; then
    GEMM_PROGRESS="{self.config.progress_file}"
fi
echo '__GPU__'
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits || true
echo '__PROCS__'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo '__TMUX__'
tmux list-windows -t {self.config.session} -F '#{{window_index}}|#{{window_name}}|#{{window_active}}|#{{pane_current_command}}|#{{pane_title}}' || true
echo '__ACTIVE__'
tmux display-message -p -t {self.config.session} '#{{window_index}}' || true
echo '__TASK_PROGRESS__'
test -n "$TASK_PROGRESS" && test -f "$TASK_PROGRESS" && cat "$TASK_PROGRESS" || true
echo '__PROGRESS__'
test -n "$GEMM_PROGRESS" && test -f "$GEMM_PROGRESS" && cat "$GEMM_PROGRESS" || true
echo '__RESULT_PATH__'
printf '%s\n' "$LATEST_RESULT"
echo '__RESULT__'
test -n "$LATEST_RESULT" && test -f "$LATEST_RESULT" && cat "$LATEST_RESULT" || true
echo '__PROFILE_PATH__'
printf '%s\n' "$LATEST_PROFILE"
echo '__PROFILE__'
test -n "$LATEST_PROFILE" && test -f "$LATEST_PROFILE" && cat "$LATEST_PROFILE" || true
echo '__TAIL__'
tmux capture-pane -pt {self.config.session} || true
"""
        proc = self.run_remote(remote_script)
        payload: dict[str, Any] = {
            "collected_utc": utc_now(),
            "host": self.config.host,
            "session": self.config.session,
            "ssh_ok": proc.returncode == 0,
            "stderr": proc.stderr.strip(),
            "gpus": [],
            "processes": [],
            "windows": [],
            "active_window_index": 0,
            "progress_state": {},
            "pane_tail": "",
            "progress_source": "none",
            "latest_result_path": "",
            "latest_result": {},
            "latest_profile_path": "",
            "latest_profile_summary": "",
        }
        if proc.returncode != 0:
            return payload

        sections = {
            "__GPU__": "",
            "__PROCS__": "",
            "__TMUX__": "",
            "__ACTIVE__": "",
            "__TASK_PROGRESS__": "",
            "__PROGRESS__": "",
            "__RESULT_PATH__": "",
            "__RESULT__": "",
            "__PROFILE_PATH__": "",
            "__PROFILE__": "",
            "__TAIL__": "",
        }
        current = None
        for line in proc.stdout.splitlines():
            marker = line.strip()
            if marker in sections:
                current = marker
                continue
            if current is not None:
                sections[current] += line + "\n"

        payload["gpus"] = parse_gpu_rows(sections["__GPU__"])
        payload["processes"] = parse_process_rows(sections["__PROCS__"])
        payload["windows"] = parse_tmux_windows(sections["__TMUX__"])
        payload["active_window_index"] = safe_int(sections["__ACTIVE__"].strip(), 0)
        try:
            task_progress = json.loads(sections["__TASK_PROGRESS__"].strip()) if sections["__TASK_PROGRESS__"].strip() else {}
        except json.JSONDecodeError:
            task_progress = {}
        try:
            gemm_progress = json.loads(sections["__PROGRESS__"].strip()) if sections["__PROGRESS__"].strip() else {}
        except json.JSONDecodeError:
            gemm_progress = {}
        try:
            latest_result = json.loads(sections["__RESULT__"].strip()) if sections["__RESULT__"].strip() else {}
        except json.JSONDecodeError:
            latest_result = {}
        payload["progress_state"], payload["progress_source"] = choose_progress_state(task_progress, gemm_progress)
        payload["latest_result_path"] = sections["__RESULT_PATH__"].strip()
        payload["latest_result"] = latest_result
        payload["latest_profile_path"] = sections["__PROFILE_PATH__"].strip()
        payload["latest_profile_summary"] = sections["__PROFILE__"].rstrip()
        payload["pane_tail"] = sections["__TAIL__"].rstrip()
        return payload

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote SSH data collector for the local Nexa Insight TUI")
    parser.add_argument("--host", default="64.247.206.171")
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--identity", default="")
    parser.add_argument("--session", default="pyc-ada")
    parser.add_argument("--repo-path", default=REMOTE_REPO)
    parser.add_argument("--progress-file", default=REMOTE_PROGRESS_FILE)
    parser.add_argument("--task-progress-file", default=REMOTE_TASK_PROGRESS_FILE)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--probe", action="store_true", help="Fetch one snapshot and print JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ObserverConfig(
        host=args.host,
        user=args.user,
        identity=args.identity,
        session=args.session,
        ssh_port=args.ssh_port,
        repo_path=args.repo_path,
        progress_file=args.progress_file,
        task_progress_file=args.task_progress_file,
    )
    observer = RemoteObserver(config)
    print(json.dumps(observer.collect_snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
