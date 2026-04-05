#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static


ROOT = Path(__file__).resolve().parents[2]
PROGRESS_STATE_DIR = ROOT / "infra" / "nexa_insight"
if str(PROGRESS_STATE_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRESS_STATE_DIR))
OBSERVER_PATH = ROOT / "infra" / "nexa_insight" / "observer_server.py"
SPEC = importlib.util.spec_from_file_location("nexa_insight_observer_server", OBSERVER_PATH)
assert SPEC is not None and SPEC.loader is not None
observer_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer_module
SPEC.loader.exec_module(observer_module)

ObserverConfig = observer_module.ObserverConfig
RemoteObserver = observer_module.RemoteObserver
import progress_state as progress_state_utils


class NexaInsightLocalApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: #11151c;
        color: #e6edf3;
    }
    .panel {
        border: round #4c6ef5;
        padding: 0 1;
        background: #161b22;
    }
    #summary {
        height: 3;
    }
    #progress {
        height: 3;
    }
    #body {
        layout: horizontal;
        height: 1fr;
    }
    #left {
        width: 2fr;
    }
    #right {
        width: 1fr;
    }
    #tail {
        height: 10;
    }
    #events {
        height: 12;
    }
    #result_summary {
        height: 8;
    }
    #profile_summary {
        height: 12;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [("q", "quit", "Quit"), ("r", "refresh_now", "Refresh")]

    def __init__(self, observer: RemoteObserver, refresh_seconds: float, process_hold_seconds: float) -> None:
        super().__init__()
        self.observer = observer
        self.refresh_seconds = refresh_seconds
        self.process_hold_seconds = max(0.5, process_hold_seconds)
        self.process_cache: dict[int, dict] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Connecting...", id="summary", classes="panel", markup=False)
        yield Static("Progress pending...", id="progress", classes="panel", markup=False)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield DataTable(id="gpus", classes="panel")
                yield Static("Waiting for recent events...", id="events", classes="panel", markup=False)
                yield Static("Waiting for pane output...", id="tail", classes="panel", markup=False)
            with Vertical(id="right"):
                yield DataTable(id="processes", classes="panel")
                yield DataTable(id="windows", classes="panel")
                yield Static("Waiting for benchmark summary...", id="result_summary", classes="panel", markup=False)
                yield Static("Waiting for profile summary...", id="profile_summary", classes="panel", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        gpu_table = self.query_one("#gpus", DataTable)
        gpu_table.add_columns("GPU", "SM %", "Mem %", "VRAM", "Power", "Temp")
        gpu_table.cursor_type = "row"

        win_table = self.query_one("#windows", DataTable)
        win_table.add_columns("Win", "Name", "Cmd", "State")
        win_table.cursor_type = "row"

        proc_table = self.query_one("#processes", DataTable)
        proc_table.add_columns("PID", "Process", "VRAM", "Type", "State")
        proc_table.cursor_type = "row"

        self.refresh_snapshot()
        self.set_interval(self.refresh_seconds, self.refresh_snapshot)

    def action_refresh_now(self) -> None:
        self.refresh_snapshot()

    def refresh_snapshot(self) -> None:
        payload = self.observer.collect_snapshot()
        now = time.monotonic()
        selected_progress = payload.get("progress_state", {})
        meta = progress_state_utils.progress_meta(selected_progress)
        prog = progress_state_utils.progress_progress(selected_progress)
        progress_source = payload.get("progress_source", "none")
        total_runs = int(meta.get("total_runs", 0) or 0)
        completed_runs = int(prog.get("completed_runs", 0) or 0)
        summary = (
            f"host={payload.get('host')} ssh_ok={payload.get('ssh_ok')} "
            f"source={progress_source} run={meta.get('tag', meta.get('task_name', 'n/a'))} status={meta.get('status', 'idle')} "
            f"shape={prog.get('current_shape_name', '-')} "
            f"adapter={prog.get('current_adapter', '-')} "
            f"repeat={prog.get('current_repeat', 0)}/{prog.get('current_repeat_total', 0)} "
            f"runs={completed_runs}/{total_runs}"
        )
        self.query_one("#summary", Static).update(summary)
        self.query_one("#progress", Static).update(self.render_progress(meta, prog))

        gpu_table = self.query_one("#gpus", DataTable)
        gpu_table.clear()
        for gpu in payload.get("gpus", []):
            gpu_table.add_row(
                f"{gpu.get('index')}: {gpu.get('name')}",
                f"{gpu.get('gpu_util', 0):.1f}",
                f"{gpu.get('mem_util', 0):.1f}",
                f"{gpu.get('mem_used_mib', 0):.0f}/{gpu.get('mem_total_mib', 0):.0f} MiB",
                f"{gpu.get('power_w', 0):.1f} W",
                f"{gpu.get('temp_c', 0):.0f} C",
            )

        win_table = self.query_one("#windows", DataTable)
        win_table.clear()
        active = payload.get("active_window_index", 0)
        for win in payload.get("windows", []):
            state = "active" if win.get("active") or win.get("index") == active else "idle"
            win_table.add_row(str(win.get("index")), win.get("name", ""), win.get("pane_command", ""), state)

        proc_table = self.query_one("#processes", DataTable)
        proc_table.clear()
        for proc in payload.get("processes", []):
            pid = int(proc.get("pid", 0) or 0)
            if pid <= 0:
                continue
            cached = dict(proc)
            cached["last_seen_monotonic"] = now
            cached["active_now"] = True
            self.process_cache[pid] = cached

        expired = []
        for pid, cached in self.process_cache.items():
            age = now - float(cached.get("last_seen_monotonic", now))
            if age > self.process_hold_seconds:
                expired.append(pid)
        for pid in expired:
            self.process_cache.pop(pid, None)

        rows = sorted(
            self.process_cache.values(),
            key=lambda proc: (
                0 if proc.get("last_seen_monotonic", 0.0) == now else 1,
                -float(proc.get("used_gpu_memory_mib", 0.0) or 0.0),
                str(proc.get("name", "")),
            ),
        )
        for proc in rows:
            age = max(0.0, now - float(proc.get("last_seen_monotonic", now)))
            state = "active" if age < max(0.2, self.refresh_seconds * 1.5) else f"recent {age:.1f}s"
            used_gpu_mib = float(proc.get("used_gpu_memory_mib", 0.0) or 0.0)
            proc_table.add_row(
                str(proc.get("pid")),
                proc.get("name", ""),
                f"{used_gpu_mib:.0f} MiB",
                proc.get("kind", ""),
                state,
            )

        self.query_one("#events", Static).update(self.render_recent_events(selected_progress))
        self.query_one("#tail", Static).update(str(payload.get("pane_tail", "")))
        self.query_one("#result_summary", Static).update(self.render_latest_result(payload))
        self.query_one("#profile_summary", Static).update(self.render_latest_profile(payload))

    @staticmethod
    def block_bar(current: int, total: int, width: int = 40) -> str:
        if total <= 0:
            return "░" * width
        current = max(0, min(current, total))
        filled = int(round((current / total) * width))
        return ("█" * filled) + ("░" * (width - filled))

    def render_progress(self, meta: dict, prog: dict) -> str:
        total_runs = int(meta.get("total_runs", 0) or 0)
        completed_runs = int(prog.get("completed_runs", 0) or 0)
        shape_name = str(prog.get("current_shape_name", "-") or "-")
        adapter = str(prog.get("current_adapter", "-") or "-")
        repeat = int(prog.get("current_repeat", 0) or 0)
        repeat_total = int(prog.get("current_repeat_total", 0) or 0)
        bar = self.block_bar(completed_runs, total_runs)
        return (
            f"Run Progress\n"
            f"{bar}  {completed_runs}/{total_runs}\n"
            f"shape={shape_name}  adapter={adapter}  repeat={repeat}/{repeat_total}"
        )

    @staticmethod
    def render_recent_events(progress_state: dict) -> str:
        rows = ["Recent Events"]
        for event in progress_state_utils.progress_recent_events(progress_state)[-8:]:
            shape = event.get("shape", "-")
            adapter = event.get("adapter", "-")
            status = event.get("status", "-")
            tflops = float(event.get("throughput_tflops", 0.0) or 0.0)
            mean_ms = float(event.get("mean_ms", 0.0) or 0.0)
            rows.append(f"{shape} | {adapter} | {status} | {tflops:.3f} TF | {mean_ms:.3f} ms")
        if len(rows) == 1:
            rows.append("No completed adapter results yet.")
        return "\n".join(rows)

    @staticmethod
    def render_latest_result(payload: dict) -> str:
        rows = ["Latest Result"]
        path = Path(str(payload.get("latest_result_path", "") or ""))
        if path.name:
            rows.append(path.name)
        data = payload.get("latest_result", {})
        if not isinstance(data, dict) or not data:
            rows.append("No parsed benchmark result.")
            return "\n".join(rows)

        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        if meta.get("tag"):
            rows.append(f"tag={meta.get('tag')}")
        if meta.get("device") or meta.get("dtype"):
            rows.append(f"device={meta.get('device', '-')} dtype={meta.get('dtype', '-')}")
        if data.get("backend") or data.get("status"):
            rows.append(f"backend={data.get('backend', '-')} status={data.get('status', '-')}")
        if data.get("task"):
            rows.append(f"task={data.get('task')}")
        adapters = data.get("adapters", [])
        if isinstance(adapters, list) and adapters:
            rows.append(f"adapters={len(adapters)}")
        return "\n".join(rows[:8])

    @staticmethod
    def render_latest_profile(payload: dict) -> str:
        rows = ["Latest Profile"]
        path = Path(str(payload.get("latest_profile_path", "") or ""))
        if path.name:
            rows.append(path.name)
        summary = str(payload.get("latest_profile_summary", "") or "").strip()
        if not summary:
            rows.append("No profile summary yet.")
            return "\n".join(rows)
        rows.extend(summary.splitlines()[-8:])
        return "\n".join(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Nexa Insight TUI for a remote SSH GPU box")
    parser.add_argument("--host", default="64.247.206.171")
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--identity", default=str(Path.home() / ".ssh" / "prime_next"))
    parser.add_argument("--session", default="pyc-ada")
    parser.add_argument("--repo-path", default="/home/ubuntu/work/PyC")
    parser.add_argument("--progress-file", default="/home/ubuntu/work/PyC/benchmark/benchmarks/results/json/latest_ada_fp32_gemm.progress.json")
    parser.add_argument("--task-progress-file", default="/home/ubuntu/work/PyC/kernels/lab/tasks/runs/latest_task_run.progress.json")
    parser.add_argument("--refresh", type=float, default=1.0)
    parser.add_argument("--process-hold", type=float, default=4.0)
    parser.add_argument("--ssh-port", type=int, default=22)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    observer = RemoteObserver(
        ObserverConfig(
            host=args.host,
            user=args.user,
            identity=args.identity,
            session=args.session,
            ssh_port=args.ssh_port,
            repo_path=args.repo_path,
            progress_file=args.progress_file,
            task_progress_file=args.task_progress_file,
        )
    )
    app = NexaInsightLocalApp(observer, max(0.5, args.refresh), args.process_hold)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
