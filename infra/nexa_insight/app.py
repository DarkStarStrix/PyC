#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pynvml  # type: ignore
except Exception:  # noqa: BLE001
    pynvml = None

TEXTUAL_IMPORT_ERROR: Exception | None = None
try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, DataTable, Footer, Header, Input, Static, TabPane, TabbedContent

    TEXTUAL_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    TEXTUAL_IMPORT_ERROR = exc
    TEXTUAL_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
INSIGHT_ROOT = ROOT / "infra" / "nexa_insight"
DEFAULT_HISTORY = INSIGHT_ROOT / ".command_history.json"
DEFAULT_EXPORTS = INSIGHT_ROOT / "exports"
DEFAULT_REFRESH_HZ = 1.0
CUDA_BIN_CANDIDATES = [
    Path("/usr/local/cuda/bin"),
    Path("/usr/local/cuda-13.2/bin"),
    Path("/usr/local/cuda-13.1/bin"),
    Path("/usr/local/cuda-13.0/bin"),
    Path("/usr/local/cuda-12.6/bin"),
    Path("/usr/local/cuda-12.5/bin"),
    Path("/usr/local/cuda-12.4/bin"),
    Path("/usr/local/cuda-12.3/bin"),
]
NVIDIA_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
]
THEMES = {
    "dracula": {
        "background": "#282a36",
        "panel": "#1f2029",
        "foreground": "#f8f8f2",
        "muted": "#bd93f9",
        "accent": "#8be9fd",
        "success": "#50fa7b",
        "warning": "#ffb86c",
        "danger": "#ff5555",
    },
    "monokai": {
        "background": "#272822",
        "panel": "#1e1f1c",
        "foreground": "#f8f8f2",
        "muted": "#a6e22e",
        "accent": "#66d9ef",
        "success": "#a6e22e",
        "warning": "#fd971f",
        "danger": "#f92672",
    },
}
SPARK_BARS = "▁▂▃▄▅▆▇█"


@dataclass
class GPUInfo:
    index: int
    name: str
    uuid: str
    memory_total_mib: int


@dataclass
class GPUProcessInfo:
    pid: int
    name: str
    memory_mib: float
    kind: str


@dataclass
class GPUSample:
    info: GPUInfo
    sm_util_pct: int
    mem_util_pct: int
    mem_used_mib: int
    power_w: float
    temp_c: int
    processes: list[GPUProcessInfo] = field(default_factory=list)


@dataclass
class KernelMetricRecord:
    key: str
    kernel_name: str
    total_time_ms: float
    throughput_pct: float
    global_loads: float
    metrics: dict[str, dict[str, Any]]
    raw_text: str


@dataclass
class SnapshotRun:
    command: str
    profiler_command: list[str]
    returncode: int
    stdout: str
    stderr: str
    records: list[KernelMetricRecord]
    started_utc: str
    finished_utc: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def resolve_tool(binary: str) -> str | None:
    direct = shutil.which(binary)
    if direct:
        return direct
    for base in CUDA_BIN_CANDIDATES:
        candidate = base / binary
        if candidate.exists():
            return str(candidate)
    return None


def safe_float(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(",", "").replace("%", "").strip()
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def sparkline(values: list[float], width: int = 24) -> str:
    if not values:
        return " " * width
    window = values[-width:]
    low = min(window)
    high = max(window)
    if high <= low:
        return SPARK_BARS[0] * len(window)
    chars = []
    for value in window:
        idx = int(round(((value - low) / (high - low)) * (len(SPARK_BARS) - 1)))
        idx = max(0, min(idx, len(SPARK_BARS) - 1))
        chars.append(SPARK_BARS[idx])
    return "".join(chars)


def format_mib(value: float) -> str:
    return f"{value:.0f} MiB"


def format_metric_value(metric_name: str, metric_value: float) -> float:
    if metric_name == "gpu__time_duration.sum":
        return metric_value / 1_000_000.0
    return metric_value


def read_process_name(pid: int) -> str:
    comm = Path(f"/proc/{pid}/comm")
    try:
        return comm.read_text(encoding="utf-8").strip()
    except OSError:
        return f"pid-{pid}"


def discover_gpus() -> tuple[list[GPUInfo], str | None]:
    if pynvml is None:
        return [], "nvidia-ml-py is not installed"
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        return [], f"nvmlInit failed: {exc}"
    infos: list[GPUInfo] = []
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            infos.append(
                GPUInfo(
                    index=index,
                    name=name.decode() if isinstance(name, bytes) else str(name),
                    uuid=uuid.decode() if isinstance(uuid, bytes) else str(uuid),
                    memory_total_mib=int(mem.total / (1024 * 1024)),
                )
            )
    except Exception as exc:  # noqa: BLE001
        return [], f"GPU discovery failed: {exc}"
    return infos, None


def _read_nvml_processes(handle, kind: str) -> dict[int, GPUProcessInfo]:
    rows: dict[int, GPUProcessInfo] = {}
    if pynvml is None:
        return rows
    getter_names = {
        "Compute": ["nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses"],
        "Graphics": ["nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses"],
    }
    getter = None
    for name in getter_names[kind]:
        getter = getattr(pynvml, name, None)
        if getter is not None:
            break
    if getter is None:
        return rows
    try:
        for proc in getter(handle) or []:
            pid = int(getattr(proc, "pid", 0) or 0)
            used = getattr(proc, "usedGpuMemory", 0) or 0
            used_mib = 0.0 if used < 0 else used / (1024 * 1024)
            rows[pid] = GPUProcessInfo(
                pid=pid,
                name=read_process_name(pid),
                memory_mib=used_mib,
                kind=kind,
            )
    except Exception:
        return rows
    return rows


def read_live_samples(gpus: list[GPUInfo]) -> tuple[list[GPUSample], str | None]:
    if pynvml is None:
        return [], "nvidia-ml-py is not installed"
    samples: list[GPUSample] = []
    try:
        for gpu in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = 0.0
            temp = 0
            try:
                power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                pass
            try:
                temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            processes = _read_nvml_processes(handle, "Compute")
            for pid, proc in _read_nvml_processes(handle, "Graphics").items():
                if pid in processes:
                    processes[pid].kind = "Compute+Graphics"
                    processes[pid].memory_mib = max(processes[pid].memory_mib, proc.memory_mib)
                else:
                    processes[pid] = proc
            samples.append(
                GPUSample(
                    info=gpu,
                    sm_util_pct=int(getattr(util, "gpu", 0) or 0),
                    mem_util_pct=int(getattr(util, "memory", 0) or 0),
                    mem_used_mib=int(mem.used / (1024 * 1024)),
                    power_w=power,
                    temp_c=temp,
                    processes=sorted(processes.values(), key=lambda row: row.memory_mib, reverse=True),
                )
            )
    except Exception as exc:  # noqa: BLE001
        return [], f"live NVML sample failed: {exc}"
    return samples, None


def parse_ncu_csv(stdout: str) -> list[KernelMetricRecord]:
    if not stdout.strip():
        return []
    rows = list(csv.reader(stdout.splitlines()))
    header_index = None
    for idx, row in enumerate(rows):
        normalized = [cell.strip() for cell in row]
        if "Kernel Name" in normalized and "Metric Name" in normalized and "Metric Value" in normalized:
            header_index = idx
            break
    if header_index is None:
        return []
    header = [cell.strip() for cell in rows[header_index]]
    index = {name: pos for pos, name in enumerate(header)}
    id_key = "ID" if "ID" in index else None
    kernel_key = "Kernel Name"
    metric_name_key = "Metric Name"
    metric_value_key = "Metric Value"
    metric_unit_key = "Metric Unit" if "Metric Unit" in index else None
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows[header_index + 1 :]:
        if not row or len(row) < len(header):
            continue
        kernel_name = row[index[kernel_key]].strip()
        metric_name = row[index[metric_name_key]].strip()
        if not kernel_name or not metric_name:
            continue
        kernel_id = row[index[id_key]].strip() if id_key else str(len(groups))
        group_key = (kernel_id, kernel_name)
        bucket = groups.setdefault(
            group_key,
            {
                "kernel_name": kernel_name,
                "metrics": {},
                "raw_lines": [],
            },
        )
        unit = row[index[metric_unit_key]].strip() if metric_unit_key else ""
        raw_value = row[index[metric_value_key]].strip()
        value = safe_float(raw_value)
        bucket["metrics"][metric_name] = {"value": value, "raw": raw_value, "unit": unit}
        bucket["raw_lines"].append(",".join(row))
    records: list[KernelMetricRecord] = []
    for (kernel_id, kernel_name), bucket in groups.items():
        metrics = bucket["metrics"]
        duration = format_metric_value("gpu__time_duration.sum", metrics.get("gpu__time_duration.sum", {}).get("value", 0.0))
        throughput = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", {}).get("value", 0.0)
        loads = metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", {}).get("value", 0.0)
        raw_text = "\n".join(bucket["raw_lines"])
        records.append(
            KernelMetricRecord(
                key=f"{kernel_id}:{kernel_name}",
                kernel_name=kernel_name,
                total_time_ms=duration,
                throughput_pct=throughput,
                global_loads=loads,
                metrics=metrics,
                raw_text=raw_text,
            )
        )
    return records


def profile_command(command: str, ncu_path: str = "ncu") -> SnapshotRun:
    base_cmd = shlex.split(command)
    profiler_cmd = [
        ncu_path,
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(NVIDIA_METRICS),
        "--kernel-name-base",
        "demangled",
    ] + base_cmd
    started = utc_now()
    proc = subprocess.run(profiler_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finished = utc_now()
    records = parse_ncu_csv(proc.stdout)
    return SnapshotRun(
        command=command,
        profiler_command=profiler_cmd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        records=records,
        started_utc=started,
        finished_utc=finished,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_history(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return []
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def save_history(path: Path, history: list[str]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(history[-50:], indent=2) + "\n", encoding="utf-8")


def append_history(path: Path, history: list[str], command: str) -> list[str]:
    command = command.strip()
    if not command:
        return history
    if history and history[-1] == command:
        return history
    history = [item for item in history if item != command]
    history.append(command)
    save_history(path, history)
    return history


def export_records_csv(path: Path, records: list[KernelMetricRecord]) -> Path:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kernel_name", "total_time_ms", "throughput_pct", "global_loads"])
        for record in records:
            writer.writerow(
                [
                    record.kernel_name,
                    f"{record.total_time_ms:.6f}",
                    f"{record.throughput_pct:.6f}",
                    f"{record.global_loads:.6f}",
                ]
            )
    return path


if TEXTUAL_AVAILABLE:
    class KernelDetailModal(ModalScreen[None]):
        BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("q", "dismiss", "Close")]

        def __init__(self, record: KernelMetricRecord, theme_name: str) -> None:
            super().__init__()
            self.record = record
            self.theme_name = theme_name

        def compose(self) -> ComposeResult:
            yield Static(
                f"Kernel: {self.record.kernel_name}\n"
                f"Duration: {self.record.total_time_ms:.6f} ms\n"
                f"Throughput: {self.record.throughput_pct:.3f}%\n"
                f"Global loads: {self.record.global_loads:.3f}\n\n"
                f"{self.record.raw_text}",
                id="kernel-detail-body",
                markup=False,
            )

    class NexaInsightApp(App[None]):
        CSS = """
        Screen {
            layout: vertical;
        }
        #top {
            height: 1fr;
        }
        #bottom {
            height: 1fr;
        }
        #command-row {
            height: 3;
            layout: horizontal;
        }
        #snapshot-status {
            height: 3;
        }
        #gpu-summary {
            height: 11;
        }
        DataTable {
            height: 1fr;
        }
        Input {
            width: 1fr;
        }
        Button {
            width: 18;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "refresh_now", "Refresh"),
            Binding("ctrl+r", "run_command", "Run Command"),
            Binding("ctrl+p", "history_prev", "Prev Command"),
            Binding("ctrl+n", "history_next", "Next Command"),
            Binding("shift+s", "export_csv", "Export CSV"),
            Binding("t", "toggle_theme", "Theme"),
            Binding("l", "sort_longest", "Sort Longest"),
            Binding("h", "sort_throughput", "Sort Throughput"),
        ]

        def __init__(self, refresh_seconds: float, history_path: Path, export_dir: Path, theme_name: str, ncu_path: str) -> None:
            super().__init__()
            self.refresh_seconds = refresh_seconds
            self.history_path = history_path
            self.export_dir = export_dir
            self.theme_name = theme_name if theme_name in THEMES else "dracula"
            self.ncu_path = ncu_path
            self.gpus, self.gpu_error = discover_gpus()
            self.histories: dict[str, dict[str, deque[float]]] = defaultdict(
                lambda: {
                    "sm": deque(maxlen=60),
                    "mem": deque(maxlen=60),
                    "power": deque(maxlen=60),
                    "temp": deque(maxlen=60),
                }
            )
            self.kernel_records: list[KernelMetricRecord] = []
            self.last_snapshot: SnapshotRun | None = None
            self.history = load_history(history_path)
            self.history_cursor = len(self.history)
            self.last_status = "Idle"

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical(id="top"):
                with TabbedContent(id="gpu-tabs"):
                    if self.gpus:
                        for gpu in self.gpus:
                            with TabPane(f"GPU {gpu.index}: {gpu.name}", id=f"gpu-{gpu.index}"):
                                yield Static("", id=f"gpu-summary-{gpu.index}")
                                yield DataTable(id=f"gpu-procs-{gpu.index}")
                    else:
                        with TabPane("No GPU", id="gpu-none"):
                            yield Static(self.gpu_error or "No NVIDIA GPU detected", id="gpu-summary-none")
            with Vertical(id="bottom"):
                with Horizontal(id="command-row"):
                    yield Input(placeholder="Run command through ncu, e.g. python3 benchmark/benchmarks/gpu/external/bench_pyc_cmd.py", id="command-input")
                    yield Button("Profile", id="profile-button", variant="primary")
                    yield Button("Theme", id="theme-button")
                    yield Button("Export CSV", id="export-button")
                yield Static("", id="snapshot-status")
                yield DataTable(id="kernel-grid")
            yield Footer()

        def on_mount(self) -> None:
            self._configure_tables()
            if self.history:
                self.query_one("#command-input", Input).value = self.history[-1]
            self._apply_theme()
            self._render_snapshot_status()
            self.collect_live()
            self.set_interval(self.refresh_seconds, self.collect_live)

        def _configure_tables(self) -> None:
            for gpu in self.gpus:
                table = self.query_one(f"#gpu-procs-{gpu.index}", DataTable)
                table.cursor_type = "row"
                table.add_columns("PID", "Process", "VRAM", "Type")
            kernel_table = self.query_one("#kernel-grid", DataTable)
            kernel_table.cursor_type = "row"
            kernel_table.add_columns("Kernel", "Time ms", "SM % Peak", "Global Loads")

        def _apply_theme(self) -> None:
            palette = THEMES[self.theme_name]
            self.screen.styles.background = palette["background"]
            self.screen.styles.color = palette["foreground"]
            for widget_id in ["snapshot-status"] + [f"gpu-summary-{gpu.index}" for gpu in self.gpus]:
                try:
                    widget = self.query_one(f"#{widget_id}", Static)
                except Exception:
                    continue
                widget.styles.background = palette["panel"]
                widget.styles.color = palette["foreground"]
            for table_id in [f"gpu-procs-{gpu.index}" for gpu in self.gpus] + ["kernel-grid"]:
                try:
                    table = self.query_one(f"#{table_id}", DataTable)
                except Exception:
                    continue
                table.styles.background = palette["panel"]
                table.styles.color = palette["foreground"]
            try:
                inp = self.query_one("#command-input", Input)
                inp.styles.background = palette["panel"]
                inp.styles.color = palette["foreground"]
            except Exception:
                pass

        def _set_status(self, message: str) -> None:
            self.last_status = message
            self._render_snapshot_status()

        def _render_snapshot_status(self) -> None:
            text = (
                f"Theme: {self.theme_name} | GPUs: {len(self.gpus)} | History: {len(self.history)} commands\n"
                f"Status: {self.last_status}\n"
                f"Hotkeys: Ctrl+R profile, Ctrl+P/N history, Shift+S export CSV, T theme, L longest, H highest throughput"
            )
            self.query_one("#snapshot-status", Static).update(text)

        def _render_gpu_summary(self, sample: GPUSample) -> None:
            history = self.histories[sample.info.uuid]
            lines = [
                f"{sample.info.name} | UUID={sample.info.uuid}",
                f"SM Util    {sample.sm_util_pct:>3}%  {sparkline(list(history['sm']))}",
                f"Mem Util   {sample.mem_util_pct:>3}%  {sparkline(list(history['mem']))}",
                f"Power      {sample.power_w:>5.1f} W  {sparkline(list(history['power']))}",
                f"Temp       {sample.temp_c:>3} C  {sparkline(list(history['temp']))}",
                f"VRAM       {sample.mem_used_mib:>5} / {sample.info.memory_total_mib:<5} MiB",
                f"Processes  {len(sample.processes)} active contexts",
            ]
            self.query_one(f"#gpu-summary-{sample.info.index}", Static).update("\n".join(lines))

        def _render_gpu_processes(self, sample: GPUSample) -> None:
            table = self.query_one(f"#gpu-procs-{sample.info.index}", DataTable)
            table.clear(columns=False)
            for proc in sample.processes:
                table.add_row(str(proc.pid), proc.name, format_mib(proc.memory_mib), proc.kind)

        def _render_kernel_records(self) -> None:
            table = self.query_one("#kernel-grid", DataTable)
            table.clear(columns=False)
            for record in self.kernel_records:
                table.add_row(
                    record.kernel_name,
                    f"{record.total_time_ms:.6f}",
                    f"{record.throughput_pct:.3f}",
                    f"{record.global_loads:.3f}",
                    key=record.key,
                )

        @work(thread=True, exclusive=True)
        def collect_live(self) -> None:
            samples, error = read_live_samples(self.gpus)
            self.call_from_thread(self._apply_live_samples, samples, error)

        def _apply_live_samples(self, samples: list[GPUSample], error: str | None) -> None:
            if error:
                self._set_status(error)
                return
            for sample in samples:
                history = self.histories[sample.info.uuid]
                history["sm"].append(float(sample.sm_util_pct))
                history["mem"].append(float(sample.mem_util_pct))
                history["power"].append(sample.power_w)
                history["temp"].append(float(sample.temp_c))
                self._render_gpu_summary(sample)
                self._render_gpu_processes(sample)
            self._set_status(f"Live telemetry refreshed at {utc_now()}")

        @work(thread=True, exclusive=True)
        def run_profile(self, command: str) -> None:
            self.call_from_thread(self._set_status, f"Profiling: {command}")
            result = profile_command(command, self.ncu_path)
            self.call_from_thread(self._apply_profile_result, result)

        def _apply_profile_result(self, result: SnapshotRun) -> None:
            self.last_snapshot = result
            self.kernel_records = sorted(result.records, key=lambda row: row.total_time_ms, reverse=True)
            self._render_kernel_records()
            if result.returncode == 0 and result.records:
                self._set_status(
                    f"Snapshot complete: {len(result.records)} kernels from `{result.command}` | profiler={shell_join(result.profiler_command)}"
                )
            elif result.returncode == 0:
                self._set_status("Snapshot finished but no kernels were captured. Check permissions or command shape.")
            else:
                message = result.stderr.strip() or "ncu profiling failed"
                self._set_status(message)

        def action_refresh_now(self) -> None:
            self.collect_live()

        def action_run_command(self) -> None:
            command = self.query_one("#command-input", Input).value.strip()
            if not command:
                self._set_status("Command runner is empty")
                return
            self.history = append_history(self.history_path, self.history, command)
            self.history_cursor = len(self.history)
            self.run_profile(command)

        def action_history_prev(self) -> None:
            if not self.history:
                return
            self.history_cursor = max(0, self.history_cursor - 1)
            self.query_one("#command-input", Input).value = self.history[self.history_cursor]

        def action_history_next(self) -> None:
            if not self.history:
                return
            self.history_cursor = min(len(self.history), self.history_cursor + 1)
            if self.history_cursor >= len(self.history):
                self.query_one("#command-input", Input).value = ""
            else:
                self.query_one("#command-input", Input).value = self.history[self.history_cursor]

        def action_export_csv(self) -> None:
            if not self.kernel_records:
                self._set_status("Kernel grid is empty; nothing to export")
                return
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = self.export_dir / f"kernel-grid-{stamp}.csv"
            export_records_csv(out_path, self.kernel_records)
            self._set_status(f"Exported kernel grid to {out_path}")

        def action_toggle_theme(self) -> None:
            self.theme_name = "monokai" if self.theme_name == "dracula" else "dracula"
            self._apply_theme()
            self._set_status(f"Theme switched to {self.theme_name}")

        def action_sort_longest(self) -> None:
            self.kernel_records.sort(key=lambda row: row.total_time_ms, reverse=True)
            self._render_kernel_records()
            self._set_status("Kernel grid sorted by longest running")

        def action_sort_throughput(self) -> None:
            self.kernel_records.sort(key=lambda row: row.throughput_pct, reverse=True)
            self._render_kernel_records()
            self._set_status("Kernel grid sorted by highest throughput")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "profile-button":
                self.action_run_command()
            elif event.button.id == "theme-button":
                self.action_toggle_theme()
            elif event.button.id == "export-button":
                self.action_export_csv()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "command-input":
                self.action_run_command()

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            if event.data_table.id != "kernel-grid":
                return
            row_key = str(event.row_key.value) if event.row_key is not None else ""
            record = next((item for item in self.kernel_records if item.key == row_key), None)
            if record is not None:
                self.push_screen(KernelDetailModal(record, self.theme_name))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nexa Insight: live GPU telemetry and snapshot profiler")
    parser.add_argument("--refresh", type=float, default=DEFAULT_REFRESH_HZ, help="telemetry refresh frequency in Hz")
    parser.add_argument("--theme", default="dracula", choices=sorted(THEMES))
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--ncu-path", default="ncu")
    return parser.parse_args(argv)


def ensure_dependencies() -> None:
    if not TEXTUAL_AVAILABLE:
        detail = f" ({TEXTUAL_IMPORT_ERROR})" if TEXTUAL_IMPORT_ERROR else ""
        raise SystemExit(f"Textual is required to run Nexa Insight{detail}")
    if pynvml is None:
        raise SystemExit("nvidia-ml-py is required to run Nexa Insight")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_dependencies()
    resolved_ncu = resolve_tool(args.ncu_path)
    if resolved_ncu is None:
        raise SystemExit(
            "Nsight Compute CLI `ncu` is required. Install the CUDA toolkit or pass --ncu-path /path/to/ncu."
        )
    app = NexaInsightApp(
        refresh_seconds=max(0.5, 1.0 / max(args.refresh, 0.1)),
        history_path=args.history_file,
        export_dir=args.export_dir,
        theme_name=args.theme,
        ncu_path=resolved_ncu,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
