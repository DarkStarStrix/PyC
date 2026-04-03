#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


@dataclass(slots=True)
class DistCtx:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    backend: str


class TelemetryWriter:
    def __init__(self, out_csv: Path, interval_sec: float, enabled: bool) -> None:
        self.out_csv = out_csv
        self.interval_sec = interval_sec
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "ts_utc",
                    "rank",
                    "gpu_id",
                    "util_gpu",
                    "util_mem",
                    "mem_used_mb",
                    "mem_total_mb",
                    "power_w",
                    "temp_c",
                ]
            )
        self._thread = threading.Thread(target=self._loop, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_sec * 2.0))

    def _loop(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            ts = datetime.now(timezone.utc).isoformat()
            rows: list[list[str]] = []
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode == 0:
                    for line in proc.stdout.strip().splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) >= 7:
                            rows.append([ts, "0"] + parts[:7])
            except Exception:
                rows = []

            if rows:
                with self.out_csv.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerows(rows)
            self._stop.wait(self.interval_sec)


def choose_backend(arg_backend: str) -> str:
    if arg_backend == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    if arg_backend == "rccl":
        return "nccl"
    return arg_backend


def init_dist(dist_mode: str, backend_name: str) -> DistCtx:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = dist_mode != "none" and world_size > 1
    backend = choose_backend(backend_name)
    if enabled:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend=backend,
                device_id=torch.device("cuda", local_rank) if torch.cuda.is_available() else None,
            )
        except TypeError:
            dist.init_process_group(backend=backend)
    return DistCtx(enabled=enabled, rank=rank, world_size=world_size, local_rank=local_rank, backend=backend)


def cleanup_dist(dctx: DistCtx) -> None:
    if dctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def pick_device(dctx: DistCtx) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", dctx.local_rank if dctx.enabled else 0)
    return torch.device("cpu")


def resolve_transformer_layer_cls(model: nn.Module, class_name: str) -> type[nn.Module]:
    for module in model.modules():
        if module.__class__.__name__ == class_name:
            return module.__class__
    raise ValueError(f"unable to find transformer layer class {class_name!r} in {model.__class__.__name__}")


def render_chat_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    prompt_chunks: list[str] = []
    response_chunks: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant" and index == len(messages) - 1:
            response_chunks.append(content)
            continue
        label = role.capitalize() if role else "Message"
        prompt_chunks.append(f"{label}:\n{content}")
    if not response_chunks:
        raise ValueError("chat example is missing a terminal assistant response")
    return "\n\n".join(prompt_chunks).strip(), "\n\n".join(response_chunks).strip()


def extract_prompt_response(example: dict[str, Any]) -> tuple[str, str]:
    if "messages" in example and isinstance(example["messages"], list):
        return render_chat_messages(example["messages"])
    if "conversations" in example and isinstance(example["conversations"], list):
        return render_chat_messages(example["conversations"])
    if "instruction" in example and "output" in example:
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        response = str(example.get("output", "")).strip()
        prompt = f"Instruction:\n{instruction}"
        if input_text:
            prompt += f"\n\nInput:\n{input_text}"
        return prompt, response
    if "input" in example and "output" in example:
        return str(example.get("input", "")).strip(), str(example.get("output", "")).strip()
    if "prompt" in example and "completion" in example:
        return str(example.get("prompt", "")).strip(), str(example.get("completion", "")).strip()
    if "question" in example and "answer" in example:
        return str(example.get("question", "")).strip(), str(example.get("answer", "")).strip()
    if "question" in example and "response" in example:
        return str(example.get("question", "")).strip(), str(example.get("response", "")).strip()
    if "text" in example and "response" in example:
        return str(example.get("text", "")).strip(), str(example.get("response", "")).strip()
    keys = ", ".join(sorted(example.keys()))
    raise ValueError(f"unsupported instruction-tuning schema; columns={keys}")


def build_sft_text(prompt: str, response: str) -> tuple[str, str]:
    rendered_prompt = f"### Instruction\n{prompt.strip()}\n\n### Response\n"
    rendered_full = f"{rendered_prompt}{response.strip()}"
    return rendered_prompt, rendered_full


def tokenize_supervised_example(
    tokenizer: Any,
    prompt: str,
    response: str,
    seq_length: int,
) -> dict[str, list[int]]:
    prompt_text, full_text = build_sft_text(prompt, response)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
        full_ids = full_ids + [eos_id]
    if eos_id is not None and (not prompt_ids or prompt_ids[-1] != eos_id):
        prompt_ids = prompt_ids

    input_ids = full_ids[:seq_length]
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)
    prompt_token_count = min(len(prompt_ids), len(labels))
    for index in range(prompt_token_count):
        labels[index] = -100

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define either pad_token_id or eos_token_id")
        pad_id = tokenizer.eos_token_id

    while len(input_ids) < seq_length:
        input_ids.append(pad_id)
        attention_mask.append(0)
        labels.append(-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collect_gpu_utilization(out_dir: Path) -> list[float]:
    values: list[float] = []
    path = out_dir / "gpu_telemetry.csv"
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(safe_float(row.get("util_gpu", 0.0), 0.0))
    return values


def summarize_run(
    *,
    out_dir: Path,
    run_id: str,
    start_iso: str,
    runtime_sec: float,
    total_samples: int,
    total_tokens: int,
    optimizer_steps: int,
    loss_curve: list[float],
    eval_loss: float | None,
    h2d_times: list[float],
    compute_times: list[float],
    comm_times: list[float],
    idle_gaps: list[float],
    extra_metrics: dict[str, Any],
) -> dict[str, Any]:
    gpu_util = collect_gpu_utilization(out_dir)
    metrics = {
        "run_id": run_id,
        "timestamp_utc": start_iso,
        "host": socket.gethostname(),
        "train_runtime_sec": round(runtime_sec, 4),
        "samples_per_sec": round(total_samples / max(runtime_sec, 1e-9), 4),
        "steps_per_sec": round(optimizer_steps / max(runtime_sec, 1e-9), 4),
        "tokens_per_sec": round(total_tokens / max(runtime_sec, 1e-9), 4),
        "loss_final": round(loss_curve[-1], 6) if loss_curve else 0.0,
        "loss_curve": [round(value, 6) for value in loss_curve[-256:]],
        "eval_loss": round(eval_loss, 6) if eval_loss is not None else None,
        "gpu_util_mean": round(sum(gpu_util) / len(gpu_util), 4) if gpu_util else 0.0,
        "gpu_util_p95": round(p95(gpu_util), 4) if gpu_util else 0.0,
        "h2d_time_ms_mean": round(sum(h2d_times) / len(h2d_times), 4) if h2d_times else 0.0,
        "compute_time_ms_mean": round(sum(compute_times) / len(compute_times), 4) if compute_times else 0.0,
        "comm_time_ms_mean": round(sum(comm_times) / len(comm_times), 4) if comm_times else 0.0,
        "idle_gap_ms_mean": round(sum(idle_gaps) / len(idle_gaps), 4) if idle_gaps else 0.0,
        "idle_gap_ms_p95": round(p95(idle_gaps), 4) if idle_gaps else 0.0,
    }
    metrics.update(extra_metrics)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
