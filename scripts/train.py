#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import socket
import subprocess
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"PyTorch is required for scripts/train.py: {exc}")

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {raw}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, math.ceil(0.95 * len(arr)) - 1))
    return float(arr[idx])


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return default


@dataclass
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
        with self.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
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
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2.0))

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
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 7:
                            rows.append([ts, "0"] + parts[:7])
            except Exception:  # noqa: BLE001
                rows = []

            if rows:
                with self.out_csv.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
            self._stop.wait(self.interval_sec)


def choose_backend(arg_backend: str) -> str:
    if arg_backend == "auto":
        if torch.cuda.is_available():
            return "nccl"
        return "gloo"
    if arg_backend == "rccl":
        return "nccl"
    return arg_backend


def init_dist(args: argparse.Namespace) -> DistCtx:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist_mode = args.dist
    if dist_mode == "none":
        enabled = False
    else:
        enabled = world_size > 1

    backend = choose_backend(args.backend)

    if enabled:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        try:
            # Newer torch can lock rank->device mapping here, which removes NCCL mapping warnings.
            dist.init_process_group(backend=backend, device_id=torch.device("cuda", local_rank) if torch.cuda.is_available() else None)
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


def infer_text_column(cols: list[str]) -> str:
    for candidate in ("text", "sentence", "content", "review", "article"):
        if candidate in cols:
            return candidate
    return cols[0]


def infer_label_column(cols: list[str]) -> str:
    for candidate in ("label", "labels", "target"):
        if candidate in cols:
            return candidate
    return cols[-1]


def render_summary_svg(metrics: dict[str, Any], out_svg: Path) -> None:
    width, height = 980, 380
    vals = [
        ("samples/s", safe_float(metrics.get("samples_per_sec"))),
        ("steps/s", safe_float(metrics.get("steps_per_sec"))),
        ("tokens/s", safe_float(metrics.get("tokens_per_sec"))),
        ("GPU util mean", safe_float(metrics.get("gpu_util_mean"))),
    ]
    vmax = max(1.0, max(v for _, v in vals))
    left = 250
    bar_space = width - left - 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="24" y="38" font-size="24" font-family="Arial, sans-serif" fill="#0f172a">PYC Distributed Training Summary</text>',
        f'<text x="24" y="60" font-size="12" font-family="Arial, sans-serif" fill="#334155">run_id={metrics.get("run_id","unknown")} mode={metrics.get("mode")} dist={metrics.get("dist")} backend={metrics.get("backend")}</text>',
    ]
    for i, (label, value) in enumerate(vals):
        y = 95 + i * 60
        w = int((value / vmax) * bar_space)
        parts.append(f'<text x="24" y="{y+20}" font-size="14" font-family="Arial, sans-serif" fill="#0f172a">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w}" height="24" rx="4" fill="#0ea5e9"/>')
        parts.append(f'<text x="{left + max(8, w + 8)}" y="{y+17}" font-size="12" font-family="Arial, sans-serif" fill="#1f2937">{value:.4f}</text>')
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agnostic training entrypoint for local/single-node/multi-node runs")
    parser.add_argument("--mode", choices=["conventional", "nexa_vortex"], default="conventional")
    parser.add_argument("--dist", choices=["none", "ddp"], default="ddp")
    parser.add_argument("--backend", choices=["auto", "nccl", "rccl", "mpi", "gloo"], default="auto")

    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--dataset-name", default="ag_news")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--max-train-samples", type=int, default=8192)
    parser.add_argument("--max-eval-samples", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)

    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--dataloader-timeout-sec", type=int, default=120)
    parser.add_argument("--pin-memory", type=parse_bool, default=True)
    parser.add_argument("--persistent-workers", type=parse_bool, default=True)
    parser.add_argument("--non-blocking-h2d", type=parse_bool, default=True)

    parser.add_argument("--torch-compile", choices=["none", "default", "reduce-overhead", "max-autotune"], default="none")
    parser.add_argument("--compile-cache-dir", default="")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-root", default="benchmark/benchmarks/results/remote_results/local")
    parser.add_argument("--telemetry-interval-sec", type=float, default=2.0)
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"datasets is required for scripts/train.py: {exc}")
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"transformers is required for scripts/train.py: {exc}")

    dctx = init_dist(args)
    rank0 = dctx.rank == 0

    run_id = args.run_id or utc_now()
    out_dir = Path(args.out_root).resolve() / run_id
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if dctx.enabled:
        dist.barrier()

    if args.compile_cache_dir:
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", args.compile_cache_dir)

    torch.manual_seed(args.seed + dctx.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + dctx.rank)

    device = pick_device(dctx)

    telemetry = TelemetryWriter(
        out_csv=out_dir / "gpu_telemetry.csv",
        interval_sec=args.telemetry_interval_sec,
        enabled=rank0,
    )
    if rank0:
        telemetry.start()

    start_ts = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()

    ds_cfg = args.dataset_config if args.dataset_config else None
    ds = load_dataset(args.dataset_name, ds_cfg)

    train_ds = ds["train"]
    eval_key = "test" if "test" in ds else "validation" if "validation" in ds else "train"
    eval_ds = ds[eval_key]

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    text_col = infer_text_column(train_ds.column_names)
    label_col = infer_label_column(train_ds.column_names)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tok_fn(batch: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenizer(batch[text_col], truncation=True, max_length=args.max_length)
        tokens["labels"] = batch[label_col]
        return tokens

    remove_cols = [c for c in train_ds.column_names if c not in {"labels"}]
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=remove_cols)
    eval_ds = eval_ds.map(tok_fn, batched=True, remove_columns=remove_cols)

    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    num_labels = None
    label_feat = train_ds.features.get("labels")
    if hasattr(label_feat, "num_classes") and getattr(label_feat, "num_classes", None):
        num_labels = int(label_feat.num_classes)
    if not num_labels:
        # Robust fallback for plain integer labels.
        max_label = 0
        sample_n = min(len(train_ds), 20000)
        for i in range(sample_n):
            v = train_ds[i]["labels"]
            iv = int(v.item()) if hasattr(v, "item") else int(v)
            if iv > max_label:
                max_label = iv
        num_labels = max_label + 1

    suppress = io.StringIO()
    with redirect_stdout(suppress), redirect_stderr(suppress):
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.to(device)

    if args.torch_compile != "none" and hasattr(torch, "compile"):
        compile_mode = None if args.torch_compile == "default" else args.torch_compile
        model = torch.compile(model, mode=compile_mode)

    if dctx.enabled:
        ddp_kwargs: dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [dctx.local_rank]
            ddp_kwargs["output_device"] = dctx.local_rank
        model = DDP(model, **ddp_kwargs)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_sampler = DistributedSampler(train_ds, num_replicas=dctx.world_size, rank=dctx.rank, shuffle=True) if dctx.enabled else None
    eval_sampler = DistributedSampler(eval_ds, num_replicas=dctx.world_size, rank=dctx.rank, shuffle=False) if dctx.enabled else None

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.per_device_batch,
        "collate_fn": collator,
        "num_workers": args.dataloader_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers if args.dataloader_workers > 0 else False,
        "timeout": max(0, args.dataloader_timeout_sec) if args.dataloader_workers > 0 else 0,
        "sampler": train_sampler,
        "shuffle": train_sampler is None,
    }
    if args.dataloader_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        # Use spawn workers to reduce fork-related deadlocks in long DDP runs.
        loader_kwargs["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_ds, **loader_kwargs)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.per_device_batch,
        collate_fn=collator,
        num_workers=max(0, min(2, args.dataloader_workers)),
        pin_memory=args.pin_memory,
        sampler=eval_sampler,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_opt_steps = max(1, int((len(train_loader) * max(1.0, args.epochs)) / max(1, args.grad_accum)))
    warmup_steps = int(total_opt_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    use_amp = args.mixed_precision in {"fp16", "bf16"} and device.type == "cuda"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision == "fp16" and device.type == "cuda"))

    step_target = max(1, int(len(train_loader) * max(0.0, args.epochs)))
    do_progress = args.progress == "on" or (args.progress == "auto" and rank0)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_curve: list[float] = []
    h2d_times: list[float] = []
    compute_times: list[float] = []
    comm_times: list[float] = []
    idle_gaps: list[float] = []
    backward_nosync_ms: list[float] = []

    global_step = 0
    local_samples = 0
    local_tokens = 0

    progress = tqdm(total=step_target, desc="train", unit="step") if (do_progress and tqdm is not None) else None

    prev_iter_end = time.perf_counter()
    train_iter = iter(train_loader)

    while global_step < step_target:
        if dctx.enabled and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(global_step // max(1, len(train_loader)))
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        iter_start = time.perf_counter()
        idle_gaps.append(max(0.0, (iter_start - prev_iter_end) * 1000.0))

        h2d_start = time.perf_counter()
        moved: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            moved[k] = v.to(device, non_blocking=args.non_blocking_h2d)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        h2d_end = time.perf_counter()
        h2d_times.append((h2d_end - h2d_start) * 1000.0)

        should_sync = ((global_step + 1) % max(1, args.grad_accum)) == 0
        sync_ctx = nullcontext()
        if dctx.enabled and isinstance(model, DDP) and not should_sync:
            sync_ctx = model.no_sync()

        compute_start = time.perf_counter()
        with sync_ctx:
            ac = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
            with ac:
                out = model(**moved)
                loss = out.loss / max(1, args.grad_accum)

            bw_start = time.perf_counter()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            bw_end = time.perf_counter()

        if should_sync:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_end = time.perf_counter()

        bw_ms = (bw_end - bw_start) * 1000.0
        compute_ms = (compute_end - compute_start) * 1000.0
        compute_times.append(compute_ms)

        if dctx.enabled and should_sync and backward_nosync_ms:
            baseline = sum(backward_nosync_ms) / len(backward_nosync_ms)
            comm_times.append(max(0.0, bw_ms - baseline))
        else:
            comm_times.append(0.0)

        if dctx.enabled and not should_sync:
            backward_nosync_ms.append(bw_ms)
            if len(backward_nosync_ms) > 64:
                backward_nosync_ms = backward_nosync_ms[-64:]

        bs = int(moved["labels"].shape[0]) if "labels" in moved else args.per_device_batch
        local_samples += bs
        if "input_ids" in moved:
            local_tokens += int(moved["input_ids"].numel())

        loss_curve.append(float(loss.detach().item() * max(1, args.grad_accum)))
        global_step += 1
        prev_iter_end = time.perf_counter()

        if progress is not None:
            progress.update(1)
            progress.set_postfix(loss=f"{loss_curve[-1]:.4f}")

    if progress is not None:
        progress.close()

    # Lightweight eval pass.
    model.eval()
    eval_loss_sum = 0.0
    eval_steps = 0
    with torch.no_grad():
        for batch in eval_loader:
            moved = {k: v.to(device, non_blocking=args.non_blocking_h2d) for k, v in batch.items()}
            ac = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
            with ac:
                out = model(**moved)
                eval_loss_sum += float(out.loss.detach().item())
                eval_steps += 1

    end_ts = time.perf_counter()
    runtime_sec = end_ts - start_ts

    # Aggregate cardinal metrics across ranks.
    total_samples = local_samples
    total_steps = global_step
    total_tokens = local_tokens
    if dctx.enabled:
        tensor = torch.tensor([local_samples, global_step, local_tokens], dtype=torch.long, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_samples = int(tensor[0].item())
        total_steps = int(tensor[1].item())
        total_tokens = int(tensor[2].item())

    if rank0:
        telemetry.stop()

    if dctx.enabled:
        dist.barrier()

    if not rank0:
        cleanup_dist(dctx)
        return 0

    gpu_util: list[float] = []
    if (out_dir / "gpu_telemetry.csv").exists():
        with (out_dir / "gpu_telemetry.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gpu_util.append(safe_float(r.get("util_gpu", 0.0), 0.0))

    metrics = {
        "run_id": run_id,
        "timestamp_utc": start_iso,
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "mode": args.mode,
        "dist": args.dist,
        "backend": dctx.backend,
        "world_size": dctx.world_size,
        "rank": dctx.rank,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_runtime_sec": round(runtime_sec, 4),
        "samples_per_sec": round(total_samples / max(runtime_sec, 1e-9), 4),
        "steps_per_sec": round(total_steps / max(runtime_sec, 1e-9), 4),
        "tokens_per_sec": round(total_tokens / max(runtime_sec, 1e-9), 4),
        "loss_final": round(loss_curve[-1], 6) if loss_curve else 0.0,
        "loss_curve": [round(x, 6) for x in loss_curve[-256:]],
        "eval_loss": round(eval_loss_sum / max(eval_steps, 1), 6),
        "gpu_util_mean": round(sum(gpu_util) / len(gpu_util), 4) if gpu_util else 0.0,
        "gpu_util_p95": round(p95(gpu_util), 4) if gpu_util else 0.0,
        "h2d_time_ms_mean": round(sum(h2d_times) / len(h2d_times), 4) if h2d_times else 0.0,
        "compute_time_ms_mean": round(sum(compute_times) / len(compute_times), 4) if compute_times else 0.0,
        "comm_time_ms_mean": round(sum(comm_times) / len(comm_times), 4) if comm_times else 0.0,
        "idle_gap_ms_mean": round(sum(idle_gaps) / len(idle_gaps), 4) if idle_gaps else 0.0,
        "idle_gap_ms_p95": round(p95(idle_gaps), 4) if idle_gaps else 0.0,
    }

    run_config = {
        "argv": os.sys.argv,
        "args": vars(args),
        "env_subset": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", ""),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", ""),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", ""),
            "TRANSFORMERS_VERBOSITY": os.environ.get("TRANSFORMERS_VERBOSITY", ""),
        },
    }

    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Training Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Mode: `{args.mode}`",
        f"- Dist: `{args.dist}`",
        f"- Backend: `{dctx.backend}`",
        f"- World size: `{dctx.world_size}`",
        f"- Model: `{args.model_name}`",
        f"- Dataset: `{args.dataset_name}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    key_order = [
        "train_runtime_sec",
        "samples_per_sec",
        "steps_per_sec",
        "tokens_per_sec",
        "loss_final",
        "eval_loss",
        "gpu_util_mean",
        "gpu_util_p95",
        "h2d_time_ms_mean",
        "compute_time_ms_mean",
        "comm_time_ms_mean",
        "idle_gap_ms_mean",
        "idle_gap_ms_p95",
    ]
    for k in key_order:
        md_lines.append(f"| {k} | {metrics.get(k)} |")
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    render_summary_svg(metrics, out_dir / "summary.svg")

    print(f"[train] wrote {(out_dir / 'run_config.json')}")
    print(f"[train] wrote {(out_dir / 'train_metrics.json')}")
    print(f"[train] wrote {(out_dir / 'summary.md')}")
    print(f"[train] wrote {(out_dir / 'summary.svg')}")

    cleanup_dist(dctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
