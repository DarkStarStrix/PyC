#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import IterableDataset as HFIterableDataset
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from train_sft_lib import (
    TelemetryWriter,
    cleanup_dist,
    extract_prompt_response,
    init_dist,
    parse_bool,
    pick_device,
    resolve_transformer_layer_cls,
    summarize_run,
    tokenize_supervised_example,
    utc_now,
)


def sft_collate_fn(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    keys = rows[0].keys() if rows else []
    for key in keys:
        values = [row[key] for row in rows]
        first = values[0]
        if isinstance(first, torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = torch.tensor(values, dtype=torch.long)
    return batch


def preprocess_sft_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    seq_length: int,
) -> dict[str, list[list[int]]]:
    rows = []
    batch_size = len(next(iter(batch.values()))) if batch else 0
    for index in range(batch_size):
        example = {key: value[index] for key, value in batch.items()}
        prompt, response = extract_prompt_response(example)
        rows.append(tokenize_supervised_example(tokenizer, prompt, response, seq_length))
    return {
        "input_ids": [row["input_ids"] for row in rows],
        "attention_mask": [row["attention_mask"] for row in rows],
        "labels": [row["labels"] for row in rows],
    }


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed supervised fine-tuning for causal LMs")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--dataset_name", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--dataset_streaming", type=parse_bool, default=False)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--lr_scheduler", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--dist", choices=["none", "ddp", "fsdp"], default="fsdp")
    parser.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="auto")
    parser.add_argument("--fsdp", nargs="*", default=["full_shard", "auto_wrap"])
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", default="Qwen2DecoderLayer")
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=parse_bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--preprocessing_workers", type=int, default=0)
    parser.add_argument("--preprocessing_batch_size", type=int, default=64)
    parser.add_argument("--compile", default="false")
    parser.add_argument("--telemetry_interval_sec", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs/qwen14b_codeinstruct")
    parser.add_argument("--run_name", default="qwen14b_codeinstruct")
    parser.add_argument("--save_total_limit", type=int, default=2)
    return parser.parse_args()


def maybe_limit_dataset(dataset: Any, max_items: int) -> Any:
    if max_items <= 0:
        return dataset
    if isinstance(dataset, HFIterableDataset):
        return dataset.take(max_items)
    if hasattr(dataset, "select"):
        return dataset.select(range(min(max_items, len(dataset))))
    return dataset


def maybe_shard_dataset(dataset: Any, dctx: Any) -> Any:
    if not dctx.enabled:
        return dataset
    if isinstance(dataset, HFIterableDataset):
        return dataset.shard(num_shards=dctx.world_size, index=dctx.rank)
    if hasattr(dataset, "shard"):
        return dataset.shard(num_shards=dctx.world_size, index=dctx.rank)
    return dataset


def prepare_train_dataset(raw_dataset: Any, tokenizer: Any, args: argparse.Namespace, dctx: Any) -> Any:
    dataset = maybe_shard_dataset(raw_dataset, dctx)
    dataset = maybe_limit_dataset(dataset, args.max_train_samples)
    preprocess = partial(preprocess_sft_batch, tokenizer=tokenizer, seq_length=args.seq_length)

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "batch_size": max(1, args.preprocessing_batch_size),
    }
    if not isinstance(dataset, HFIterableDataset):
        map_kwargs["desc"] = "tokenizing train split"
    if hasattr(dataset, "column_names"):
        map_kwargs["remove_columns"] = dataset.column_names
    if not isinstance(dataset, HFIterableDataset) and args.preprocessing_workers > 1:
        map_kwargs["num_proc"] = args.preprocessing_workers
    mapped = dataset.map(preprocess, **map_kwargs)
    if hasattr(mapped, "set_format"):
        mapped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return mapped


def prepare_eval_dataset(raw_dataset: Any, tokenizer: Any, args: argparse.Namespace, dctx: Any) -> Any:
    dataset = maybe_shard_dataset(raw_dataset, dctx)
    dataset = maybe_limit_dataset(dataset, args.max_eval_samples)
    preprocess = partial(preprocess_sft_batch, tokenizer=tokenizer, seq_length=args.seq_length)

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "batch_size": max(1, args.preprocessing_batch_size),
    }
    if not isinstance(dataset, HFIterableDataset):
        map_kwargs["desc"] = "tokenizing eval split"
    if hasattr(dataset, "column_names"):
        map_kwargs["remove_columns"] = dataset.column_names
    if not isinstance(dataset, HFIterableDataset) and args.preprocessing_workers > 1:
        map_kwargs["num_proc"] = min(args.preprocessing_workers, 8)
    mapped = dataset.map(preprocess, **map_kwargs)
    if hasattr(mapped, "set_format"):
        mapped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return mapped


def build_loader(dataset: Any, args: argparse.Namespace, dctx: Any, shuffle: bool) -> DataLoader:
    sampler = None
    is_iterable = isinstance(dataset, HFIterableDataset)
    if not is_iterable and hasattr(dataset, "__len__") and dctx.enabled:
        sampler = DistributedSampler(dataset, num_replicas=dctx.world_size, rank=dctx.rank, shuffle=shuffle)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.per_device_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": sft_collate_fn,
        "sampler": sampler,
        "shuffle": sampler is None and shuffle and not is_iterable and hasattr(dataset, "__len__"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **loader_kwargs)


def resolve_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    if args.lr_scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )


def evaluate(model: Any, eval_loader: DataLoader | None, device: torch.device, precision: str, non_blocking: bool) -> float | None:
    if eval_loader is None:
        return None
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model.eval()
    loss_sum = 0.0
    steps = 0
    with torch.no_grad():
        for batch in eval_loader:
            moved = {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
            ac = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
            with ac:
                out = model(**moved)
                loss_sum += float(out.loss.detach().item())
                steps += 1
    model.train()
    return loss_sum / max(steps, 1)


def write_live_metrics(
    *,
    out_dir: Path,
    run_id: str,
    start_iso: str,
    runtime_sec: float,
    total_samples: int,
    total_tokens: int,
    optimizer_steps: int,
    loss_curve: list[float],
    h2d_times: list[float],
    compute_times: list[float],
    comm_times: list[float],
    idle_gaps: list[float],
    extra_metrics: dict[str, Any],
) -> None:
    summarize_run(
        out_dir=out_dir,
        run_id=run_id,
        start_iso=start_iso,
        runtime_sec=runtime_sec,
        total_samples=total_samples,
        total_tokens=total_tokens,
        optimizer_steps=optimizer_steps,
        loss_curve=loss_curve,
        eval_loss=None,
        h2d_times=h2d_times,
        compute_times=compute_times,
        comm_times=comm_times,
        idle_gaps=idle_gaps,
        extra_metrics=extra_metrics,
    )


def cleanup_old_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if len(checkpoints) <= keep:
        return
    for path in checkpoints[:-keep]:
        if path.is_dir():
            for subpath in sorted(path.rglob("*"), reverse=True):
                if subpath.is_file() or subpath.is_symlink():
                    subpath.unlink()
                elif subpath.is_dir():
                    subpath.rmdir()
            path.rmdir()


def save_checkpoint(model: Any, tokenizer: Any, out_dir: Path, optimizer_step: int, dctx: Any) -> None:
    checkpoint_dir = out_dir / f"checkpoint-{optimizer_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state = model.state_dict()
        if dctx.rank == 0:
            model.module.save_pretrained(checkpoint_dir, state_dict=state, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
    elif dctx.rank == 0:
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)


def main() -> int:
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = resolve_hf_token()

    dctx = init_dist(args.dist, args.backend)
    rank0 = dctx.rank == 0
    torch.manual_seed(args.seed + dctx.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + dctx.rank)

    device = pick_device(dctx)
    run_id = args.run_name or utc_now()
    out_dir = Path(args.output_dir).resolve() / run_id
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if dctx.enabled:
        dist.barrier()

    telemetry = TelemetryWriter(
        out_csv=out_dir / "gpu_telemetry.csv",
        interval_sec=args.telemetry_interval_sec,
        enabled=rank0,
    )
    if rank0:
        telemetry.start()

    start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_ts = time.perf_counter()

    ds_config = args.dataset_config or None
    raw_train = load_dataset(
        args.dataset_name,
        ds_config,
        split=args.train_split,
        streaming=args.dataset_streaming,
        token=hf_token,
    )
    eval_split = args.eval_split
    raw_eval = None
    if eval_split:
        raw_eval = load_dataset(args.dataset_name, ds_config, split=eval_split, streaming=False)
    elif not args.dataset_streaming:
        try:
            raw_eval = load_dataset(args.dataset_name, ds_config, split="validation", streaming=False, token=hf_token)
        except Exception:
            try:
                raw_eval = load_dataset(args.dataset_name, ds_config, split="test", streaming=False, token=hf_token)
            except Exception:
                raw_eval = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = prepare_train_dataset(raw_train, tokenizer, args, dctx)
    eval_dataset = prepare_eval_dataset(raw_eval, tokenizer, args, dctx) if raw_eval is not None else None
    train_loader = build_loader(train_dataset, args, dctx, shuffle=not args.dataset_streaming)
    eval_loader = build_loader(eval_dataset, args, dctx, shuffle=False) if eval_dataset is not None else None

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=hf_token,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    layer_cls = resolve_transformer_layer_cls(model, args.fsdp_transformer_layer_cls_to_wrap)

    if args.gradient_checkpointing:
        wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper,
            check_fn=lambda module: isinstance(module, layer_cls),
        )

    if args.dist == "fsdp" and dctx.enabled:
        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
    else:
        model.to(device)
        if args.dist == "ddp" and dctx.enabled:
            ddp_kwargs: dict[str, Any] = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [dctx.local_rank]
                ddp_kwargs["output_device"] = dctx.local_rank
            model = DDP(model, **ddp_kwargs)

    if args.compile != "false" and hasattr(torch, "compile"):
        compile_mode = None if args.compile == "default" else args.compile
        model = torch.compile(model, mode=compile_mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = resolve_scheduler(optimizer, args)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"))

    loss_curve: list[float] = []
    h2d_times: list[float] = []
    compute_times: list[float] = []
    comm_times: list[float] = []
    idle_gaps: list[float] = []
    backward_nosync_ms: list[float] = []

    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    optimizer_step = 0
    micro_step = 0
    total_local_samples = 0
    total_local_tokens = 0
    prev_iter_end = time.perf_counter()

    try:
        while optimizer_step < args.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            iter_start = time.perf_counter()
            idle_gaps.append(max(0.0, (iter_start - prev_iter_end) * 1000.0))

            h2d_start = time.perf_counter()
            moved = {key: value.to(device, non_blocking=args.pin_memory) for key, value in batch.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_end = time.perf_counter()
            h2d_times.append((h2d_end - h2d_start) * 1000.0)

            should_sync = ((micro_step + 1) % max(1, args.gradient_accumulation_steps)) == 0
            sync_ctx = nullcontext()
            if dctx.enabled and hasattr(model, "no_sync") and not should_sync:
                sync_ctx = model.no_sync()

            compute_start = time.perf_counter()
            with sync_ctx:
                ac = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
                with ac:
                    out = model(**moved)
                    loss = out.loss / max(1, args.gradient_accumulation_steps)

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
                optimizer_step += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_end = time.perf_counter()

            bw_ms = (bw_end - bw_start) * 1000.0
            compute_times.append((compute_end - compute_start) * 1000.0)
            if dctx.enabled and should_sync and backward_nosync_ms:
                baseline = sum(backward_nosync_ms) / len(backward_nosync_ms)
                comm_times.append(max(0.0, bw_ms - baseline))
            else:
                comm_times.append(0.0)
            if dctx.enabled and not should_sync:
                backward_nosync_ms.append(bw_ms)
                if len(backward_nosync_ms) > 64:
                    backward_nosync_ms = backward_nosync_ms[-64:]

            total_local_samples += int(moved["input_ids"].shape[0])
            total_local_tokens += int(moved["attention_mask"].sum().item())
            loss_curve.append(float(loss.detach().item() * max(1, args.gradient_accumulation_steps)))
            micro_step += 1
            prev_iter_end = time.perf_counter()

            if rank0 and optimizer_step > 0 and should_sync and optimizer_step % args.logging_steps == 0:
                runtime_sec = time.perf_counter() - start_ts
                write_live_metrics(
                    out_dir=out_dir,
                    run_id=run_id,
                    start_iso=start_iso,
                    runtime_sec=runtime_sec,
                    total_samples=total_local_samples,
                    total_tokens=total_local_tokens,
                    optimizer_steps=optimizer_step,
                    loss_curve=loss_curve,
                    h2d_times=h2d_times,
                    compute_times=compute_times,
                    comm_times=comm_times,
                    idle_gaps=idle_gaps,
                    extra_metrics={
                        "mode": "sft",
                        "dist": args.dist,
                        "backend": dctx.backend,
                        "world_size": dctx.world_size,
                        "model_name": args.model_name,
                        "dataset_name": args.dataset_name,
                        "dataset_config": args.dataset_config,
                        "precision": args.precision,
                        "seq_length": args.seq_length,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    },
                )
                print(
                    f"[train_sft] step={optimizer_step}/{args.max_steps} "
                    f"loss={loss_curve[-1]:.4f} tokens={total_local_tokens}"
                    f" samples_per_sec={total_local_samples/max(runtime_sec,1e-9):.2f}"
                    f" tokens_per_sec={total_local_tokens/max(runtime_sec,1e-9):.2f}",
                    flush=True,
                )

            if should_sync and args.eval_steps > 0 and optimizer_step > 0 and optimizer_step % args.eval_steps == 0:
                eval_loss = evaluate(model, eval_loader, device, args.precision, args.pin_memory)
                if rank0:
                    print(f"[train_sft] eval step={optimizer_step} loss={eval_loss if eval_loss is not None else 'n/a'}", flush=True)

            if should_sync and args.save_steps > 0 and optimizer_step > 0 and optimizer_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, out_dir, optimizer_step, dctx)
                if rank0:
                    cleanup_old_checkpoints(out_dir, args.save_total_limit)
    finally:
        runtime_sec = time.perf_counter() - start_ts
        eval_loss = evaluate(model, eval_loader, device, args.precision, args.pin_memory)

        total_samples = total_local_samples
        total_tokens = total_local_tokens
        if dctx.enabled:
            tensor = torch.tensor([total_local_samples, total_local_tokens], dtype=torch.long, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_samples = int(tensor[0].item())
            total_tokens = int(tensor[1].item())

        if rank0:
            telemetry.stop()
            config = {
                "argv": os.sys.argv,
                "args": vars(args),
                "dist_backend": dctx.backend,
                "world_size": dctx.world_size,
            }
            (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            summarize_run(
                out_dir=out_dir,
                run_id=run_id,
                start_iso=start_iso,
                runtime_sec=runtime_sec,
                total_samples=total_samples,
                total_tokens=total_tokens,
                optimizer_steps=optimizer_step,
                loss_curve=loss_curve,
                eval_loss=eval_loss,
                h2d_times=h2d_times,
                compute_times=compute_times,
                comm_times=comm_times,
                idle_gaps=idle_gaps,
                extra_metrics={
                    "mode": "sft",
                    "dist": args.dist,
                    "backend": dctx.backend,
                    "world_size": dctx.world_size,
                    "model_name": args.model_name,
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "precision": args.precision,
                    "seq_length": args.seq_length,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                },
            )
            print(f"[train_sft] wrote {out_dir / 'run_config.json'}")
            print(f"[train_sft] wrote {out_dir / 'train_metrics.json'}")
        cleanup_dist(dctx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
