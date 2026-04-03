#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from train_sft_lib import extract_prompt_response, init_dist, parse_bool, pick_device, tokenize_supervised_example


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight for Qwen/OpenCodeInstruct supervised fine-tuning")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--dataset_name", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--dataset_streaming", type=parse_bool, default=False)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="auto")
    return parser.parse_args()


def main() -> int:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    dctx = init_dist("fsdp", args.backend)
    rank0 = dctx.rank == 0
    device = pick_device(dctx)
    hf_token = resolve_hf_token()

    try:
        tensor = torch.ones(1, device=device)
        if dctx.enabled:
            dist.all_reduce(tensor)
        if rank0:
            print("NCCL communication OK")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = load_dataset(
            args.dataset_name,
            args.dataset_config or None,
            split=args.train_split,
            streaming=args.dataset_streaming,
            token=hf_token,
        )
        if args.dataset_streaming:
            rows = list(ds.take(args.max_samples))
        else:
            rows = ds.select(range(min(args.max_samples, len(ds))))

        prompt, response = extract_prompt_response(rows[0])
        tokenized = tokenize_supervised_example(tokenizer, prompt, response, args.seq_length)
        batch = {
            key: torch.tensor(value, device=device).unsqueeze(0)
            for key, value in tokenized.items()
        }

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=hf_token,
        ).to(device)
        model.config.use_cache = False
        out = model(**batch)
        loss = out.loss
        loss.backward()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if rank0:
            print("Preflight check passed.")
    finally:
        if dctx.enabled and dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
