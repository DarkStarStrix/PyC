#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

try:
    from tqdm import tqdm
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"tqdm is required: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal tqdm probe for tmux/TTY validation")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--delay", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[tqdm-probe] tty_stdout={sys.stdout.isatty()} tty_stderr={sys.stderr.isatty()} steps={args.steps}", flush=True)
    bar = tqdm(
        total=args.steps,
        desc="tqdm-probe",
        unit="step",
        file=sys.stderr,
        ascii=" ▏▎▍▌▋▊▉█",
        dynamic_ncols=True,
        mininterval=0.1,
        miniters=1,
        leave=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    try:
        for index in range(args.steps):
            time.sleep(args.delay)
            bar.set_postfix_str(f"step={index + 1}/{args.steps}")
            bar.update(1)
    finally:
        bar.close()
    print("[tqdm-probe] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
