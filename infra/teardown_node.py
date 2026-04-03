#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def build_ssh_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-i",
        args.key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(args.port),
        f"{args.user}@{args.host}",
    ]


def ssh(args: argparse.Namespace, remote_script: str, *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = build_ssh_prefix(args) + ["bash", "-s", "--"]
    return subprocess.run(cmd, input=remote_script, text=True, check=True, capture_output=capture_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teardown a remote GPU node and snapshot run artifacts.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--key", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-root", default="/root/PyC")
    parser.add_argument("--session", default="qwen")
    parser.add_argument("--run-subdir", default="runs/qwen14b_codeinstruct/qwen14b_codeinstruct")
    parser.add_argument("--stage-root", default="/tmp")
    parser.add_argument("--hf-repo", default="")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--stop-insight", action="store_true")
    parser.add_argument("--kill-tmux", action="store_true")
    parser.add_argument("--detach-upload", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def build_remote_script(args: argparse.Namespace) -> str:
    include_checkpoints = "1" if args.include_checkpoints else "0"
    stop_insight = "1" if args.stop_insight else "0"
    kill_tmux = "1" if args.kill_tmux else "0"
    hf_repo = args.hf_repo
    hf_private = "1" if args.hf_private else "0"
    detach_upload = "1" if args.detach_upload else "0"

    return textwrap.dedent(
        f"""\
        set -euo pipefail

        REMOTE_ROOT={shlex.quote(args.remote_root)}
        RUN_SUBDIR={shlex.quote(args.run_subdir)}
        SESSION_NAME={shlex.quote(args.session)}
        STAGE_ROOT={shlex.quote(args.stage_root)}
        INCLUDE_CHECKPOINTS={include_checkpoints}
        STOP_INSIGHT={stop_insight}
        KILL_TMUX={kill_tmux}
        HF_REPO={shlex.quote(hf_repo)}
        HF_PRIVATE={hf_private}
        DETACH_UPLOAD={detach_upload}

        TS="$(date -u +%Y%m%dT%H%M%SZ)"
        STAGE_DIR="${{STAGE_ROOT}}/pyc_teardown_${{TS}}"
        META_DIR="${{STAGE_DIR}}/metadata"
        LOG_DIR="${{STAGE_DIR}}/logs"
        RUN_DIR="${{STAGE_DIR}}/run"
        mkdir -p "${{META_DIR}}" "${{LOG_DIR}}" "${{RUN_DIR}}"

        if tmux has-session -t "${{SESSION_NAME}}" 2>/dev/null; then
          tmux list-windows -t "${{SESSION_NAME}}" > "${{META_DIR}}/tmux_windows.txt" || true
          tmux list-panes -t "${{SESSION_NAME}}" -F '#{{session_name}}:#{{window_index}}.#{{pane_index}} #{{pane_title}} #{{pane_current_command}} dead=#{{pane_dead}}' > "${{META_DIR}}/tmux_panes.txt" || true
          tmux capture-pane -p -t "${{SESSION_NAME}}:0.0" | tail -n 400 > "${{LOG_DIR}}/tmux_run_pane.txt" || true
          tmux capture-pane -p -t "${{SESSION_NAME}}:0.1" | tail -n 400 > "${{LOG_DIR}}/tmux_insight_pane.txt" || true
        fi

        ps -ef > "${{META_DIR}}/ps_full.txt" || true
        pgrep -af 'torchrun|train_sft.py|nexa-insight-tui|tmux' > "${{META_DIR}}/process_focus.txt" || true
        uptime > "${{META_DIR}}/uptime.txt" || true
        free -h > "${{META_DIR}}/free_h.txt" || true
        df -h > "${{META_DIR}}/df_h.txt" || true
        nvidia-smi > "${{META_DIR}}/nvidia_smi.txt" 2>&1 || true
        nvidia-smi topo -m > "${{META_DIR}}/nvidia_topo.txt" 2>&1 || true
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits > "${{META_DIR}}/nvidia_gpu_snapshot.csv" 2>&1 || true

        if [ -d "${{REMOTE_ROOT}}/.git" ]; then
          git -C "${{REMOTE_ROOT}}" rev-parse HEAD > "${{META_DIR}}/git_head.txt" || true
          git -C "${{REMOTE_ROOT}}" status --short > "${{META_DIR}}/git_status.txt" || true
          git -C "${{REMOTE_ROOT}}" diff --stat > "${{META_DIR}}/git_diff_stat.txt" || true
          git -C "${{REMOTE_ROOT}}" diff > "${{META_DIR}}/git_diff.patch" || true
        fi

        if [ -f "${{REMOTE_ROOT}}/.env" ]; then
          grep -vE 'TOKEN|SECRET|KEY|PASSWORD' "${{REMOTE_ROOT}}/.env" > "${{META_DIR}}/env_redacted.env" || true
        fi

        tar -czf "${{STAGE_DIR}}/code_snapshot.tgz" \\
          --exclude='.git' \\
          --exclude='.venv' \\
          --exclude='runs' \\
          --exclude='build*' \\
          --exclude='__pycache__' \\
          -C "${{REMOTE_ROOT}}" . || true

        for file in /root/qwen_full.log /root/qwen_smoke.log /root/qwen_hf_upload.log; do
          if [ -f "$file" ]; then
            cp -f "$file" "${{LOG_DIR}}/" || true
          fi
        done

        REMOTE_RUN_DIR="${{REMOTE_ROOT}}/${{RUN_SUBDIR}}"
        if [ -d "${{REMOTE_RUN_DIR}}" ]; then
          for name in run_config.json train_metrics.json gpu_telemetry.csv; do
            if [ -f "${{REMOTE_RUN_DIR}}/$name" ]; then
              cp -f "${{REMOTE_RUN_DIR}}/$name" "${{RUN_DIR}}/" || true
            fi
          done
          find "${{REMOTE_RUN_DIR}}" -maxdepth 1 -type d -name 'checkpoint-*' | sort > "${{RUN_DIR}}/checkpoint_listing.txt" || true
          if [ "${{INCLUDE_CHECKPOINTS}}" = "1" ]; then
            mkdir -p "${{RUN_DIR}}/checkpoints"
            while IFS= read -r checkpoint_dir; do
              [ -n "$checkpoint_dir" ] || continue
              cp -a "$checkpoint_dir" "${{RUN_DIR}}/checkpoints/" || true
            done < "${{RUN_DIR}}/checkpoint_listing.txt"
          fi
        fi

        pkill -TERM -f 'torchrun' || true
        pkill -TERM -f 'scripts/train_sft.py' || true
        sleep 3
        pkill -KILL -f 'torchrun' || true
        pkill -KILL -f 'scripts/train_sft.py' || true

        if [ "${{STOP_INSIGHT}}" = "1" ]; then
          pkill -TERM -f 'nexa-insight-tui' || true
          pkill -TERM -f 'nexa-insight' || true
        fi

        if [ "${{KILL_TMUX}}" = "1" ] && tmux has-session -t "${{SESSION_NAME}}" 2>/dev/null; then
          tmux kill-session -t "${{SESSION_NAME}}" || true
        fi

        pgrep -af 'torchrun|train_sft.py|nexa-insight-tui|tmux' > "${{META_DIR}}/process_focus_after_stop.txt" || true

        if [ -n "${{HF_REPO}}" ]; then
          cat > "${{STAGE_DIR}}/upload_to_hf.py" <<'PY'
        from __future__ import annotations
        import os
        from pathlib import Path
        from huggingface_hub import HfApi

        stage_dir = Path(os.environ["PYC_STAGE_DIR"])
        repo_id = os.environ["PYC_HF_REPO"]
        token = os.environ["HUGGINGFACE_TOKEN"]
        private = os.environ.get("PYC_HF_PRIVATE", "0") == "1"

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=stage_dir.name,
        )
        print(f"uploaded {{stage_dir}} -> {{repo_id}}")
        PY

          set +u
          [ -f "${{REMOTE_ROOT}}/.env" ] && source "${{REMOTE_ROOT}}/.env"
          set -u
          export HUGGINGFACE_TOKEN="${{HUGGINGFACE_TOKEN:-${{HF_TOKEN:-}}}}"
          export PYC_STAGE_DIR="${{STAGE_DIR}}"
          export PYC_HF_REPO="${{HF_REPO}}"
          export PYC_HF_PRIVATE="${{HF_PRIVATE}}"
          if [ -z "${{HUGGINGFACE_TOKEN:-}}" ]; then
            echo "missing Hugging Face token on remote host" >&2
            exit 1
          fi
          if [ "${{DETACH_UPLOAD}}" = "1" ]; then
            nohup "${{REMOTE_ROOT}}/.venv/bin/python" "${{STAGE_DIR}}/upload_to_hf.py" > "${{STAGE_DIR}}/upload.log" 2>&1 < /dev/null &
            echo "stage_dir=${{STAGE_DIR}}"
            echo "upload_log=${{STAGE_DIR}}/upload.log"
            echo "upload_mode=background"
          else
            "${{REMOTE_ROOT}}/.venv/bin/python" "${{STAGE_DIR}}/upload_to_hf.py"
            echo "stage_dir=${{STAGE_DIR}}"
            echo "upload_mode=foreground"
          fi
        else
          echo "stage_dir=${{STAGE_DIR}}"
          echo "upload_mode=none"
        fi
        """
    )


def main() -> int:
    args = parse_args()
    remote_script = build_remote_script(args)
    if args.print_only:
        print(remote_script)
        return 0
    result = ssh(args, remote_script, capture_output=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
