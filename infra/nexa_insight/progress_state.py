from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROGRESS_META_KEY = "meta"
PROGRESS_PROGRESS_KEY = "progress"
PROGRESS_RECENT_EVENTS_KEY = "recent_events"


def write_progress_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def progress_updated_utc(payload: dict[str, Any]) -> str:
    meta = payload.get(PROGRESS_META_KEY, {}) if isinstance(payload, dict) else {}
    return str(meta.get("updated_utc") or meta.get("completed_utc") or meta.get("started_utc") or "")


def choose_progress_state(
    task_progress: dict[str, Any],
    gemm_progress: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    task_ts = progress_updated_utc(task_progress)
    gemm_ts = progress_updated_utc(gemm_progress)
    if task_ts and (not gemm_ts or task_ts >= gemm_ts):
        return task_progress, "task"
    if gemm_ts:
        return gemm_progress, "gemm"
    if task_progress:
        return task_progress, "task"
    return gemm_progress, "gemm"


def progress_meta(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    meta = payload.get(PROGRESS_META_KEY, {})
    return meta if isinstance(meta, dict) else {}


def progress_progress(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    progress = payload.get(PROGRESS_PROGRESS_KEY, {})
    return progress if isinstance(progress, dict) else {}


def progress_recent_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    events = payload.get(PROGRESS_RECENT_EVENTS_KEY, [])
    return events if isinstance(events, list) else []
