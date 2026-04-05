from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "infra" / "nexa_insight" / "observer_server.py"
SPEC = importlib.util.spec_from_file_location("nexa_insight_observer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)


def test_parse_gpu_rows():
    text = "\n".join(
        [
            "0, NVIDIA RTX 6000 Ada Generation, 17, 12, 632, 49140, 42, 29.2",
            "1, NVIDIA H100, 99, 87, 80210, 81559, 67, 287.4",
        ]
    )
    rows = observer.parse_gpu_rows(text)
    assert len(rows) == 2
    assert rows[0]["index"] == 0
    assert rows[0]["gpu_util"] == 17.0
    assert rows[1]["mem_total_mib"] == 81559.0


def test_parse_tmux_windows():
    text = "\n".join(
        [
            "0|bash|0|python3|shape-sweep",
            "1|insight|1|bash|observer",
        ]
    )
    rows = observer.parse_tmux_windows(text)
    assert len(rows) == 2
    assert rows[1]["name"] == "insight"
    assert rows[1]["active"] is True
