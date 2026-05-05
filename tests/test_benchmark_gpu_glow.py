from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "benchmark" / "benchmarks" / "gpu" / "adapters" / "adapter_glow.py"
COMMON_PATH = ROOT / "benchmark" / "benchmarks" / "gpu" / "adapters" / "common.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_glow_autowires_bundled_helper(monkeypatch):
    common = load_module("gpu_adapters_common_for_glow_test", COMMON_PATH)
    sys.modules["common"] = common
    adapter_glow = load_module("gpu_adapter_glow_for_test", ADAPTER_PATH)

    captured: dict[str, str] = {}

    def fake_run_external_json_command(command: str) -> dict:
        captured["command"] = command
        return {"status": "ok", "backend": "glow_proxy", "mode": "proxy"}

    monkeypatch.delenv("GLOW_BENCH_CMD", raising=False)
    monkeypatch.setattr(adapter_glow, "run_external_json_command", fake_run_external_json_command)
    monkeypatch.setattr(
        adapter_glow,
        "emit",
        lambda payload: payload,
    )
    monkeypatch.setattr(sys, "argv", ["adapter_glow.py"])

    payload = adapter_glow.main()

    helper = ROOT / "benchmark" / "benchmarks" / "gpu" / "external" / "bench_glow_cmd.py"
    assert "command" in captured
    assert str(helper) in captured["command"]
    assert os.environ["GLOW_BENCH_CMD"].endswith(str(helper))
    assert payload["adapter"] == "glow"
