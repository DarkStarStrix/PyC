from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmark" / "benchmarks" / "gpu" / "arena.py"
SPEC = importlib.util.spec_from_file_location("gpu_arena", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gpu_arena = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gpu_arena
SPEC.loader.exec_module(gpu_arena)


def test_resolve_adapter_plan_arena_mode_uses_expected_tiers():
    plan = gpu_arena.resolve_adapter_plan("", gpu_arena.DEFAULT_ADAPTERS, arena_mode=True)

    assert plan["mode"] == "arena"
    assert plan["adapters"] == ["torch_compile", "torch_eager", "glow", "tvm", "xla", "pyc"]
    assert [tier["id"] for tier in plan["tiers"]] == ["tier-1", "tier-2", "tier-3", "tier-4"]


def test_resolve_adapter_plan_maps_prod_alias_to_pyc():
    plan = gpu_arena.resolve_adapter_plan("torch_eager,prod,xla", gpu_arena.DEFAULT_ADAPTERS)

    assert plan["mode"] == "explicit"
    assert plan["adapters"] == ["torch_eager", "pyc", "xla"]
    assert plan["tiers"][-1]["id"] == "tier-4"
    assert plan["tiers"][-1]["adapters"] == ["pyc"]


def test_resolve_adapter_plan_arena_tier_one_adds_prod_lane():
    plan = gpu_arena.resolve_adapter_plan("", gpu_arena.DEFAULT_ADAPTERS, arena_tier="1")

    assert plan["mode"] == "arena:tier-1"
    assert plan["adapters"] == ["torch_compile", "torch_eager", "pyc"]
    assert [tier["id"] for tier in plan["tiers"]] == ["tier-1", "tier-4"]


def test_resolve_adapter_plan_hopper_profile_excludes_glow_and_adds_cutlass():
    plan = gpu_arena.resolve_adapter_plan(
        "",
        gpu_arena.DEFAULT_ADAPTERS,
        arena_mode=True,
        arena_profile="hopper",
    )

    assert plan["mode"] == "arena"
    assert plan["profile"] == "hopper"
    assert plan["adapters"] == ["torch_compile", "torch_eager", "cutlass", "tensorrt", "tvm", "xla", "pyc"]
    assert "glow" not in plan["adapters"]
    assert [tier["id"] for tier in plan["tiers"]] == ["tier-1", "tier-2", "tier-3", "tier-4"]
