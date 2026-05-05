from __future__ import annotations

from typing import Iterable

DEFAULT_ADAPTERS = [
    "torch_eager",
    "torch_compile",
    "pyc",
    "tvm",
    "xla",
    "tensorrt",
    "glow",
]

ADAPTER_ALIASES = {
    "prod": "pyc",
}

DEFAULT_ARENA_PROFILE = "legacy"

ARENA_PROFILES = {
    "legacy": [
        {
            "id": "tier-1",
            "label": "Tier 1",
            "adapters": ["torch_compile", "torch_eager"],
        },
        {
            "id": "tier-2",
            "label": "Tier 2",
            "adapters": ["glow"],
        },
        {
            "id": "tier-3",
            "label": "Tier 3",
            "adapters": ["tvm", "xla"],
        },
        {
            "id": "tier-4",
            "label": "Tier 4",
            "adapters": ["pyc"],
        },
    ],
    "hopper": [
        {
            "id": "tier-1",
            "label": "Tier 1",
            "adapters": ["torch_compile", "torch_eager"],
        },
        {
            "id": "tier-2",
            "label": "Tier 2",
            "adapters": ["cutlass", "tensorrt"],
        },
        {
            "id": "tier-3",
            "label": "Tier 3",
            "adapters": ["tvm", "xla"],
        },
        {
            "id": "tier-4",
            "label": "Tier 4",
            "adapters": ["pyc"],
        },
    ],
}


def canonicalize_adapter_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    return ADAPTER_ALIASES.get(normalized, normalized)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def resolve_arena_profile(name: str = "") -> tuple[str, list[dict]]:
    normalized = (name or "").strip().lower() or DEFAULT_ARENA_PROFILE
    if normalized not in ARENA_PROFILES:
        raise ValueError(f"unknown arena profile: {name}")
    return normalized, ARENA_PROFILES[normalized]


def arena_adapters(profile_name: str = "") -> list[str]:
    _, tiers = resolve_arena_profile(profile_name)
    return [adapter for tier in tiers for adapter in tier["adapters"]]


def classify_adapter_plan(adapters: Iterable[str], mode: str, arena_profile: str = "") -> dict:
    profile_name, tiers_for_profile = resolve_arena_profile(arena_profile)
    ordered = dedupe_preserve_order(canonicalize_adapter_name(item) for item in adapters)
    ranked = {adapter: index for index, adapter in enumerate(ordered)}
    tiers = []
    arena_members = {adapter for tier in tiers_for_profile for adapter in tier["adapters"]}
    for tier in tiers_for_profile:
        members = [adapter for adapter in tier["adapters"] if adapter in ranked]
        if not members:
            continue
        tiers.append(
            {
                "id": tier["id"],
                "label": tier["label"],
                "adapters": members,
            }
        )
    unranked = [adapter for adapter in ordered if adapter not in arena_members]
    if unranked:
        tiers.append(
            {
                "id": "unranked",
                "label": "Unranked",
                "adapters": unranked,
            }
        )
    return {
        "mode": mode,
        "adapters": ordered,
        "tiers": tiers,
        "profile": profile_name,
    }


def normalize_arena_tier(value: str, arena_profile: str = "") -> str:
    _, tiers_for_profile = resolve_arena_profile(arena_profile)
    tier_ids = {tier["id"] for tier in tiers_for_profile}
    normalized = (value or "").strip().lower()
    if not normalized:
        raise ValueError("arena tier must be non-empty")
    if normalized.isdigit():
        normalized = f"tier-{normalized}"
    if normalized not in tier_ids:
        raise ValueError(f"unknown arena tier: {value}")
    return normalized


def resolve_arena_tier_plan(tier_value: str, arena_profile: str = "") -> dict:
    profile_name, tiers_for_profile = resolve_arena_profile(arena_profile)
    tier_id = normalize_arena_tier(tier_value, profile_name)
    selected = next(tier for tier in tiers_for_profile if tier["id"] == tier_id)
    adapters = list(selected["adapters"])
    if "pyc" not in adapters:
        adapters.append("pyc")
    return classify_adapter_plan(adapters, f"arena:{tier_id}", profile_name)


def resolve_adapter_plan(
    requested: str,
    fallback: Iterable[str],
    *,
    arena_mode: bool = False,
    arena_tier: str = "",
    arena_profile: str = "",
) -> dict:
    profile_name, _ = resolve_arena_profile(arena_profile)
    if arena_tier.strip():
        return resolve_arena_tier_plan(arena_tier, profile_name)
    raw = [item.strip() for item in requested.split(",") if item.strip()]
    raw_mode = "explicit"
    if arena_mode or (len(raw) == 1 and canonicalize_adapter_name(raw[0]) == "arena"):
        return classify_adapter_plan(arena_adapters(profile_name), "arena", profile_name)
    if not raw:
        raw = [str(item).strip() for item in fallback if str(item).strip()]
        raw_mode = "default"
    canonical = dedupe_preserve_order(canonicalize_adapter_name(item) for item in raw)
    return classify_adapter_plan(canonical, raw_mode, profile_name)
