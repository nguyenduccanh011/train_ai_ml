from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.components.fusion.base import FusionLayer, FusionStrategy


@dataclass(frozen=True, slots=True)
class StrategyEntry:
    name: str
    layer: FusionLayer
    builder: Any  # StrategyBuilder
    always_on: bool = False


_REGISTRY: dict[str, StrategyEntry] = {}


def register_strategy(
    name: str,
    layer: FusionLayer,
    builder: Any,
    *,
    always_on: bool = False,
    replace: bool = False,
) -> None:
    if not replace and name in _REGISTRY:
        raise ValueError(f"strategy {name!r} already registered")
    _REGISTRY[name] = StrategyEntry(name=name, layer=layer, builder=builder, always_on=always_on)


def get_strategy(name: str, **params: Any) -> FusionStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown fusion strategy: {name!r}")
    return _REGISTRY[name].builder(**params)


def list_strategies(layer: FusionLayer | None = None) -> list[str]:
    if layer is None:
        return sorted(_REGISTRY)
    return sorted(n for n, e in _REGISTRY.items() if e.layer == layer)


def list_always_on(layer: FusionLayer | None = None) -> list[str]:
    return sorted(
        n for n, e in _REGISTRY.items() if e.always_on and (layer is None or e.layer == layer)
    )


def get_entry(name: str) -> StrategyEntry:
    if name not in _REGISTRY:
        raise KeyError(f"unknown fusion strategy: {name!r}")
    return _REGISTRY[name]


def clear_registry() -> None:
    """Test-only helper. Do not call in production code paths."""
    _REGISTRY.clear()
