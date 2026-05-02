from __future__ import annotations

from pathlib import Path
from typing import Any

import src.components.fusion.strategies  # noqa: F401
import yaml
from src.components.fusion.registry import get_entry, get_strategy
from src.components.runners.generic_fusion import (
    DEFAULT_V22_MODS,
    V22_ACTIVE_EXIT_STRATEGY_NAMES,
    V22_DEFAULTS,
    V22_FORCE_EXIT_STRATEGY_NAMES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
V22_YAML = REPO_ROOT / "stock_ml" / "config" / "experiments" / "champions" / "v22.yaml"


def _load_v22_config() -> dict[str, Any]:
    return yaml.safe_load(V22_YAML.read_text())


def _names(items: list[dict[str, Any]]) -> list[str]:
    return [str(item["name"]) for item in items]


def test_v22_yaml_strategies_are_registered() -> None:
    cfg = _load_v22_config()
    fusion = cfg["fusion"]
    names = [
        *_names(fusion["entry"]),
        *_names(fusion["force_exit"]),
        *_names(fusion["active_exit"]),
        *_names(fusion["hold"]),
    ]

    for name in names:
        assert get_strategy(name).name


def test_v22_yaml_exit_order_matches_runner_contract() -> None:
    cfg = _load_v22_config()
    fusion = cfg["fusion"]

    assert tuple(_names(fusion["force_exit"])) == V22_FORCE_EXIT_STRATEGY_NAMES
    assert tuple(_names(fusion["active_exit"])) == V22_ACTIVE_EXIT_STRATEGY_NAMES


def test_v22_registry_layers_match_yaml_sections() -> None:
    cfg = _load_v22_config()
    fusion = cfg["fusion"]
    section_layers = {
        "entry": "entry",
        "force_exit": "exit_override",
        "active_exit": "exit_override",
        "hold": "hold",
    }

    for section, layer in section_layers.items():
        for name in _names(fusion[section]):
            assert get_entry(name).layer == layer


def test_v22_yaml_defaults_match_runner_contract() -> None:
    cfg = _load_v22_config()

    assert cfg["mods"] == DEFAULT_V22_MODS
    for name, value in cfg["params"].items():
        assert V22_DEFAULTS[name] == value
