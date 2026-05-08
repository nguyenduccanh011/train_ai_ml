from __future__ import annotations

import src.components.fusion.strategies  # noqa: F401
from src.components.fusion.registry import get_entry, get_strategy
from src.pipeline.config import ExperimentConfig


def test_runner_path_legacy_prefix_is_canonicalized() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "v22_compat",
            "strategy": "v22",
            "runner": "components.runners.run_v22",
        }
    )
    assert cfg.runner == "src.components.runners.run_v22"


def test_exit_model_rule_alias_in_fusion_is_canonicalized() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "v22_alias_fusion",
            "strategy": "v22",
            "fusion": {
                "active_exit": [
                    {"name": "exit_model_exit"},
                    {"name": "exit_model"},
                ]
            },
        }
    )
    assert [item.name for item in cfg.fusion.active_exit] == ["exit_model"]
    assert cfg.strategy_v3 is not None
    assert cfg.strategy_v3.active_exit_rules == ["exit_model"]


def test_exit_model_rule_alias_in_strategy_v3_is_canonicalized() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "v22_alias_v3",
            "strategy": "v22",
            "strategy_v3": {
                "active_exit_rules": ["exit_model_exit", "exit_model"],
                "exit_rules": ["exit_model_exit"],
            },
        }
    )
    assert cfg.strategy_v3 is not None
    assert cfg.strategy_v3.active_exit_rules == ["exit_model"]
    assert cfg.strategy_v3.exit_rules == ["exit_model"]


def test_registry_keeps_backward_compatible_exit_model_alias() -> None:
    entry = get_entry("exit_model_exit")
    assert entry.layer == "exit_override"
    strategy = get_strategy("exit_model_exit")
    assert strategy.name == "exit_model"
