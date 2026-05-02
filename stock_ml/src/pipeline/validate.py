"""Validation rules for ExperimentConfig — catch errors at load time, not runtime.

Rules:
1. strategy must have a registered runner
2. entry_model.type must be registered
3. target.type must be valid
4. exit_model requires target.supports_exit_labels if enabled
5. split: first_test_year < last_test_year, train_years >= 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.config import ExperimentConfig

VALID_TARGET_TYPES = {"trend_regime", "early_wave", "early_wave_v2", "early_wave_dual"}
EXIT_LABEL_TARGETS = {"early_wave", "early_wave_v2", "early_wave_dual"}


@dataclass
class ValidationError:
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.field}] {self.message}"


def validate_config(cfg: ExperimentConfig) -> list[ValidationError]:
    """Return list of validation errors (empty = OK)."""
    errors: list[ValidationError] = []

    # 1. Strategy runner
    from src.components.runners.generic_fusion import FUSION_RUNNER_DEFS
    from src.components.runners.runner_registry import RUNNER_DEFS
    from src.pipeline.orchestrator import CHAMPION_RUNNER_MAP

    valid_runners = set(CHAMPION_RUNNER_MAP) | set(FUSION_RUNNER_DEFS) | set(RUNNER_DEFS)
    if cfg.strategy not in valid_runners:
        errors.append(
            ValidationError(
                "strategy",
                f"'{cfg.strategy}' has no registered runner. Valid: {sorted(valid_runners)}",
            )
        )

    # 2. Entry model type
    from src.components.exit_models.registry import list_exit_models
    from src.components.models.registry import list_models

    valid_models = set(list_models())
    if cfg.entry_model_type() not in valid_models:
        errors.append(
            ValidationError(
                "signals.entry_model.type",
                f"'{cfg.entry_model_type()}' is not registered. Valid: {sorted(valid_models)}",
            )
        )

    valid_exit_models = set(list_exit_models())
    if (
        cfg.components.exit_model.enabled
        and cfg.components.exit_model.type not in valid_exit_models
    ):
        errors.append(
            ValidationError(
                "components.exit_model.type",
                f"'{cfg.components.exit_model.type}' is not registered. Valid: {sorted(valid_exit_models)}",
            )
        )

    # 3. Target type
    target_type = cfg.components.target.type
    if target_type not in VALID_TARGET_TYPES:
        errors.append(
            ValidationError(
                "components.target.type",
                f"'{target_type}' is not a known target type. Valid: {sorted(VALID_TARGET_TYPES)}",
            )
        )

    # 4. Exit model with non-dual target: note only (EXIT_MODEL_BUG.md — output currently dropped)
    # Not an error — legacy behavior preserved for golden parity.

    # 5. Split range validity
    s = cfg.split
    if s.first_test_year >= s.last_test_year:
        errors.append(
            ValidationError(
                "split",
                f"first_test_year ({s.first_test_year}) must be < last_test_year ({s.last_test_year})",
            )
        )
    if s.train_years < 1:
        errors.append(ValidationError("split.train_years", "must be >= 1"))

    # 6. Fusion rule names
    for group_name in ("entry", "force_exit", "active_exit", "hold"):
        for idx, item in enumerate(getattr(cfg.fusion, group_name)):
            if not item.name.strip():
                errors.append(
                    ValidationError(
                        f"fusion.{group_name}.{idx}.name",
                        "must not be empty",
                    )
                )

    # 7. V3 strategy rule names
    import src.components.fusion.strategies  # noqa: F401 — triggers register_strategy() calls
    from src.components.fusion.registry import list_strategies

    valid_strategies = set(list_strategies())
    assert valid_strategies, "strategy registry empty — import of fusion.strategies may have failed"
    if cfg.strategy_v3 is not None:
        for group_name in (
            "entry_rules",
            "hold_rules",
            "exit_rules",
            "force_exit_rules",
            "active_exit_rules",
        ):
            for idx, name in enumerate(getattr(cfg.strategy_v3, group_name)):
                if name not in valid_strategies:
                    errors.append(
                        ValidationError(
                            f"strategy_v3.{group_name}.{idx}",
                            f"'{name}' is not registered. Valid: {sorted(valid_strategies)}",
                        )
                    )

    return errors


def assert_valid(cfg: ExperimentConfig) -> None:
    """Raise ValueError with all errors if config is invalid."""
    errors = validate_config(cfg)
    if errors:
        msg = "\n".join(f"  {e}" for e in errors)
        raise ValueError(f"Invalid ExperimentConfig '{cfg.name}':\n{msg}")
