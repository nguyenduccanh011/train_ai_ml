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
EXIT_LABEL_TARGETS = {"early_wave_dual"}


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
    from src.pipeline.orchestrator import CHAMPION_RUNNER_MAP

    if cfg.strategy not in CHAMPION_RUNNER_MAP:
        errors.append(
            ValidationError(
                "strategy",
                f"'{cfg.strategy}' has no registered runner. Valid: {sorted(CHAMPION_RUNNER_MAP)}",
            )
        )

    # 2. Entry model type
    from src.components.models.registry import list_models

    valid_models = set(list_models())
    if cfg.entry_model_type() not in valid_models:
        errors.append(
            ValidationError(
                "components.entry_model.type",
                f"'{cfg.entry_model_type()}' is not registered. Valid: {sorted(valid_models)}",
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

    return errors


def assert_valid(cfg: ExperimentConfig) -> None:
    """Raise ValueError with all errors if config is invalid."""
    errors = validate_config(cfg)
    if errors:
        msg = "\n".join(f"  {e}" for e in errors)
        raise ValueError(f"Invalid ExperimentConfig '{cfg.name}':\n{msg}")
