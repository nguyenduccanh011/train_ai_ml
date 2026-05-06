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

VALID_TARGET_TYPES = {
    "trend_regime",
    "early_wave",
    "early_wave_v2",
    "early_wave_dual",
    "return_classification",
    "forward_risk_reward",
    "forward_return",
}
EXIT_LABEL_TARGETS = {"early_wave", "early_wave_v2", "early_wave_dual"}


@dataclass
class ValidationError:
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.field}] {self.message}"


@dataclass
class ValidationWarning:
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.field}] {self.message}"


def validate_config(
    cfg: ExperimentConfig, strict: bool = False
) -> tuple[list[ValidationError], list[ValidationWarning]]:
    """Return validation errors and warnings."""
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    # 0. Market profile exists
    from src.market_profile import load_market_profile, resolve_market_name

    resolved_market = resolve_market_name(cfg.market)
    try:
        load_market_profile(resolved_market)
    except FileNotFoundError as e:
        errors.append(ValidationError("market", str(e)))
    except Exception as e:
        errors.append(ValidationError("market", f"invalid market profile: {e}"))

    # 1. Strategy runner
    from src.components.runners.runner_registry import list_runners
    from src.pipeline.orchestrator import STRATEGIES_WITHOUT_PREDICTION_CACHE

    valid_runners = set(list_runners())
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
    if (
        cfg.strategy not in STRATEGIES_WITHOUT_PREDICTION_CACHE
        and cfg.entry_model_type() not in valid_models
    ):
        errors.append(
            ValidationError(
                "signals.entry_model.type",
                f"'{cfg.entry_model_type()}' is not registered. Valid: {sorted(valid_models)}",
            )
        )

    valid_exit_models = set(list_exit_models())
    if cfg.signals.exit_model.enabled and cfg.signals.exit_model.type not in valid_exit_models:
        errors.append(
            ValidationError(
                "signals.exit_model.type",
                f"'{cfg.signals.exit_model.type}' is not registered. Valid: {sorted(valid_exit_models)}",
            )
        )

    # 3. Target type — extend with MarketProfile target.type if different
    target_type = cfg.signals.target.type
    effective_valid_target_types = set(VALID_TARGET_TYPES)
    try:
        from src.market_profile import load_market_profile

        mprofile = load_market_profile(resolved_market)
        if mprofile.target.type:
            effective_valid_target_types.add(mprofile.target.type)
    except Exception:
        pass
    if target_type not in effective_valid_target_types:
        errors.append(
            ValidationError(
                "signals.target.type",
                f"'{target_type}' is not a known target type. Valid: {sorted(effective_valid_target_types)}",
            )
        )

    if strict and cfg.signals.exit_model.enabled:
        if target_type not in EXIT_LABEL_TARGETS:
            errors.append(
                ValidationError(
                    "signals.exit_model",
                    f"exit_model enabled but target '{target_type}' does not support exit labels",
                )
            )
        if cfg.strategy_v3 is not None and not cfg.strategy_v3.active_exit_rules:
            warnings.append(
                ValidationWarning(
                    "signals.exit_model",
                    "exit_model enabled but strategy_v3.active_exit_rules is empty - exit signal may not be consumed",
                )
            )

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

    return errors, warnings


def assert_valid(cfg: ExperimentConfig, strict: bool = False) -> None:
    """Raise ValueError with all errors if config is invalid."""
    errors, _ = validate_config(cfg, strict=strict)
    if errors:
        msg = "\n".join(f"  {e}" for e in errors)
        raise ValueError(f"Invalid ExperimentConfig '{cfg.name}':\n{msg}")
