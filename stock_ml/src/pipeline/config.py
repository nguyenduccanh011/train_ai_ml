"""ExperimentConfig — typed config loaded from champion YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.backtest.defaults import DEFAULT_TRADING_COST
from src.market_profile import resolve_market_name

_RUNNER_PREFIX_ALIASES: dict[str, str] = {
    "components.runners.": "src.components.runners.",
}
_FUSION_RULE_ALIASES: dict[str, str] = {
    "exit_model_exit": "exit_model",
}


def _canonical_runner_path(runner: str) -> str:
    raw = str(runner or "").strip()
    for legacy, canonical in _RUNNER_PREFIX_ALIASES.items():
        if raw.startswith(legacy):
            return canonical + raw[len(legacy) :]
    return raw


def _canonical_rule_name(name: str) -> str:
    raw = str(name or "").strip()
    return _FUSION_RULE_ALIASES.get(raw, raw)


def _canonical_rule_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        canonical = _canonical_rule_name(name)
        if canonical and canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


class TargetConfig(BaseModel):
    type: str = "trend_regime"
    trend_method: str = "dual_ma"
    short_window: int = 5
    long_window: int = 20
    classes: int = 3
    forward_window: int = 8
    horizon: int | None = None
    unit: str = "bars"
    gain_threshold: float = 0.06
    loss_threshold: float = 0.03

    @model_validator(mode="before")
    @classmethod
    def _normalize_horizon(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values is None:
            return {}
        horizon = values.get("horizon")
        forward_window = values.get("forward_window")
        if forward_window is None and horizon is not None:
            values["forward_window"] = horizon
        if values.get("horizon") is None and values.get("forward_window") is not None:
            values["horizon"] = values["forward_window"]
        target_type = str(values.get("type", "trend_regime")).strip().lower()
        if target_type == "forward_return":
            # Backward-compatible alias for return target family.
            values["type"] = "return_classification"
        return values

    def to_legacy_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EntryModelConfig(BaseModel):
    type: str = "lightgbm"
    device: str = "cpu"
    extras: dict[str, Any] = Field(default_factory=dict)


class ExitModelConfig(BaseModel):
    enabled: bool = False
    type: str = "lightgbm"
    forward_window: int = 15
    loss_threshold: float = 0.05
    extras: dict[str, Any] = Field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "type": self.type,
            "forward_window": self.forward_window,
            "loss_threshold": self.loss_threshold,
            "extras": self.extras,
        }


class ComponentsConfig(BaseModel):
    features: str = "leading_v2"
    target: TargetConfig = Field(default_factory=TargetConfig)
    entry_model: EntryModelConfig = Field(default_factory=EntryModelConfig)
    exit_model: ExitModelConfig = Field(default_factory=ExitModelConfig)


class SplitConfig(BaseModel):
    method: str = "walk_forward"
    train_years: int = 4
    test_years: int = 1
    gap_days: int = 25
    first_test_year: int = 2020
    last_test_year: int = 2025


class FusionItemConfig(BaseModel):
    name: str

    model_config = ConfigDict(extra="allow")


class FusionGroupConfig(BaseModel):
    entry: list[FusionItemConfig] = Field(default_factory=list)
    force_exit: list[FusionItemConfig] = Field(default_factory=list)
    active_exit: list[FusionItemConfig] = Field(default_factory=list)
    hold: list[FusionItemConfig] = Field(default_factory=list)


class SignalsConfig(BaseModel):
    features: str = "leading_v2"
    target: TargetConfig = Field(default_factory=TargetConfig)
    entry_model: EntryModelConfig = Field(default_factory=EntryModelConfig)
    exit_model: ExitModelConfig = Field(default_factory=ExitModelConfig)


class StrategyV3Config(BaseModel):
    entry_rules: list[str] = Field(default_factory=list)
    hold_rules: list[str] = Field(default_factory=list)
    exit_rules: list[str] = Field(default_factory=list)
    force_exit_rules: list[str] = Field(default_factory=list)
    active_exit_rules: list[str] = Field(default_factory=list)
    mods: dict[str, bool] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class ExecutionConfig(BaseModel):
    backtester: str = ""
    pnl_mode: str = "equity_spot"
    currency: str = "VND"
    capital: float = 100_000_000
    commission: float = DEFAULT_TRADING_COST["commission"]
    tax: float = DEFAULT_TRADING_COST["tax"]
    slippage: float = DEFAULT_TRADING_COST["slippage"]
    split: SplitConfig = Field(default_factory=SplitConfig)


class ExperimentConfig(BaseModel):
    name: str
    strategy: str
    market: str = "vn_stock"
    runner: str = ""
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    mods: dict[str, bool] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    fusion: FusionGroupConfig = Field(default_factory=FusionGroupConfig)
    signals: SignalsConfig | None = None
    strategy_v3: StrategyV3Config | None = None
    execution: ExecutionConfig | None = None

    @field_validator("fusion", mode="before")
    @classmethod
    def _coerce_fusion(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "name" not in values:
            values["name"] = values.get("strategy", "unknown")
        if "market" not in values or not str(values.get("market", "")).strip():
            values["market"] = resolve_market_name(None)
        if "runner" not in values or not values.get("runner"):
            strategy = values.get("strategy", "")
            values["runner"] = f"src.components.runners.{strategy}_runner" if strategy else ""
        else:
            values["runner"] = _canonical_runner_path(str(values["runner"]))

        has_components = "components" in values and values["components"] is not None
        has_signals = "signals" in values and values["signals"] is not None
        if has_components and has_signals:
            components = ComponentsConfig.model_validate(values["components"])
            signals = SignalsConfig.model_validate(values["signals"])
            if components.model_dump() != signals.model_dump():
                raise ValueError(
                    "components and signals config sections must match when both are provided"
                )
        elif has_signals:
            signals = SignalsConfig.model_validate(values["signals"])
            values["components"] = signals.model_dump()
        return values

    @model_validator(mode="after")
    def _fill_v3_sections(self) -> ExperimentConfig:
        self.runner = _canonical_runner_path(self.runner)

        for group_name in ("entry", "force_exit", "active_exit", "hold"):
            group = getattr(self.fusion, group_name)
            seen: set[str] = set()
            normalized_group: list[FusionItemConfig] = []
            for item in group:
                canonical = _canonical_rule_name(item.name)
                if canonical in seen:
                    continue
                if canonical != item.name:
                    item = item.model_copy(update={"name": canonical})
                normalized_group.append(item)
                seen.add(canonical)
            setattr(self.fusion, group_name, normalized_group)

        if self.signals is None:
            self.signals = SignalsConfig(
                features=self.components.features,
                target=self.components.target,
                entry_model=self.components.entry_model,
                exit_model=self.components.exit_model,
            )
        self.components = ComponentsConfig(
            features=self.signals.features,
            target=self.signals.target,
            entry_model=self.signals.entry_model,
            exit_model=self.signals.exit_model,
        )
        if self.strategy_v3 is None:
            force_exit_rules = [item.name for item in self.fusion.force_exit]
            active_exit_rules = [item.name for item in self.fusion.active_exit]
            self.strategy_v3 = StrategyV3Config(
                entry_rules=[item.name for item in self.fusion.entry],
                hold_rules=[item.name for item in self.fusion.hold],
                exit_rules=force_exit_rules + active_exit_rules,
                force_exit_rules=force_exit_rules,
                active_exit_rules=active_exit_rules,
                mods=self.mods,
                params=self.params,
            )
        else:
            self.strategy_v3.entry_rules = _canonical_rule_names(self.strategy_v3.entry_rules)
            self.strategy_v3.hold_rules = _canonical_rule_names(self.strategy_v3.hold_rules)
            self.strategy_v3.exit_rules = _canonical_rule_names(self.strategy_v3.exit_rules)
            self.strategy_v3.force_exit_rules = _canonical_rule_names(
                self.strategy_v3.force_exit_rules
            )
            self.strategy_v3.active_exit_rules = _canonical_rule_names(
                self.strategy_v3.active_exit_rules
            )
        if self.execution is None:
            self.execution = ExecutionConfig(
                backtester=self._normalize_runner(self.runner),
                split=self.split,
            )
        return self

    @staticmethod
    def _normalize_runner(runner: str) -> str:
        return runner.rsplit(".", 1)[-1] if runner else ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "market" not in data or not str(data.get("market", "")).strip():
            data["market"] = resolve_market_name(None)
        return cls.model_validate(data)

    def entry_model_type(self) -> str:
        return self.signals.entry_model.type

    def feature_set(self) -> str:
        return self.signals.features

    def target_dict(self) -> dict[str, Any]:
        return self.signals.target.to_legacy_dict()

    def exit_model_dict(self) -> dict[str, Any] | None:
        return self.signals.exit_model.to_legacy_dict()
