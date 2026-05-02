"""ExperimentConfig — typed config loaded from champion YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TargetConfig(BaseModel):
    type: str = "trend_regime"
    trend_method: str = "dual_ma"
    short_window: int = 5
    long_window: int = 20
    classes: int = 3
    forward_window: int = 8
    gain_threshold: float = 0.06
    loss_threshold: float = 0.03

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
    gap_days: int = 0
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
    capital: float = 100_000_000
    split: SplitConfig = Field(default_factory=SplitConfig)


class ExperimentConfig(BaseModel):
    name: str
    strategy: str
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
        if "runner" not in values or not values.get("runner"):
            strategy = values.get("strategy", "")
            values["runner"] = f"src.components.runners.{strategy}_runner" if strategy else ""
        return values

    @model_validator(mode="after")
    def _fill_v3_sections(self) -> ExperimentConfig:
        if self.signals is None:
            self.signals = SignalsConfig(
                features=self.components.features,
                target=self.components.target,
                entry_model=self.components.entry_model,
                exit_model=self.components.exit_model,
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
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def entry_model_type(self) -> str:
        return self.signals.entry_model.type

    def feature_set(self) -> str:
        return self.components.features

    def target_dict(self) -> dict[str, Any]:
        return self.components.target.to_legacy_dict()

    def exit_model_dict(self) -> dict[str, Any] | None:
        return self.components.exit_model.to_legacy_dict()
