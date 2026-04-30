"""ExperimentConfig — typed config loaded from champion YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


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
    forward_window: int = 15
    loss_threshold: float = 0.05

    def to_legacy_dict(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {"forward_window": self.forward_window, "loss_threshold": self.loss_threshold}


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


class ExperimentConfig(BaseModel):
    name: str
    strategy: str
    runner: str = ""
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    mods: dict[str, bool] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    fusion: dict[str, Any] = Field(default_factory=dict)
    enable_model_b_exit: bool = False

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "name" not in values:
            values["name"] = values.get("strategy", "unknown")
        if "runner" not in values or not values.get("runner"):
            strategy = values.get("strategy", "")
            values["runner"] = f"src.components.runners.{strategy}_runner" if strategy else ""
        return values

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def entry_model_type(self) -> str:
        return self.components.entry_model.type

    def feature_set(self) -> str:
        return self.components.features

    def target_dict(self) -> dict[str, Any]:
        return self.components.target.to_legacy_dict()

    def exit_model_dict(self) -> dict[str, Any] | None:
        return self.components.exit_model.to_legacy_dict()
