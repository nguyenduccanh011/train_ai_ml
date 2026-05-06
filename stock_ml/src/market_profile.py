"""Market profile loader and resolver for multi-market pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.backtest.pnl import SUPPORTED_PNL_MODES

_STRATEGY_NAMES: set[str] | None = None


def _normalize_symbols(value: Any, none_value: Any) -> Any:
    if value is None:
        return none_value
    return [str(sym).strip().upper() for sym in value if str(sym).strip()]


def _registered_strategy_names() -> set[str]:
    global _STRATEGY_NAMES
    if _STRATEGY_NAMES is None:
        import src.components.fusion.strategies  # noqa: F401
        from src.components.fusion.registry import list_strategies

        _STRATEGY_NAMES = set(list_strategies())
    return _STRATEGY_NAMES


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _base_config_path() -> Path:
    return _repo_root() / "config" / "base.yaml"


def markets_dir() -> Path:
    return _repo_root() / "config" / "markets"


class MarketDataConfig(BaseModel):
    data_dir: str | None = None
    schema: str | None = None
    default_timeframe: str = "1D"
    benchmark_symbol: str | None = None
    timestamp_column: str = "timestamp"
    timezone: str | None = None
    required_columns: list[str] = Field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )
    optional_columns: list[str] = Field(default_factory=list)
    volume_unit: str | None = None


class MarketExecutionConfig(BaseModel):
    instrument_type: str | None = None
    pnl_mode: str = "equity_spot"
    commission: float = 0.0015
    tax: float = 0.001
    slippage: float = 0.0
    initial_capital: float = 100_000_000
    currency: str = "VND"
    contract_multiplier: float = 1.0
    funding_rate_column: str | None = None
    borrow_rate_column: str | None = None
    borrow_available_column: str | None = None
    margin_mode: str | None = None
    leverage: float = 1.0
    maintenance_margin_rate: float = 0.0
    liquidation_fee: float = 0.0
    short_enabled: bool = False
    short_position_size: float | None = None
    short_hard_cap: float | None = None
    short_squeeze_exit: bool = False
    short_squeeze_vol_mult: float = 3.0
    short_squeeze_price_pct: float = 0.03
    max_short_notional: float | None = None
    max_total_short_notional: float | None = None
    roll_cost_rate: float = 0.0
    expiry_date_column: str | None = None
    roll_rule: str | None = None
    roll_days_before_expiry: int = 3
    next_volume_column: str | None = None
    next_oi_column: str | None = None

    @field_validator("pnl_mode")
    @classmethod
    def _validate_pnl_mode(cls, value: str) -> str:
        if value not in SUPPORTED_PNL_MODES:
            available = ", ".join(sorted(SUPPORTED_PNL_MODES))
            raise ValueError(f"unsupported pnl_mode {value!r}. Available: {available}")
        return value


class MarketSymbolsConfig(BaseModel):
    default_list: list[str] = Field(default_factory=list)
    groups: dict[str, str] = Field(default_factory=dict)
    default_group: str = "balanced"

    @field_validator("default_list", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: Any) -> Any:
        return _normalize_symbols(value, [])


class MarketFeaturesConfig(BaseModel):
    enabled_blocks: list[str] | None = None


class MarketModelsConfig(BaseModel):
    default_stack: list[str] | None = None


class MarketTargetConfig(BaseModel):
    type: str = "trend_regime"
    horizon: int | None = None
    unit: str = "bars"
    forward_window: int | None = None


class StrategyOverrideConfig(BaseModel):
    rule_priority: list[str] | None = None
    score5_risky: list[str] | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("rule_priority", "score5_risky", mode="before")
    @classmethod
    def _normalize_symbol_list(cls, value: Any) -> Any:
        return _normalize_symbols(value, None)


@dataclass(frozen=True)
class ResolvedRunContext:
    experiment_cfg: Any | None
    market_profile: MarketProfile
    market: str
    resolved_data_dir: str | None
    resolved_symbols: list[str]
    resolved_symbol_groups: dict[str, str]
    execution_costs: dict[str, Any]
    timeframe: str
    schema: str | None
    feature_set: list[str] | None
    model_stack: list[str] | None
    target_config: dict[str, Any]
    run_identity: dict[str, Any]


class MarketProfile(BaseModel):
    name: str
    market_type: str | None = None
    data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    execution: MarketExecutionConfig = Field(default_factory=MarketExecutionConfig)
    symbols: MarketSymbolsConfig = Field(default_factory=MarketSymbolsConfig)
    strategy_overrides: dict[str, StrategyOverrideConfig] = Field(default_factory=dict)
    features: MarketFeaturesConfig = Field(default_factory=MarketFeaturesConfig)
    models: MarketModelsConfig = Field(default_factory=MarketModelsConfig)
    target: MarketTargetConfig = Field(default_factory=MarketTargetConfig)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_strategy_overrides(self) -> MarketProfile:
        if not self.strategy_overrides:
            return self
        unknown = sorted(set(self.strategy_overrides) - _registered_strategy_names())
        if unknown:
            raise ValueError(f"unknown strategy_overrides for market {self.name!r}: {unknown}")
        return self


def _read_base_market() -> str | None:
    path = _base_config_path()
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    market = cfg.get("market")
    if market is None:
        return None
    market_str = str(market).strip()
    return market_str or None


def resolve_market_name(experiment_market: str | None = None) -> str:
    """Resolve market name by priority: experiment > base.yaml > vn_stock."""
    if experiment_market is not None and str(experiment_market).strip():
        return str(experiment_market).strip()
    return _read_base_market() or "vn_stock"


def market_profile_path(market: str | None = None) -> Path:
    resolved = resolve_market_name(market)
    return markets_dir() / f"{resolved}.yaml"


def load_market_profile(market: str | None = None) -> MarketProfile:
    resolved = resolve_market_name(market)
    path = markets_dir() / f"{resolved}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Market profile not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "name" not in raw:
        raw["name"] = resolved
    return MarketProfile.model_validate(raw)


def _experiment_value(experiment_cfg: Any | None, name: str, default: Any = None) -> Any:
    if experiment_cfg is None:
        return default
    if isinstance(experiment_cfg, dict):
        return experiment_cfg.get(name, default)
    return getattr(experiment_cfg, name, default)


def _resolve_feature_set(
    experiment_cfg: Any | None, profile: MarketProfile
) -> list[str] | str | None:
    """Resolve feature set: profile.enabled_blocks > experiment string > None."""
    if profile.features.enabled_blocks is not None:
        return profile.features.enabled_blocks
    if experiment_cfg is None:
        return None
    # ExperimentConfig has feature_set() method; dict-like access for legacy
    if hasattr(experiment_cfg, "feature_set"):
        return experiment_cfg.feature_set()
    if isinstance(experiment_cfg, dict):
        signals = experiment_cfg.get("signals") or experiment_cfg.get("components") or {}
        return signals.get("features") if isinstance(signals, dict) else None
    return None


def _resolve_model_stack(experiment_cfg: Any | None, profile: MarketProfile) -> list[str] | None:
    """Resolve model stack: profile.default_stack > [experiment entry_model_type] > None."""
    if profile.models.default_stack is not None:
        return profile.models.default_stack
    if experiment_cfg is None:
        return None
    if hasattr(experiment_cfg, "entry_model_type"):
        return [experiment_cfg.entry_model_type()]
    return None


def _resolve_target_config(experiment_cfg: Any | None, profile: MarketProfile) -> dict[str, Any]:
    """Merge target config: experiment overrides per-field wins over profile defaults."""
    profile_target = profile.target.model_dump(exclude_none=True)
    if experiment_cfg is None:
        return profile_target
    if hasattr(experiment_cfg, "target_dict"):
        exp_target = experiment_cfg.target_dict()
    elif isinstance(experiment_cfg, dict):
        signals = experiment_cfg.get("signals") or experiment_cfg.get("components") or {}
        exp_target = signals.get("target", {}) if isinstance(signals, dict) else {}
    else:
        exp_target = {}
    if not exp_target:
        return profile_target
    # Experiment wins per-field; profile fills gaps
    return {**profile_target, **exp_target}


def resolve_run_context(experiment_cfg: Any | None = None) -> ResolvedRunContext:
    market = resolve_market_name(_experiment_value(experiment_cfg, "market"))
    profile = load_market_profile(market)
    execution = profile.execution.model_dump()

    feature_set = _resolve_feature_set(experiment_cfg, profile)
    model_stack = _resolve_model_stack(experiment_cfg, profile)
    target_config = _resolve_target_config(experiment_cfg, profile)

    run_identity = {
        "market": market,
        "schema": profile.data.schema,
        "timeframe": profile.data.default_timeframe,
        "symbols": profile.symbols.default_list,
        "features": feature_set,
        "target": target_config,
        "models": model_stack,
        "strategy_overrides": {
            name: override.model_dump(exclude_none=True)
            for name, override in sorted(profile.strategy_overrides.items())
        },
    }
    return ResolvedRunContext(
        experiment_cfg=experiment_cfg,
        market_profile=profile,
        market=market,
        resolved_data_dir=profile.data.data_dir,
        resolved_symbols=profile.symbols.default_list,
        resolved_symbol_groups=profile.symbols.groups,
        execution_costs=execution,
        timeframe=profile.data.default_timeframe,
        schema=profile.data.schema,
        feature_set=feature_set,
        model_stack=model_stack,
        target_config=target_config,
        run_identity=run_identity,
    )


def list_markets() -> list[str]:
    root = markets_dir()
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.yaml"))
