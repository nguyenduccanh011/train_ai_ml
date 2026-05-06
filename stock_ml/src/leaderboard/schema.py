"""LeaderboardRow canonical schema (Phase 0).

Single source of truth for leaderboard data contract.
Export JSON Schema via LeaderboardRow.model_json_schema().
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator


class TargetConfig(BaseModel):
    type: str
    forward_window: int
    gain_threshold: float | None = None
    loss_threshold: float | None = None


class CostProfile(BaseModel):
    commission: float | str = "unknown"
    tax: float | str = "unknown"
    slippage: float | str = "unknown"

    @field_validator("commission", "tax", "slippage", mode="before")
    @classmethod
    def allow_unknown(cls, v):
        if v == "unknown":
            return v
        return float(v)


class LeaderboardRow(BaseModel):
    # Identity
    run_id: str = Field(description="f'{bundle}/{run_name}#{config_hash[:8]}'")
    bundle: str
    run_name: str
    config_hash: str
    generated_at: str = Field(description="ISO-8601 timestamp")
    superseded: bool = False

    # Strategy / model identity
    market: str = "unknown"
    currency: str = "unknown"
    pnl_mode: str = "unknown"
    schema: str = "unknown"
    timeframe: str = "unknown"
    strategy: str
    feature_set: str
    entry_model: str
    exit_model_type: str
    exit_model_enabled: bool
    target: TargetConfig

    # Trading metrics (recomputed from trades.csv)
    trades: int
    wr: float
    avg_pnl: float
    total_pnl: float
    pf: float
    avg_hold: float
    sharpe: float

    # Risk
    max_drawdown: float
    mdd_per_symbol: float
    yearly_consistency: float

    # Score
    composite_score: float
    score_mode: str = Field(default="live", description="'live' | 'legacy'")

    # Fairness metadata
    n_symbols: int
    first_test_year: int
    last_test_year: int
    cost_profile: CostProfile = Field(default_factory=CostProfile)
    fairness_group_key: str = Field(
        description="sha1 of {market, currency, schema, symbols_set, window, cost_profile, target}"
    )
    is_baseline: bool = False
    same_symbols_as_baseline: bool | None = None
    same_window_as_baseline: bool | None = None
    same_cost_as_baseline: bool | None = None
    same_target_as_baseline: bool | None = None

    # Diagnostics
    warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


def export_json_schema(path: str | None = None) -> dict:
    """Export JSON Schema for the leaderboard row. Optionally write to file."""
    schema = LeaderboardRow.model_json_schema()
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
    return schema
