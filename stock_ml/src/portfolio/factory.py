"""Build a portfolio constructor + context from a config dict.

Config shape (lives under engine_config["portfolio"]):

    portfolio:
      enabled: true
      policy: top_k            # threshold_binary | score_proportional | top_k
                               # | market_neutral | vol_target | kelly
      params: {k: 10, weighting: equal}
      max_gross: 1.0
      max_per_name: 0.1
      direction: long          # falls back to the template-level direction
"""

from __future__ import annotations

from stock_ml.src.portfolio.base import PortfolioConstructor, PortfolioContext
from stock_ml.src.portfolio.policies import (
    KellyFractionalPolicy,
    MarketNeutralPolicy,
    ScoreProportionalPolicy,
    ThresholdBinaryPolicy,
    TopKPolicy,
    VolTargetPolicy,
)

_POLICIES = {
    "threshold_binary": ThresholdBinaryPolicy,
    "score_proportional": ScoreProportionalPolicy,
    "top_k": TopKPolicy,
    "market_neutral": MarketNeutralPolicy,
    "vol_target": VolTargetPolicy,
    "kelly": KellyFractionalPolicy,
}


def build_portfolio_constructor(portfolio_cfg: dict) -> PortfolioConstructor:
    """Instantiate the configured sizing policy. Defaults to threshold_binary."""
    policy = (portfolio_cfg or {}).get("policy", "threshold_binary")
    params = (portfolio_cfg or {}).get("params", {}) or {}
    if policy not in _POLICIES:
        raise ValueError(
            f"Unknown portfolio policy '{policy}'. Available: {sorted(_POLICIES)}"
        )
    return _POLICIES[policy](**params)


def build_portfolio_context(
    portfolio_cfg: dict,
    direction: str = "long",
    regime_signal=None,
    size_signal=None,
    ohlcv=None,
) -> PortfolioContext:
    """Assemble the shared PortfolioContext. Policy-level direction overrides template."""
    cfg = portfolio_cfg or {}
    return PortfolioContext(
        direction=cfg.get("direction", direction),
        max_gross=cfg.get("max_gross", 1.0),
        max_per_name=cfg.get("max_per_name", 1.0),
        regime_signal=regime_signal,
        size_signal=size_signal,
        ohlcv=ohlcv,
    )
