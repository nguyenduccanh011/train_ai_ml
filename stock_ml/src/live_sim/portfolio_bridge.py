"""Bridge from a live FrozenSignalSet to portfolio target weights.

Lets the live simulator size positions with the SAME policy library the batch
backtest uses (stock_ml.src.portfolio), guaranteeing that backtest and live agree on
how a day's continuous alpha scores become a book — the divergence the refactor set
out to remove. The live loop stays day-by-day and frozen; only the sizing is shared.
"""

from __future__ import annotations

import pandas as pd

from stock_ml.src.portfolio.factory import build_portfolio_constructor, build_portfolio_context


def frozen_to_targets(
    frozen,
    portfolio_cfg: dict | None = None,
    direction: str = "long",
) -> pd.DataFrame:
    """Convert one day's frozen scores into a single-date TargetWeightFrame.

    Args:
        frozen: a FrozenSignalSet (must carry `.scores`).
        portfolio_cfg: portfolio sub-config (policy/params/max_gross/max_per_name).
        direction: long | short | long_short | market_neutral.

    Returns:
        TargetWeightFrame for `frozen.for_execution_date`. Empty if no scores.
    """
    if not getattr(frozen, "scores", None):
        return pd.DataFrame(
            columns=["date", "symbol", "target_weight", "side", "rank", "gated", "score"]
        )
    alpha = pd.DataFrame(
        {
            "symbol": list(frozen.scores.keys()),
            "date": pd.Timestamp(frozen.for_execution_date),
            "score": list(frozen.scores.values()),
        }
    )
    constructor = build_portfolio_constructor(portfolio_cfg or {})
    ctx = build_portfolio_context(portfolio_cfg or {}, direction=direction)
    return constructor.build(alpha, ctx)
