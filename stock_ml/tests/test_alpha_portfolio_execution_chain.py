"""End-to-end chain: ML alpha -> portfolio weights -> time-stepped execution (Phase 3).

Exercises the three tiers composing on realistic model output (not hand-made scores).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.backtest.engine import EngineConfig  # noqa: E402
from src.execution import run_portfolio_backtest  # noqa: E402
from src.features.catalog import set_members  # noqa: E402
from src.features.resolver import add_features  # noqa: E402
from src.pipeline.experiment import ExperimentConfig, train_fold  # noqa: E402

FEATURE_COLS = set_members("basic_v1")
from src.portfolio.base import PortfolioContext  # noqa: E402
from src.portfolio.factory import build_portfolio_constructor  # noqa: E402
from src.targets.forward_regression import ForwardReturnRegressionTarget  # noqa: E402


def _synthetic_ohlcv(symbols, start, end, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    frames = []
    for s, sym in enumerate(symbols):
        rets = rng.normal(0.0003 + 0.0001 * s, 0.02, size=len(dates))
        close = 100.0 * np.exp(np.cumsum(rets))
        opn = close * (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
        high = np.maximum(opn, close) * 1.004
        low = np.minimum(opn, close) * 0.996
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": opn,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 100_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_alpha_to_portfolio_to_execution():
    symbols = ["A", "B", "C", "D", "E"]
    # Burn-in buffer: start before the window so basic_v1 rolling features (max
    # lookback 20 bars) are warm inside it; the window is selected by date, nothing
    # is dropped as NaN.
    bars = _synthetic_ohlcv(symbols, "2019-10-01", "2021-06-30")
    data = ForwardReturnRegressionTarget(horizon=5).apply(add_features(bars))

    train = data[(data["date"] >= "2020-01-01") & (data["date"] < "2021-01-01")]
    test = data[data["date"] >= "2021-01-01"]

    cfg = ExperimentConfig(
        name="chain",
        strategy="entry_exit",
        market="test",
        feature_set="basic_v1",
        target={"type": "forward_return_regression", "horizon": 5},
        entry_model={"type": "lightgbm", "params": {}},
        exit_model={"type": "none", "enabled": False},
        split={},
        engine={},
        seed=42,
    )
    _, _, signals = train_fold(train, test, FEATURE_COLS, cfg)

    # Tier 1 output carries the continuous alpha.
    assert "score" in signals.columns
    alpha = signals[["symbol", "date", "score"]].copy()

    # Tier 2: top-2 long book.
    constructor = build_portfolio_constructor({"policy": "top_k", "params": {"k": 2}})
    targets = constructor.build(alpha, PortfolioContext(direction="long", max_gross=1.0, max_per_name=0.6))
    assert not targets.empty
    # Each rebalance date deploys at most full gross.
    gross = targets.groupby("date")["target_weight"].apply(lambda w: w.abs().sum())
    assert (gross <= 1.0 + 1e-6).all()

    # Tier 3: capital-aware execution produces a finite NAV curve.
    trades, equity = run_portfolio_backtest(
        targets, bars, EngineConfig(max_hold_bars=10, hard_stop_pct=-0.08), initial_capital=1_000_000.0
    )
    assert not equity.empty
    assert np.isfinite(equity["nav"]).all()
    assert (equity["nav"] > 0).all()
