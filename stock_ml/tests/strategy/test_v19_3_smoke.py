"""Smoke tests for componentized v19_3 runner using synthetic prediction cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.components.runners import run_v19_3, trades_to_v19_3_dataframe


def _synthetic_cache() -> list[dict]:
    n = 90
    close = np.linspace(100, 125, n)
    close[45:60] = np.linspace(125, 108, 15)
    close[60:] = np.linspace(108, 130, n - 60)
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "close": close,
            "high": close + 0.8,
            "low": close - 0.8,
            "volume": np.full(n, 2_000_000.0),
            "symbol": ["TEST"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "rsi_slope_5d": np.full(n, 5.0),
            "vol_surge_ratio": np.full(n, 1.2),
            "range_position_20d": np.full(n, 0.4),
            "dist_to_resistance": np.full(n, 0.05),
            "breakout_setup_score": np.full(n, 4.0),
            "bb_width_percentile": np.full(n, 0.5),
            "higher_lows_count": np.full(n, 3.0),
            "obv_price_divergence": np.zeros(n),
            "return_1d": np.r_[0.0, close[1:] / close[:-1] - 1],
        }
    )
    y_pred = np.zeros(n, dtype=int)
    y_pred[10:45] = 1
    y_pred[65:80] = 1
    return [
        {
            "symbol": "TEST",
            "y_pred": y_pred,
            "returns": df["return_1d"].values,
            "sym_test_df": df,
            "feature_cols": [],
        }
    ]


def test_run_v19_3_smoke_synthetic_cache() -> None:
    trades = run_v19_3(
        symbols=["TEST"],
        data_dir="unused",
        prediction_cache=_synthetic_cache(),
    )
    df = trades_to_v19_3_dataframe(trades)

    assert not df.empty
    assert list(df.columns) == [
        "entry_day",
        "exit_day",
        "holding_days",
        "pnl_pct",
        "exit_reason",
        "exit_date",
        "entry_wp",
        "entry_dp",
        "entry_rs",
        "entry_vs",
        "entry_bs",
        "entry_hl",
        "entry_od",
        "entry_bb",
        "entry_score",
        "entry_date",
        "entry_symbol",
        "position_size",
        "entry_trend",
        "quick_reentry",
        "breakout_entry",
        "vshape_entry",
        "entry_ret_5d",
        "entry_drop20d",
        "entry_dist_sma20",
        "entry_profile",
        "entry_choppy_regime",
        "symbol",
        "max_profit_pct",
    ]
    assert df["pnl_pct"].notna().all()
    assert set(df["exit_reason"]).issubset(
        {
            "signal",
            "hard_stop",
            "signal_hard_cap",
            "fast_exit_loss",
            "stop_loss",
            "peak_protect_dist",
            "peak_protect_ema",
            "hybrid_exit",
            "trailing_stop",
            "profit_lock",
            "zombie_exit",
            "end",
        }
    )
