"""Smoke + leakage tests for the rewritten pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.backtest.engine import EngineConfig, run_backtest, trades_to_dataframe  # noqa: E402
from src.backtest.integrity import audit_report  # noqa: E402
from src.backtest.stats import aggregate_stats, per_day_stats, per_year_stats  # noqa: E402
from src.data.splitter import YearSplitter  # noqa: E402
from src.features.basic import FEATURE_COLS, add_features  # noqa: E402
from src.targets.forward import ForwardReturnTarget  # noqa: E402


def _synthetic_ohlcv(symbols: list[str], start: str, end: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    frames = []
    for s, sym in enumerate(symbols):
        drift = 0.0003 + 0.0001 * s
        vol = 0.02
        rets = rng.normal(drift, vol, size=len(dates))
        close = 100.0 * np.exp(np.cumsum(rets))
        opn = close * (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
        high = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=len(dates))))
        low = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=len(dates))))
        volume = rng.integers(1_000_00, 1_000_000, size=len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": opn,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_year_splitter_no_overlap_and_gap():
    sp = YearSplitter(
        train_years=4, test_years=1, gap_days=25, first_test_year=2020, last_test_year=2024
    )
    windows = sp.windows()
    assert len(windows) == 5
    for w in windows:
        assert w.train_end <= w.test_start
        assert (w.test_start - w.train_end).days + 1 >= 25


def test_target_does_not_peek_future():
    df = _synthetic_ohlcv(["X"], "2020-01-01", "2020-03-31")
    tgt = ForwardReturnTarget(horizon=5, gain_threshold=0.02, loss_threshold=0.02)
    out = tgt.apply(df)
    # last 5 rows per symbol have no future window -> target NaN
    tail = out.groupby("symbol").tail(5)
    assert tail["target"].isna().all()


def test_engine_next_bar_fill_no_lookahead():
    bars = _synthetic_ohlcv(["X"], "2020-01-01", "2020-02-28")
    bars = bars[bars["symbol"] == "X"].reset_index(drop=True)
    # one buy signal on the 10th bar
    sig_date = bars.loc[10, "date"]
    signals = pd.DataFrame(
        [
            {"symbol": "X", "date": sig_date, "signal": 1},
            {"symbol": "X", "date": bars.loc[20, "date"], "signal": -1},
        ]
    )
    trades = run_backtest(signals, bars, EngineConfig(max_hold_bars=50, hard_stop_pct=None))
    assert len(trades) == 1
    t = trades[0]
    # entry on bar 11 (next day after signal at bar 10)
    assert t.entry_date == bars.loc[11, "date"]
    # exit one bar after sell signal at bar 20 -> bar 21
    assert t.exit_date == bars.loc[21, "date"]


def test_audit_passes_on_clean_run():
    bars = _synthetic_ohlcv(["X"], "2020-01-01", "2020-04-30")
    bars = bars[bars["symbol"] == "X"].reset_index(drop=True)
    signals = pd.DataFrame(
        [
            {"symbol": "X", "date": bars.loc[5, "date"], "signal": 1},
            {"symbol": "X", "date": bars.loc[15, "date"], "signal": -1},
        ]
    )
    trades = run_backtest(signals, bars, EngineConfig(max_hold_bars=50, hard_stop_pct=None))
    df = trades_to_dataframe(trades)
    sp = YearSplitter(
        train_years=2, test_years=1, gap_days=25, first_test_year=2020, last_test_year=2020
    )
    report = audit_report(df, signals, windows=sp.windows(), min_gap_days=15)
    assert report["overall"] in {"PASS", "WARN"}
    assert report["n_fail"] == 0


def test_costs_applied():
    bars = pd.DataFrame(
        {
            "symbol": "X",
            "date": pd.date_range("2020-01-01", periods=10, freq="B"),
            "open": np.full(10, 100.0),
            "high": np.full(10, 100.0),
            "low": np.full(10, 100.0),
            "close": np.full(10, 100.0),
            "volume": np.full(10, 1000),
        }
    )
    signals = pd.DataFrame(
        [
            {"symbol": "X", "date": bars.loc[1, "date"], "signal": 1},
            {"symbol": "X", "date": bars.loc[5, "date"], "signal": -1},
        ]
    )
    trades = run_backtest(signals, bars, EngineConfig(max_hold_bars=50, hard_stop_pct=None))
    assert len(trades) == 1
    # flat market: gross == 0, net should equal -(round_trip_cost + slippage_effects)
    # slippage round-trip on equal prices is approximately +slippage on buy and -slippage on sell
    # so combined drag ≈ commission*2 + tax + slippage*2
    drag = 2 * 0.0015 + 0.001 + 2 * 0.0010
    # allow tiny multiplicative correction
    assert trades[0].pnl_pct < 0
    assert abs(trades[0].pnl_pct + drag) < 1e-3


def test_features_pipeline_runs():
    bars = _synthetic_ohlcv(["A", "B"], "2019-01-01", "2020-06-30")
    feat = add_features(bars)
    assert set(FEATURE_COLS).issubset(feat.columns)
    # last row per symbol should have non-NaN features (sufficient warmup)
    last = feat.groupby("symbol").tail(1)
    assert last[FEATURE_COLS].notna().all().all()


def test_per_day_and_per_year_stats():
    trades = pd.DataFrame(
        [
            {
                "symbol": "X",
                "entry_date": "2020-01-10",
                "exit_date": "2020-01-15",
                "pnl_pct": 0.05,
                "holding_days": 3,
            },
            {
                "symbol": "X",
                "entry_date": "2020-01-16",
                "exit_date": "2020-01-20",
                "pnl_pct": -0.02,
                "holding_days": 2,
            },
            {
                "symbol": "Y",
                "entry_date": "2021-02-01",
                "exit_date": "2021-02-10",
                "pnl_pct": 0.03,
                "holding_days": 5,
            },
        ]
    )
    daily = per_day_stats(trades)
    yearly = per_year_stats(trades)
    assert len(daily) == 3
    assert "win_rate" in daily.columns
    assert set(yearly["year"]) == {2020, 2021}
    agg = aggregate_stats(trades)
    assert agg["n_trades"] == 3
